//! Radix-2 DIT FFT over two-adic coset domains (WP §5.3): the evaluation
//! engine for trace LDEs and quotient chunks, over Goldilocks and the
//! challenge field alike.
//!
//! Coset law: a polynomial's coefficient j is scaled by shift^j before the
//! subgroup transform (evaluation at shift·g^i), and unscaled by shift^{-j}
//! after the inverse. Output index i is dom.point(i) — pinned by Horner
//! differentials in both fields (DSR-7).

use crate::stark::domain::TwoAdicDomain;
use crate::stark::ext_field::ExtF;
use nerv_core::field::Goldilocks;

/// Values the FFT runs over: the base field and the challenge field.
/// Twiddle factors are always base-field scalars.
pub trait FftValue: Copy + Sized {
    fn zero() -> Self;
    fn add(self, rhs: Self) -> Self;
    fn sub(self, rhs: Self) -> Self;
    fn scale(self, g: Goldilocks) -> Self;
}

impl FftValue for Goldilocks {
    fn zero() -> Self {
        Goldilocks::ZERO
    }
    fn add(self, rhs: Self) -> Self {
        Goldilocks::add(self, rhs)
    }
    fn sub(self, rhs: Self) -> Self {
        Goldilocks::sub(self, rhs)
    }
    fn scale(self, g: Goldilocks) -> Self {
        self.mul(g)
    }
}

impl FftValue for ExtF {
    fn zero() -> Self {
        ExtF::ZERO
    }
    fn add(self, rhs: Self) -> Self {
        ExtF::add(self, rhs)
    }
    fn sub(self, rhs: Self) -> Self {
        ExtF::sub(self, rhs)
    }
    fn scale(self, g: Goldilocks) -> Self {
        ExtF::scale(self, g)
    }
}

fn bit_reverse(i: usize, log: usize) -> usize {
    (i as u32).reverse_bits() as usize >> (32 - log)
}

fn bit_reverse_permute<V>(vals: &mut [V]) {
    let n = vals.len();
    if n <= 1 {
        return;
    }
    let log = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_reverse(i, log);
        if i < j {
            vals.swap(i, j);
        }
    }
}

/// In-place DIT: input `vals` as coefficients in natural order, output the
/// evaluations at gen^i. `gen` must have order exactly `vals.len()`.
fn dit_inplace<V: FftValue>(vals: &mut [V], gen: Goldilocks) {
    let n = vals.len();
    assert!(n.is_power_of_two() && n <= (1 << 32), "FFT length must be a power of two");
    if n == 1 {
        return;
    }
    debug_assert_eq!(gen.pow(n as u64), Goldilocks::ONE);
    debug_assert_ne!(gen.pow(n as u64 / 2), Goldilocks::ONE);
    bit_reverse_permute(vals);
    let mut m = 2usize;
    while m <= n {
        let half = m >> 1;
        let w_step = gen.pow((n / m) as u64);
        for start in (0..n).step_by(m) {
            let mut w = Goldilocks::ONE;
            for k in 0..half {
                let u = vals[start + k];
                let t = vals[start + k + half].scale(w);
                vals[start + k] = u.add(t);
                vals[start + k + half] = u.sub(t);
                w = w * w_step;
            }
        }
        m <<= 1;
    }
}

/// Evaluate the polynomial with `coeffs` (natural order, zero-padded) over
/// `dom`: output i is the value at dom.point(i). `coeffs.len() <= dom.size()`.
pub fn evaluate<V: FftValue>(coeffs: &[V], dom: &TwoAdicDomain) -> Vec<V> {
    let size = dom.size();
    assert!(
        coeffs.len() <= size,
        "polynomial with {} coefficients exceeds the domain of size {size}",
        coeffs.len()
    );
    let mut vals = Vec::with_capacity(size);
    vals.extend_from_slice(coeffs);
    vals.resize(size, V::zero());
    let s = dom.shift();
    if s != Goldilocks::ONE {
        let mut spow = Goldilocks::ONE;
        for v in vals.iter_mut() {
            *v = v.scale(spow);
            spow = spow * s;
        }
    }
    dit_inplace(&mut vals, dom.gen());
    vals
}

/// Interpolate `evals` (index i = value at dom.point(i), exactly
/// dom.size() of them) into natural-order coefficients.
pub fn interpolate<V: FftValue>(evals: &[V], dom: &TwoAdicDomain) -> Vec<V> {
    let size = dom.size();
    assert_eq!(
        evals.len(),
        size,
        "interpolation needs exactly {size} evaluations"
    );
    let mut vals = evals.to_vec();
    if size > 1 {
        dit_inplace(&mut vals, dom.gen().inverse());
    }
    let n_inv = Goldilocks::from_u64_reduce(size as u64).inverse();
    let s = dom.shift();
    if s == Goldilocks::ONE {
        for v in vals.iter_mut() {
            *v = v.scale(n_inv);
        }
    } else {
        let s_inv = s.inverse();
        let mut scalar = n_inv;
        for v in vals.iter_mut() {
            *v = v.scale(scalar);
            scalar = scalar * s_inv;
        }
    }
    vals
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;

    fn gf(rng: &mut SplitMix64) -> Goldilocks {
        Goldilocks::from_u64_reduce(rng.next_u64())
    }

    fn ef(rng: &mut SplitMix64) -> ExtF {
        ExtF::new(gf(rng), gf(rng))
    }

    fn random_coset(rng: &mut SplitMix64, log: usize) -> TwoAdicDomain {
        let mut s = gf(rng);
        while s.is_zero() {
            s = gf(rng);
        }
        TwoAdicDomain::coset(log, s)
    }

    fn naive_eval_g(coeffs: &[Goldilocks], dom: &TwoAdicDomain) -> Vec<Goldilocks> {
        dom.points()
            .map(|x| {
                let mut acc = Goldilocks::ZERO;
                for &c in coeffs.iter().rev() {
                    acc = acc * x + c;
                }
                acc
            })
            .collect()
    }

    fn naive_eval_e(coeffs: &[ExtF], dom: &TwoAdicDomain) -> Vec<ExtF> {
        dom.points()
            .map(|x| {
                let x = ExtF::from_base(x);
                let mut acc = ExtF::ZERO;
                for &c in coeffs.iter().rev() {
                    acc = acc * x + c;
                }
                acc
            })
            .collect()
    }

    #[test]
    fn bit_reverse_pins() {
        assert_eq!(bit_reverse(0, 0), 0);
        assert_eq!(bit_reverse(0, 3), 0);
        assert_eq!(bit_reverse(1, 3), 4);
        assert_eq!(bit_reverse(3, 3), 6);
        assert_eq!(bit_reverse(4, 3), 1);
        assert_eq!(bit_reverse(6, 3), 3);
        assert_eq!(bit_reverse(7, 3), 7);
        for log in 1..=12usize {
            for i in 0..(1usize << log) {
                assert_eq!(bit_reverse(bit_reverse(i, log), log), i, "log={log} i={i}");
            }
        }
    }

    #[test]
    fn naive_dft_differential_goldilocks() {
        let mut rng = SplitMix64::new(0xDA1);
        for log in 0..=8usize {
            let n = 1usize << log;
            for dom in [
                TwoAdicDomain::subgroup(log),
                TwoAdicDomain::standard_coset(log),
                random_coset(&mut rng, log),
            ] {
                let coeffs: Vec<Goldilocks> = (0..n).map(|_| gf(&mut rng)).collect();
                assert_eq!(evaluate(&coeffs, &dom), naive_eval_g(&coeffs, &dom), "log={log}");
            }
        }
    }

    #[test]
    fn naive_dft_differential_ext() {
        let mut rng = SplitMix64::new(0xDA2);
        for log in 0..=6usize {
            let n = 1usize << log;
            for dom in [
                TwoAdicDomain::subgroup(log),
                TwoAdicDomain::standard_coset(log),
                random_coset(&mut rng, log),
            ] {
                let coeffs: Vec<ExtF> = (0..n).map(|_| ef(&mut rng)).collect();
                assert_eq!(evaluate(&coeffs, &dom), naive_eval_e(&coeffs, &dom), "log={log}");
            }
        }
    }

    #[test]
    fn round_trip_both_directions_both_fields() {
        let mut rng = SplitMix64::new(0xF37C_A1B2_C3D4_E5F6);
        for log in 0..=12usize {
            let n = 1usize << log;
            for dom in [
                TwoAdicDomain::subgroup(log),
                TwoAdicDomain::standard_coset(log),
                random_coset(&mut rng, log),
            ] {
                let cg: Vec<Goldilocks> = (0..n).map(|_| gf(&mut rng)).collect();
                assert_eq!(interpolate(&evaluate(&cg, &dom), &dom), cg, "g log={log}");
                let eg: Vec<Goldilocks> = (0..n).map(|_| gf(&mut rng)).collect();
                assert_eq!(evaluate(&interpolate(&eg, &dom), &dom), eg, "g log={log}");

                let ce: Vec<ExtF> = (0..n).map(|_| ef(&mut rng)).collect();
                assert_eq!(interpolate(&evaluate(&ce, &dom), &dom), ce, "e log={log}");
                let ee: Vec<ExtF> = (0..n).map(|_| ef(&mut rng)).collect();
                assert_eq!(evaluate(&interpolate(&ee, &dom), &dom), ee, "e log={log}");
            }
        }
    }

    #[test]
    fn lde_matches_horner_on_the_coset() {
        let mut rng = SplitMix64::new(0x1D);
        for log_trace in 0..=6usize {
            for blowup in 1..=3usize {
                let trace = TwoAdicDomain::subgroup(log_trace);
                let lde = trace.lde_coset(blowup);
                let n = trace.size();
                let evals: Vec<Goldilocks> = (0..n).map(|_| gf(&mut rng)).collect();
                let coeffs = interpolate(&evals, &trace);
                let ext = evaluate(&coeffs, &lde);
                assert_eq!(ext.len(), lde.size());
                for _ in 0..64 {
                    let i = (rng.next_u64() as usize) % lde.size();
                    let x = lde.point(i);
                    let mut acc = Goldilocks::ZERO;
                    for &c in coeffs.iter().rev() {
                        acc = acc * x + c;
                    }
                    assert_eq!(ext[i], acc, "log={log_trace} blowup={blowup} i={i}");
                }
            }
        }
    }

    #[test]
    fn padding_is_the_zero_extension() {
        let mut rng = SplitMix64::new(0xB4D);
        let dom = TwoAdicDomain::standard_coset(6);
        let full: Vec<Goldilocks> = (0..dom.size()).map(|_| gf(&mut rng)).collect();
        for k in [0usize, 1, 7, 63, 64] {
            let got = evaluate(&full[..k], &dom);
            let mut padded = full[..k].to_vec();
            padded.resize(dom.size(), Goldilocks::ZERO);
            assert_eq!(got, evaluate(&padded, &dom), "k={k}");
            assert_eq!(got, naive_eval_g(&full[..k], &dom), "k={k}");
        }
        // the empty coefficient list is the zero polynomial
        assert!(evaluate::<Goldilocks>(&[], &dom).iter().all(|v| v.is_zero()));
    }

    #[test]
    fn basis_and_constant_pins() {
        // X over subgroup(1): evals [1, g]; over coset(1, s): [s, s·g].
        let g = TwoAdicDomain::subgroup(1).gen();
        let poly_x = [Goldilocks::ZERO, Goldilocks::ONE];
        assert_eq!(evaluate(&poly_x, &TwoAdicDomain::subgroup(1)), vec![Goldilocks::ONE, g]);
        let s = TwoAdicDomain::subgroup(5).gen();
        assert_eq!(
            evaluate(&poly_x, &TwoAdicDomain::coset(1, s)),
            vec![s, s * g]
        );
        // inverse pin: interpolating X's evals recovers X.
        assert_eq!(
            interpolate(&[Goldilocks::ONE, g], &TwoAdicDomain::subgroup(1)),
            poly_x
        );
        // degree-0 polynomial: every point evaluates to the constant.
        let mut rng = SplitMix64::new(5);
        let c = gf(&mut rng);
        for dom in [TwoAdicDomain::subgroup(5), TwoAdicDomain::standard_coset(5)] {
            assert!(evaluate(&[c], &dom).iter().all(|&v| v == c));
        }
    }

    #[test]
    fn size_one_is_identity() {
        let mut rng = SplitMix64::new(0x11);
        let a = gf(&mut rng);
        let mut s = gf(&mut rng);
        while s.is_zero() {
            s = gf(&mut rng);
        }
        for dom in [TwoAdicDomain::subgroup(0), TwoAdicDomain::coset(0, s)] {
            assert_eq!(evaluate(&[a], &dom), vec![a]);
            assert_eq!(interpolate(&[a], &dom), vec![a]);
        }
        // the zero polynomial on the trivial domain
        assert_eq!(evaluate::<Goldilocks>(&[], &TwoAdicDomain::subgroup(0)), vec![Goldilocks::ZERO]);
    }

    #[test]
    #[should_panic(expected = "exceeds the domain")]
    fn evaluate_rejects_oversized_polynomial() {
        let dom = TwoAdicDomain::subgroup(2);
        let _ = evaluate(&[Goldilocks::ONE; 5], &dom);
    }

    #[test]
    #[should_panic(expected = "exactly 4 evaluations")]
    fn interpolate_rejects_wrong_count() {
        let dom = TwoAdicDomain::subgroup(2);
        let _ = interpolate(&[Goldilocks::ONE; 3], &dom);
    }
}

