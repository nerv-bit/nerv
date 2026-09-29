//! Two-adic evaluation domains over Goldilocks (WP §5.3): the trace
//! subgroup H and its cosets. p − 1 = 2^32·(2^32 − 1), so the field
//! contains exactly one subgroup of each order 2^n (n ≤ 32), and they
//! nest: H_m ⊇ H_n iff m ≥ n.
//!
//! Canonical generators: ω_n = 7^((p−1)/2^n) — order exactly 2^n (pinned:
//! ω^(2^(n−1)) = −1, the field's unique order-2 element; follows from 7
//! being a quadratic non-residue, Euler-pinned in ext_field's tests).
//! The standard coset shift is ω_32: outside every proper two-adic
//! subgroup, so `standard_coset(m)` is disjoint from every H_n, n ≤ m —
//! the disjointness the quotient evaluation requires (Z_H ≠ 0 at every
//! evaluation point; no Lagrange denominator vanishes).
//!
//! FRI pairing on any domain of size M ≥ 2 is index (i, i + M/2)
//! (ω^(M/2) = −1); folding maps c·⟨ω⟩ onto c²·⟨ω²⟩ (`halve`) — no
//! bit-reversal anywhere in the engine.

use crate::stark::ext_field::ExtF;
use nerv_core::field::{Goldilocks, GOLDILOCKS_PRIME};

pub const TWO_ADICITY: usize = 32;

const _: () = assert!((GOLDILOCKS_PRIME - 1).trailing_zeros() == 32);

const fn const_pow(mut base: Goldilocks, mut exp: u64) -> Goldilocks {
    let mut acc = Goldilocks::ONE;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = acc.const_mul(base);
        }
        base = base.const_mul(base);
        exp >>= 1;
    }
    acc
}

/// The order-2^32 subgroup generator: 7^((p−1)/2^32) = 7^(2^32 − 1).
pub const FULL_TWO_ADIC_GEN: Goldilocks = const_pow(Goldilocks::from_u32(7), (1u64 << 32) - 1);

/// ω_n — the canonical generator of the order-2^log_size subgroup.
pub fn two_adic_generator(log_size: usize) -> Goldilocks {
    assert!(log_size <= TWO_ADICITY, "log_size {log_size} exceeds two-adicity {TWO_ADICITY}");
    FULL_TWO_ADIC_GEN.pow(1u64 << (TWO_ADICITY - log_size))
}

/// A two-adic evaluation domain: the points {shift·gen^i : i < 2^log_size}.
/// `shift = 1` is the subgroup itself; any other nonzero shift is a coset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwoAdicDomain {
    log_size: usize,
    gen: Goldilocks,
    shift: Goldilocks,
}

impl TwoAdicDomain {
    pub fn subgroup(log_size: usize) -> TwoAdicDomain {
        TwoAdicDomain { log_size, gen: two_adic_generator(log_size), shift: Goldilocks::ONE }
    }

    pub fn coset(log_size: usize, shift: Goldilocks) -> TwoAdicDomain {
        assert!(!shift.is_zero(), "coset shift must be nonzero");
        TwoAdicDomain { log_size, gen: two_adic_generator(log_size), shift }
    }

    /// ω_32·H — disjoint from every proper two-adic subgroup, hence from
    /// every trace subgroup it extends.
    pub fn standard_coset(log_size: usize) -> TwoAdicDomain {
        assert!(
            log_size < TWO_ADICITY,
            "the standard shift lies inside the full subgroup"
        );
        TwoAdicDomain::coset(log_size, FULL_TWO_ADIC_GEN)
    }

    /// The canonical LDE/FRI commitment domain for this trace domain at the
    /// given blowup: standard_coset(log_size + log_blowup).
    pub fn lde_coset(&self, log_blowup: usize) -> TwoAdicDomain {
        TwoAdicDomain::standard_coset(self.log_size + log_blowup)
    }

    pub const fn log_size(&self) -> usize {
        self.log_size
    }

    pub const fn size(&self) -> usize {
        1usize << self.log_size
    }

    pub const fn gen(&self) -> Goldilocks {
        self.gen
    }

    pub const fn shift(&self) -> Goldilocks {
        self.shift
    }

    /// The i-th point: shift·gen^i (i < size).
    pub fn point(&self, i: usize) -> Goldilocks {
        assert!(i < self.size(), "index {i} out of range");
        self.shift * self.gen.pow(i as u64)
    }

    /// All points in index order: shift, shift·gen, shift·gen², …
    pub fn points(&self) -> impl Iterator<Item = Goldilocks> {
        let mut cur = self.shift;
        let g = self.gen;
        std::iter::from_fn(move || {
            let out = cur;
            cur = cur * g;
            Some(out)
        })
        .take(self.size())
    }

    /// The index paired with i under x ↦ −x (size ≥ 2): i + size/2.
    pub fn sibling_index(&self, i: usize) -> usize {
        assert!(self.log_size >= 1, "the trivial domain has no negation pairing");
        assert!(i < self.size(), "index {i} out of range");
        i + self.size() / 2
    }

    /// c²·⟨gen²⟩ — the folded domain (size ≥ 2).
    pub fn halve(&self) -> TwoAdicDomain {
        assert!(self.log_size >= 1, "cannot halve the trivial domain");
        TwoAdicDomain {
            log_size: self.log_size - 1,
            gen: self.gen * self.gen,
            shift: self.shift * self.shift,
        }
    }

    /// x^size − 1: the vanishing polynomial of the underlying subgroup
    /// (the domain's own, for subgroups).
    pub fn vanishing(&self, x: Goldilocks) -> Goldilocks {
        x.pow(self.size() as u64) - Goldilocks::ONE
    }

    /// x^size − 1 lifted to the challenge field (the DEEP denominator check).
    pub fn vanishing_ext(&self, x: ExtF) -> ExtF {
        x.pow(self.size() as u64) - ExtF::ONE
    }

    /// Membership in the underlying subgroup (x^size = 1). Coset points of
    /// a disjoint coset are never members.
    pub fn contains(&self, x: Goldilocks) -> bool {
        self.vanishing(x).is_zero()
    }

    /// The index step equal to multiplication by the trace generator
    /// ω_{log_trace}: point(j + step) = point(j)·ω_{log_trace}, indices
    /// wrapping mod size. Requires log_trace ≤ log_size (subgroups nest).
    pub fn shift_step_for(&self, log_trace: usize) -> usize {
        assert!(log_trace <= self.log_size, "trace subgroup is larger than this domain");
        1usize << (self.log_size - log_trace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;

    fn fe(rng: &mut SplitMix64) -> Goldilocks {
        Goldilocks::from_u64_reduce(rng.next_u64())
    }

    #[test]
    fn generator_orders_are_exact() {
        assert_eq!(two_adic_generator(0), Goldilocks::ONE);
        assert_eq!(FULL_TWO_ADIC_GEN, two_adic_generator(32));
        let minus_one = Goldilocks::from_u64_reduce(GOLDILOCKS_PRIME - 1);
        for log in 0..=20usize {
            let g = two_adic_generator(log);
            assert_eq!(g.pow(1u64 << log), Goldilocks::ONE, "log={log}");
            if log >= 1 {
                assert_eq!(g.pow(1u64 << (log - 1)), minus_one, "log={log}");
            }
        }
        assert_eq!(FULL_TWO_ADIC_GEN.pow(1u64 << 31), minus_one);
    }

    #[test]
    fn subgroups_nest() {
        for big in 1..=12usize {
            for small in 0..=big {
                assert!(
                    TwoAdicDomain::subgroup(big).contains(two_adic_generator(small)),
                    "H_{small} not inside H_{big}"
                );
            }
        }
    }

    #[test]
    fn points_agree_with_pow_and_are_distinct() {
        for log in 0..=10usize {
            for dom in [TwoAdicDomain::subgroup(log), TwoAdicDomain::standard_coset(log)] {
                let pts: Vec<Goldilocks> = dom.points().collect();
                assert_eq!(pts.len(), dom.size());
                for (i, &p) in pts.iter().enumerate() {
                    assert_eq!(p, dom.point(i), "log={log} i={i}");
                }
                let mut sorted = pts.clone();
                sorted.sort();
                sorted.dedup();
                assert_eq!(sorted.len(), pts.len(), "log={log}: duplicate points");
            }
        }
    }

    #[test]
    fn subgroup_membership_and_vanishing() {
        for log in 0..=8usize {
            let d = TwoAdicDomain::subgroup(log);
            for p in d.points() {
                assert!(d.contains(p));
                assert!(d.vanishing(p).is_zero());
            }
            // ord(7) has full 2-part 2^32, so 7 lies in no H_log, log < 32.
            let seven = Goldilocks::from_u32(7);
            assert!(!d.contains(seven));
            assert!(!d.vanishing(seven).is_zero());
        }
    }

    #[test]
    fn standard_cosets_are_disjoint_from_subgroups() {
        for log in 0..=20usize {
            assert!(!TwoAdicDomain::subgroup(log).contains(FULL_TWO_ADIC_GEN));
        }
        for log_big in 0..=10usize {
            let cos = TwoAdicDomain::standard_coset(log_big);
            for log_small in 0..=log_big {
                let h = TwoAdicDomain::subgroup(log_small);
                for p in cos.points() {
                    assert!(!h.contains(p), "coset point landed in H_{log_small}");
                }
            }
        }
    }

    #[test]
    fn coset_points_start_at_shift() {
        let c = two_adic_generator(20);
        let d = TwoAdicDomain::coset(5, c);
        assert_eq!(d.point(0), c);
        assert_eq!(d.point(1), c * two_adic_generator(5));
        assert_eq!(d.points().next(), Some(c));
    }

    #[test]
    fn sibling_is_negation() {
        for log in 1..=10usize {
            for dom in [TwoAdicDomain::subgroup(log), TwoAdicDomain::standard_coset(log)] {
                let pts: Vec<Goldilocks> = dom.points().collect();
                for (i, &p) in pts.iter().enumerate() {
                    assert_eq!(pts[dom.sibling_index(i)], -p, "log={log} i={i}");
                }
            }
        }
    }

    #[test]
    fn halve_squares_points() {
        for log in 1..=9usize {
            for parent in [TwoAdicDomain::subgroup(log), TwoAdicDomain::standard_coset(log)] {
                let child = parent.halve();
                assert_eq!(child.log_size(), log - 1);
                assert_eq!(child.size(), parent.size() / 2);
                let half = child.size();
                let pts: Vec<Goldilocks> = parent.points().collect();
                for (i, &x) in pts.iter().enumerate() {
                    assert_eq!(x * x, child.point(i % half), "log={log} i={i}");
                }
                if parent.shift() == Goldilocks::ONE {
                    assert_eq!(child.shift(), Goldilocks::ONE);
                } else {
                    assert_eq!(child.shift(), parent.shift() * parent.shift());
                }
            }
        }
    }

    #[test]
    fn shift_step_matches_trace_generator() {
        for log in 0..=8usize {
            assert_eq!(TwoAdicDomain::subgroup(log).shift_step_for(log), 1);
        }
        let mut rng = SplitMix64::new(0xD0);
        for log_trace in 1..=6usize {
            for blowup in 1..=3usize {
                let lde = TwoAdicDomain::subgroup(log_trace).lde_coset(blowup);
                let g = two_adic_generator(log_trace);
                let step = lde.shift_step_for(log_trace);
                assert_eq!(step, 1usize << blowup);
                let m = lde.size();
                for _ in 0..64 {
                    let j = (rng.next_u64() as usize) % m;
                    let x = lde.point(j);
                    let nxt = lde.point((j + step) % m);
                    assert_eq!(nxt, x * g, "log_trace={log_trace} blowup={blowup} j={j}");
                }
            }
        }
    }

    #[test]
    fn vanishing_ext_agrees_with_base() {
        let mut rng = SplitMix64::new(0xE7);
        for log in 0..=8usize {
            let d = TwoAdicDomain::subgroup(log);
            for _ in 0..16 {
                let b = fe(&mut rng);
                assert_eq!(d.vanishing_ext(ExtF::from_base(b)), ExtF::from_base(d.vanishing(b)));
            }
            for p in d.points() {
                assert!(d.vanishing_ext(ExtF::from_base(p)).is_zero());
            }
        }
    }

    #[test]
    fn lde_coset_is_the_standard_coset_of_the_sum() {
        let trace = TwoAdicDomain::subgroup(10);
        let lde = trace.lde_coset(2);
        assert_eq!(lde, TwoAdicDomain::standard_coset(12));
        assert_eq!(lde.shift_step_for(10), 4);
    }

    #[test]
    #[should_panic(expected = "exceeds two-adicity")]
    fn generator_rejects_oversized_log() {
        let _ = two_adic_generator(33);
    }

    #[test]
    #[should_panic(expected = "exceeds two-adicity")]
    fn subgroup_rejects_oversized_log() {
        let _ = TwoAdicDomain::subgroup(33);
    }

    #[test]
    #[should_panic(expected = "nonzero")]
    fn coset_rejects_zero_shift() {
        let _ = TwoAdicDomain::coset(4, Goldilocks::ZERO);
    }

    #[test]
    #[should_panic(expected = "proper two-adic subgroup")]
    fn standard_coset_rejects_full_log() {
        let _ = TwoAdicDomain::standard_coset(32);
    }

    #[test]
    #[should_panic]
    fn lde_coset_rejects_overflow() {
        let _ = TwoAdicDomain::subgroup(32).lde_coset(1);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn point_rejects_out_of_range() {
        let _ = TwoAdicDomain::subgroup(3).point(8);
    }

    #[test]
    #[should_panic(expected = "no negation pairing")]
    fn sibling_rejects_trivial_domain() {
        let _ = TwoAdicDomain::subgroup(0).sibling_index(0);
    }

    #[test]
    #[should_panic(expected = "cannot halve")]
    fn halve_rejects_trivial_domain() {
        let _ = TwoAdicDomain::subgroup(0).halve();
    }

    #[test]
    #[should_panic(expected = "larger than this domain")]
    fn shift_step_rejects_oversized_trace() {
        let _ = TwoAdicDomain::subgroup(3).shift_step_for(4);
    }
}

