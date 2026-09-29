//! NERV-Poseidon2-G16 — the frozen tree-hash permutation (native reference).
//!
//! Structure per the published Poseidon2 design over the Goldilocks prover
//! field (WP §5.3): state width t = 16; S-box x^7; 8 full external rounds
//! (4 + 4) with M_E = (J4 + I4) ⊗ M4; 56 partial internal rounds with
//! M_I = blockdiag(M4 × 4); M4 = I4 + J4; round constants added to the full
//! state every round. Constants and per-domain capacity IVs derive from
//! BLAKE3-XOF ("nothing up my sleeve"), pinned by the M1 conformance freeze.
//!
//! DSR-7: this module IS the reference. The in-circuit Merkle chip
//! (nerv-proofs) must match it bit-for-bit; it is a [BUILD] of this
//! permutation, not a p3-poseidon2 wrap, because plonky3's precomputed
//! constants do not pin NERV's frozen parameterization.

use std::sync::OnceLock;
use nerv_core::constants::{NCT_EMPTY, NCT_LEAF, NCT_NODE, POSEIDON2};
use nerv_core::field::Goldilocks;
use nerv_core::hash::Xof;
use nerv_core::params::{
    PROOFS_FRI_POSEIDON2_FULL_ROUNDS, PROOFS_FRI_POSEIDON2_PARTIAL_ROUNDS,
    PROOFS_FRI_POSEIDON2_SBOX_DEGREE, PROOFS_FRI_POSEIDON2_T,
};

pub const T: usize = PROOFS_FRI_POSEIDON2_T as usize;
pub const FULL_ROUNDS: usize = PROOFS_FRI_POSEIDON2_FULL_ROUNDS as usize;
pub const PARTIAL_ROUNDS: usize = PROOFS_FRI_POSEIDON2_PARTIAL_ROUNDS as usize;
pub const ROUNDS: usize = FULL_ROUNDS + PARTIAL_ROUNDS;

const _: () = assert!(T == 16);
const _: () = assert!(FULL_ROUNDS >= 2 && FULL_ROUNDS % 2 == 0);
const _: () = assert!(PARTIAL_ROUNDS >= 1);
const _: () = assert!(PROOFS_FRI_POSEIDON2_SBOX_DEGREE == 7);
const HALF: usize = FULL_ROUNDS / 2;

static ROUND_CONSTANTS: OnceLock<Vec<[Goldilocks; T]>> = OnceLock::new();

pub fn round_constants() -> &'static Vec<[Goldilocks; T]> {
    ROUND_CONSTANTS.get_or_init(|| {
        let mut xof = Xof::new(&POSEIDON2, b"nerv.poseidon2.rounds.v1");
        (0..ROUNDS)
            .map(|_| {
                let mut r = [Goldilocks::ZERO; T];
                for e in r.iter_mut() {
                    *e = Goldilocks::from_u64_reduce(xof.next_u64());
                }
                r
            })
            .collect()
    })
}

/// Capacity IV for a domain: first four XOF-derived field elements.
pub fn domain_iv(domain: &nerv_core::constants::Domain) -> [Goldilocks; 4] {
    let mut xof = Xof::new(domain, b"nerv.poseidon2.iv.v1");
    let mut iv = [Goldilocks::ZERO; 4];
    for e in iv.iter_mut() {
        *e = Goldilocks::from_u64_reduce(xof.next_u64());
    }
    iv
}

static LEAF_IV: OnceLock<[Goldilocks; 4]> = OnceLock::new();
static NODE_IV: OnceLock<[Goldilocks; 4]> = OnceLock::new();
static EMPTY_IV: OnceLock<[Goldilocks; 4]> = OnceLock::new();

pub fn leaf_iv() -> &'static [Goldilocks; 4] {
    LEAF_IV.get_or_init(|| domain_iv(&NCT_LEAF))
}
pub fn node_iv() -> &'static [Goldilocks; 4] {
    NODE_IV.get_or_init(|| domain_iv(&NCT_NODE))
}
pub fn empty_iv() -> &'static [Goldilocks; 4] {
    EMPTY_IV.get_or_init(|| domain_iv(&NCT_EMPTY))
}

fn sbox(x: Goldilocks) -> Goldilocks {
    x.pow7()
}

/// M4 = I4 + J4: v_i ↦ v_i + Σv.
fn m4(v: &mut [Goldilocks]) {
    let t = v[0] + v[1] + v[2] + v[3];
    for e in v.iter_mut() {
        *e = *e + t;
    }
}

/// M_I = blockdiag(M4, M4, M4, M4).
fn internal_linear(x: &mut [Goldilocks; T]) {
    for g in 0..4 {
        m4(&mut x[g * 4..g * 4 + 4]);
    }
}

/// M_E = (J4 + I4) ⊗ M4: y_g = M4(x_g + S) with S the element-wise group sum.
fn external_linear(x: &mut [Goldilocks; T]) {
    let mut s = [Goldilocks::ZERO; 4];
    for g in 0..4 {
        for i in 0..4 {
            s[i] = s[i] + x[g * 4 + i];
        }
    }
    for g in 0..4 {
        let mut v = [Goldilocks::ZERO; 4];
        for i in 0..4 {
            v[i] = x[g * 4 + i] + s[i];
        }
        m4(&mut v);
        x[g * 4..g * 4 + 4].copy_from_slice(&v);
    }
}

fn full_round(x: &mut [Goldilocks; T], rc: &[Goldilocks; T]) {
    for i in 0..T {
        x[i] = x[i] + rc[i];
    }
    for e in x.iter_mut() {
        *e = sbox(*e);
    }
    external_linear(x);
}

fn partial_round(x: &mut [Goldilocks; T], rc: &[Goldilocks; T]) {
    for i in 0..T {
        x[i] = x[i] + rc[i];
    }
    x[0] = sbox(x[0]);
    internal_linear(x);
}

/// The permutation P. Deterministic, total, platform-stable.
pub fn permute(state: &mut [Goldilocks; T]) {
    let rc = round_constants();
    for r in 0..HALF {
        full_round(state, &rc[r]);
    }
    for r in HALF..HALF + PARTIAL_ROUNDS {
        partial_round(state, &rc[r]);
    }
    for r in HALF + PARTIAL_ROUNDS..ROUNDS {
        full_round(state, &rc[r]);
    }
}

/// One-shot compression: 8 input elements, domain IV in the capacity words,
/// 4-element (256-bit) digest out.
pub fn compress(iv: &[Goldilocks; 4], input: &[Goldilocks; 8]) -> [Goldilocks; 4] {
    let mut x = [Goldilocks::ZERO; T];
    x[0..8].copy_from_slice(input);
    x[12..16].copy_from_slice(iv);
    permute(&mut x);
    [x[0], x[1], x[2], x[3]]
}

/// The full permutation round trace: the input state followed by the state
/// after each of the ROUNDS rounds (ROUNDS + 1 = 65 states) — the native
/// twin of the in-circuit permutation rows (DSR-7). The Merkle chip's
/// witness rows are exactly these states.
pub fn round_trace(state: &[Goldilocks; T]) -> Vec<[Goldilocks; T]> {
    let rc = round_constants();
    let mut x = *state;
    let mut out = Vec::with_capacity(ROUNDS + 1);
    out.push(x);
    for r in 0..HALF {
        full_round(&mut x, &rc[r]);
        out.push(x);
    }
    for r in HALF..HALF + PARTIAL_ROUNDS {
        partial_round(&mut x, &rc[r]);
        out.push(x);
    }
    for r in HALF + PARTIAL_ROUNDS..ROUNDS {
        full_round(&mut x, &rc[r]);
        out.push(x);
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;
    use nerv_core::field::GOLDILOCKS_PRIME;

    fn fe(rng: &mut SplitMix64) -> Goldilocks {
        Goldilocks::from_u64_reduce(rng.next_u64())
    }

    fn random_state(rng: &mut SplitMix64) -> [Goldilocks; T] {
        let mut s = [Goldilocks::ZERO; T];
        for e in s.iter_mut() {
            *e = fe(rng);
        }
        s
    }

    // -- differential references (explicit matrix products) ------------------

    fn me_matrix() -> [[u64; 16]; 16] {
        let mut m = [[0u64; 16]; 16];
        for bi in 0..4 {
            for bj in 0..4 {
                for r in 0..4 {
                    for c in 0..4 {
                        let m4: u64 = if r == c { 2 } else { 1 };
                        m[bi * 4 + r][bj * 4 + c] = if bi == bj { 2 * m4 } else { m4 };
                    }
                }
            }
        }
        m
    }

    fn mi_matrix() -> [[u64; 16]; 16] {
        let mut m = [[0u64; 16]; 16];
        for bi in 0..4 {
            for r in 0..4 {
                for c in 0..4 {
                    m[bi * 4 + r][bi * 4 + c] = if r == c { 2 } else { 1 };
                }
            }
        }
        m
    }

    fn mat_vec(m: &[[u64; 16]; 16], v: &[Goldilocks; T]) -> [Goldilocks; T] {
        let mut out = [Goldilocks::ZERO; T];
        for i in 0..16 {
            let mut acc = Goldilocks::ZERO;
            for j in 0..16 {
                if m[i][j] != 0 {
                    acc = acc + v[j] * Goldilocks::from_u32(m[i][j] as u32);
                }
            }
            out[i] = acc;
        }
        out
    }

    #[test]
    fn external_linear_matches_matrix_reference() {
        let m = me_matrix();
        let mut rng = SplitMix64::new(0x901D);
        for _ in 0..50 {
            let x = random_state(&mut rng);
            let want = mat_vec(&m, &x);
            let mut got = x;
            external_linear(&mut got);
            assert_eq!(got, want);
        }
        // linearity: M_E(a + b) == M_E(a) + M_E(b)
        let a = random_state(&mut rng);
        let b = random_state(&mut rng);
        let mut sum = [Goldilocks::ZERO; T];
        for i in 0..T {
            sum[i] = a[i] + b[i];
        }
        let (mut la, mut lb, mut lsum) = (a, b, sum);
        external_linear(&mut la);
        external_linear(&mut lb);
        external_linear(&mut lsum);
        for i in 0..T {
            assert_eq!(lsum[i], la[i] + lb[i]);
        }
    }

    #[test]
    fn internal_linear_matches_matrix_reference() {
        let m = mi_matrix();
        let mut rng = SplitMix64::new(0x902E);
        for _ in 0..50 {
            let x = random_state(&mut rng);
            let want = mat_vec(&m, &x);
            let mut got = x;
            internal_linear(&mut got);
            assert_eq!(got, want);
        }
        let a = random_state(&mut rng);
        let b = random_state(&mut rng);
        let mut sum = [Goldilocks::ZERO; T];
        for i in 0..T {
            sum[i] = a[i] + b[i];
        }
        let (mut la, mut lb, mut lsum) = (a, b, sum);
        internal_linear(&mut la);
        internal_linear(&mut lb);
        internal_linear(&mut lsum);
        for i in 0..T {
            assert_eq!(lsum[i], la[i] + lb[i]);
        }
    }

    // -- permutation properties -----------------------------------------------

    fn egcd(a: u64, b: u64) -> (i128, i128, i64) {
        let (mut old_r, mut r) = (a as i128, b as i128);
        let (mut old_s, mut s) = (1i128, 0i128);
        let (mut old_t, mut t) = (0i128, 1i128);
        while r != 0 {
            let q = old_r / r;
            (old_r, r) = (r, old_r - q * r);
            (old_s, s) = (s, old_s - q * s);
            (old_t, t) = (t, old_t - q * t);
        }
        (old_s, old_t, old_r as i64)
    }

    #[test]
    fn sbox_is_a_field_permutation() {
        let (x, _, g) = egcd(7, GOLDILOCKS_PRIME - 1);
        assert_eq!(g, 1, "gcd(7, p-1) must be 1 for x^7 to be bijective");
        let d = x.rem_euclid((GOLDILOCKS_PRIME - 1) as i128) as u64;
        let mut rng = SplitMix64::new(0x7B0C);
        for _ in 0..300 {
            let a = fe(&mut rng);
            assert_eq!(a.pow7().pow(d), a);
        }
        assert_eq!(Goldilocks::ZERO.pow7(), Goldilocks::ZERO);
        assert_eq!(Goldilocks::ONE.pow7(), Goldilocks::ONE);
    }

    #[test]
    fn permute_is_deterministic_and_input_sensitive() {
        let mut rng = SplitMix64::new(0xA1AC);
        let x = random_state(&mut rng);
        let (mut a, mut b) = (x, x);
        permute(&mut a);
        permute(&mut b);
        assert_eq!(a, b);
        assert_ne!(a, x);
        let mut changed = 0usize;
        for _ in 0..40 {
            let base = random_state(&mut rng);
            let pos = (rng.next_u64() % T as u64) as usize;
            let mut flipped = base;
            flipped[pos] = flipped[pos] + Goldilocks::ONE;
            let (mut pa, mut pb) = (base, flipped);
            permute(&mut pa);
            permute(&mut pb);
            changed += (0..T).filter(|&i| pa[i] != pb[i]).count();
        }
        assert!(changed >= 7 * 40, "avalanche too weak: {changed}/640");
    }

    #[test]
    fn distinct_states_yield_distinct_outputs() {
        let mut rng = SplitMix64::new(0xD157);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..256 {
            let mut x = random_state(&mut rng);
            permute(&mut x);
            let bytes: Vec<u8> = x.iter().flat_map(|e| e.as_u64().to_le_bytes()).collect();
            assert!(seen.insert(bytes));
        }
    }

    #[test]
    fn round_constants_shape_and_derivation() {
        let rc = round_constants();
        assert_eq!(rc.len(), ROUNDS);
        let mut nonzero = 0usize;
        for r in rc.iter() {
            for e in r.iter() {
                assert!(e.as_u64() < GOLDILOCKS_PRIME);
                if !e.is_zero() {
                    nonzero += 1;
                }
            }
        }
        assert!(nonzero > ROUNDS * T / 2);
        assert_ne!(rc[0], rc[1]);
        assert_ne!(rc[0], rc[ROUNDS - 1]);
        let mut xof = Xof::new(&POSEIDON2, b"nerv.poseidon2.rounds.v1");
        for r in rc.iter() {
            for e in r.iter() {
                assert_eq!(e.as_u64(), Goldilocks::from_u64_reduce(xof.next_u64()).as_u64());
            }
        }
    }

    #[test]
    fn compression_domains_and_determinism() {
        let zero = [Goldilocks::ZERO; 8];
        let mut one = zero;
        one[0] = Goldilocks::ONE;
        assert_eq!(compress(leaf_iv(), &zero), compress(leaf_iv(), &zero));
        assert_ne!(compress(leaf_iv(), &zero), compress(node_iv(), &zero));
        assert_ne!(compress(node_iv(), &zero), compress(empty_iv(), &zero));
        assert_ne!(compress(leaf_iv(), &zero), compress(leaf_iv(), &one));
        for iv in [leaf_iv(), node_iv(), empty_iv()] {
            for e in iv.iter() {
                assert!(e.as_u64() < GOLDILOCKS_PRIME);
            }
        }
        assert_ne!(*leaf_iv(), *node_iv());
        assert_ne!(*node_iv(), *empty_iv());
        assert_eq!(domain_iv(&NCT_LEAF), *leaf_iv());
        assert_eq!(domain_iv(&NCT_NODE), *node_iv());
        assert_eq!(domain_iv(&NCT_EMPTY), *empty_iv());
    }

    #[test]
    fn round_trace_matches_permute() {
        let mut rng = SplitMix64::new(0x7AC3);
        for _ in 0..20 {
            let x = random_state(&mut rng);
            let mut want = x;
            permute(&mut want);
            let trace = round_trace(&x);
            assert_eq!(trace.len(), ROUNDS + 1);
            assert_eq!(trace[0], x);
            assert_eq!(trace[ROUNDS], want);
        }
    }

}
