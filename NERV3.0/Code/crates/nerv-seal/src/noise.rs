//! Noise-budget accounting for NERV-Seal (WP §6.3.3, §6.3.7–§6.3.8; errata
//! 30–31, 40–41).
//!
//! The budget, honestly closed (exact integers throughout):
//!
//! * Distributions: encryptor-side CBD(η = 2) (variance 1, support ±2,
//!   statement-10 circuit bound ±3). DKG-side ternary CBD(η = 1)
//!   (variance 1/2, support ±1) — the committee key is sum-structured:
//!   s = Σᵢ aᵢ, E₀ = Σᵢ Eᵢ over n = 10 members, so key variance is
//!   n/2 = 5 and honest |key|∞ ≤ n = 10 (a bound the DKG's FS proofs
//!   establish per member: ternary).
//!
//! * Honest noise: per component, ν = E₀·r + e₂ − s·e₁. Each rank-8 dot
//!   product contributes 8 convolutions of a key-variance-5 polynomial
//!   with a CBD(2) polynomial (per-coefficient variance 8·256·5 = 10,240):
//!   VAR_LEG = 2·10,240 + 1 = 20,481, σ ≈ 143. Chunk of c legs (the key
//!   enters through the summed encryptor randomness): σ² = c·VAR_LEG —
//!   at 128: σ = 1,619.
//!
//! * Scale/margin (erratum 40): scale = 2^15; decode margin scale/2 =
//!   16,384 is 10.12σ of encryption noise, 9.49σ after the committee
//!   smudging allocation — per-slot honest failure ≈ 10⁻²¹.
//!
//! * Adversarial closure (compile-time): with key coefficients at the
//!   DKG-proven bound 10, encryptors at ±3, and full smudging, a chunk
//!   cannot wrap the centered decode range. Remaining adversarial modes:
//!   in-envelope digit corruption of an advisory aggregate (≥ t
//!   collusion — priced by slashing and rotation, WP §2.5) or
//!   out-of-envelope sums — a detectable invalid reveal. Custody is
//!   untouched in every branch (§6.3.6).

use crate::digitize::DIGIT_MAX;
use crate::ring::{N, Q};

/// Plaintext scale (erratum 40): the digit unit's weight in the ciphertext.
pub const SCALE: u64 = 1 << 15;
/// The decode margin |ν| < SCALE/2.
pub const HALF_SCALE: i64 = SCALE as i64 / 2;
/// Maximum decodable per-slot digit sum: the centered range over scale.
pub const DIGIT_SUM_HEADROOM: u64 = (Q - 1) / 2 / SCALE;

/// CBD(η = 2) per-coefficient variance.
pub const VAR_CBD2: u64 = 1;
/// Committee size (params pin) — the sum-structure factor.
pub const COMMITTEE_N: u64 = nerv_core::params::SEAL_COMMITTEE_N as u64;
/// DKG key-side per-coefficient variance: n members × ternary variance 1/2.
pub const KEY_VARIANCE_DKG: u64 = 5;
/// Single-party reference-key variance (the DSR-7 twin used by tests and
/// conformance vectors).
pub const KEY_VARIANCE_REF: u64 = 1;

/// Per-leg, per-component decryption-noise variance for a key whose
/// coefficients have per-coefficient variance `key_var`:
/// 2·(rank 8)·(n 256)·key_var·1 + 1.
pub const fn leg_variance(key_var: u64) -> u64 {
    2 * 8 * N as u64 * key_var + 1
}

/// Production (DKG-key) per-leg noise variance: 20,481.
pub const VAR_LEG: u64 = leg_variance(KEY_VARIANCE_DKG);
/// Reference (single-party) per-leg noise variance: 4,097.
pub const VAR_LEG_REF: u64 = leg_variance(KEY_VARIANCE_REF);

/// DKG-proven honest bound on key-side coefficients (s, E₀): n × ternary.
pub const KEY_BOUND: u64 = COMMITTEE_N;
/// Statement-10 in-circuit bound on encryptor-side coefficients (±3σ at η=2).
pub const CIRCUIT_NOISE_BOUND: u64 = 3;

/// Committee smudging budget (§6.3.3; erratum 37): the per-coefficient
/// bound on |su_combined − s·u_B|∞ — the summed ε of the t contributing
/// partials. The VPD proofs (part 2) enforce ε per member at 128, and
/// t·128 = 896 ≤ 1,024.
pub const SMUDGING_BUDGET: i64 = 1 << 10;

/// Provable worst-case per-leg noise with honest (bounded) short inputs.
pub const HONEST_LEG_WORST_NOISE: i64 =
    (2 * 8 * N as u64 * KEY_BOUND * 2 + 2) as i64;
/// Provable worst-case per-leg noise: key at the DKG bound, encryptor at
/// the circuit bound.
pub const ADVERSARIAL_LEG_WORST_NOISE: i64 =
    (2 * 8 * N as u64 * KEY_BOUND * CIRCUIT_NOISE_BOUND + CIRCUIT_NOISE_BOUND) as i64;

/// Variance of the chunk decryption noise: c·VAR_LEG (DKG key).
pub const fn chunk_noise_variance(legs: u64) -> u64 {
    legs * VAR_LEG
}

/// The decode margin (scale/2).
pub const fn decode_margin() -> i64 {
    HALF_SCALE
}

// --- compile-time budget closure (errata 30–31, 40) ------------------------

const _: () = assert!(SCALE == 1u64 << (nerv_core::params::SEAL_SCALE_LOG2 as u64));
const _: () = assert!(KEY_VARIANCE_DKG * 2 == COMMITTEE_N);

// Even worst-case adversarial noise plus full smudging cannot wrap the
// centered decode range.
const _: () = {
    let chunk = nerv_core::params::SEAL_CHUNK_MAX as u64;
    let plaintext_worst = SCALE * (chunk * DIGIT_MAX);
    let noise_worst = chunk * ADVERSARIAL_LEG_WORST_NOISE as u64 + SMUDGING_BUDGET as u64;
    assert!(plaintext_worst + noise_worst < (Q - 1) / 2);
};

// Encryption-noise margin floor at the genesis chunk: ≥ 102 σ² (≈ 10.1σ).
const _: () = {
    let chunk = nerv_core::params::SEAL_CHUNK_MAX as u64;
    assert!((HALF_SCALE as u64 * HALF_SCALE as u64) / (chunk * VAR_LEG) >= 102);
};

// Post-smudging honest margin floor: ≥ 89 σ² (≈ 9.43σ).
const _: () = {
    let chunk = nerv_core::params::SEAL_CHUNK_MAX as u64;
    let effective = (HALF_SCALE - SMUDGING_BUDGET) as u64;
    assert!((effective * effective) / (chunk * VAR_LEG) >= 89);
};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn variance_model_identities() {
        assert_eq!(leg_variance(KEY_VARIANCE_DKG), 2 * 8 * 256 * 5 + 1);
        assert_eq!(VAR_LEG, 20_481);
        assert_eq!(VAR_LEG_REF, 4_097);
        assert_eq!(chunk_noise_variance(128), 2_621_568);
        assert_eq!(chunk_noise_variance(0), 0);
        assert_eq!(decode_margin(), 16_384);
    }

    #[test]
    fn provable_bounds_arithmetic() {
        let n = N as u64;
        assert_eq!(HONEST_LEG_WORST_NOISE as u64, 2 * 8 * n * KEY_BOUND * 2 + 2);
        assert_eq!(
            ADVERSARIAL_LEG_WORST_NOISE as u64,
            2 * 8 * n * KEY_BOUND * CIRCUIT_NOISE_BOUND + CIRCUIT_NOISE_BOUND
        );
        assert_eq!(HONEST_LEG_WORST_NOISE, 81_922);
        assert_eq!(ADVERSARIAL_LEG_WORST_NOISE, 122_883);
    }

    #[test]
    fn scale_margin_and_headroom_pins() {
        assert_eq!(SCALE, 32_768);
        assert_eq!(HALF_SCALE, 16_384);
        assert_eq!(DIGIT_SUM_HEADROOM, 65_520);
        assert!(COMMITTEE_N * DIGIT_MAX < DIGIT_SUM_HEADROOM);
    }

    #[test]
    fn budget_closed_even_adversarially() {
        let chunk = nerv_core::params::SEAL_CHUNK_MAX as u64;
        assert_eq!(chunk, 128);
        let total = SCALE * (chunk * DIGIT_MAX)
            + chunk * ADVERSARIAL_LEG_WORST_NOISE as u64
            + SMUDGING_BUDGET as u64;
        assert!(total < (Q - 1) / 2);
        assert_eq!((HALF_SCALE as u64 * HALF_SCALE as u64) / (chunk * VAR_LEG), 102);
    }

    #[test]
    fn smudging_budget_pins() {
        assert_eq!(SMUDGING_BUDGET, 1_024);
        let t = nerv_core::params::SEAL_COMMITTEE_THRESHOLD as u64;
        assert!(t * 128 <= SMUDGING_BUDGET as u64);
        let effective = HALF_SCALE - SMUDGING_BUDGET;
        assert_eq!(effective, 15_360);
        let chunk = nerv_core::params::SEAL_CHUNK_MAX as u64;
        assert_eq!((effective as u64 * effective as u64) / (chunk * VAR_LEG), 89);
    }
}
