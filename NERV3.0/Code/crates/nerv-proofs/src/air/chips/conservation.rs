//! The conservation chip (WP §5.1 statements 4 and 5): Σ inputs =
//! Σ outputs + Σ fees over the whole transaction, exactly, plus every
//! value in [0, 2^60] — differentially tested against native u64/u128
//! sums (DSR-7).
//!
//! Design (erratum 66):
//! * One row per value slot: inputs first (add rows), then outputs and
//!   fees (subtract rows). Row p carries the value's 64-bit decomposition
//!   (limbs + bits), the accumulator BEFORE the update (acc, a copy of the
//!   previous row's acc'; row 0's acc = 0), the accumulator AFTER (acc',
//!   range-checked), and the carry/borrow bits.
//! * Three 32-bit limbs = a 96-bit accumulator. Given every limb in
//!   [0, 2^32) (value limbs by decomposition; acc' limbs by range check;
//!   acc limbs by induction — acc(p) = acc'(p−1) by the copy constraint,
//!   acc(0) = 0), each limb equation's carry is unique and the row models
//!   acc' ≡ acc ± v (mod 2^96) exactly.
//! * Soundness of the final `acc' = 0`: it yields Σin − Σout − Σfee ≡ 0
//!   (mod 2^96). The shell's caps (≤ 256 legs × 8 values ≤ 2^60 each;
//!   ≤ 256 fees < 2^63) bound |Σin − Σout − Σfee| < 2^73 < 2^95, so the
//!   congruence implies exact integer equality — no wraparound forgery
//!   exists.
//! * The 64-bit boundary: the top limb after the LAST input row is
//!   constrained to 0, i.e. Σin < 2^64 — matching native checked u64
//!   arithmetic (and implied anyway by T3's supply bound; proven locally,
//!   belt-and-braces).
//! * Statement 5 as `b_60 · b_j = 0` for every decomposition bit j ≠ 60:
//!   exactly v ∈ [0, 2^60] INCLUSIVE (§5.1/§3.2) — zeroing bits 61–63
//!   would wrongly exclude the legal v = 2^60. Fees: bit 63 = 0
//!   (fee < 2^63; a documented circuit-level validity bound — u64 fees
//!   ≥ 2^63 are shell-representable but unprovable and economically
//!   absurd).
//!
//! Layout (171 columns): acc[3] ‖ acc'[3] ‖ v_hi ‖ v_lo ‖ c[3] ‖
//! v_bits[64] ‖ acc'_bits[96]. Preprocessed (6, offset by prep_base):
//! active, phase (1 = add), is_fee, boundary, final, step. Inactive rows
//! are zero-filled: every unconditional constraint (range recomposition,
//! bit booleans) passes on zeros; all others are prep-gated.
//!
//! The values/fees are WITNESSES here; their binding to the note
//! commitments (leaf inputs) and the public shell (via the shell digest)
//! is the commitment/BLAKE3 chips' obligation (chunk 11 — the typed
//! contract is `custody_air::InputWitness`/`OutputWitness`).

use nerv_core::field::Goldilocks;
use crate::air::builder::{Air, AirBuilder};
use crate::air::chips::range::RangeChip;

pub const ACC_BASE: usize = 0;
pub const ACCP_BASE: usize = 3;
pub const V_HI: usize = 6;
pub const V_LO: usize = 7;
pub const CARRY_BASE: usize = 8;
pub const V_BITS_BASE: usize = 11;
pub const ACCP_BITS_BASE: usize = 75;
pub const WIDTH: usize = 171;

pub const PREP_ACTIVE: usize = 0;
pub const PREP_PHASE: usize = 1;
pub const PREP_IS_FEE: usize = 2;
pub const PREP_BOUNDARY: usize = 3;
pub const PREP_FINAL: usize = 4;
pub const PREP_STEP: usize = 5;
pub const PREP_WIDTH: usize = 6;

const _: () = assert!(WIDTH == 11 + 64 + 96);
const MASK32: u64 = 0xFFFF_FFFF;

/// The conservation chip: statements 4 and 5 over one column slice.
#[derive(Clone, Copy, Debug)]
pub struct ConservationChip {
    pub col_base: usize,
    pub prep_base: usize,
}

impl ConservationChip {
    pub const fn new() -> ConservationChip {
        ConservationChip { col_base: 0, prep_base: 0 }
    }

    pub const fn at(col_base: usize, prep_base: usize) -> ConservationChip {
        ConservationChip { col_base, prep_base }
    }
}

impl Default for ConservationChip {
    fn default() -> Self {
        ConservationChip::new()
    }
}

impl<B: AirBuilder> Air<B> for ConservationChip {
    fn eval(&self, builder: &mut B) {
        let cb = self.col_base;
        let pb = self.prep_base;
        let active = builder.preprocessed(pb + PREP_ACTIVE);
        let phase = builder.preprocessed(pb + PREP_PHASE);
        let is_fee = builder.preprocessed(pb + PREP_IS_FEE);
        let boundary = builder.preprocessed(pb + PREP_BOUNDARY);
        let final_row = builder.preprocessed(pb + PREP_FINAL);
        let step = builder.preprocessed(pb + PREP_STEP);
        let one = B::constant(1);
        let not_phase = one.clone() - phase.clone();
        let not_fee = one.clone() - is_fee.clone();

        // 64-bit decomposition of the row's value (unconditional; zeros pass).
        RangeChip::new(cb + V_HI, cb + V_BITS_BASE + 32, 32).eval(builder);
        RangeChip::new(cb + V_LO, cb + V_BITS_BASE, 32).eval(builder);
        // acc' limbs in [0, 2^32) (unconditional; zeros pass).
        for k in 0..3 {
            RangeChip::new(cb + ACCP_BASE + k, cb + ACCP_BITS_BASE + 32 * k, 32).eval(builder);
        }

        // Statement 5: b_60·b_j = 0 for all j ≠ 60 on value rows —
        // exactly v ∈ [0, 2^60]. Fee rows: b_63 = 0 (fee < 2^63).
        let b60 = builder.witness(0, cb + V_BITS_BASE + 60);
        for j in 0..64 {
            if j != 60 {
                let bj = builder.witness(0, cb + V_BITS_BASE + j);
                builder.assert_zero(
                    active.clone() * not_fee.clone() * (b60.clone() * bj),
                    "cons_value_cap",
                );
            }
        }
        let b63 = builder.witness(0, cb + V_BITS_BASE + 63);
        builder.assert_zero(active.clone() * is_fee.clone() * b63, "cons_fee_cap");

        // Carries boolean on active rows.
        for k in 0..3 {
            let c = builder.witness(0, cb + CARRY_BASE + k);
            let e = c.clone() * (c - one.clone());
            builder.assert_zero(active.clone() * e, "cons_carry_bool");
        }

        let acc = [
            builder.witness(0, cb + ACC_BASE),
            builder.witness(0, cb + ACC_BASE + 1),
            builder.witness(0, cb + ACC_BASE + 2),
        ];
        let accp = [
            builder.witness(0, cb + ACCP_BASE),
            builder.witness(0, cb + ACCP_BASE + 1),
            builder.witness(0, cb + ACCP_BASE + 2),
        ];
        let vlo = builder.witness(0, cb + V_LO);
        let vhi = builder.witness(0, cb + V_HI);
        let c = [
            builder.witness(0, cb + CARRY_BASE),
            builder.witness(0, cb + CARRY_BASE + 1),
            builder.witness(0, cb + CARRY_BASE + 2),
        ];
        let two32 = B::constant(1u64 << 32);

        // add: acc' = acc + v with carries.
        builder.assert_zero(
            active.clone() * phase.clone()
                * (accp[0].clone() - acc[0].clone() - vlo.clone() + two32.clone() * c[0].clone()),
            "cons_add0",
        );
        builder.assert_zero(
            active.clone() * phase.clone()
                * (accp[1].clone() - acc[1].clone() - vhi.clone() - c[0].clone()
                    + two32.clone() * c[1].clone()),
            "cons_add1",
        );
        builder.assert_zero(
            active.clone() * phase.clone()
                * (accp[2].clone() - acc[2].clone() - c[1].clone() + two32.clone() * c[2].clone()),
            "cons_add2",
        );

        // sub: acc' = acc − v with borrows.
        builder.assert_zero(
            active.clone() * not_phase.clone()
                * (accp[0].clone() - acc[0].clone() + vlo.clone() - two32.clone() * c[0].clone()),
            "cons_sub0",
        );
        builder.assert_zero(
            active.clone() * not_phase.clone()
                * (accp[1].clone() - acc[1].clone() + vhi.clone() + c[0].clone()
                    - two32.clone() * c[1].clone()),
            "cons_sub1",
        );
        builder.assert_zero(
            active.clone() * not_phase.clone()
                * (accp[2].clone() - acc[2].clone() + c[1].clone() - two32.clone() * c[2].clone()),
            "cons_sub2",
        );

        // Copy: next row's acc = this row's acc' (both rows active).
        for k in 0..3 {
            let nxt = builder.witness(1, cb + ACC_BASE + k);
            let cur = builder.witness(0, cb + ACCP_BASE + k);
            builder.assert_zero(step.clone() * (nxt - cur), "cons_copy");
        }

        // Row 0: acc = 0 (the accumulator chain's base).
        let first_gate = active.clone() * builder.is_first_row();
        for k in 0..3 {
            let a = builder.witness(0, cb + ACC_BASE + k);
            builder.assert_zero(first_gate.clone() * a, "cons_first_acc");
        }

        // Boundary: top limb 0 after the last input row (Σin < 2^64).
        builder.assert_zero(
            boundary.clone() * builder.witness(0, cb + ACCP_BASE + 2),
            "cons_boundary",
        );

        // Final: acc' = 0 after the last value row — the conservation law.
        for k in 0..3 {
            let a = builder.witness(0, cb + ACCP_BASE + k);
            builder.assert_zero(final_row.clone() * a, "cons_final");
        }
    }
}

/// Witness rows for `entries` = (value, is_input), in order. The last
/// input row is the boundary; the last row is final. Deterministic; the
/// limb arithmetic mirrors the constraints' unique carry solutions.
pub fn gen_cons_trace(entries: &[(u64, bool)]) -> Vec<Vec<Goldilocks>> {
    let mut rows = Vec::with_capacity(entries.len());
    let mut acc = [0u64, 0u64, 0u64];
    for &(v, is_input) in entries {
        let v_lo = v & MASK32;
        let v_hi = v >> 32;
        let (c0, c1, c2, a0, a1, a2);
        if is_input {
            let t0 = acc[0] + v_lo;
            c0 = t0 >> 32;
            a0 = t0 & MASK32;
            let t1 = acc[1] + v_hi + c0;
            c1 = t1 >> 32;
            a1 = t1 & MASK32;
            let t2 = acc[2] + c1;
            c2 = t2 >> 32;
            a2 = t2 & MASK32;
        } else {
            let t0 = acc[0] as i128 - v_lo as i128;
            if t0 < 0 {
                c0 = 1;
                a0 = (t0 + (1i128 << 32)) as u64;
            } else {
                c0 = 0;
                a0 = t0 as u64;
            }
            let t1 = acc[1] as i128 - v_hi as i128 - c0 as i128;
            if t1 < 0 {
                c1 = 1;
                a1 = (t1 + (1i128 << 32)) as u64;
            } else {
                c1 = 0;
                a1 = t1 as u64;
            }
            let t2 = acc[2] as i128 - c1 as i128;
            if t2 < 0 {
                c2 = 1;
                a2 = (t2 + (1i128 << 32)) as u64;
            } else {
                c2 = 0;
                a2 = t2 as u64;
            }
        }
        let mut row = vec![Goldilocks::ZERO; WIDTH];
        row[ACC_BASE..ACC_BASE + 3].copy_from_slice(&[
            Goldilocks::from_u64_reduce(acc[0]),
            Goldilocks::from_u64_reduce(acc[1]),
            Goldilocks::from_u64_reduce(acc[2]),
        ]);
        row[ACCP_BASE..ACCP_BASE + 3].copy_from_slice(&[
            Goldilocks::from_u64_reduce(a0),
            Goldilocks::from_u64_reduce(a1),
            Goldilocks::from_u64_reduce(a2),
        ]);
        row[V_HI] = Goldilocks::from_u64_reduce(v_hi);
        row[V_LO] = Goldilocks::from_u64_reduce(v_lo);
        row[CARRY_BASE..CARRY_BASE + 3].copy_from_slice(&[
            Goldilocks::from_u32(c0 as u32),
            Goldilocks::from_u32(c1 as u32),
            Goldilocks::from_u32(c2 as u32),
        ]);
        for j in 0..64 {
            row[V_BITS_BASE + j] = Goldilocks::from_u32(((v >> j) & 1) as u32);
        }
        for k in 0..3 {
            let limb = [a0, a1, a2][k];
            for j in 0..32 {
                row[ACCP_BITS_BASE + 32 * k + j] =
                    Goldilocks::from_u32(((limb >> j) & 1) as u32);
            }
        }
        acc = [a0, a1, a2];
        rows.push(row);
    }
    rows
}

/// Preprocessed rows (6-wide) for `n_in + n_out + n_fee` value rows:
/// active rows contiguous from 0; adds first; fee rows after outputs.
pub fn gen_cons_prep(n_in: usize, n_out: usize, n_fee: usize) -> Vec<Vec<Goldilocks>> {
    let n = n_in + n_out + n_fee;
    let mut prep = vec![vec![Goldilocks::ZERO; PREP_WIDTH]; n];
    for r in 0..n {
        let e = &mut prep[r];
        e[PREP_ACTIVE] = Goldilocks::ONE;
        if r < n_in {
            e[PREP_PHASE] = Goldilocks::ONE;
        }
        if r >= n_in + n_out {
            e[PREP_IS_FEE] = Goldilocks::ONE;
        }
        if n_in >= 1 && r + 1 == n_in {
            e[PREP_BOUNDARY] = Goldilocks::ONE;
        }
        if r + 1 == n {
            e[PREP_FINAL] = Goldilocks::ONE;
        }
        if r + 1 < n {
            e[PREP_STEP] = Goldilocks::ONE;
        }
    }
    prep
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::air::builder::NativeEval;

    fn check(
        entries: &[(u64, bool)],
        n_in: usize,
        n_out: usize,
        n_fee: usize,
    ) -> Result<(), Vec<crate::air::builder::ConstraintFailure>> {
        let trace = gen_cons_trace(entries);
        let prep = gen_cons_prep(n_in, n_out, n_fee);
        NativeEval::check_with_prep(trace, prep, vec![], &ConservationChip::new(), 8)
    }

    fn balanced(seed: u64, n_in: usize, n_out: usize) -> Vec<(u64, bool)> {
        let mut rng = crate::testutil::SplitMix64::new(seed);
        let ins: Vec<u64> = (0..n_in)
            .map(|_| 1 + rng.next_u64() % (1u64 << 59))
            .collect();
        let total: u128 = ins.iter().map(|&v| u128::from(v)).sum();
        let mut outs: Vec<u64> = (0..n_out.saturating_sub(1))
            .map(|_| 1 + (rng.next_u64() as u128 % (total / n_out as u128).max(1)) as u64)
            .collect();
        let spent: u128 = outs.iter().map(|&v| u128::from(v)).sum();
        outs.push((total - spent) as u64);
        let mut entries: Vec<(u64, bool)> = ins.into_iter().map(|v| (v, true)).collect();
        entries.extend(outs.into_iter().map(|v| (v, false)));
        entries
    }

    #[test]
    fn layout_pins() {
        assert_eq!(WIDTH, 171);
        assert_eq!(PREP_WIDTH, 6);
        assert_eq!(ACCP_BITS_BASE, 75);
    }

    #[test]
    fn random_balanced_and_native_differential() {
        for seed in 0..40u64 {
            let n_in = 1 + (seed % 6) as usize;
            let n_out = 1 + (seed % 4) as usize;
            let entries = balanced(seed, n_in, n_out);
            // Native differential: the post-input accumulator equals the
            // u128 sum exactly, limbs and top-zero.
            let ins: Vec<u64> =
                entries.iter().take(n_in).map(|&(v, _)| v).collect();
            let total: u128 = ins.iter().map(|&v| u128::from(v)).sum();
            assert!(total < 1u128 << 64);
            let trace = gen_cons_trace(&entries);
            let r = &trace[n_in - 1];
            assert_eq!(r[ACCP_BASE + 2], Goldilocks::ZERO);
            assert_eq!(r[ACCP_BASE], Goldilocks::from_u64_reduce(total as u64 & MASK32));
            assert_eq!(
                r[ACCP_BASE + 1],
                Goldilocks::from_u64_reduce((total >> 32) as u64 & MASK32)
            );
            // And the AIR accepts.
            assert!(check(&entries, n_in, n_out, 0).is_ok(), "seed {seed}");
        }
    }

    #[test]
    fn exact_cap_boundaries() {
        // v = 2^60 exactly: legal (§3.2 inclusive), passes.
        let e = vec![(1u64 << 60, true), (1u64 << 60, false)];
        assert!(check(&e, 1, 1, 0).is_ok());
        // v = 2^60 + 1: rejected (statement 5).
        let e = vec![(1u64 << 60, true), ((1u64 << 60) + 1, false)];
        let errs = check(&e, 1, 1, 0).unwrap_err();
        assert!(errs.iter().any(|f| f.name == "cons_value_cap" || f.name == "cons_final"));
        // v = 2^61: rejected.
        let e = vec![(1u64 << 61, true), (1u64 << 61, false)];
        assert!(check(&e, 1, 1, 0).is_err());
        // fee = 2^63 − 1: passes; fee = 2^63: rejected.
        let f = (1u64 << 63) - 1;
        let e = vec![(f, true), (1u64 << 62, false), (f - (1u64 << 62), false)];
        assert!(check(&e, 1, 1, 1).is_ok());
        let f2 = 1u64 << 63;
        let e = vec![(f2, true), (f2 - 1, false), (1u64, false)];
        let errs = check(&e, 1, 1, 1).unwrap_err();
        assert!(errs.iter().any(|x| x.name == "cons_fee_cap"));
    }

    #[test]
    fn unbalanced_fails_at_final() {
        let e = vec![(100u64, true), (99u64, false)];
        let errs = check(&e, 1, 1, 0).unwrap_err();
        assert!(errs.iter().any(|f| f.name == "cons_final"));
        let e = vec![(100u64, true), (100u64, false), (1u64, false)];
        let errs = check(&e, 1, 1, 1).unwrap_err();
        assert!(errs.iter().any(|f| f.name == "cons_final"));
    }

    #[test]
    fn underflow_wraps_and_fails_at_final() {
        // Subtraction exceeding the accumulator wraps mod 2^96 (the limb
        // equations accept it — the soundness margin makes the wrap
        // unexploitable) and the final check rejects.
        let e = vec![(10u64, true), (20u64, false)];
        let trace = gen_cons_trace(&e);
        // The wrapped accumulator: 2^96 − 10.
        assert_eq!(trace[1][ACCP_BASE + 2], Goldilocks::from_u64_reduce(MASK32));
        assert_eq!(trace[1][ACCP_BASE + 1], Goldilocks::from_u64_reduce(MASK32));
        assert_eq!(trace[1][ACCP_BASE], Goldilocks::from_u64_reduce(MASK32 - 9));
        let errs = check(&e, 1, 1, 0).unwrap_err();
        assert!(errs.iter().any(|f| f.name == "cons_final"));
    }

    #[test]
    fn boundary_64bit_check() {
        let v = (1u64 << 60) - 1;
        // 16 inputs: sum = 2^64 − 16 < 2^64 → boundary holds.
        let mut e: Vec<(u64, bool)> = (0..16).map(|_| (v, true)).collect();
        e.extend((0..16).map(|_| (v, false)));
        assert!(check(&e, 16, 16, 0).is_ok());
        // 17 inputs: sum = 2^64 + 2^60 − 17 ≥ 2^64 → boundary fires even
        // though conservation balances.
        let mut e: Vec<(u64, bool)> = (0..17).map(|_| (v, true)).collect();
        e.extend((0..17).map(|_| (v, false)));
        let errs = check(&e, 17, 17, 0).unwrap_err();
        assert!(errs.iter().any(|f| f.name == "cons_boundary"));
    }

    #[test]
    fn tamper_cells() {
        let e = vec![(0x1_2345_6789u64, true), (0x1_2345_6789u64, false)];
        let trace = gen_cons_trace(&e);
        let prep = gen_cons_prep(1, 1, 0);
        let chip = ConservationChip::new();
        assert!(NativeEval::check_with_prep(trace.clone(), prep.clone(), vec![], &chip, 8).is_ok());

        // acc' limb.
        let mut bad = trace.clone();
        bad[0][ACCP_BASE] = bad[0][ACCP_BASE] + Goldilocks::ONE;
        assert!(NativeEval::check_with_prep(bad, prep.clone(), vec![], &chip, 8).is_err());
        // acc limb (breaks the copy / update).
        let mut bad = trace.clone();
        bad[1][ACC_BASE] = bad[1][ACC_BASE] + Goldilocks::ONE;
        assert!(NativeEval::check_with_prep(bad, prep.clone(), vec![], &chip, 8).is_err());
        // value limb.
        let mut bad = trace.clone();
        bad[0][V_LO] = bad[0][V_LO] + Goldilocks::ONE;
        assert!(NativeEval::check_with_prep(bad, prep.clone(), vec![], &chip, 8).is_err());
        // carry bit.
        let mut bad = trace.clone();
        bad[0][CARRY_BASE] = Goldilocks::from_u32(2);
        assert!(NativeEval::check_with_prep(bad, prep.clone(), vec![], &chip, 8).is_err());
        // decomposition bit.
        let mut bad = trace.clone();
        bad[1][V_BITS_BASE + 3] = bad[1][V_BITS_BASE + 3] + Goldilocks::ONE;
        assert!(NativeEval::check_with_prep(bad, prep.clone(), vec![], &chip, 8).is_err());
        // acc' range bit.
        let mut bad = trace.clone();
        bad[0][ACCP_BITS_BASE] = bad[0][ACCP_BITS_BASE] + Goldilocks::ONE;
        assert!(NativeEval::check_with_prep(bad, prep, vec![], &chip, 8).is_err());
    }

    #[test]
    fn inactive_zero_rows_pass() {
        // The composition invariant: zero-filled rows beyond the value
        // region pass (unconditional constraints see zeros; gated ones off).
        let e = vec![(50u64, true), (30u64, false), (20u64, false)];
        let mut trace = gen_cons_trace(&e);
        let mut prep = gen_cons_prep(1, 2, 0);
        for _ in 0..9 {
            trace.push(vec![Goldilocks::ZERO; WIDTH]);
            prep.push(vec![Goldilocks::ZERO; PREP_WIDTH]);
        }
        assert!(NativeEval::check_with_prep(trace, prep, vec![], &ConservationChip::new(), 8).is_ok());
    }
}

