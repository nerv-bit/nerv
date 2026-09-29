//! The windowed range chip (WP §5.3; D.2's windowed range checks): proves
//! a field element lies in [0, 2^bits) via full bit decomposition. The
//! lookup-table optimization (D.2's 16-bit windows) is a prover-side
//! concern (chunk 11); the CONSTRAINTS are identical either way.
//!
//! Layout: value at `value_col`, bits at `first_bit_col .. +bits`.
//! Constraints: `bits` boolean checks (degree 2) + 1 recomposition
//! (degree 1). Values must satisfy bits ≤ 63 (the Goldilocks field
//! accommodates 2^63 < p; 64-bit values need the two-limb approach used
//! by the digit chip).

use nerv_core::field::Goldilocks;
use crate::air::builder::{Air, AirBuilder};
pub const MAX_BITS: u32 = 63;

/// A range-check chip: proves value ∈ [0, 2^bits).
pub struct RangeChip {
    pub value_col: usize,
    pub first_bit_col: usize,
    pub bits: u32,
}

impl RangeChip {
    pub const fn new(value_col: usize, first_bit_col: usize, bits: u32) -> RangeChip {
        RangeChip { value_col, first_bit_col, bits }
    }

    pub const fn width(&self) -> usize {
        1 + self.bits as usize
    }

    pub const fn value_col(&self) -> usize {
        self.value_col
    }

    pub const fn first_bit_col(&self) -> usize {
        self.first_bit_col
    }
}

impl<B: AirBuilder> Air<B> for RangeChip {
    fn eval(&self, builder: &mut B) {
        for i in 0..self.bits as usize {
            let b = builder.witness(0, self.first_bit_col + i);
            builder.assert_bool(b, "range_bit");
        }

        let value = builder.witness(0, self.value_col);
        let mut sum = B::constant(0);
        for i in 0..self.bits as usize {
            let b = builder.witness(0, self.first_bit_col + i);
            sum = sum + b * B::constant(1u64 << i);
        }
        builder.assert_eq(value, sum, "range_recompose");
    }
}

/// Generates one row of range-check witness for `value` with `bits` bits.
pub fn gen_range_witness(value: u64, bits: u32) -> Vec<Goldilocks> {
    debug_assert!(bits <= MAX_BITS);
    let mut row = Vec::with_capacity(bits as usize + 1);
    row.push(Goldilocks::from_u64_reduce(value));
    for i in 0..bits {
        row.push(Goldilocks::from_u32(((value >> i) & 1) as u32));
    }
    row
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::air::builder::{ConstraintFailure, NativeEval};
    use crate::testutil::SplitMix64;

    fn check_value(value: u64, bits: u32) -> Result<(), Vec<ConstraintFailure>> {
        let trace = vec![gen_range_witness(value, bits)];
        let chip = RangeChip::new(0, 1, bits);
        NativeEval::check(trace, vec![], &chip, 16)
    }

    #[test]
    fn edge_values_pass() {
        assert!(check_value(0, 60).is_ok());
        assert!(check_value(1, 60).is_ok());
        assert!(check_value((1u64 << 60) - 1, 60).is_ok());
        assert!(check_value(u64::MAX & ((1u64 << 63) - 1), 63).is_ok());
        assert!(check_value(1u64 << 62, 63).is_ok());
    }

    #[test]
    fn random_values_in_range_pass() {
        let mut rng = SplitMix64::new(0xA0);
        for bits in [8u32, 16, 32, 60, 63] {
            let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
            for _ in 0..50 {
                let v = rng.next_u64() & mask;
                assert!(check_value(v, bits).is_ok(), "v = {v}, bits = {bits}");
            }
        }
    }

    #[test]
    fn out_of_range_fails() {
        // The witness can't represent a value ≥ 2^bits: recomposition fails.
        let v = 1u64 << 20;
        let bits = 16u32;
        let mut row = gen_range_witness(v & ((1 << bits) - 1), bits); // valid bits
        row[0] = Goldilocks::from_u64_reduce(v); // but wrong value
        let trace = vec![row];
        let chip = RangeChip::new(0, 1, bits);
        let errs = NativeEval::check(trace, vec![], &chip, 16).unwrap_err();
        assert!(errs.iter().any(|e| e.name == "range_recompose"));
    }

    #[test]
    fn tampered_bit_fails() {
        let bits = 16u32;
        let mut row = gen_range_witness(0x5A5A, bits);
        row[3] = Goldilocks::from_u32(5); // not boolean
        let trace = vec![row];
        let chip = RangeChip::new(0, 1, bits);
        let errs = NativeEval::check(trace, vec![], &chip, 16).unwrap_err();
        assert!(errs.iter().any(|e| e.name == "range_bit" && e.row == 0));
    }

    #[test]
    fn multi_row_trace() {
        let bits = 32u32;
        let chip = RangeChip::new(0, 1, bits);
        let values = [0u64, 1, 255, 65535, 12345678, u32::MAX as u64, 42];
        let trace: Vec<Vec<Goldilocks>> =
            values.iter().map(|&v| gen_range_witness(v, bits)).collect();
        assert!(NativeEval::check(trace, vec![], &chip, 16).is_ok());

        // One bad row.
        let mut bad_trace = trace.clone();
        bad_trace[3][0] = Goldilocks::from_u64_reduce(1 << 31); // out of range
        let errs = NativeEval::check(bad_trace, vec![], &chip, 16).unwrap_err();
        assert!(errs.iter().any(|e| e.row == 3));
    }

    #[test]
    fn column_offset_composition() {
        // The chip at a nonzero base column (composition scenario).
        let bits = 8u32;
        let chip = RangeChip::new(5, 10, bits);
        let mut row = vec![Goldilocks::ZERO; 20];
        let value = 0xABu64;
        row[5] = Goldilocks::from_u64_reduce(value);
        for i in 0..bits {
            row[10 + i as usize] = Goldilocks::from_u32(((value >> i) & 1) as u32);
        }
        assert!(NativeEval::check(vec![row], vec![], &chip, 16).is_ok());
    }
}


