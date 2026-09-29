//! The 8×8-bit digit-decomposition chip (statement 9; WP §5.3; errata 40, 55):
//! proves that 8 witness digits are the canonical base-256 decomposition of
//! a u64 coordinate, represented in the Goldilocks field as the pair
//! (hi, lo) of 32-bit halves — because u64 values ≥ p (the prime is
//! 2^64 − 2^32 + 1) cannot be represented in a single field element.
//!
//! Layout (74 columns):
//!   col 0: hi (upper 32 bits of the u64)
//!   col 1: lo (lower 32 bits)
//!   cols 2..9: d_0..d_7 (the 8 digits, each < 256)
//!   cols 10..73: 64 bits (8 bits per digit)
//!
//! Constraints (72 per coordinate, max degree 2):
//!   * 64 boolean checks (bits)
//!   * 8 digit recompositions: d_k = Σ_j b_{k,j}·2^j
//!   * 2 half recompositions: lo = Σ_{k<4} d_k·256^k; hi = Σ_{k≥4} d_k·256^k
//!
//! Differentially tested against `nerv_seal::digitize` (DSR-7): the chip's
//! digit extraction is the seal's slot layout for one coordinate.

use nerv_core::field::Goldilocks;
use crate::air::builder::{Air, AirBuilder};

pub const HI_COL: usize = 0;
pub const LO_COL: usize = 1;
pub const DIGIT_BASE: usize = 2;
pub const BITS_BASE: usize = 10;
pub const DIGITS: usize = 8;
pub const DIGIT_BITS: u32 = 8;
pub const WIDTH: usize = 2 + DIGITS + (DIGITS * DIGIT_BITS as usize);

const _: () = assert!(WIDTH == 74);
const _: () = assert!(DIGITS * (DIGIT_BITS as usize) == 64);

/// The digit-decomposition chip for one coordinate per row.
#[derive(Clone, Copy, Debug)]
pub struct DigitChip;

impl DigitChip {
    pub const fn new() -> DigitChip {
        DigitChip
    }

    pub const fn width(&self) -> usize {
        WIDTH
    }
}

impl Default for DigitChip {
    fn default() -> Self {
        DigitChip
    }
}

impl<B: AirBuilder> Air<B> for DigitChip {
    fn eval(&self, builder: &mut B) {
        // 64 boolean constraints on the bits.
        for i in 0..64 {
            let b = builder.witness(0, BITS_BASE + i);
            builder.assert_bool(b, "digit_bit");
        }

        // 8 digit recompositions from bits.
        for k in 0..DIGITS {
            let d = builder.witness(0, DIGIT_BASE + k);
            let mut sum = B::constant(0);
            for j in 0..DIGIT_BITS as usize {
                let bit = builder.witness(0, BITS_BASE + k * DIGIT_BITS as usize + j);
                sum = sum + bit * B::constant(1u64 << j);
            }
            builder.assert_eq(d, sum, "digit_from_bits");
        }

        // lo = d_0 + 256·d_1 + 65536·d_2 + 2^24·d_3.
        let lo = builder.witness(0, LO_COL);
        let mut lo_sum = B::constant(0);
        for k in 0..4 {
            let d = builder.witness(0, DIGIT_BASE + k);
            lo_sum = lo_sum + d * B::constant(1u64 << (DIGIT_BITS * k as u32));
        }
        builder.assert_eq(lo, lo_sum, "lo_from_digits");

        // hi = d_4 + 256·d_5 + 65536·d_6 + 2^24·d_7.
        let hi = builder.witness(0, HI_COL);
        let mut hi_sum = B::constant(0);
        for k in 0..4 {
            let d = builder.witness(0, DIGIT_BASE + 4 + k);
            hi_sum = hi_sum + d * B::constant(1u64 << (DIGIT_BITS * k as u32));
        }
        builder.assert_eq(hi, hi_sum, "hi_from_digits");
    }
}

/// Generates one row of digit-decomposition witness for `coord`.
pub fn gen_digit_witness(coord: u64) -> Vec<Goldilocks> {
    let mut row = vec![Goldilocks::ZERO; WIDTH];
    row[HI_COL] = Goldilocks::from_u64_reduce(coord >> 32);
    row[LO_COL] = Goldilocks::from_u64_reduce(coord & 0xFFFF_FFFF);
    for k in 0..DIGITS {
        let d = (coord >> (DIGIT_BITS * k as u32)) & 255;
        row[DIGIT_BASE + k] = Goldilocks::from_u32(d as u32);
        for j in 0..DIGIT_BITS as usize {
            row[BITS_BASE + k * DIGIT_BITS as usize + j] =
                Goldilocks::from_u32(((d >> j) & 1) as u32);
        }
    }
    row
}

/// Extracts digit k of a u64 coordinate (matches nerv-seal::digitize's
/// slot layout for one coordinate: `(coord >> 8k) & 255`).
pub const fn extract_digit(coord: u64, k: usize) -> u64 {
    (coord >> (DIGIT_BITS * k as u32)) & 255
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::air::builder::{ConstraintFailure, NativeEval};
    use crate::testutil::SplitMix64;

    fn check_coord(coord: u64) -> Result<(), Vec<ConstraintFailure>> {
        let trace = vec![gen_digit_witness(coord)];
        NativeEval::check(trace, vec![], &DigitChip, 16)
    }

    #[test]
    fn edge_values_pass() {
        assert!(check_coord(0).is_ok());
        assert!(check_coord(1).is_ok());
        assert!(check_coord(u64::MAX).is_ok());
        assert!(check_coord(1u64 << 32).is_ok());
        assert!(check_coord((1u64 << 32) - 1).is_ok());
        assert!(check_coord(0xDEAD_BEEF_1234_5678).is_ok());
    }

    #[test]
    fn random_values_pass() {
        let mut rng = SplitMix64::new(0xD16);
        for _ in 0..200 {
            let coord = rng.next_u64();
            assert!(check_coord(coord).is_ok(), "coord = {coord:#x}");
        }
    }

    #[test]
    fn tampered_digit_fails() {
        let coord = 0x1234_5678_9ABC_DEF0u64;
        let mut row = gen_digit_witness(coord);
        row[DIGIT_BASE + 3] = row[DIGIT_BASE + 3] + Goldilocks::ONE; // wrong digit
        let errs = NativeEval::check(vec![row], vec![], &DigitChip, 16).unwrap_err();
        assert!(errs.iter().any(|e| e.name == "digit_from_bits" || e.name == "lo_from_digits"));
    }

    #[test]
    fn tampered_half_fails() {
        let coord = 0xABCD_EF01_2345_6789u64;
        let mut row = gen_digit_witness(coord);
        row[HI_COL] = Goldilocks::from_u64_reduce((coord >> 32) ^ 1); // wrong hi
        let errs = NativeEval::check(vec![row], vec![], &DigitChip, 16).unwrap_err();
        assert!(errs.iter().any(|e| e.name == "hi_from_digits"));
    }

    #[test]
    fn tampered_bit_fails() {
        let coord = 42u64;
        let mut row = gen_digit_witness(coord);
        row[BITS_BASE + 5] = Goldilocks::from_u32(3); // not boolean
        let errs = NativeEval::check(vec![row], vec![], &DigitChip, 16).unwrap_err();
        assert!(errs.iter().any(|e| e.name == "digit_bit"));
    }

    #[test]
    fn digit_above_255_fails_through_bits() {
        // A digit of 256+ can't be represented by 8 bits: recomposition fails.
        let coord = 0u64;
        let mut row = gen_digit_witness(coord);
        row[DIGIT_BASE + 0] = Goldilocks::from_u64_reduce(256);
        let errs = NativeEval::check(vec![row], vec![], &DigitChip, 16).unwrap_err();
        assert!(errs.iter().any(|e| e.name == "digit_from_bits"));
    }

    #[test]
    fn multi_row_trace() {
        let mut rng = SplitMix64::new(0xD17);
        let coords: Vec<u64> = (0..16).map(|_| rng.next_u64()).collect();
        let trace: Vec<Vec<Goldilocks>> =
            coords.iter().map(|&c| gen_digit_witness(c)).collect();
        assert!(NativeEval::check(trace, vec![], &DigitChip, 16).is_ok());

        let mut bad = trace.clone();
        bad[7][LO_COL] = bad[7][LO_COL] + Goldilocks::ONE;
        let errs = NativeEval::check(bad, vec![], &DigitChip, 16).unwrap_err();
        assert!(errs.iter().any(|e| e.row == 7));
    }

    #[test]
    fn extract_digit_matches_shift() {
        let coord = 0xFEDC_BA98_7654_3210u64;
        for k in 0..8 {
            assert_eq!(extract_digit(coord, k), (coord >> (8 * k)) & 255);
        }
        assert_eq!(extract_digit(0, 0), 0);
        assert_eq!(extract_digit(u64::MAX, 7), 255);
    }

    // -- DSR-7 differential against nerv-seal::digitize ----------------------

    #[test]
    fn differential_against_seal_digitize() {
        use nerv_seal::digitize::{digitize, COORDS, DIGITS_PER_COORD};

        let mut rng = SplitMix64::new(0xD5A7);
        for _ in 0..100 {
            let coord = rng.next_u64();
            let mut coords = [0u64; COORDS];
            coords[0] = coord;
            let pt = digitize(&coords);

            // The seal's digit k of coordinate 0 (slot layout).
            for k in 0..DIGITS_PER_COORD {
                assert_eq!(
                    pt.digit(0, k),
                    extract_digit(coord, k),
                    "coord {coord:#x}, digit {k}"
                );
            }

            // The chip's witness passes the native evaluator.
            assert!(check_coord(coord).is_ok());
        }

        // Edge coordinates.
        for coord in [0u64, 1, u64::MAX, 1 << 32, (1 << 32) - 1, 255, 256, 1 << 63] {
            let mut coords = [0u64; COORDS];
            coords[0] = coord;
            let pt = digitize(&coords);
            for k in 0..DIGITS_PER_COORD {
                assert_eq!(pt.digit(0, k), extract_digit(coord, k));
            }
            assert!(check_coord(coord).is_ok());
        }
    }

    #[test]
    fn seal_plaintext_digits_match_chip_layout() {
        // Full round-trip: random coordinates → seal digitize → extract all
        // digits → generate chip witnesses for each → verify all pass.
        use nerv_seal::digitize::{digitize, COORDS, DIGITS_PER_COORD};

        let mut rng = SplitMix64::new(0xD5A8);
        let coords: [u64; COORDS] = std::array::from_fn(|_| rng.next_u64());
        let pt = digitize(&coords);

        // 64 rows, one per coordinate.
        let trace: Vec<Vec<Goldilocks>> =
            (0..COORDS).map(|j| gen_digit_witness(coords[j])).collect();
        assert!(NativeEval::check(trace, vec![], &DigitChip, 16).is_ok());

        // Cross-check: every digit the seal placed matches the chip's extraction.
        for j in 0..COORDS {
            for k in 0..DIGITS_PER_COORD {
                assert_eq!(pt.digit(j, k), extract_digit(coords[j], k));
            }
        }
    }
}
