
//! Constraint chips (WP §5.3): sub-AIRs that generate constraints over a
//! designated slice of witness columns. Each chip is independently
//! testable via `NativeEval` and composes into the transaction AIR.

pub mod digit;
pub mod merkle_poseidon2;
pub mod range;
pub mod conservation;
pub mod blake3;
pub mod encoder;
pub mod seal_chip;

pub use digit::{
    gen_digit_witness, DigitChip, BITS_BASE, DIGIT_BASE, HI_COL, LO_COL, WIDTH as DIGIT_WIDTH,
};
pub use merkle_poseidon2::{
    gen_merkle_trace, MerkleChip, BIT_COL, PERM_ROWS, PREP_COLS, SIB_BASE, STATE_COLS, T2_BASE,
    T4_BASE, WITNESS_COLS,
};
pub use range::{gen_range_witness, RangeChip};
pub use conservation::{gen_cons_prep, gen_cons_trace, ConservationChip};
pub use blake3::{
    compress_native, gen_prep, gen_trace, hash_native, Blake3Compression, PREP_COLS as BL3_PREP,
    ROWS as BL3_ROWS, WITNESS_COLS as BL3_COLS,
};
pub use encoder::{EncoderChip, E_W, MAC_ROWS};
pub use seal_chip::{
    gen_seal_trace, SealChip, SealLegInput, SealTrace, LEG_W as SEAL_LEG_W,
    PREP_COLS as SEAL_PREP, ROWS as SEAL_ROWS,
};
