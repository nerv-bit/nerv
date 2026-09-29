//! The AIR layer (WP §5.2–§5.4).


pub mod builder;
pub mod chips;
pub mod fs;
pub mod custody_air;
pub mod delta_air;
pub mod sym;
pub mod tx_air;


pub use builder::{
    Air, AirBuilder, AirExpr, ConstraintFailure, NativeEval,
};


pub use chips::merkle_poseidon2::{gen_merkle_trace, MerkleChip};


pub use fs::{
    bind, bind_transaction, canonical_nullifiers, shell_digest, FsTranscript, TxPublicInputs,
};


pub use chips::conservation::{gen_cons_prep, gen_cons_trace, ConservationChip};


pub use sym::{ConstraintSet, SymBuilder, SymExpr, MeasureBuilder, measure};


pub use custody_air::{
    bind_witness_to_shell, gen_custody_trace, BurnWitness, CustodyAir, CustodyTrace,
    CustodyWitness, InputWitness, OutputWitness, RevertWitness,
};


pub use chips::blake3::{compress_native, hash_native, Blake3Compression};


pub use chips::seal_chip::{
    gen_seal_trace, SealChip, SealLegInput, SealTrace, LEG_W as SEAL_LEG_W,
    PREP_COLS as SEAL_PREP, ROWS as SEAL_ROWS,
};


pub use delta_air::{gen_delta_trace, DeltaAir, DeltaTrace, LegShape, LEG_W as DELTA_LEG_W};


pub use tx_air::{
    build_tx_air, check_ct_binding, gen_tx_prep, gen_tx_trace, prove_transaction,
    serialize_uv, verify_transaction, TransactionProof, TxAir, TxError, TxTrace,
};
