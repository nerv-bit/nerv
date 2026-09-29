#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum WitnessError {
    #[error("leg {key:?} not found in the block")]
    LegNotFound { key: nerv_core::types::LegKey },
    #[error("block resolution failed: {0}")]
    BlockUnresolvable(String),
    #[error("anchor: {0}")]
    Anchor(#[from] anchor::AnchorError),
}
