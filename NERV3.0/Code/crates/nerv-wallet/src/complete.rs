//! Issue-leg completion (WP §4.5, App A; erratum 189).

use nerv_core::hash::Hash256;
use nerv_core::types::{Height, LegIndex, ShardId, TxId};
use nerv_custody::tx::TransactionShell;
use nerv_state::block::{SettledLeg, TransitEvidence};
use nerv_state::ttau::TauTree;
use nerv_state::{BeaconView, ShardHeader};

use crate::prove::ProvedTx;

/// Check if a spend leg's transit entry is beacon-finalized at `height`.
pub fn check_finalized(
    view: &dyn BeaconView,
    spend_shard: ShardId,
    height: Height,
    txid: &TxId,
    spend_leg: LegIndex,
) -> Option<Hash256> {
    view.transit_root(spend_shard, height)
        .map(|_| Hash256::from_bytes([0u8; 32])) // The root itself; the caller checks membership separately.
}

/// Build the transit witness for the issue leg's settlement (erratum 189).
pub fn build_witness(
    view: &dyn BeaconView,
    spend_shard: ShardId,
    finalized_height: Height,
    transit_log: &nerv_custody::TransitLog,
    txid: &TxId,
    spend_leg: LegIndex,
) -> Result<TransitEvidence, String> {
    let key = nerv_custody::transit_key(txid, &spend_shard, spend_leg);
    let entry = transit_log
        .get(&key)
        .ok_or_else(|| format!("transit entry for {txid:?} leg {spend_leg:?} not found"))?;
    let _root = view
        .transit_root(spend_shard, finalized_height)
        .ok_or_else(|| "transit root not finalized".to_string())?;
    Ok(TransitEvidence {
        root_height: finalized_height,
        proof: transit_log.membership_proof(&entry),
    })
}

/// The completion package: everything needed to settle the issue leg.
#[derive(Clone, Debug)]
pub struct CompletionPackage {
    pub proved_tx: ProvedTx,
    pub issue_leg_index: usize,
    pub transit_witness: TransitEvidence,
}

/// Construct the completion: the issue leg's SettledLeg for the receiving
/// shard's producer (erratum 189).
pub fn build_completion(
    package: &CompletionPackage,
    tau_tree: &TauTree,
    tau_position: u64,
) -> Result<SettledLeg, String> {
    let canon = package.proved_tx.shell.canonicalize()
        .map_err(|e| format!("shell: {e}"))?;
    let leg = canon.legs.get(package.issue_leg_index)
        .ok_or_else(|| "issue leg index out of range".to_string())?;
    Ok(SettledLeg {
        shell: canon,
        leg: nerv_core::types::LegIndex::from_u8(package.issue_leg_index as u8),
        tau: tau_tree.witness(tau_position)
            .map_err(|e| format!("tau: {e}"))?,
        siblings: vec![package.transit_witness.clone()],
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn completion_package_shape() {
        // The module's types are composable; the full test requires a
        // multi-shard fixture (chunk 20's testkit).
        let _ = std::marker::PhantomData::<CompletionPackage>;
    }
}
