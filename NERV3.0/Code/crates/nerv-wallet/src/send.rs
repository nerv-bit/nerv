//! Mixnet submission (WP §2.3, §5.5, D.2; erratum 188): the proved
//! transaction through the 5-relay path to 3 aggregators.

use nerv_core::codec::Encode;
use nerv_core::types::TxId;
use nerv_net::host::{Host, PeerInfo};
use nerv_net::relay_registry::{RelayRegistry, select_path};
use nerv_net::sphinx::{
    build, fragment, fragment_class, PathSpec, TerminalAction, PATH_RELAYS,
};
use nerv_net::submission::SubmissionMessage;
use nerv_registry::mempool::PoolEntry;

use crate::prove::ProvedTx;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SendError {
    #[error("sphinx: {0}")]
    Sphinx(#[from] nerv_net::sphinx::SphinxError),
    #[error("registry: {0}")]
    Registry(#[from] nerv_net::relay_registry::RegistryError),
    #[error("not connected to the first relay {relay}")]
    NotConnected { relay: String },
    #[error("send failed for aggregator {aggregator}")]
    SendFailed { aggregator: String },
    #[error("no aggregators available")]
    NoAggregators,
}

pub struct SendConfig {
    pub relay_registry: RelayRegistry,
    pub aggregators: Vec<PeerInfo>,
    pub fanout: usize,
}

pub struct SendReport {
    pub txid: TxId,
    pub fragments: usize,
    pub class: usize,
    pub aggregators_reached: usize,
}

/// Round the fee to a coarse bucket (optional; wallet policy, D.2).
pub fn bucket_fee(fee_nano: u64, bucket_count: u64) -> u64 {
    if bucket_count <= 1 || fee_nano == 0 {
        return fee_nano;
    }
    let step = (fee_nano + bucket_count - 1) / bucket_count;
    ((fee_nano + step / 2) / step) * step
}

/// Encode the submission payload for one aggregator.
fn encode_submission(proved: &ProvedTx) -> Vec<u8> {
    SubmissionMessage::Transaction {
        entry: PoolEntry {
            txid: proved.txid,
            shell: proved.shell.clone(),
            proof: proved.proof.clone(),
        },
    }
    .encode()
}

/// Send a proved transaction through the mixnet to `fanout` aggregators
/// (erratum 188). Returns the report; a partial fan-out (≥ 1 aggregator
/// reached) is a success.
pub fn send_transaction(
    host: &Host,
    proved: &ProvedTx,
    config: &SendConfig,
    wallet_seed: &[u8; 32],
    entropy: &mut dyn FnMut() -> [u8; 32],
) -> Result<SendReport, SendError> {
    if config.aggregators.is_empty() {
        return Err(SendError::NoAggregators);
    }

    // 1. Encode and fragment.
    let payload = encode_submission(proved);
    let class = fragment_class(payload.len())?;
    let frames = fragment(&payload, &proved.txid)?;

    // 2. For each aggregator: build a Sphinx path and send all fragments.
    let mut reached = 0;
    let fanout = config.fanout.min(config.aggregators.len());
    for agg_idx in 0..fanout {
        let aggregator = &config.aggregators[agg_idx];

        // Select a relay path to this aggregator.
        let path = select_path(&config.relay_registry, wallet_seed)?;
        let spec = PathSpec::new(path, aggregator.id(), TerminalAction::Deliver);

        // Build and send one Sphinx packet per fragment.
        let mut sent_any = false;
        for frame in &frames {
            let frame_bytes = frame.encode();
            let packet = build(&spec, &frame_bytes, &[
                entropy(); PATH_RELAYS
            ])?;
            // Send to the first relay in the path.
            let first_relay_peer = config
                .relay_registry
                .get(&path[0].0)
                .map(|r| r.record.vk);
            let _ = first_relay_peer;
            // The host sends by PeerId; we look up the first relay's id.
            let first_id = path[0].0;
            if host.send(&first_id, packet.to_frame()).is_ok() {
                sent_any = true;
            }
        }
        if sent_any {
            reached += 1;
        }
    }

    if reached == 0 {
        return Err(SendError::SendFailed {
            aggregator: "all".into(),
        });
    }

    Ok(SendReport {
        txid: proved.txid,
        fragments: frames.len(),
        class,
        aggregators_reached: reached,
    })
}

// ---- Gap 6: end-to-end send pipeline ------------------------------------

/// The pipeline stages for the GUI/CLI progress display. Mirrors
/// `nerv_wallet_core::action::SendStage` (here as a string label so the
/// wallet crate stays decoupled from the GUI/CLI types).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineStage {
    Constructing,
    Proving,
    Sealing,
    Routing,
    Submitted,
}

impl PipelineStage {
    /// Human-readable label (used by every UI; the wallet-core `SendStage`
    /// carries the canonical user-facing string).
    pub fn label(self) -> &'static str {
        match self {
            PipelineStage::Constructing => "Constructing transaction…",
            PipelineStage::Proving => "Generating proof (0.8–2 s)…",
            PipelineStage::Sealing => "Sealing delta…",
            PipelineStage::Routing => "Routing through mixnet…",
            PipelineStage::Submitted => "Submitted to aggregators",
        }
    }
}

/// All inputs the pipeline needs. Cloned into the spawned task — keep it
/// `Send` + `'static`; the platform shell owns the canonical copies.
pub struct SendPipeline {
    pub spec: crate::construct::PaymentSpec,
    pub wallet: crate::scan::WalletNoteSet,
    pub addresses: crate::keys::AddressSet,
    pub keys: nerv_custody::WalletKeys,
    pub active: nerv_core::types::ShardSet,
    pub codec: nerv_codec::codec_w::CodecW,
    pub epoch_pk: nerv_seal::encrypt::PublicKey,
    /// The frozen `[proofs.fri]` shape for this network (mirrors
    /// `bin/nerv-node`'s `default_fri_shape` — see Gap 2).
    pub fri: nerv_proofs::FriShape,
    pub current_height: u64,
    pub host: nerv_net::host::Host,
    pub send_config: SendConfig,
    pub wallet_seed: [u8; 32],
}

/// Error variants the pipeline can surface. Stringly-typed so we don't
/// leak every leaf error type's `Debug` into the wallet API.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PipelineError {
    #[error("construct: {0}")]
    Construct(String),
    #[error("prove: {0}")]
    Prove(String),
    #[error("send: {0}")]
    Send(#[from] SendError),
}

/// The end-to-end pipeline. The caller supplies an `on_stage` callback
/// that the platform shell turns into a `WalletAction::SendProgress(stage)`.
///
/// Stage sequence:
/// 1. `Constructing` → `construct_payment` (notes + envelope)
/// 2. `Proving`      → `prove` (tx STARK)
/// 3. `Sealing`      → part of `prove` (the delta's seal leg); we report a
///                       stage transition here for the UI even though it's
///                       folded into the previous step's elapsed time
/// 4. `Routing`      → `send_transaction` (Sphinx mixnet fan-out)
/// 5. `Submitted`    → success report built
///
/// Any stage failure short-circuits with `PipelineError`; the caller maps
/// that into `WalletAction::SendFailed(reason)`.
pub fn run_send_pipeline<'a, F: FnMut(PipelineStage)>(
    pipeline: &SendPipeline,
    entropy: &mut crate::construct::WalletEntropy<'a>,
    mut on_stage: F,
) -> Result<SendReport, PipelineError> {
    on_stage(PipelineStage::Constructing);
    let constructed = crate::construct::construct_payment(
        &pipeline.spec,
        &pipeline.wallet,
        &pipeline.addresses,
        &pipeline.keys,
        &pipeline.active,
        &pipeline.codec,
        &pipeline.epoch_pk,
        pipeline.current_height,
        entropy,
    )
    .map_err(|e| PipelineError::Construct(format!("{e:?}")))?;

    on_stage(PipelineStage::Proving);
    let proved = crate::prove::prove(
        &constructed,
        &pipeline.codec,
        &pipeline.epoch_pk,
        &pipeline.fri,
    )
    .map_err(|e| PipelineError::Prove(format!("{e:?}")))?;

    // The seal leg is folded into the prove step (statement-11 binding);
    // surface a stage transition so the UI shows the right granularity.
    on_stage(PipelineStage::Sealing);

    on_stage(PipelineStage::Routing);
    let report = send_transaction(
        &pipeline.host,
        &proved,
        &pipeline.send_config,
        &pipeline.wallet_seed,
        entropy,
    )?;

    on_stage(PipelineStage::Submitted);
    Ok(report)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn fee_bucketing() {
        assert_eq!(bucket_fee(700, 1), 700);
        assert_eq!(bucket_fee(0, 4), 0);
        assert_eq!(bucket_fee(1000, 4), 1000);
        assert_eq!(bucket_fee(999, 4), 750);
        assert_eq!(bucket_fee(1001, 4), 1250);
        // Coarse: round to nearest 500.
        assert_eq!(bucket_fee(700, 4), 750);
        assert_eq!(bucket_fee(1249, 4), 1250);
        assert_eq!(bucket_fee(1251, 4), 1500);
    }

    #[test]
    fn fragment_class_pins() {
        assert_eq!(fragment_class(100).unwrap(), 1);
        assert_eq!(fragment_class(14_113).unwrap(), 1);
        assert_eq!(fragment_class(14_114).unwrap(), 2);
    }

    // ---- Gap 6: pipeline orchestration ----

    #[test]
    fn pipeline_stage_label_is_pinned() {
        // Pin every label so a UI test can match against it.
        assert_eq!(PipelineStage::Constructing.label(), "Constructing transaction…");
        assert_eq!(PipelineStage::Proving.label(), "Generating proof (0.8–2 s)…");
        assert_eq!(PipelineStage::Sealing.label(), "Sealing delta…");
        assert_eq!(PipelineStage::Routing.label(), "Routing through mixnet…");
        assert_eq!(PipelineStage::Submitted.label(), "Submitted to aggregators");
    }

    #[test]
    fn pipeline_stage_ordering() {
        // The platform shells rely on the order:
        // Constructing → Proving → Sealing → Routing → Submitted.
        let order = [
            PipelineStage::Constructing,
            PipelineStage::Proving,
            PipelineStage::Sealing,
            PipelineStage::Routing,
            PipelineStage::Submitted,
        ];
        // Adjacent stages must differ.
        for w in order.windows(2) {
            assert_ne!(w[0], w[1], "stages must be distinct");
        }
        // Round-trip the order into the wallet-core's SendStage.
        use nerv_wallet_core::action::SendStage;
        let mapped: Vec<SendStage> = order
            .iter()
            .map(|s| match s {
                PipelineStage::Constructing => SendStage::Constructing,
                PipelineStage::Proving => SendStage::Proving,
                PipelineStage::Sealing => SendStage::Sealing,
                PipelineStage::Routing => SendStage::Routing,
                PipelineStage::Submitted => SendStage::Submitted,
            })
            .collect();
        assert_eq!(
            mapped,
            vec![
                SendStage::Constructing,
                SendStage::Proving,
                SendStage::Sealing,
                SendStage::Routing,
                SendStage::Submitted,
            ]
        );
    }

    #[test]
    fn pipeline_error_stringly_typed_for_shell_friendly_message() {
        // The platform shell surfaces `PipelineError` to the user via
        // `WalletAction::SendFailed(reason)`. Pin the `Display` impl so
        // the user-visible message format is stable.
        let e = PipelineError::Construct("insufficient funds".into());
        assert_eq!(e.to_string(), "construct: insufficient funds");
        let e = PipelineError::Prove("witness mismatch".into());
        assert_eq!(e.to_string(), "prove: witness mismatch");
    }
}
