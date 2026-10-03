//! Transaction submission (WP §5.5): the client's N-aggregator fan-out —
//! the censorship-insurance path — and the aggregator ingress. The
//! relay-egress seam: [`deliver`] is the single-target primitive chunk
//! 16's final mixnet relay calls with the unwrapped, already-framed
//! payload (the onion's carried payload IS the encoded
//! [`SubmissionMessage`]).
//!
//! The network layer carries bytes; the aggregator's ingress runs the
//! full verification gate. Fee bucketing is wallet-side (chunk 19).

use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::error::CodecError;
use nerv_core::types::TxId;
use nerv_registry::mempool::{Mempool, PoolEntry, VerifyContext};
use nerv_registry::MempoolError;

use crate::host::{Host, PeerId, PeerInfo};

/// The submission frame's tag (erratum 147's allocation).
pub const SUBMISSION_TAG: u8 = 3;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubmissionMessage {
    Transaction { entry: PoolEntry },
}

impl Encode for SubmissionMessage {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.push(SUBMISSION_TAG);
        match self {
            SubmissionMessage::Transaction { entry } => entry.encode_into(out),
        }
    }
    fn encoded_len(&self) -> usize {
        1 + match self {
            SubmissionMessage::Transaction { entry } => entry.encoded_len(),
        }
    }
}

impl Decode for SubmissionMessage {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        match r.read_u8()? {
            SUBMISSION_TAG => Ok(SubmissionMessage::Transaction {
                entry: PoolEntry::decode_from(r)?,
            }),
            tag => Err(CodecError::InvalidOptionTag { tag }),
        }
    }
}

/// The params' censorship-insurance N (mixnet.submission_fanout).
pub const DEFAULT_FANOUT: usize = nerv_core::params::MIXNET_SUBMISSION_FANOUT as usize;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SubmissionError {
    #[error("malformed submission: {0}")]
    Malformed(#[from] CodecError),
    #[error("no aggregator was reachable")]
    NoAggregator,
    #[error("delivery to {0} failed: not connected")]
    NotConnected(PeerId),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FanOutReport {
    pub attempted: usize,
    pub delivered: Vec<PeerId>,
}

/// The client's fan-out: deliver to the first `fanout` CONNECTED
/// aggregators in list order. The caller pre-shuffles for load balance
/// (client policy); zero delivery is an error (§5.5's censorship
/// insurance demands at least one path).
pub fn fan_out(
    host: &Host,
    aggregators: &[PeerInfo],
    entry: &PoolEntry,
    fanout: usize,
) -> Result<FanOutReport, SubmissionError> {
    let payload = SubmissionMessage::Transaction { entry: entry.clone() }.encode();
    let mut report = FanOutReport { attempted: aggregators.len(), delivered: Vec::new() };
    for agg in aggregators {
        if report.delivered.len() >= fanout {
            break;
        }
        let id = agg.id();
        if host.is_connected(&id) {
            if host.send(&id, payload.clone()).is_ok() {
                report.delivered.push(id);
            }
        }
    }
    if report.delivered.is_empty() {
        return Err(SubmissionError::NoAggregator);
    }
    Ok(report)
}

/// The relay-egress primitive: deliver an already-framed submission
/// payload to one aggregator. Chunk 16's final mixnet relay calls this
/// with the onion's unwrapped payload.
pub fn deliver(host: &Host, aggregator: &PeerId, frame: &[u8]) -> Result<(), SubmissionError> {
    host.send(aggregator, frame.to_vec())
        .map_err(|_| SubmissionError::NotConnected(*aggregator))
}

#[derive(Debug)]
pub enum IngestOutcome {
    Admitted(TxId),
    Duplicate(TxId),
    Rejected(MempoolError),
}

/// The aggregator ingress: decode the wire transaction and run the full
/// verification gate. The duplicate check precedes verification, so
/// re-submissions of a pooled txid are idempotent (erratum 148).
pub fn ingest(
    ctx: &VerifyContext,
    pool: &mut Mempool,
    payload: &[u8],
) -> Result<IngestOutcome, SubmissionError> {
    let message = SubmissionMessage::decode(payload)?;
    let SubmissionMessage::Transaction { entry } = message;
    Ok(ingest_entry(ctx, pool, entry))
}

/// The ingress over an already-decoded entry (the node's
/// HostFrame-routed path).
pub fn ingest_entry(ctx: &VerifyContext, pool: &mut Mempool, entry: PoolEntry) -> IngestOutcome {
    let txid = entry.txid;
    match pool.admit(ctx, entry.shell, entry.proof) {
        Ok(true) => IngestOutcome::Admitted(txid),
        Ok(false) => IngestOutcome::Duplicate(txid),
        Err(e) => IngestOutcome::Rejected(e),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::host::HostEvent;
    use crate::testutil::harness::{node, tx_entry, verify_ctx};
    use std::time::Duration;

    fn payload_of(entry: &PoolEntry) -> Vec<u8> {
        SubmissionMessage::Transaction { entry: entry.clone() }.encode()
    }

    async fn next_event(events: &mut tokio::sync::mpsc::Receiver<HostEvent>) -> HostEvent {
        tokio::time::timeout(Duration::from_secs(5), events.recv())
            .await
            .expect("event timeout")
            .expect("channel open")
    }

    async fn silence(events: &mut tokio::sync::mpsc::Receiver<HostEvent>) {
        assert!(
            tokio::time::timeout(Duration::from_millis(300), events.recv())
                .await
                .is_err(),
            "unexpected event"
        );
    }

    async fn await_connected(
        events: &mut tokio::sync::mpsc::Receiver<HostEvent>,
        n: usize,
    ) -> Vec<PeerId> {
        let mut seen = Vec::new();
        while seen.len() < n {
            match next_event(events).await {
                HostEvent::Connected(p) => seen.push(p),
                _ => {}
            }
        }
        seen
    }

    #[tokio::test]
    async fn fan_out_to_first_n_connected() {
        let mut aggs: Vec<_> = Vec::new();
        for i in 0..5u64 {
            let mut n = node(100 + i).await;
            await_connected(&mut n.events, 1).await;
            aggs.push(n);
        }
        let mut client = node(1).await;
        for a in &aggs {
            client.host.dial(a.info.clone()).unwrap();
        }
        await_connected(&mut client.events, 5).await;

        let entry = tx_entry(42);
        let infos: Vec<PeerInfo> = aggs.iter().map(|a| a.info.clone()).collect();
        let report = fan_out(&client.host, &infos, &entry, 3).unwrap();
        assert_eq!(report.attempted, 5);
        assert_eq!(report.delivered.len(), 3);
        assert_eq!(report.delivered, infos[..3].iter().map(|i| i.id()).collect::<Vec<_>>());

        let expected = payload_of(&entry);
        let client_id = client.info.id();
        for (i, agg) in aggs.iter_mut().enumerate() {
            if i < 3 {
                assert_eq!(
                    next_event(&mut agg.events).await,
                    HostEvent::Frame(client_id, expected.clone())
                );
            } else {
                silence(&mut agg.events).await;
            }
        }
    }

    #[tokio::test]
    async fn fan_out_skips_unconnected() {
        let mut aggs: Vec<_> = Vec::new();
        for i in 0..3u64 {
            let mut n = node(200 + i).await;
            await_connected(&mut n.events, 1).await;
            aggs.push(n);
        }
        let mut client = node(2).await;
        // Dial only the first two.
        for a in &aggs[..2] {
            client.host.dial(a.info.clone()).unwrap();
        }
        await_connected(&mut client.events, 2).await;

        let entry = tx_entry(43);
        let infos: Vec<PeerInfo> = aggs.iter().map(|a| a.info.clone()).collect();
        let report = fan_out(&client.host, &infos, &entry, 3).unwrap();
        assert_eq!(report.delivered.len(), 2, "the third is skipped, not an error");

        let expected = payload_of(&entry);
        let client_id = client.info.id();
        for (i, agg) in aggs.iter_mut().enumerate() {
            if i < 2 {
                assert_eq!(
                    next_event(&mut agg.events).await,
                    HostEvent::Frame(client_id, expected.clone())
                );
            } else {
                silence(&mut agg.events).await;
            }
        }
    }

    #[tokio::test]
    async fn fan_out_zero_reachable_is_an_error() {
        let mut agg = node(300).await;
        await_connected(&mut agg.events, 1).await;
        let client = node(3).await;
        let entry = tx_entry(44);
        let report = fan_out(&client.host, &[agg.info.clone()], &entry, 3);
        assert!(matches!(report, Err(SubmissionError::NoAggregator)));
    }

    #[tokio::test]
    async fn deliver_to_single_aggregator() {
        let mut agg = node(301).await;
        await_connected(&mut agg.events, 1).await;
        let mut client = node(4).await;
        client.host.dial(agg.info.clone()).unwrap();
        await_connected(&mut client.events, 1).await;

        let entry = tx_entry(45);
        let frame = payload_of(&entry);
        deliver(&client.host, &agg.info.id(), &frame).unwrap();
        assert_eq!(
            next_event(&mut agg.events).await,
            HostEvent::Frame(client.info.id(), frame)
        );

        // Not connected: an explicit error.
        let (_, vk_x, _) = crate::testutil::node_keys(90);
        let mut d = [0u8; 1184];
        d.copy_from_slice(vk_x.as_bytes());
        let _ = d;
        assert!(matches!(
            deliver(&client.host, &crate::host::PeerId::of(&vk_x), &frame),
            Err(SubmissionError::NotConnected(_))
        ));
    }

    #[test]
    fn default_fanout_is_params() {
        assert_eq!(DEFAULT_FANOUT, 3);
    }

    #[test]
    fn ingest_rejects_garbage_proof() {
        let ctx = verify_ctx();
        let mut pool = Mempool::new(64);
        let entry = tx_entry(7);
        let payload = payload_of(&entry);
        match ingest(&ctx, &mut pool, &payload).unwrap() {
            IngestOutcome::Rejected(e) => {
                assert!(matches!(e, MempoolError::Verification(_)));
            }
            other => panic!("expected rejection, got {other:?}"),
        }
        assert!(pool.is_empty());
    }

    #[test]
    fn ingest_duplicate_is_idempotent() {
        let ctx = verify_ctx();
        let mut pool = Mempool::new(64);
        let entry = tx_entry(8);
        pool.insert(entry.shell.clone(), entry.proof.clone()).unwrap();
        assert_eq!(pool.len(), 1);
        let payload = payload_of(&entry);
        match ingest(&ctx, &mut pool, &payload).unwrap() {
            IngestOutcome::Duplicate(txid) => assert_eq!(txid, entry.txid),
            other => panic!("expected duplicate, got {other:?}"),
        }
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn ingest_malformed_and_foreign_frames() {
        let ctx = verify_ctx();
        let mut pool = Mempool::new(64);
        assert!(matches!(ingest(&ctx, &mut pool, &[]), Err(SubmissionError::Malformed(_))));
        let gossip_frame = crate::gossip::GossipMessage::Reveal {
            shard: nerv_core::types::ShardSet::genesis().ids()[0],
            height: nerv_core::types::Height::from_u64(1),
            batch: 0,
            reveal: crate::testutil::harness::test_reveal(),
        }
        .encode();
        assert!(matches!(
            ingest(&ctx, &mut pool, &gossip_frame),
            Err(SubmissionError::Malformed(CodecError::InvalidOptionTag { tag: 2 }))
        ));
        assert!(pool.is_empty());
    }

    #[test]
    fn submission_codec_roundtrip() {
        let entry = tx_entry(9);
        let msg = SubmissionMessage::Transaction { entry };
        let enc = msg.encode();
        assert_eq!(enc.len(), msg.encoded_len());
        assert_eq!(enc[0], SUBMISSION_TAG);
        assert_eq!(SubmissionMessage::decode(&enc).unwrap(), msg);
        assert!(SubmissionMessage::decode(&enc[..enc.len() - 1]).is_err());
        let mut ext = enc.clone();
        ext.push(0);
        assert!(SubmissionMessage::decode(&ext).is_err());
    }
}
