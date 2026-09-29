
//! # nerv-net — transport (WP §6.2, D-04, DSR-8)
//!
//! * `wire`       — the 1-RTT mutual PQ handshake and framed AEAD
//!                  sessions, generic over AsyncRead + AsyncWrite.
//! * `host`       — the TCP mesh: peer identity, the address book, the
//!                  connection registry, the event surface.
//! * `gossip`     — topic-tagged mesh messages and the DSR-8 ordering
//!                  rule (headers before decryption partials, always).
//! * `submission` — client fan-out to N aggregators (censorship
//!                  insurance) and the aggregator ingress; the
//!                  relay-egress seam chunk 16's mixnet plugs into.
//!
//! [`HostFrame`] is the frame router: tag 0–2 → gossip, 3 → submission,
//! 4+ reserved for chunk 16's relay topics.


#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic, clippy::float_cmp)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]


pub mod gossip;
pub mod host;
pub mod submission;
pub mod wire;


#[cfg(test)]
mod testutil;


pub use gossip::{
    message_digest, GossipEngine, GossipMessage, GossipStats, Inbound, PublishOutcome, Rejection,
    DEDUP_CAPACITY, HEADER_WINDOW, MAX_BLOCK_DATA_LEN,
};
pub use host::{Host, HostConfig, HostError, HostEvent, PeerBook, PeerId, PeerInfo};
pub use submission::{
   default_fanout, deliver, fan_out, ingest, ingest_entry, FanOutReport, IngestOutcome,
   SubmissionError, SubmissionMessage, SUBMISSION_TAG,
};
pub use wire::{accept_handshake, dial_handshake, Session, WireError, MAX_FRAME};


const _: () = assert!(nerv_core::params::MIXNET_SUBMISSION_FANOUT >= 1);


/// The single frame taxonomy of the host mesh. The node dispatches every
/// `HostEvent::Frame` through [`HostFrame::decode`] before routing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostFrame {
   Gossip(GossipMessage),
   Submission(SubmissionMessage),
}


impl HostFrame {
   pub fn encode(&self) -> Vec<u8> {
       match self {
           HostFrame::Gossip(m) => m.encode(),
           HostFrame::Submission(m) => m.encode(),
       }
   }


   pub fn decode(payload: &[u8]) -> Result<HostFrame, nerv_core::error::CodecError> {
       match payload.first() {
           None => Err(nerv_core::error::CodecError::Truncated),
           Some(0..=2) => Ok(HostFrame::Gossip(GossipMessage::decode(payload)?)),
           Some(&crate::submission::SUBMISSION_TAG) => Ok(HostFrame::Submission(
               SubmissionMessage::decode(payload)?,
           )),
           Some(&tag) => Err(nerv_core::error::CodecError::InvalidOptionTag { tag }),
       }
   }
}


#[cfg(test)]
mod tests {
   use super::*;
   use nerv_core::error::CodecError;


   #[test]
   fn host_frame_dispatch() {
       let f = HostFrame::Gossip(GossipMessage::Reveal {
           shard: nerv_core::types::ShardSet::genesis().ids()[0],
           height: nerv_core::types::Height::from_u64(3),
           batch: 0,
           reveal: crate::testutil::harness::test_reveal(),
       });
       let enc = f.encode();
       assert_eq!(HostFrame::decode(&enc).unwrap(), f);
       assert_eq!(enc[0], 2);


       let s = HostFrame::Submission(SubmissionMessage::Transaction {
           entry: crate::testutil::harness::tx_entry(1),
       });
       let enc = s.encode();
       assert_eq!(HostFrame::decode(&enc).unwrap(), s);
       assert_eq!(enc[0], SUBMISSION_TAG);


       assert!(matches!(HostFrame::decode(&[]), Err(CodecError::Truncated)));
       let mut foreign = vec![9u8, 0, 0];
       foreign[0] = 9;
       assert!(matches!(
           HostFrame::decode(&foreign),
           Err(CodecError::InvalidOptionTag { tag: 9 })
       ));
   }
}

