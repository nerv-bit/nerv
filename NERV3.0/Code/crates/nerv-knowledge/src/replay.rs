//! The D_t re-derivation (WP §10.1, §4.2; erratum 170): fold the block
//! loop over the header-derived event stream and compare each block's
//! derived root against the committed chain. Mismatch = the advisory
//! fault — a public, governance-slashable determinism bug in the
//! advisory layer; custody untouched by construction.


use nerv_core::hash::Hash256;


use crate::block_loop::{BlockEvent, KnowledgeState, LoopError};


#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("D_t mismatch at height {height}: committed {expected}, replayed {found}")]
pub struct ReplayFault {
    pub height: u64,
    pub expected: Hash256,
    pub found: Hash256,
}


#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ReplayError {
    #[error(transparent)]
    Loop(#[from] LoopError),
    #[error(transparent)]
    Fault(#[from] ReplayFault),
    #[error("{events} events vs {committed} committed roots")]
    LengthMismatch { events: usize, committed: usize },
}


fn height_of(e: &BlockEvent) -> u64 {
    match e {
        BlockEvent::Reveal { height, .. } | BlockEvent::Miss { height, .. } => *height,
    }
}


/// Fold the loop over the events from `initial`.
pub fn replay(initial: &KnowledgeState, events: &[BlockEvent]) -> Result<KnowledgeState, LoopError> {
    let mut s = initial.clone();
    for e in events {
        s.process(e.clone())?;
    }
    Ok(s)
}


/// The per-block D_t verification: one committed root per event, in
/// order; the root after applying event i must equal committed[i].
pub fn verify_d_t_chain(
    initial: &KnowledgeState,
    events: &[BlockEvent],
    committed: &[Hash256],
) -> Result<(), ReplayError> {
    if events.len() != committed.len() {
        return Err(ReplayError::LengthMismatch {
            events: events.len(),
            committed: committed.len(),
        });
    }
    let mut s = initial.clone();
    for (i, e) in events.iter().enumerate() {
        s.process(e.clone())?;
        let found = s.derived_root();
        if found != committed[i] {
            return Err(ReplayFault {
                height: height_of(e),
                expected: committed[i],
                found,
            }
            .into());
        }
    }
    Ok(())
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::block_loop::BlockEvent;
    use crate::forecaster::DIMS;
    use nerv_codec::codec_w::Delta;
    use crate::testutil::SplitMix64;


    fn events(seed: u64, n: u64) -> Vec<BlockEvent> {
        let mut rng = SplitMix64::new(seed);
        (1..=n)
            .map(|h| {
                if rng.next_u64() % 5 == 0 {
                    BlockEvent::Miss { height: h, legs: 128 }
                } else {
                    BlockEvent::Reveal {
                        height: h,
                        delta: Delta(std::array::from_fn(|_| rng.next_u64() % 500_000)),
                        fee_sum: rng.next_u64() % 1_000_000,
                        bucket: (rng.next_u64() % 16) as u16,
                    }
                }
            })
            .collect()
    }


    fn roots(initial: &KnowledgeState, events: &[BlockEvent]) -> Vec<Hash256> {
        let mut s = initial.clone();
        events
            .iter()
            .map(|e| {
                s.process(e.clone()).unwrap();
                s.derived_root()
            })
            .collect()
    }


    #[test]
    fn the_honest_chain_verifies() {
        let ev = events(0x2A, 60);
        let committed = roots(&KnowledgeState::genesis(), &ev);
        verify_d_t_chain(&KnowledgeState::genesis(), &ev, &committed).unwrap();
        assert_eq!(committed.len(), 60);
        // The empty chain.
        verify_d_t_chain(&KnowledgeState::genesis(), &[], &[]).unwrap();
    }


    #[test]
    fn a_tampered_root_faults_at_the_height() {
        let ev = events(0x2B, 40);
        let mut committed = roots(&KnowledgeState::genesis(), &ev);
        committed[17] = Hash256::from_bytes([0xEE; 32]);
        let err = verify_d_t_chain(&KnowledgeState::genesis(), &ev, &committed).unwrap_err();
        match err {
            ReplayError::Fault(f) => {
                assert_eq!(f.height, 18);
                assert_eq!(f.expected, committed[17]);
            }
            other => panic!("{other:?}"),
        }
    }


    #[test]
    fn a_divergent_event_stream_faults() {
        let ev = events(0x2C, 30);
        let committed = roots(&KnowledgeState::genesis(), &ev);
        // The same stream but one delta altered.
        let mut divergent = ev.clone();
        if let BlockEvent::Reveal { delta, .. } = &mut divergent[9] {
            delta.0[0] = delta.0[0].wrapping_add(1);
        } else {
            panic!("expected a reveal");
        }
        let err = verify_d_t_chain(&KnowledgeState::genesis(), &divergent, &committed).unwrap_err();
        assert!(matches!(err, ReplayError::Fault(ReplayFault { height: 10, .. })));


        // A miss where the committed chain had a reveal (or vice versa)
        // also faults: a reveal changes D_t, a miss carries it.
        let mut swapped = ev.clone();
        let h = height_of(&swapped[9]);
        swapped[9] = BlockEvent::Miss { height: h, legs: 128 };
        let err = verify_d_t_chain(&KnowledgeState::genesis(), &swapped, &committed).unwrap_err();
        assert!(matches!(err, ReplayError::Fault(_)));
    }


    #[test]
    fn a_different_initial_state_faults_immediately() {
        let ev = events(0x2D, 10);
        let committed = roots(&KnowledgeState::genesis(), &ev);
        let mut other = KnowledgeState::genesis();
        other.process(BlockEvent::Reveal {
            height: 1,
            delta: Delta([9u64; DIMS]),
            fee_sum: 0,
            bucket: 0,
        })
        .unwrap();
        assert!(verify_d_t_chain(&other, &ev, &committed).is_err());
    }


    #[test]
    fn length_mismatch_and_replay_fold() {
        let ev = events(0x2E, 10);
        let committed = roots(&KnowledgeState::genesis(), &ev);
        assert!(matches!(
            verify_d_t_chain(&KnowledgeState::genesis(), &ev, &committed[..9]),
            Err(ReplayError::LengthMismatch { events: 10, committed: 9 })
        ));
        let s = replay(&KnowledgeState::genesis(), &ev).unwrap();
        assert_eq!(s.derived_root(), *committed.last().unwrap());
        assert_eq!(replay(&KnowledgeState::genesis(), &[]).unwrap(), KnowledgeState::genesis());
    }
}
