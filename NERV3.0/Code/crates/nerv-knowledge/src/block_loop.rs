//! The §10.3 per-block loop (erratum 170): commit → reveal → score →
//! update → record, over (embedding, forecaster). Atomic per event; the
//! commit()/reveal() split is the node's temporal discipline.


use nerv_codec::codec_w::Delta;
use nerv_core::constants::DERIVED_PRED;
use nerv_core::hash::Hash256;


use crate::adam;
use crate::embedding::{derived_state_root, EmbedError, Embedding};
use crate::forecaster::{DIMS, Observation};
use crate::huber;


#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum LoopError {
    #[error(transparent)]
    Embed(#[from] EmbedError),
    #[error(transparent)]
    Adam(#[from] adam::AdamError),
    #[error("the state moved between commit and reveal")]
    CommitmentMismatch { expected: Hash256, found: Hash256 },
    #[error("time bucket {found} outside [0, 16)")]
    BadBucket { found: u16 },
}


/// The §10.3 Commit: the deterministic prediction from the current
/// state, made before the reveal exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Commitment {
    pub prediction: Delta,
    pub hash: Hash256,
}


#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlockEvent {
    Reveal { height: u64, delta: Delta, fee_sum: u64, bucket: u16 },
    Miss { height: u64, legs: u64 },
}


/// The public per-block derived record (advisory).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockRecord {
    pub height: u64,
    pub missed: bool,
    pub legs: Option<u64>,
    pub prediction: Option<Delta>,
    pub prediction_hash: Option<Hash256>,
    pub residual: Option<[i64; DIMS]>,
    pub loss_q32: Option<u128>,
}


/// The per-shard knowledge state: e_t and F.
#[derive(Clone, Debug, PartialEq)]
pub struct KnowledgeState {
    embedding: Embedding,
    forecaster: crate::forecaster::Forecaster,
}


impl KnowledgeState {
    pub fn genesis() -> KnowledgeState {
        KnowledgeState {
            embedding: Embedding::genesis(),
            forecaster: crate::forecaster::Forecaster::reference(),
        }
    }


    pub fn new(embedding: Embedding, forecaster: crate::forecaster::Forecaster) -> KnowledgeState {
        KnowledgeState { embedding, forecaster }
    }


    pub fn embedding(&self) -> &Embedding {
        &self.embedding
    }


    pub fn forecaster(&self) -> &crate::forecaster::Forecaster {
        &self.forecaster
    }


    /// D_t = H("nerv.derived" ‖ e_t ‖ fs_root) (WP §4.2).
    pub fn derived_root(&self) -> Hash256 {
        derived_state_root(self.embedding.coord(), self.forecaster.state_root().as_bytes())
    }


    pub fn commit(&self) -> Commitment {
        let prediction = self.forecaster.predict();
        let hash = Hash256::concat(&DERIVED_PRED, &prediction.canonical_bytes());
        Commitment { prediction, hash }
    }


    /// Reveal → score → update. `committed` must be the pre-reveal
    /// commitment of this state (re-derived and checked).
    #[allow(clippy::too_many_lines)]
    pub fn reveal(
        &mut self,
        committed: &Commitment,
        height: u64,
        delta: Delta,
        fee_sum: u64,
        bucket: u16,
    ) -> Result<BlockRecord, LoopError> {
        if bucket >= 16 {
            return Err(LoopError::BadBucket { found: bucket });
        }
        let check = self.commit();
        if check.hash != committed.hash {
            return Err(LoopError::CommitmentMismatch {
                expected: committed.hash,
                found: check.hash,
            });
        }


        let r = huber::residual(&delta, &committed.prediction);
        let mut scales = [0u64; DIMS];
        for (j, s) in scales.iter_mut().enumerate() {
            *s = self.forecaster.weights().scale(j);
        }
        let b = huber::blame(&r, &scales);
        let loss = huber::total_loss(&r, &scales);


        let grads = assemble_grads(&self.forecaster, &b);
        self.forecaster.adam_step(&grads)?;
        for j in 0..DIMS {
            let s = huber::scale_ema(scales[j], r[j].unsigned_abs());
            self.forecaster.weights_mut().set_scale(j, s);
        }


        self.forecaster.push_observation(Observation { delta: delta.clone(), fee_sum, bucket });
        self.embedding.apply(height, &delta)?;


        Ok(BlockRecord {
            height,
            missed: false,
            legs: None,
            prediction: Some(committed.prediction.clone()),
            prediction_hash: Some(committed.hash),
            residual: Some(r),
            loss_q32: Some(loss),
        })
    }


    /// The skip-and-carry rule (erratum 170): the height advances, the
    /// state carries unchanged.
    pub fn miss(&mut self, height: u64, legs: u64) -> Result<BlockRecord, LoopError> {
        self.embedding.apply_miss(height, legs)?;
        Ok(BlockRecord {
            height,
            missed: true,
            legs: Some(legs),
            prediction: None,
            prediction_hash: None,
            residual: None,
            loss_q32: None,
        })
    }


    pub fn process(&mut self, event: BlockEvent) -> Result<BlockRecord, LoopError> {
        match event {
            BlockEvent::Reveal { height, delta, fee_sum, bucket } => {
                let c = self.commit();
                self.reveal(&c, height, delta, fee_sum, bucket)
            }
            BlockEvent::Miss { height, legs } => self.miss(height, legs),
        }
    }
}


impl Default for KnowledgeState {
    fn default() -> Self {
        KnowledgeState::genesis()
    }
}


/// The gradient vector in the canonical order (ar ‖ fee ‖ bucket ‖ bias),
/// over the PRE-push window — the same lagged features the prediction
/// consumed (erratum 170).
fn assemble_grads(
    f: &crate::forecaster::Forecaster,
    b: &[i128; DIMS],
) -> Vec<i128> {
    use crate::forecaster::{AR_ORDER, LEARNABLE, PER_CHANNEL};
    use nerv_core::fixed_point::round_half_even_pow2;


    let mut grads = vec![0i128; LEARNABLE];
    let n = f.window().len().min(AR_ORDER);
    for j in 0..DIMS {
        let bj = b[j];
        grads[3 * PER_CHANNEL + j] = bj.saturating_neg();
        for lag in 1..=n {
            let obs = f.window().lag(lag).expect("lag within the window");
            let x = obs.delta.0[j] as i64;
            let idx = j * AR_ORDER + lag - 1;
            grads[idx] = -round_half_even_pow2(bj.saturating_mul(i128::from(x)), 15);
            grads[PER_CHANNEL + idx] = -(bj.saturating_mul(i128::from(obs.fee_sum)));
            grads[2 * PER_CHANNEL + idx] =
                -(bj.saturating_mul(i128::from(u64::from(obs.bucket))));
        }
    }
    grads
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;


    fn delta(v: u64) -> Delta {
        Delta([v; DIMS])
    }


    fn reveal_event(height: u64, v: u64) -> BlockEvent {
        BlockEvent::Reveal { height, delta: delta(v), fee_sum: 5_000, bucket: 3 }
    }


    #[test]
    fn genesis_derived_root_composition() {
        let s = KnowledgeState::genesis();
        assert_eq!(
            s.derived_root(),
            derived_state_root(
                s.embedding().coord(),
                s.forecaster().state_root().as_bytes()
            )
        );
        assert_eq!(s.embedding().next_height(), 1);
        assert!(s.forecaster().window().is_empty());
        assert_eq!(s.commit().prediction, Delta([0u64; DIMS]));
    }


    #[test]
    fn first_block_prediction_and_scoring() {
        let mut s = KnowledgeState::genesis();
        let c = s.commit();
        assert_eq!(c.prediction, Delta([0u64; DIMS]), "the empty window predicts the bias");
        let mut msg = Vec::new();
        msg.extend_from_slice(DERIVED_PRED.as_bytes());
        msg.extend_from_slice(&c.prediction.canonical_bytes());
        assert_eq!(c.hash.as_bytes(), blake3::hash(&msg).as_bytes());


        let rec = s.reveal(&c, 1, delta(100), 5_000, 3).unwrap();
        assert_eq!(rec.residual.unwrap(), [100i64; DIMS]);
        assert_eq!(rec.loss_q32.unwrap() > 0, true);
        assert_eq!(s.embedding().coord().0[0], 100);
        assert_eq!(s.forecaster().window().len(), 1);
        assert_eq!(s.forecaster().moments().step, 1, "one Adam step");


        // Positive residual → the blame is positive → the bias gradient is
        // negative → descent INCREASES the bias (§10.3's direction).
        assert!(s.forecaster().weights().bias(0) > 0, "bias moved up");
        assert!(
            s.forecaster().weights().ar(0, 1).to_bits() > 8192,
            "the lag-1 weight moved up"
        );
        assert_eq!(s.forecaster().weights().scale(0), huber::scale_ema(1 << 20, 100));
    }


    #[test]
   fn second_block_reads_lag_one() {
       let mut s = KnowledgeState::genesis();
       s.process(reveal_event(1, 1_000)).unwrap();
       let c = s.commit();
       // The exact signed reconstruction: pred = wrap(bias + round½even(
       // Σ_channels w·x)) over the single lag-1 observation.
       let w = s.forecaster().weights();
       let ar = i128::from(w.ar(0, 1).to_bits()) * 1_000;
       let fee = i128::from(w.fee(0, 1).to_bits()) * 5_000;
       let bucket = i128::from(w.bucket(0, 1).to_bits()) * 3;
       let want = nerv_core::wrap_mod_2_64(
           i128::from(w.bias(0))
               + nerv_core::round_half_even_pow2(ar + fee + bucket, 15),
       );
       assert_eq!(c.prediction.0[0], want);
       // The untouched lags contributed nothing: only lag 1 is in the
       // window, and the reference lag-1 weight family proves the AR
       // term is w(0,1)·1000/2^15.
       assert_eq!(
           ar / 32_768,
           i128::from(w.ar(0, 1).to_bits()) * 1000 / 32768
       );
       assert!(w.ar(0, 2).to_bits() >= 8192, "the lag-2 weight survived untouched or rose");
       s.reveal(&c, 2, delta(1_000), 5_000, 3).unwrap();
       assert_eq!(s.embedding().coord().0[0], 2_000);
       assert_eq!(s.forecaster().window().len(), 2);
       assert_eq!(s.forecaster().moments().step, 2);
   }



    #[test]
    fn miss_carries_the_state() {
        let mut s = KnowledgeState::genesis();
        s.process(reveal_event(1, 77)).unwrap();
        let d_before = s.derived_root();
        let fs_before = s.forecaster().state_root();
        let rec = s.process(BlockEvent::Miss { height: 2, legs: 128 }).unwrap();
        assert!(rec.missed);
        assert_eq!(rec.legs, Some(128));
        assert!(rec.prediction.is_none() && rec.residual.is_none());
        assert_eq!(s.derived_root(), d_before, "the miss carries e and the state");
        assert_eq!(s.forecaster().state_root(), fs_before);
        assert_eq!(s.embedding().misses(), &[crate::embedding::MissedReveal {
            height: 2,
            legs: 128,
        }]);
        assert_eq!(s.embedding().next_height(), 3, "the height advanced");
        assert_eq!(s.forecaster().moments().step, 1, "no update on a miss");
        // The chain continues.
        s.process(reveal_event(3, 79)).unwrap();
        assert_eq!(s.embedding().coord().0[0], 156);
    }


    #[test]
    fn commitment_mismatch_and_input_errors() {
        let mut s = KnowledgeState::genesis();
        let c = s.commit();
        s.process(reveal_event(1, 5)).unwrap();
        assert!(matches!(
            s.reveal(&c, 2, delta(5), 0, 0),
            Err(LoopError::CommitmentMismatch { .. })
        ));


        let mut s = KnowledgeState::genesis();
        let c = s.commit();
        assert!(matches!(
            s.reveal(&c, 1, delta(5), 0, 16),
            Err(LoopError::BadBucket { found: 16 })
        ));
        assert!(matches!(
            s.reveal(&c, 2, delta(5), 0, 0),
            Err(LoopError::Embed(crate::embedding::EmbedError::NonContiguous { .. }))
        ));
        let _ = c;
    }


    #[test]
    fn long_chain_determinism_and_embedding_identity() {
        let mut rng = SplitMix64::new(0x1009);
        let events: Vec<BlockEvent> = (1..=300u64)
            .map(|h| {
                if rng.next_u64() % 7 == 0 {
                    BlockEvent::Miss { height: h, legs: 128 }
                } else {
                    let v = std::array::from_fn(|_| rng.next_u64() % 100_000);
                    BlockEvent::Reveal {
                        height: h,
                        delta: Delta(v),
                        fee_sum: rng.next_u64() % 1_000_000,
                        bucket: (rng.next_u64() % 16) as u16,
                    }
                }
            })
            .collect();


        let mut a = KnowledgeState::genesis();
        let mut b = KnowledgeState::genesis();
        let mut reference = [0u64; DIMS];
        let mut roots = Vec::new();
        for e in &events {
            let rec = a.process(e.clone()).unwrap();
            if let BlockEvent::Reveal { delta, .. } = e {
                for (r, &v) in reference.iter_mut().zip(delta.0.iter()) {
                    *r = r.wrapping_add(v);
                }
            }
            roots.push(a.derived_root());
            let rec_b = b.process(e.clone()).unwrap();
            assert_eq!(rec, rec_b);
        }
        assert_eq!(a.embedding().coord().0, reference);
        assert_eq!(a, b, "full state determinism");
        assert_eq!(a.derived_root(), *roots.last().unwrap());
        assert!(!roots.windows(2).all(|w| w[0] == w[1]), "reveals move D_t");
    }
}
