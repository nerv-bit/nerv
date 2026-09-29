//! The full transaction witness (WP §5.1's private inputs, assembled and
//! validated against the natives): custody (statements 1–5 + revert/burn
//! binding) + the delta module's data (feature vectors, W·ΔS deltas,
//! per-leg digitized plaintexts) + the seal openings (statement-10
//! triplets and expected ciphertexts). AIR-independent by design: the
//! witness is the DATA the AIRs constrain; generation and validation run
//! against nerv-codec / nerv-seal / nerv-custody natives only (DSR-7).

use nerv_core::hash::Hash256;
use nerv_core::INTERVALS_PER_EPOCH;
use nerv_codec::codec_w::CodecW;
use nerv_codec::features::{
    build_leg_features, LegKind, LegMovement,
};
use nerv_seal::circuit_stmt::{apply_statement_10, check_statement_10, epoch_key_identifier};
use nerv_seal::digitize::Plaintext;
use nerv_seal::encrypt::PublicKey;
use nerv_seal::ring::{Vec2, Vec8};
use nerv_seal::sampling::{derive_short_triplet, NoiseSeed};

use nerv_custody::tx::TransactionShell;

use crate::air::custody_air::{bind_witness_to_shell, CustodyWitness, InputWitness, OutputWitness};
use crate::error::WitnessGenError;

/// Build a `LegMovement` from the leg's `LegShell` plus the wallet-side
/// `InputWitness`/`OutputWitness` slices. The classification (`LegKind`)
/// derives from the canonical shell's per-leg pattern: empty-input
/// conditional-output legs are the canonical reversion-bearing kind;
/// everything else follows the movement tuple.
///
/// The `total_legs` argument is reserved for future cross-leg consistency
/// checks (e.g., slot-budget reconciliation across conditional outputs);
/// for now it's a documentation witness.
fn derive_movement(
    leg: &nerv_custody::tx::LegShell,
    total_legs: usize,
    inputs: &[crate::air::custody_air::InputWitness],
    outputs: &[crate::air::custody_air::OutputWitness],
    epoch_length_blocks: u64,
) -> LegMovement {
    let _ = total_legs;
    let kind = if leg.outputs.iter().any(|o| o.conditional) && leg.inputs.nullifiers.is_empty() {
        LegKind::Claim
    } else if !leg.inputs.nullifiers.is_empty() {
        LegKind::CrossShardSpend
    } else {
        LegKind::CrossShardIssue
    };
    LegMovement {
        inputs: inputs
            .iter()
            .map(|iw| (iw.opening.delivery.to_vec(), iw.opening.value))
            .collect(),
        outputs: outputs
            .iter()
            .map(|ow| (ow.opening.delivery.to_vec(), ow.opening.value))
            .collect(),
        fee_nano: leg.fee.as_u64(),
        kind,
        expiry_height: leg.expiry.as_u64(),
        epoch_length_blocks,
    }
}

#[derive(Clone, Debug)]
pub struct DeltaLegWitness {
    pub leg_index: usize,
    pub movement: LegMovement,
    pub features: nerv_codec::features::FeatureVector,
    pub delta: nerv_codec::codec_w::Delta,
}

#[derive(Clone, Debug)]
pub struct DeltaWitness {
    pub weight_version: u64,
    pub legs: Vec<DeltaLegWitness>,
}

#[derive(Clone, Debug)]
pub struct SealLegWitness {
    pub leg_index: usize,
    pub noise_seed: NoiseSeed,
    pub r: Vec8,
    pub e1: Vec8,
    pub e2: Vec2,
    pub plaintext: Plaintext,
    /// The expected ciphertext (u, v) — the statement-10 native output;
    /// the leg's published ct must equal it (binding lands with tx_air).
    pub u: Vec8,
    pub v: Vec2,
}

#[derive(Clone, Debug)]
pub struct TransactionWitness {
    pub custody: CustodyWitness,
    pub delta: DeltaWitness,
    pub seal: Vec<SealLegWitness>,
    pub epoch_key_id: Hash256,
}

impl TransactionWitness {
    pub fn generate(
        shell: &TransactionShell,
        custody: CustodyWitness,
        w: &CodecW,
        epoch_pk: &PublicKey,
        noise_seeds: &[NoiseSeed],
    ) -> Result<TransactionWitness, WitnessGenError> {
        bind_witness_to_shell(&custody, shell)?;
        let canon = shell.canonicalize()?;
        if noise_seeds.len() != canon.legs.len() {
            return Err(WitnessGenError::SeedCount {
                expected: canon.legs.len(),
                found: noise_seeds.len(),
            });
        }
        let wv = canon.legs[0].weight_version;
        if wv != w.version().0 {
            return Err(WitnessGenError::WeightVersion { shell: wv, codec: w.version().0 });
        }
        for leg in canon.legs.iter().skip(1) {
            if leg.weight_version != wv {
                return Err(WitnessGenError::WeightVersion {
                    shell: leg.weight_version,
                    codec: w.version().0,
                });
            }
        }
        let epoch_key_id = Hash256::from_bytes(epoch_key_identifier(epoch_pk.a_seed(), epoch_pk.t()));

        let mut in_cur = 0usize;
        let mut out_cur = 0usize;
        let mut delta_legs = Vec::with_capacity(canon.legs.len());
        let mut seal = Vec::with_capacity(canon.legs.len());
        for (li, leg) in canon.legs.iter().enumerate() {
            let n_in_l = leg.inputs.nullifiers.len();
            let n_out_l = leg.outputs.len();
            let inputs = &custody.inputs[in_cur..in_cur + n_in_l];
            let outputs = &custody.outputs[out_cur..out_cur + n_out_l];
            in_cur += n_in_l;
            out_cur += n_out_l;

            let movement = derive_movement(leg, canon.legs.len(), inputs, outputs, INTERVALS_PER_EPOCH);
            let features = build_leg_features(&movement)?;
            features.check_admissible()?;
            let delta = w.apply(&features);
            if delta.is_zero() {
                return Err(WitnessGenError::ZeroDelta { leg: li });
            }
            delta_legs.push(DeltaLegWitness { leg_index: li, movement, features, delta });

            let (r, e1, e2) = derive_short_triplet(&noise_seeds[li])?;
            let plaintext = nerv_seal::digitize::digitize(&delta.0);
            let (u, v) =
                apply_statement_10(epoch_pk.a_seed(), epoch_pk.t(), &r, &e1, &e2, &plaintext)?;
            check_statement_10(
                epoch_pk.a_seed(), epoch_pk.t(), &u, &v, &plaintext, &r, &e1, &e2,
            )?;
            seal.push(SealLegWitness {
                leg_index: li,
                noise_seed: noise_seeds[li],
                r,
                e1,
                e2,
                plaintext,
                u,
                v,
            });
        }
        Ok(TransactionWitness {
            custody,
            delta: DeltaWitness { weight_version: wv, legs: delta_legs },
            seal,
            epoch_key_id,
        })
    }

    /// Full native re-validation: every derived field recomputed and
    /// compared. The wallet-side double-check and the tamper oracle.
    pub fn validate(
        &self,
        shell: &TransactionShell,
        w: &CodecW,
        epoch_pk: &PublicKey,
    ) -> Result<(), WitnessGenError> {
        bind_witness_to_shell(&self.custody, shell)?;
        let canon = shell.canonicalize()?;
        if self.delta.weight_version != w.version().0 {
            return Err(WitnessGenError::WeightVersion {
                shell: self.delta.weight_version,
                codec: w.version().0,
            });
        }
        if self.seal.len() != canon.legs.len() || self.delta.legs.len() != canon.legs.len() {
            return Err(WitnessGenError::SeedCount {
                expected: canon.legs.len(),
                found: self.seal.len(),
            });
        }
        if self.epoch_key_id.as_bytes() != &epoch_key_identifier(epoch_pk.a_seed(), epoch_pk.t()) {
            return Err(WitnessGenError::EpochKeyId);
        }
        let mut in_cur = 0usize;
        let mut out_cur = 0usize;
        for (li, leg) in canon.legs.iter().enumerate() {
            let n_in_l = leg.inputs.nullifiers.len();
            let n_out_l = leg.outputs.len();
            let inputs = &self.custody.inputs[in_cur..in_cur + n_in_l];
            let outputs = &self.custody.outputs[out_cur..out_cur + n_out_l];
            in_cur += n_in_l;
            out_cur += n_out_l;

            let dw = &self.delta.legs[li];
            if dw.leg_index != li {
                return Err(WitnessGenError::LegIndex { found: dw.leg_index, expected: li });
            }
            let movement = derive_movement(leg, canon.legs.len(), inputs, outputs, INTERVALS_PER_EPOCH);
            if dw.movement != movement {
                return Err(WitnessGenError::MovementMismatch { leg: li });
            }
            let features = build_leg_features(&movement)?;
            if dw.features != features {
                return Err(WitnessGenError::FeaturesMismatch { leg: li });
            }
            features.check_admissible()?;
            let delta = w.apply(&features);
            if dw.delta != delta {
                return Err(WitnessGenError::DeltaMismatch { leg: li });
            }
            if delta.is_zero() {
                return Err(WitnessGenError::ZeroDelta { leg: li });
            }

            let sw = &self.seal[li];
            if sw.leg_index != li {
                return Err(WitnessGenError::LegIndex { found: sw.leg_index, expected: li });
            }
            let (r, e1, e2) = derive_short_triplet(&sw.noise_seed)?;
            if sw.r != r || sw.e1 != e1 || sw.e2 != e2 {
                return Err(WitnessGenError::TripletMismatch { leg: li });
            }
            let plaintext = nerv_seal::digitize::digitize(&delta.0);
            if sw.plaintext != plaintext {
                return Err(WitnessGenError::PlaintextMismatch { leg: li });
            }
            check_statement_10(
                epoch_pk.a_seed(), epoch_pk.t(), &sw.u, &sw.v, &plaintext, &r, &e1, &e2,
            )?;
            let (u, v) =
                apply_statement_10(epoch_pk.a_seed(), epoch_pk.t(), &r, &e1, &e2, &plaintext)?;
            if sw.u != u || sw.v != v {
                return Err(WitnessGenError::CiphertextMismatch { leg: li });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::air::custody_air::{BurnWitness, InputWitness, OutputWitness, RevertWitness};
    use crate::testutil::SplitMix64;
    use nerv_codec::codec_w::WeightVersion;
    use nerv_codec::features::RAIL_COUNT;
    use nerv_codec::weight_gen::{certify, expand, BeaconRandomness, CertConfig};
    use nerv_core::types::{FeeSats, Height, ShardSet};
    use nerv_custody::nct::NoteCommitmentTree;
    use nerv_custody::tx::{InputSet, LegShell, Output};
    use nerv_custody::{MasterSeed, WalletKeys};
    use nerv_seal::encrypt::derive_reference_keypair;

    const DEPTH: usize = nerv_custody::nct::DEPTH;

    struct Fixture {
        shell: TransactionShell,
        custody: CustodyWitness,
        seeds: Vec<NoiseSeed>,
    }

    fn fixture(seed: u64) -> Fixture {
        let mut rng = SplitMix64::new(seed);
        let wk = WalletKeys::from_master(&MasterSeed::from_bytes(rng.bytes32()));
        let det = wk.viewing();
        let g = ShardSet::genesis();
        let addr = |i: u64| nerv_custody::Address::generate(det, wk.nullifier_key(), i, &g).unwrap();
        let opening = |v: u64, i: u64| nerv_custody::commitment::NoteOpening {
            value: v,
            rho: rng.bytes32(),
            delivery: *addr(i).delivery().as_bytes(),
            blinding: rng.bytes32(),
            pk_n: *addr(i).pk_n(),
        };
        let nanos = 1_000_000_000u64;
        let in1 = opening(50 * nanos, 0);
        let in2 = opening(35 * nanos, 1);
        let out_carol = opening(25 * nanos, 2);
        let out_change = opening(9_999 * nanos / 1000, 3);
        let out_bob = opening(50 * nanos, 4);
        let revert_bob = opening(50 * nanos, 4);

        let mut tree = NoteCommitmentTree::new();
        for _ in 0..6 {
            tree.append(&Hash256::from_bytes(rng.bytes32())).unwrap();
        }
        let idx1 = tree.append(&in1.commitment().unwrap()).unwrap();
        let idx2 = tree.append(&in2.commitment().unwrap()).unwrap();
        let root = tree.root();
        let sib = |idx: u64| -> Vec<[nerv_core::field::Goldilocks; 4]> {
            tree.witness(idx).unwrap().siblings.iter().map(|d| d.to_elements().unwrap()).collect()
        };
        let inputs = vec![
            InputWitness {
                opening: in1,
                nullifier_key: wk.nullifier_key_at(0),
                leaf_index: idx1,
                siblings: sib(idx1),
                anchor: root,
            },
            InputWitness {
                opening: in2,
                nullifier_key: wk.nullifier_key_at(1),
                leaf_index: idx2,
                siblings: sib(idx2),
                anchor: root,
            },
        ];
        let nf1 = nerv_custody::nullifier::derive_nullifier(&inputs[0].nullifier_key, &inputs[0].opening.rho);
        let nf2 = nerv_custody::nullifier::derive_nullifier(&inputs[1].nullifier_key, &inputs[1].opening.rho);
        let mk_out = |o: &nerv_custody::commitment::NoteOpening, cond: bool| Output {
            cm: o.commitment().unwrap(),
            sealed_note: vec![0xA5; 48],
            value: o.value,
            conditional: cond,
            revert_cm: if cond { Some(revert_bob.commitment().unwrap()) } else { None },
        };

       let leg7 = LegShell {
            shard: g.ids()[7],
            inputs: InputSet::new(vec![nf1, nf2]),
            outputs: vec![mk_out(&out_carol, false), mk_out(&out_change, false)],
            fee: FeeSats::from_u64(600_000),
            anchor: root.as_hash256(),
            expiry: Height::from_u64(5_000),
            weight_version: 1,
            ct: vec![],
            burns: vec![],
        };
        let leg40 = LegShell {
            shard: g.ids()[40],
            inputs: InputSet::new(vec![]),
            outputs: vec![mk_out(&out_bob, true)],
            fee: FeeSats::from_u64(400_000),
            anchor: Hash256::from_bytes(rng.bytes32()),
            expiry: Height::from_u64(5_000),
            weight_version: 1,
            ct: vec![],
            burns: vec![],
        };

        let shell = TransactionShell { legs: vec![leg7, leg40] };
        let custody = CustodyWitness {
            inputs,
            outputs: vec![
                OutputWitness { opening: out_carol },
                OutputWitness { opening: out_change },
                OutputWitness { opening: out_bob },
            ],
            reverts: vec![RevertWitness { opening: revert_bob }],
            burns: vec![],
            fees: vec![600_000, 400_000],
        };
        let seeds = vec![NoiseSeed::from_bytes(rng.bytes32()), NoiseSeed::from_bytes(rng.bytes32())];
        Fixture { shell, custody, seeds }
    }

    fn w() -> nerv_codec::codec_w::CodecW {
        let w = expand(&BeaconRandomness::from_bytes([0x57; 32]), WeightVersion(1));
        certify(&w, CertConfig { spark_samples_per_size: 2, column_rail_samples: 1 }).unwrap();
        w
    }

    fn epoch_pk() -> PublicKey {
        derive_reference_keypair(&[0xE0; 32]).unwrap().0
    }

    #[test]
    fn generate_and_validate_full_differential() {
        let f = fixture(0x57A1);
        let w = w();
        let pk = epoch_pk();
        let tw = TransactionWitness::generate(&f.shell, f.custody.clone(), &w, &pk, &f.seeds)
            .unwrap();
        tw.validate(&f.shell, &w, &pk).unwrap();

        // Determinism.
        let tw2 = TransactionWitness::generate(&f.shell, f.custody, &w, &pk, &f.seeds).unwrap();
        assert_eq!(tw.delta.legs.len(), 2);
        assert_eq!(tw.seal.len(), 2);
        assert_eq!(tw.seal[0].plaintext, tw2.seal[0].plaintext);
        assert_eq!(tw.seal[0].u, tw2.seal[0].u);
        assert_eq!(tw.delta.legs[0].delta, tw2.delta.legs[0].delta);
        assert_eq!(tw.epoch_key_id, tw2.epoch_key_id);

        // Delta properties: non-zero, features admissible, kind derivation.
        assert!(!tw.delta.legs[0].delta.is_zero());
        assert_eq!(tw.delta.legs[0].movement.kind, LegKind::CrossShardSpend);
        assert_eq!(tw.delta.legs[1].movement.kind, LegKind::CrossShardIssue);
        assert_eq!(tw.delta.legs[0].features.value(RAIL_COUNT), 1);
    }

    #[test]
    fn tampering_is_caught_by_validate() {
        let f = fixture(0x57A2);
        let (w, pk) = (w(), epoch_pk());
        let mut tw =
            TransactionWitness::generate(&f.shell, f.custody.clone(), &w, &pk, &f.seeds).unwrap();

        let mut t = tw.clone();
        t.seal[0].r = crate_seal_tamper(&t.seal[0].r);
        assert!(matches!(t.validate(&f.shell, &w, &pk), Err(WitnessGenError::TripletMismatch { leg: 0 })));

        let mut t = tw.clone();
        t.seal[1].plaintext =
            nerv_seal::digitize::digitize(&[7u64; 64]);
        assert!(matches!(t.validate(&f.shell, &w, &pk), Err(WitnessGenError::PlaintextMismatch { .. })));

        let mut t = tw.clone();
        t.delta.legs[0].delta = nerv_codec::codec_w::Delta([1u64; 64]);
        assert!(matches!(t.validate(&f.shell, &w, &pk), Err(WitnessGenError::DeltaMismatch { .. })));

        let mut t = tw.clone();
        t.custody.inputs[0].nullifier_key[0] ^= 1;
        assert!(t.validate(&f.shell, &w, &pk).is_err());

        let mut t = tw.clone();
        t.epoch_key_id = Hash256::from_bytes([9u8; 32]);
        assert!(matches!(t.validate(&f.shell, &w, &pk), Err(WitnessGenError::EpochKeyId)));
        let _ = &mut tw;
    }

    fn crate_seal_tamper(r: &Vec8) -> Vec8 {
        let mut polys = *r.polys();
        let cl = polys[0].centerlift();
        let mut cl2 = cl;
        cl2[0] += 1;
        polys[0] = nerv_seal::ring::Poly::from_centered(&cl2);
        Vec8::new(polys)
    }

    #[test]
    fn inadmissible_features_rejected() {
        // 1 input, 7 outputs to 7 distinct addresses: 8 active slots > 6.
        let mut rng = SplitMix64::new(0x57A3);
        let wk = WalletKeys::from_master(&MasterSeed::from_bytes(rng.bytes32()));
        let det = wk.viewing();
        let g = ShardSet::genesis();
        let addr = |i: u64| nerv_custody::Address::generate(det, wk.nullifier_key(), i, &g).unwrap();
        let mk = |i: u64| nerv_custody::commitment::NoteOpening {
            value: 100,
            rho: rng.bytes32(),
            delivery: *addr(i).delivery().as_bytes(),
            blinding: rng.bytes32(),
            pk_n: *addr(i).pk_n(),
        };
        let input = mk(0);
        let outs: Vec<_> = (1..=7).map(mk).collect();
        let mut tree = NoteCommitmentTree::new();
        let idx = tree.append(&input.commitment().unwrap()).unwrap();
        let root = tree.root();
        let sibs: Vec<[nerv_core::field::Goldilocks; 4]> = tree
            .witness(idx)
            .unwrap()
            .siblings
            .iter()
            .map(|d| d.to_elements().unwrap())
            .collect();
        let nf = nerv_custody::nullifier::derive_nullifier(&wk.nullifier_key_at(0), &input.rho);
         let shell = TransactionShell {
                legs: vec![LegShell {
                    shard: g.ids()[7],
                    inputs: InputSet::new(vec![nf]),
                    outputs: outs
                        .iter()
                        .map(|o| Output {
                            cm: o.commitment().unwrap(),
                            sealed_note: vec![0; 16],
                            value: o.value,
                            conditional: false,
                            revert_cm: None,
                        })
                        .collect(),
                    fee: FeeSats::from_u64(1000),
                    anchor: root.as_hash256(),
                    expiry: Height::from_u64(5_000),
                    weight_version: 1,
                    ct: vec![],
                    burns: vec![],
                }],
            };

        let custody = CustodyWitness {
            inputs: vec![InputWitness {
                opening: input,
                nullifier_key: wk.nullifier_key_at(0),
                leaf_index: idx,
                siblings: sibs,
                anchor: root,
            }],
            outputs: outs.into_iter().map(|o| OutputWitness { opening: o }).collect(),
            reverts: vec![],
            burns: vec![],
            fees: vec![700],
        };
        let seeds = vec![NoiseSeed::from_bytes(rng.bytes32())];
        let (w, pk) = (w(), epoch_pk());
        assert!(matches!(
            TransactionWitness::generate(&shell, custody, &w, &pk, &seeds),
            Err(WitnessGenError::Inadmissible(_))
        ));
    }

    #[test]
    fn zero_delta_and_version_and_seed_errors() {
        let f = fixture(0x57A4);
        let (w, pk) = (w(), epoch_pk());
        let zero_w = nerv_codec::codec_w::CodecW::new(
            WeightVersion(1),
            [[0i16; 256]; 64],
        );
        assert!(matches!(
            TransactionWitness::generate(&f.shell, f.custody.clone(), &zero_w, &pk, &f.seeds),
            Err(WitnessGenError::ZeroDelta { leg: 0 })
        ));
        let bad_seeds = vec![NoiseSeed::from_bytes([1u8; 32])];
        assert!(matches!(
            TransactionWitness::generate(&f.shell, f.custody.clone(), &w, &pk, &bad_seeds),
            Err(WitnessGenError::SeedCount { expected: 2, found: 1 })
        ));
        let mut shell = f.shell.clone();
        shell.legs[0].weight_version = 2;
        assert!(matches!(
            TransactionWitness::generate(&shell, f.custody.clone(), &w, &pk, &f.seeds),
            Err(WitnessGenError::WeightVersion { shell: 2, codec: 1 })
        ));
    }
}
