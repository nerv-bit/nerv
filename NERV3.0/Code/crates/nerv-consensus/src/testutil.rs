#[derive(Clone)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> SplitMix64 {
        SplitMix64 { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(crate) fn bytes32(&mut self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for chunk in out.chunks_exact_mut(8) {
            chunk.copy_from_slice(&self.next_u64().to_le_bytes());
        }
        out
    }
}

/// The chunk-14 part-2 test harness: deterministic headers, key rosters,
/// QCs, and the slash fixtures built on real state/registry surfaces.
pub(crate) mod harness {
    use super::SplitMix64;
    use crate::finality::ShardFinality;
    use nerv_codec::codec_w::{CodecW, WeightVersion};
    use nerv_codec::weight_gen::{certify, expand, BeaconRandomness, CertConfig};
    use nerv_core::constants::CT_BATCH;
    use nerv_core::field::Goldilocks;
    use nerv_core::hash::Hash256;
    use nerv_core::types::{
        Epoch, FeeSats, Height, Interval, kappa, ShardId, ShardSet, TxId,
    };
    use nerv_crypto::mldsa::{EncapsulationKey, SigningKey, VerifyingKey};
    use nerv_crypto::sigaggr::{vote_bytes, QuorumCertificate, VoteCollector};
    use nerv_custody::nct::{NctDigest, NoteCommitmentTree};
    use nerv_custody::tx::{InputSet, LegShell, Output, TransactionShell};
    use nerv_custody::Address;
    use nerv_proofs::stark::ext_field::ExtF;
    use nerv_proofs::stark::fri::FriProof;
    use nerv_proofs::stark::compose::ComposedProof;
    use nerv_proofs::stark::prover::Proved;
    use nerv_proofs::{FriShape, TransactionProof};
    use nerv_registry::challenge::InclusionChallenge;
    use nerv_registry::mempool::{PoolEntry, VerifyContext};
    use nerv_seal::circuit_stmt::epoch_key_identifier;
    use nerv_seal::encrypt::{derive_reference_keypair, Ciphertext, PublicKey};
    use nerv_state::block::{SettledLeg, ShardBlock};
    use nerv_state::fraud::{BeaconFacts, FraudCondition, FraudProof, ReexecutionFraud};
    use nerv_state::header::{RegistryRef, ShardHeader};
    use nerv_state::ttau::{RegistryWitness, TauTree};
    use nerv_state::{BeaconView, ChainSource, ShardState};
    use std::collections::BTreeMap;
    use std::sync::OnceLock;

    pub(crate) fn params_root() -> Hash256 {
        Hash256::from_bytes([0x9C; 32])
    }

    pub(crate) fn h(rng: &mut SplitMix64) -> Hash256 {
        Hash256::from_bytes(rng.bytes32())
    }

    /// A chain source with no history (no escrows to recover).
   pub(crate) struct NoChain;


   impl ChainSource for NoChain {
       fn settled_leg(
           &self,
           _height: Height,
           _key: &nerv_core::types::LegKey,
       ) -> Option<nerv_state::block::ResolvedLeg> {
           None
       }
   }


   pub(crate) fn dummy_chain() -> NoChain {
       NoChain
   }


   /// The producer payout address.
   pub(crate) fn payout() -> Address {
       static PAYOUT: OnceLock<Address> = OnceLock::new();
       PAYOUT
           .get_or_init(|| {
               let mut rng = SplitMix64::new(0xFA9);
               let mut d = [0u8; 1184];
               for c in d.chunks_exact_mut(8) {
                   c.copy_from_slice(&rng.next_u64().to_le_bytes());
               }
               let ek = EncapsulationKey::from_bytes(d);
               let set = ShardSet::genesis();
               let tag = set.home_kappa(&kappa(ek.as_bytes())).unwrap();
               Address::new(ek, tag, rng.bytes32()).unwrap()
           })
           .clone()
   }

    pub(crate) fn header(shard: ShardId, height: u64, seed: u64) -> ShardHeader {
        let mut rng = SplitMix64::new(seed ^ 0xCC00_0000);
       ShardHeader {
           prev: h(&mut rng),
           height: Height::from_u64(height),
           nct_root: NctDigest::from_elements(&[
               Goldilocks::from_u32((height % 4_000_000_000) as u32),
               Goldilocks::ONE,
               Goldilocks::ZERO,
               Goldilocks::ZERO,
           ]),

            nullifier_root: h(&mut rng),
            transit_root: h(&mut rng),
            params_root: params_root(),
            derived: rng.bytes32(),
            ct_batch_hash: h(&mut rng),
            prev_reveal: None,
            registry: RegistryRef { interval: Interval::from_u64(86_401), root: h(&mut rng) },
            fee_total: FeeSats::from_u64(1000),
            producer_payout: payout(),
            qc_hash: h(&mut rng),
        }
    }

    /// n aligned (signing, verifying) key pairs.
    pub(crate) fn keys(n: u64) -> (Vec<SigningKey>, Vec<VerifyingKey>) {
        let keys: Vec<SigningKey> = (0..n)
            .map(|i| {
                let mut b = [0u8; 32];
                b[..8].copy_from_slice(&(0x517_u64 + i).to_le_bytes());
                b[24..32].copy_from_slice(&i.to_le_bytes());
                SigningKey::from_seed(&b).unwrap()
            })
            .collect();
        let vks: Vec<VerifyingKey> = keys.iter().map(|k| *k.verifying_key()).collect();
        (keys, vks)
    }

    pub(crate) fn vks_of(keys: &[SigningKey]) -> Vec<VerifyingKey> {
        keys.iter().map(|k| *k.verifying_key()).collect()
    }

    pub(crate) fn qc_for(
        subject: Hash256,
        epoch: Epoch,
        signers: impl Iterator<Item = usize>,
        keys: &[SigningKey],
        quorum: usize,
    ) -> QuorumCertificate {
        let roster = vks_of(keys);
        let mut vc = VoteCollector::new(epoch, subject);
        for i in signers {
            let sig = keys[i].sign(&vote_bytes(epoch, &subject)).unwrap();
            vc.add(i, sig, &roster).unwrap();
        }
        vc.assemble(quorum).unwrap()
    }

     /// A per-epoch randomness map for committee selection. Call sites
   /// close over it: `let r = &map; ... &|e| r.get(e)`.
   #[derive(Clone, Default)]
   pub(crate) struct RandomnessMap {
       pub map: BTreeMap<u64, Hash256>,
   }


   impl RandomnessMap {
       pub(crate) fn with(entries: Vec<(u64, Hash256)>) -> RandomnessMap {
           RandomnessMap { map: entries.into_iter().collect() }
       }


       pub(crate) fn get(&self, e: Epoch) -> Option<Hash256> {
           self.map.get(&e.as_u64()).copied()
       }
   }


    /// A beacon view over registered roots.
    #[derive(Default)]
    pub(crate) struct TestView {
        pub tau: BTreeMap<u64, Hash256>,
        pub transit: BTreeMap<(ShardId, u64), Hash256>,
    }

    impl BeaconView for TestView {
        fn tau_root(&self, interval: Interval) -> Option<Hash256> {
            self.tau.get(&interval.as_u64()).copied()
        }
        fn transit_root(&self, shard: ShardId, height: Height) -> Option<Hash256> {
            self.transit.get(&(shard, height.as_u64())).copied()
        }
    }

    /// A chain source with no history (no escrows to recover).
    pub(crate) struct NoChain;

    impl ChainSource for NoChain {
        fn settled_leg(
            &self,
            _height: Height,
            _key: &nerv_core::types::LegKey,
        ) -> Option<nerv_state::block::ResolvedLeg> {
            None
        }
    }

    // -- slash fixtures ------------------------------------------------------

    pub(crate) fn fri() -> FriShape {
        FriShape {
            log_blowup: 4,
            num_queries: 56,
            log_final_poly_len: 1,
            max_log_arity: FriShape::DEFAULT_ARITY,
            commit_pow_bits: 0,
            query_pow_bits: 0,
        }
    }

    fn w() -> CodecW {
        let w = expand(&BeaconRandomness::from_bytes([0x57; 32]), WeightVersion(1));
        certify(&w, CertConfig { spark_samples_per_size: 2, column_rail_samples: 1 }).unwrap();
        w
    }

    fn epoch_pk() -> PublicKey {
        derive_reference_keypair(&[0xE3; 32]).unwrap().0
    }

    pub(crate) fn verify_ctx() -> VerifyContext {
        VerifyContext::new(fri(), w(), epoch_pk())
    }

    /// A shape-inconsistent garbage transaction proof: verification
    /// rejects at the publics-shape check.
    pub(crate) fn garbage_proof() -> TransactionProof {
        TransactionProof {
            proved: Proved {
                proof: ComposedProof {
                    trace_root: Hash256::from_bytes([1u8; 32]),
                    quotient_root: Hash256::from_bytes([2u8; 32]),
                    trace_zeta: Vec::new(),
                    trace_zeta_next: Vec::new(),
                    quotient_zeta: ExtF::ZERO,
                    num_assertions: 0,
                    fri: FriProof {
                        roots: Vec::new(),
                        final_coeffs: Vec::new(),
                        pow_nonce: 0,
                        queries: Vec::new(),
                        positions: Vec::new(),
                    },
                    outer: Vec::new(),
                },
                log_n: 2,
                width: 33,
                prep: Vec::new(),
            },
            publics: Vec::new(),
        }
    }

    fn dummy_ct(rng: &mut SplitMix64) -> Vec<u8> {
        let mut ct = vec![0u8; Ciphertext::WIRE_SIZE];
        for c in ct.chunks_exact_mut(4) {
            c.copy_from_slice(&(rng.next_u32() & 0xFFF0_0000).to_le_bytes());
        }
        ct
    }

    fn shell(seed: u64) -> TransactionShell {
        let mut rng = SplitMix64::new(seed ^ 0x5EED_0000);
        let set = ShardSet::genesis();
        TransactionShell {
            legs: vec![LegShell {
                shard: set.ids()[7],
                inputs: InputSet::new(vec![h(&mut rng)]),
                outputs: vec![Output {
                    cm: h(&mut rng),
                    sealed_note: vec![0xA5; 48],
                    value: 1_000_000_000,
                    conditional: false,
                    revert_cm: None,
                }],
                fee: FeeSats::from_u64(1000),
                anchor: h(&mut rng),
                expiry: Height::from_u64(5_000),
                weight_version: 1,
                ct: dummy_ct(&mut rng),
                burns: vec![],
            }],
        }
    }

    /// A block at height 1 over one settled single-shard leg, with a
    /// header whose fee total disagrees with the leg's fee — the
    /// FeeTotalMismatch fraud condition — plus its view registration.
    fn fee_mismatch_block(seed: u64) -> (ShardBlock, TestView, ShardState) {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let st = ShardState::genesis(shard, params_root());
        let canon = shell(seed).canonicalize().unwrap();
        let txid = nerv_state::canonical_txid(&canon);
        let tree = TauTree::from_sorted(&[txid]).unwrap();
        let mut view = TestView::default();
        let interval = 86_401u64;
        view.tau.insert(interval, tree.root());

        let mut rng = SplitMix64::new(seed ^ 0xB10C);
        let (keys, _) = keys(21);
        let mut hdr = header(shard, 1, seed);
        hdr.prev = st.state_commitment();
        hdr.registry = RegistryRef { interval: Interval::from_u64(interval), root: tree.root() };
        hdr.fee_total = FeeSats::from_u64(1299); // ≠ the leg's 1000 (mismatch probe)
        hdr.nct_root = st.nct().root(); // unchanged: no outputs apply
        hdr.nullifier_root = st.nullifiers().root();
        hdr.transit_root = st.transit().root();
        let leg = SettledLeg {
            shell: canon,
            leg: nerv_core::types::LegIndex::FIRST,
            tau: tree.witness(0).unwrap(),
            siblings: vec![],
        };
        let qc = qc_for(hdr.header_hash(), Epoch::from_u64(1), 0..15, &keys, 15);
        hdr.qc_hash = qc.qc_hash();
        let block = ShardBlock {
            shard,
            header: hdr,
            legs: vec![leg],
            reversions: vec![],
            claims: vec![],
            qc,
        };
        let _ = &mut rng;
        (block, view, st)
    }

    /// The minimal fraud class: an invalid block + matching facts.
    pub(crate) fn fee_fraud_fixture(seed: u64) -> (FraudProof, TestView) {
        let (block, view, _) = fee_mismatch_block(seed);
        let proof = FraudProof {
            facts: BeaconFacts::for_block(&block, &view),
            condition: FraudCondition::FeeTotalMismatch,
            block,
        };
        (proof, view)
    }

    /// The reexecution class: the same invalid block, to be run against
    /// its genesis predecessor.
    pub(crate) fn reexec_fixture(seed: u64) -> (ReexecutionFraud, ShardState, TestView) {
        let (block, view, st) = fee_mismatch_block(seed);
        (ReexecutionFraud { block }, st, view)
    }

    /// A sustained inclusion challenge: a real T_τ witness over the
    /// shell's txid and a garbage proof (verification fails at the
    /// publics shape — the challenge's outcome is Sustained).
    pub(crate) fn inclusion_fixture(seed: u64) -> (InclusionChallenge, Hash256, VerifyContext) {
        let canon = shell(seed ^ 0x1DCE).canonicalize().unwrap();
        let txid = nerv_state::canonical_txid(&canon);
        let tree = TauTree::from_sorted(&[txid]).unwrap();
        let interval = Interval::from_u64(5);
        let witness = RegistryWitness::new(interval, tree.witness(0).unwrap());
        let challenge = InclusionChallenge {
            witness,
            shell: canon,
            proof: garbage_proof(),
        };
        (challenge, tree.root(), verify_ctx())
    }

    /// A bundle of two transactions whose proofs are garbage —
    /// verification fails at index 0.
    pub(crate) fn bundle_fixture(seed: u64) -> (nerv_registry::Bundle, VerifyContext) {
        let mut entries = Vec::new();
        for k in 0..2u64 {
            let canon = shell(seed ^ 0xB0B0_0000 ^ k).canonicalize().unwrap();
            entries.push(PoolEntry {
                txid: nerv_state::canonical_txid(&canon),
                shell: canon,
                proof: garbage_proof(),
            });
        }
        let mut b = [0u8; 32];
        b[0] = 0xA7;
        b[24..32].copy_from_slice(&0xA66_u64.to_le_bytes());
        let sk = SigningKey::from_seed(&b).unwrap();
        let bundle = nerv_registry::Bundle::build(&sk, entries).unwrap();
        (bundle, verify_ctx())
    }

    // Re-exported for the tests that walk a finalized chain.
    pub(crate) use nerv_state::canonical_txid;
    pub(crate) fn _unused(_: &ShardFinality) {}
}
