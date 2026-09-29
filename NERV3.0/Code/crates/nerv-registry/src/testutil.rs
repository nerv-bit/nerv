

/// The chunk-14 test harness: real epoch keys, real certified codecs, the
/// frozen FRI shape, real-seal shells, and shape-consistent garbage
/// proofs that exercise the full verification pipeline before failing.
pub(crate) mod harness {
    use super::SplitMix64;
    use crate::mempool::{PoolEntry, VerifyContext};
    use nerv_codec::codec_w::{CodecW, WeightVersion};
    use nerv_codec::weight_gen::{certify, expand, BeaconRandomness, CertConfig};
    use nerv_core::field::Goldilocks;
    use nerv_core::hash::Hash256;
    use nerv_core::types::{FeeSats, Height, ShardSet, TxId};
    use nerv_crypto::mldsa::{SigningKey, VerifyingKey};
    use nerv_custody::tx::{InputSet, LegShell, Output, TransactionShell};
    use nerv_proofs::air::chips::{gen_range_witness, RangeChip};
    use nerv_proofs::air::tx_air::{build_tx_air, SEAL_PUBS_PER_LEG};
    use nerv_proofs::{ComposedProof, FriShape, FsTranscript, Prover, Proved, TransactionProof};
    use nerv_seal::digitize::{digitize, COORDS};
    use nerv_seal::encrypt::{derive_reference_keypair, Ciphertext, PublicKey};
    use nerv_seal::sampling::NoiseSeed;
    use std::sync::OnceLock;


    /// The frozen [proofs.fri] configuration.
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


    pub(crate) fn w() -> &'static CodecW {
        static W: OnceLock<CodecW> = OnceLock::new();
        W.get_or_init(|| {
            let w = expand(&BeaconRandomness::from_bytes([0x57; 32]), WeightVersion(1));
            certify(&w, CertConfig { spark_samples_per_size: 2, column_rail_samples: 1 }).unwrap();
            w
        })
    }


    pub(crate) fn epoch_pk() -> &'static PublicKey {
        static PK: OnceLock<PublicKey> = OnceLock::new();
        PK.get_or_init(|| derive_reference_keypair(&[0xE0; 32]).unwrap().0)
    }


    pub(crate) fn ctx() -> VerifyContext {
        VerifyContext::new(fri(), w().clone(), epoch_pk().clone())
    }


    fn ct_key() -> &'static PublicKey {
        static K: OnceLock<PublicKey> = OnceLock::new();
        K.get_or_init(|| derive_reference_keypair(&[0xE1; 32]).unwrap().0)
    }


    fn ct_bytes(rng: &mut SplitMix64) -> Vec<u8> {
        let mut coords = [0u64; COORDS];
        for v in coords.iter_mut() {
            *v = rng.next_u64();
        }
        let m = digitize(&coords);
        let mut nb = [0u8; 32];
        nb[..8].copy_from_slice(&rng.next_u64().to_le_bytes());
        Ciphertext::encrypt(ct_key(), &NoiseSeed::from_bytes(nb), &m)
            .unwrap()
            .to_bytes()
            .to_vec()
    }


    fn leg(rng: &mut SplitMix64, shard: nerv_core::types::ShardId, n_in: usize, fee: u64, ct: &[u8]) -> LegShell {
        let set = ShardSet::genesis();
        let _ = set;
        LegShell {
            shard,
            inputs: InputSet::new((0..n_in).map(|_| Hash256::from_bytes(rng.bytes32())).collect()),
            outputs: vec![Output {
                cm: Hash256::from_bytes(rng.bytes32()),
                sealed_note: vec![0xA5; 48],
                value: 1_000_000_000,
                conditional: false,
                revert_cm: None,
            }],
            fee: FeeSats::from_u64(fee),
            anchor: Hash256::from_bytes(rng.bytes32()),
            expiry: Height::from_u64(5_000),
            weight_version: 1,
            ct: ct.to_vec(),
            burns: vec![],
        }
    }


    /// A single-shard 1-in/1-out shell with a real sealed-delta ciphertext.
    pub(crate) fn shell(seed: u64) -> TransactionShell {
        let mut rng = SplitMix64::new(seed ^ 0xB0B0_0000);
        let set = ShardSet::genesis();
        let ct = ct_bytes(&mut rng);
        TransactionShell { legs: vec![leg(&mut rng, set.ids()[7], 1, 1000, &ct)] }
    }


    /// A cross-shard 2-leg shell (spend@7, issue@40).
    pub(crate) fn shell2(seed: u64) -> TransactionShell {
        let mut rng = SplitMix64::new(seed ^ 0xB0B0_0001);
        let set = ShardSet::genesis();
        let ct = ct_bytes(&mut rng);
        TransactionShell {
            legs: vec![
                leg(&mut rng, set.ids()[7], 1, 600, &ct),
                leg(&mut rng, set.ids()[40], 0, 400, &ct),
            ],
        }
    }


    pub(crate) fn txid_of(shell: &TransactionShell) -> TxId {
        let canon = shell.canonicalize().unwrap();
        nerv_state::canonical_txid(&canon)
    }


    pub(crate) fn entry_for(shell: &TransactionShell) -> PoolEntry {
        let canon = shell.canonicalize().unwrap();
        PoolEntry {
            txid: nerv_state::canonical_txid(&canon),
            shell: canon,
            proof: shallow_proof(),
        }
    }


    /// A real small-AIR STARK (the composed proof of a verified RangeChip
    /// statement under the frozen FRI shape) — the invalid transaction
    /// proofs' cryptographic payload.
    pub(crate) fn small_composed() -> &'static ComposedProof {
        static P: OnceLock<ComposedProof> = OnceLock::new();
        P.get_or_init(|| {
            let mut rng = SplitMix64::new(0x5C);
            let rows: Vec<Vec<Goldilocks>> = (0..4)
                .map(|_| gen_range_witness(rng.next_u64() % (1 << 20), 32))
                .collect();
            let mut t = FsTranscript::new();
            t.absorb_bytes(&0x5C_u64.to_le_bytes());
            Prover::new(fri())
                .prove(&RangeChip::new(0, 1, 32), &rows, &[], &[], &mut t)
                .unwrap()
                .proof
        })
    }


    /// A shape-inconsistent garbage proof: fails at the first statement
    /// check (publics count).
    pub(crate) fn shallow_proof() -> TransactionProof {
        TransactionProof {
            proved: Proved {
                proof: small_composed().clone(),
                log_n: 2,
                width: 33,
                prep: Vec::new(),
            },
            publics: Vec::new(),
        }
    }


    /// A shape-CONSISTENT garbage proof for `canon`: correct log_n, width,
    /// and publics count; the seal publics are the shell's own ct words,
    /// so the ct-binding PASSES and verification runs gen_tx_prep and the
    /// engine before rejecting at the STARK layer.
    pub(crate) fn deep_garbage(canon: &TransactionShell) -> TransactionProof {
        let air = build_tx_air(canon).unwrap();
        let log_n = air.rows().max(2).next_power_of_two().ilog2() as usize;
        let mut publics = vec![Goldilocks::ZERO; air.pub_count()];
        for (l, leg) in canon.legs.iter().enumerate() {
            let spb = air.seal_public_base + SEAL_PUBS_PER_LEG * l;
            for (i, w) in leg.ct.chunks_exact(4).enumerate() {
                let mut b = [0u8; 4];
                b.copy_from_slice(w);
                publics[spb + i] = Goldilocks::from_u32(u32::from_le_bytes(b));
            }
        }
        TransactionProof {
            proved: Proved {
                proof: small_composed().clone(),
                log_n,
                width: air.cols(),
                prep: Vec::new(),
            },
            publics,
        }
    }


    pub(crate) fn agg_key(tag: u8) -> SigningKey {
        let mut b = [0u8; 32];
        b[0] = tag;
        b[24..32].copy_from_slice(&0xA66_u64.to_le_bytes());
        SigningKey::from_seed(&b).unwrap()
    }


    /// A 21-member committee roster (registry parameters) with 15 signers.
    pub(crate) fn committee() -> (Vec<SigningKey>, Vec<VerifyingKey>) {
        committee_offset(0x600)
    }


    pub(crate) fn committee_offset(base: u64) -> (Vec<SigningKey>, Vec<VerifyingKey>) {
        let sks: Vec<SigningKey> = (0..21u64)
            .map(|i| {
                let mut b = [0u8; 32];
                b[..8].copy_from_slice(&(base + i).to_le_bytes());
                b[24..32].copy_from_slice(&i.to_le_bytes());
                SigningKey::from_seed(&b).unwrap()
            })
            .collect();
        let vks: Vec<VerifyingKey> = sks.iter().map(|k| *k.verifying_key()).collect();
        (sks, vks)
    }


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

}
