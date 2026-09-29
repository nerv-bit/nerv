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

    pub(crate) fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    pub(crate) fn bytes32(&mut self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for chunk in out.chunks_exact_mut(8) {
            chunk.copy_from_slice(&self.next_u64().to_le_bytes());
        }
        out
    }
}

pub(crate) fn node_keys(seed: u64) -> (
    nerv_crypto::mldsa::SigningKey,
    nerv_crypto::mlkem::EncapsulationKey,
    nerv_crypto::mlkem::DecapsulationKey,
) {
    let mut rng = SplitMix64::new(seed);
    let mut sign_seed = [0u8; 32];
    for c in sign_seed.chunks_mut(8) {
        c.copy_from_slice(&rng.next_u64().to_le_bytes());
    }
    let mut kem_seed = [0u8; 64];
    for c in kem_seed.chunks_mut(8) {
        c.copy_from_slice(&rng.next_u64().to_le_bytes());
    }
    let (ek, dk) = nerv_crypto::mlkem::keypair_from_seed(&kem_seed).unwrap();
    (nerv_crypto::mldsa::SigningKey::from_seed(&sign_seed).unwrap(), ek, dk)
}

/// The chunk-15 test harness: bound nodes, transaction entries, and the
/// gossip/submission fixtures.
pub(crate) mod harness {
    use super::{node_keys, SplitMix64};
    use crate::host::{Host, HostConfig, HostEvent, PeerInfo};
    use std::net::SocketAddr;
    use std::sync::OnceLock;
    use tokio::sync::mpsc;

    /// A bound node: its host handle, event stream, and dialable info.
    pub(crate) struct Node {
        pub host: Host,
        pub events: mpsc::Receiver<HostEvent>,
        pub info: PeerInfo,
    }

     pub(crate) async fn node(seed: u64) -> Node {
       node_full(seed).await.0
   }


   /// The node plus its signing key and static KEM decapsulation key —
   /// the relay-runtime and registry fixtures need them.
   pub(crate) async fn node_full(
       seed: u64,
   ) -> (Node, nerv_crypto::mldsa::SigningKey, nerv_crypto::mlkem::DecapsulationKey) {
       let (sk, ek, dk) = node_keys(seed);
       let vk = *sk.verifying_key();
       let cfg = HostConfig::new(sk.clone(), dk.clone());
       let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
       let (host, events, local) = Host::bind(cfg, addr).await.unwrap();
       (Node { host, events, info: PeerInfo { vk, kem_ek: ek, addrs: vec![local] } }, sk, dk)
   }


    fn dummy_ct(rng: &mut SplitMix64) -> Vec<u8> {
        let mut ct = vec![0u8; nerv_seal::encrypt::Ciphertext::WIRE_SIZE];
        for c in ct.chunks_exact_mut(4) {
            c.copy_from_slice(&(rng.next_u32() & 0xFFF0_0000).to_le_bytes());
        }
        ct
    }

    /// A canonicalizable single-leg shell with a parseable dummy ct and
    /// a shape-consistent garbage proof — the network layer carries it;
    /// the aggregator's gate rejects it.
    pub(crate) fn tx_entry(seed: u64) -> nerv_registry::mempool::PoolEntry {
        let mut rng = SplitMix64::new(seed ^ 0xB0B0_0000);
        let set = nerv_core::types::ShardSet::genesis();
        let shell = nerv_custody::tx::TransactionShell {
            legs: vec![nerv_custody::tx::LegShell {
                shard: set.ids()[7],
                inputs: nerv_custody::tx::InputSet::new(vec![
                    nerv_core::hash::Hash256::from_bytes(rng.bytes32()),
                ]),
                outputs: vec![nerv_custody::tx::Output {
                    cm: nerv_core::hash::Hash256::from_bytes(rng.bytes32()),
                    sealed_note: vec![0xA5; 48],
                    value: 1_000_000_000,
                    conditional: false,
                    revert_cm: None,
                }],
                fee: nerv_core::types::FeeSats::from_u64(1000),
                anchor: nerv_core::hash::Hash256::from_bytes(rng.bytes32()),
                expiry: nerv_core::types::Height::from_u64(5_000),
                weight_version: 1,
                ct: dummy_ct(&mut rng),
                burns: vec![],
            }],
        };
        let canon = shell.canonicalize().unwrap();
        let txid = nerv_state::canonical_txid(&canon);
        nerv_registry::mempool::PoolEntry { txid, shell: canon, proof: garbage_proof() }
    }

    /// An empty-but-well-shaped TransactionProof: encodes and decodes,
    /// fails the verification gate (the deep-garbage pattern).
    pub(crate) fn garbage_proof() -> nerv_proofs::TransactionProof {
        nerv_proofs::TransactionProof {
            proved: nerv_proofs::Proved {
                proof: empty_composed(),
                log_n: 2,
                width: 33,
                prep: Vec::new(),
            },
            publics: Vec::new(),
        }
    }

    fn empty_composed() -> nerv_proofs::ComposedProof {
        use nerv_proofs::stark::ext_field::ExtF;
        use nerv_proofs::stark::fri::FriProof;
        nerv_proofs::ComposedProof {
            trace_root: nerv_core::hash::Hash256::from_bytes([1u8; 32]),
            quotient_root: nerv_core::hash::Hash256::from_bytes([2u8; 32]),
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
        }
    }

    /// The frozen FRI shape (params' [proofs.fri]).
    fn fri() -> nerv_proofs::FriShape {
        nerv_proofs::FriShape {
            log_blowup: 4,
            num_queries: 56,
            log_final_poly_len: 1,
            max_log_arity: nerv_proofs::FriShape::DEFAULT_ARITY,
            commit_pow_bits: 0,
            query_pow_bits: 0,
        }
    }

    /// The registry verification context (the real certified codec and
    /// the real reference epoch key).
    pub(crate) fn verify_ctx() -> nerv_registry::mempool::VerifyContext {
        static CTX: OnceLock<nerv_registry::mempool::VerifyContext> = OnceLock::new();
        CTX.get_or_init(|| {
            let w = nerv_codec::weight_gen::expand(
                &nerv_codec::weight_gen::BeaconRandomness::from_bytes([0x57; 32]),
                nerv_codec::codec_w::WeightVersion(1),
            );
            nerv_codec::weight_gen::certify(
                &w,
                nerv_codec::weight_gen::CertConfig {
                    spark_samples_per_size: 2,
                    column_rail_samples: 1,
                },
            )
            .unwrap();
            let pk = nerv_seal::encrypt::derive_reference_keypair(&[0xE0; 32]).unwrap().0;
            nerv_registry::mempool::VerifyContext::new(fri(), w, pk)
        })
        .clone()
    }

    /// A structurally valid header at `height` (deterministic).
    pub(crate) fn test_header(
        shard: nerv_core::types::ShardId,
        height: u64,
    ) -> nerv_state::ShardHeader {
        use nerv_core::field::Goldilocks;
        use nerv_core::hash::Hash256;
        let mut rng = SplitMix64::new(height ^ 0xCC00_0000);
        nerv_state::ShardHeader {
            prev: Hash256::from_bytes(rng.bytes32()),
            height: nerv_core::types::Height::from_u64(height),
            nct_root: nerv_custody::NctDigest::from_elements(&[
                Goldilocks::from_u32((height % 1_000_000) as u32),
                Goldilocks::ONE,
                Goldilocks::ZERO,
                Goldilocks::ZERO,
            ]),
            nullifier_root: Hash256::from_bytes(rng.bytes32()),
            transit_root: Hash256::from_bytes(rng.bytes32()),
            params_root: Hash256::from_bytes([0x9C; 32]),
            derived: rng.bytes32(),
            ct_batch_hash: Hash256::from_bytes(rng.bytes32()),
            prev_reveal: None,
            registry: nerv_state::header::RegistryRef {
                interval: nerv_core::types::Interval::from_u64(86_401),
                root: Hash256::from_bytes(rng.bytes32()),
            },
            fee_total: nerv_core::types::FeeSats::from_u64(1000),
            producer_payout: payout(),
            qc_hash: Hash256::from_bytes(rng.bytes32()),
        }
    }

    fn payout() -> nerv_custody::Address {
        static PAYOUT: OnceLock<nerv_custody::Address> = OnceLock::new();
        PAYOUT
            .get_or_init(|| {
                let mut rng = SplitMix64::new(0xFA0);
                let mut d = [0u8; 1184];
                for c in d.chunks_exact_mut(8) {
                    c.copy_from_slice(&rng.next_u64().to_le_bytes());
                }
                let ek = nerv_crypto::mlkem::EncapsulationKey::from_bytes(d);
                let set = nerv_core::types::ShardSet::genesis();
                let tag = set
                    .home_kappa(&nerv_core::types::kappa(ek.as_bytes()))
                    .unwrap();
                nerv_custody::Address::new(ek, tag, rng.bytes32()).unwrap()
            })
            .clone()
    }

    /// A structurally valid decryption partial (a zero polynomial and an
    /// empty proof — gossip carries it; the ceremony verifies it).
    pub(crate) fn test_partial(member: u8) -> nerv_seal::vpd::PartialDecryption {
        nerv_seal::vpd::PartialDecryption {
            member,
            partial: nerv_seal::ring::Poly::zero(),
            proof: nerv_seal::dkg::sigma::Proof { h: Vec::new(), z: Vec::new() },
        }
    }

    /// A valid zero reveal (legs = 1, all-zero digit sums).
    pub(crate) fn test_reveal() -> nerv_seal::decrypt::ChunkReveal {
        let mut b = [0u8; nerv_seal::decrypt::ChunkReveal::WIRE_SIZE];
        b[0] = 1;
        nerv_seal::decrypt::ChunkReveal::from_bytes(&b).unwrap()
    }
}

