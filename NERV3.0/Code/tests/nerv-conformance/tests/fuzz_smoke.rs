#![allow(clippy::unwrap_used, clippy::expect_used)]

//! D2: fuzz-corpus smoke harness.
//!
//! The `fuzz/` cargo-fuzz workspace holds the long-tail libFuzzer targets
//! (nightly-only, CI scheduled). This file is the *stable-toolchain* smoke
//! pass: one `#[ignore]`d stress test per crate target that drives the same
//! invariant in a bounded loop with pseudo-random inputs, so the harness
//! can be booted and measured on the 8 GB box.
//!
//! Run (about a minute for all six):
//! ```text
//! cargo test -p nerv-conformance --test fuzz_smoke --ignore-rust-version \
//!     -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! Each iteration is O(1) allocations and drops its locals, so peak RSS
//! stays flat over the loop (the §4 long-tail discipline).

/// Deterministic xorshift64* — no external RNG dependency, reproducible
/// from the seed printed on failure.
struct Prng(u64);

impl Prng {
    fn new(seed: u64) -> Prng {
        Prng(seed | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn bytes32(&mut self) -> [u8; 32] {
        let mut b = [0u8; 32];
        for chunk in b.chunks_mut(8) {
            chunk.copy_from_slice(&self.next().to_le_bytes());
        }
        b
    }
}

// Iteration counts are sized so the whole six-target smoke pass finishes in
// about a minute on the 8 GB box. Per-iteration cost is dominated by the
// target's intrinsic work: the NCT uses Poseidon2 (~50 µs/hash, and each
// witness + root is ~32 hashes), and the Sphinx/seal targets each do several
// ML-KEM operations — so their counts are the smallest.
const ITERS_NCT: usize = 30;
const ITERS_CODEC: usize = 200;
const ITERS_DA: usize = 50;
const ITERS_SPHINX: usize = 30;
const ITERS_SEAL: usize = 30;
const ITERS_FRI: usize = 30;

/// nerv-custody: NCT append → witness → verify round-trip. Every appended
/// leaf's witness must reconstruct the live root.
#[test]
#[ignore = "D2 fuzz smoke: NCT insert/witness round-trip"]
fn fuzz_smoke_custody_nct() {
    use nerv_core::hash::Hash256;
    use nerv_custody::nct::{verify_witness, NoteCommitmentTree};

    let mut prng = Prng::new(0x4E43_5400);
    for _ in 0..ITERS_NCT {
        let mut tree = NoteCommitmentTree::new();
        let n = (prng.next() % 64) as usize + 1;
        let mut leaves: Vec<(Hash256, u64)> = Vec::with_capacity(n);
        for _ in 0..n {
            let cm = Hash256::from_bytes(prng.bytes32());
            let idx = tree.append(&cm).unwrap();
            let wit = tree.witness(idx).unwrap();
            assert!(
                verify_witness(&tree.root(), wit.leaf_index, &cm, &wit.siblings),
                "NCT witness failed to verify at index {idx}"
            );
            leaves.push((cm, idx));
        }
        // Re-verify a random leaf against the final root (index stability).
        let (cm, idx) = leaves[(prng.next() as usize) % leaves.len()];
        let wit = tree.witness(idx).unwrap();
        assert!(verify_witness(&tree.root(), wit.leaf_index, &cm, &wit.siblings));
    }
}

/// nerv-codec: `CodecW` canonical serialization round-trip. Arbitrary i16
/// weights are valid 1.15 values by type, so every matrix round-trips.
#[test]
#[ignore = "D2 fuzz smoke: CodecW canonical round-trip"]
fn fuzz_smoke_codec_w() {
    use nerv_codec::codec_w::{CodecW, WeightVersion, EMBEDDING_DIM};

    let mut prng = Prng::new(0x434F_4445);
    for _ in 0..ITERS_CODEC {
        let mut weights = [[0i16; 256]; EMBEDDING_DIM];
        for row in weights.iter_mut() {
            for w in row.iter_mut() {
                *w = prng.next() as i16;
            }
        }
        let c = CodecW::new(WeightVersion(prng.next()), weights);
        let parsed = CodecW::parse(&c.canonical_bytes()).unwrap();
        assert_eq!(c, parsed, "CodecW canonical round-trip differs");
        assert_eq!(c.commitment(), parsed.commitment());
    }
}

/// nerv-da: `BlobSet` build → `cell_auth` → verify round-trip, plus the
/// data-region reconstruction.
#[test]
#[ignore = "D2 fuzz smoke: BlobSet / cell_auth round-trip"]
fn fuzz_smoke_da_blobs() {
    use nerv_core::codec::{Decode, Encode};
    use nerv_core::types::{Height, ShardId, ShardSet};
    use nerv_da::blobs::{BlobSet, Square};

    let active = ShardSet::genesis();
    let shard: ShardId = active.ids()[7];
    let height = Height::from_u64(1);
    let mut prng = Prng::new(0x4441_0000);

    for _ in 0..ITERS_DA {
        let chunks = (prng.next() % 6) as usize + 1;
        let len = chunks * nerv_da::blobs::CHUNK_LEN;
        let mut data = vec![0u8; len];
        for b in data.iter_mut() {
            *b = prng.next() as u8;
        }

        let set = BlobSet::encode(shard, height, &data).expect("BlobSet::encode");
        let commitment = set.commitment();
        let auth = set.cell_auth(0, 0, 0);
        assert!(auth.verify(&commitment), "honest cell_auth must verify");

        // A flipped payload byte must NOT verify (row-path mismatch).
        let mut bad = auth.clone();
        bad.chunk[0] ^= 0x01;
        assert!(!bad.verify(&commitment), "tampered cell must NOT verify");

        // Square encode/decode round-trip: the data region is preserved.
        let sq = Square::build(&data).expect("Square::build");
        let enc = sq.encode();
        let dec = Square::decode(&enc).expect("Square::decode");
        assert_eq!(sq.data(len as u32), dec.data(len as u32));
    }
}

/// nerv-net: Sphinx build → peel round-trip. Building a 5-relay packet and
/// peeling its first layer must yield a `Relay` action with forward parts.
#[test]
#[ignore = "D2 fuzz smoke: Sphinx build/peel round-trip"]
fn fuzz_smoke_net_sphinx() {
    use nerv_net::sphinx::{
        build, encap_rand, peel, relays, Action, PathSpec, TerminalAction,
    };

    let mut prng = Prng::new(0x5350_484E);
    for _ in 0..ITERS_SPHINX {
        let seed = prng.next();
        let nodes = relays(5, seed);
        let mut hop = [(nodes[0].0, nodes[0].1); 5];
        for (i, slot) in hop.iter_mut().enumerate() {
            *slot = (nodes[i].0, nodes[i].1);
        }
        let dest = relays(5, seed ^ 0xFFFF).remove(0).0;
        let spec = PathSpec::new(hop, dest, TerminalAction::Deliver);

        let len = (prng.next() % 512) as usize + 1;
        let mut data = vec![0u8; len];
        for b in data.iter_mut() {
            *b = prng.next() as u8;
        }
        let randomness = encap_rand(prng.next());
        let packet = build(&spec, &data, &randomness).expect("sphinx::build");

        // Peel the first layer with relay 0's decapsulation key.
        let peeled = peel(&packet, &nodes[0].2).expect("sphinx::peel");
        assert!(
            matches!(peeled.action, Action::Relay),
            "first hop of a 5-relay path must relay"
        );
        assert!(peeled.forward.is_some(), "relay must carry forward parts");
    }
}

/// nerv-seal: `Ciphertext` encrypt determinism + wire round-trip, and the
/// native decrypt path is total (never panics).
#[test]
#[ignore = "D2 fuzz smoke: seal encrypt/decrypt + wire round-trip"]
fn fuzz_smoke_seal_ciphertext() {
    use nerv_seal::decrypt::decrypt_native;
    use nerv_seal::digitize::digitize;
    use nerv_seal::encrypt::{derive_reference_keypair, Ciphertext};
    use nerv_seal::sampling::NoiseSeed;

    let mut prng = Prng::new(0x5345_414C);
    for _ in 0..ITERS_SEAL {
        let entropy = prng.bytes32();
        let (pk, sk) = derive_reference_keypair(&entropy).unwrap();

        let mut coords = [0u64; 64];
        for c in coords.iter_mut() {
            *c = prng.next();
        }
        let m = digitize(&coords);
        let seed = NoiseSeed::from_bytes(prng.bytes32());

        let ct1 = Ciphertext::encrypt(&pk, &seed, &m).unwrap();
        let ct2 = Ciphertext::encrypt(&pk, &seed, &m).unwrap();
        assert_eq!(ct1, ct2, "seal encrypt must be deterministic in (pk, seed, m)");

        let wire = ct1.to_bytes();
        assert_eq!(Ciphertext::from_bytes(&wire).unwrap(), ct1, "ciphertext wire round-trip");

        // Native single-key decryption is total: Ok or Err, never a panic.
        let _ = decrypt_native(1, &sk, ct1.u(), ct1.v());
    }
}

/// nerv-proofs: FRI prove → verify round-trip on a small domain. The zero
/// codeword is the lowest-degree codeword, so an honest prover must always
/// produce a proof the verifier accepts (mirrors `zero_codeword_roundtrip`).
#[test]
#[ignore = "D2 fuzz smoke: FRI query round-trip"]
fn fuzz_smoke_proofs_fri() {
    use nerv_proofs::security::FriShape;
    use nerv_proofs::stark::domain::TwoAdicDomain;
    use nerv_proofs::stark::ext_field::ExtF;
    use nerv_proofs::stark::fri::{prove_fri, verify_fri};
    use nerv_proofs::FsTranscript;

    let mut prng = Prng::new(0x4652_4900);
    for _ in 0..ITERS_FRI {
        // Small domains only: FRI alone is cheap; the full-STARK prove path
        // is #[ignore]d on the 8 GB box and must never be pulled in here.
        let m = 4 + (prng.next() % 4) as usize; // log2 domain size ∈ [4, 7]
        let dom = TwoAdicDomain::standard_coset(m);
        let values = vec![ExtF::ZERO; dom.size()];
        let shape = FriShape {
            log_blowup: 2,
            num_queries: 8,
            log_final_poly_len: 1,
            max_log_arity: FriShape::DEFAULT_ARITY,
            commit_pow_bits: 0,
            query_pow_bits: 0,
        };

        let seed = prng.next();
        let mut tp = FsTranscript::new();
        tp.absorb_bytes(&seed.to_le_bytes());
        let proof = prove_fri(&values, &dom, &shape, &mut tp);

        let mut tv = FsTranscript::new();
        tv.absorb_bytes(&seed.to_le_bytes());
        assert!(verify_fri(&proof, &dom, &shape, &mut tv), "FRI m={m} round-trip");

        // Determinism: an identically seeded transcript reproduces the proof.
        let mut tp2 = FsTranscript::new();
        tp2.absorb_bytes(&seed.to_le_bytes());
        assert_eq!(prove_fri(&values, &dom, &shape, &mut tp2), proof);
    }
}
