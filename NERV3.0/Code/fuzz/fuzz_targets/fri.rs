#![no_main]
use libfuzzer_sys::fuzz_target;
use nerv_proofs::security::FriShape;
use nerv_proofs::stark::domain::TwoAdicDomain;
use nerv_proofs::stark::ext_field::ExtF;
use nerv_proofs::stark::fri::{prove_fri, verify_fri};
use nerv_proofs::stark::transcript_common::FsTranscript;

// nerv-proofs: FRI prove → verify round-trip on a small domain. The zero
// codeword is the lowest-degree codeword, so an honest prover always yields
// a proof the verifier accepts. Small domains only — the full-STARK prove
// path is #[ignore]d on the 8 GB box and must never be pulled in here.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let m = 4 + (data[0] as usize % 4); // log2 domain size ∈ [4, 7]
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

    let mut seed = [0u8; 8];
    for (i, byte) in seed.iter_mut().enumerate() {
        *byte = data[(i * 5) % data.len()];
    }
    let seed = u64::from_le_bytes(seed);

    let mut tp = FsTranscript::new();
    tp.absorb_bytes(&seed.to_le_bytes());
    let proof = prove_fri(&values, &dom, &shape, &mut tp);

    let mut tv = FsTranscript::new();
    tv.absorb_bytes(&seed.to_le_bytes());
    assert!(verify_fri(&proof, &dom, &shape, &mut tv), "FRI m={m} round-trip");
});
