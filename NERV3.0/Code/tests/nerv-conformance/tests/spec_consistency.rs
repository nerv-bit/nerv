#![allow(clippy::unwrap_used, clippy::expect_used)]

//! D5: spec consistency — `specs/params.toml` vs the codegen'd
//! `nerv_core::params`.
//!
//! `zero_drift.rs` already proves the *leaf* inventory (`PARAMS_MAP`)
//! matches the live file exactly, path-for-path and value-for-value. This
//! file proves the two remaining renderings the code generator emits:
//!
//!   1. the **structured families** (`ECONOMY_BUCKETS`, `ERRATA`) — a
//!      build.rs bug that swaps two bucket/erratum fields within a record
//!      would still leave `PARAMS_MAP` correct while corrupting the typed
//!      arrays consumers actually iterate;
//!   2. the **named `pub const`s** that other crates import directly — a
//!      build.rs codegen bug (e.g. the Session 10 `PROOFS_FRI_POSEIDON2_*`
//!      infix drift) would leave `PARAMS_MAP` correct while a named const
//!      silently reads the wrong value.
//!
//! Every check compares against a *fresh, validated* parse of the live
//! TOML (not a second copy of the expected values), so this test fails
//! exactly when the generated `nerv_core::params` diverges from the
//! single source of truth.

use nerv_conformance::spec::LoadedSpec;
use nerv_conformance::util::locate;
use nerv_core::params as P;

fn load_spec() -> LoadedSpec {
    let p = locate("specs/params.toml").expect("spec file");
    LoadedSpec::load(&p).expect("validate spec")
}

/// `economy.buckets` → the generated `ECONOMY_BUCKETS` structured array.
#[test]
fn economy_buckets_match_live_spec() {
    let loaded = load_spec();
    let live = &loaded.spec.economy.buckets;

    assert_eq!(P::ECONOMY_BUCKET_COUNT, live.len(), "bucket count drift");
    assert_eq!(P::ECONOMY_BUCKETS.len(), live.len(), "bucket array len drift");

    for (i, (gen, live)) in P::ECONOMY_BUCKETS.iter().zip(live.iter()).enumerate() {
        assert_eq!(gen.name, live.name, "bucket[{i}].name");
        assert_eq!(gen.kind, live.kind, "bucket[{i}].kind");
        assert_eq!(gen.total_nerv, live.total_nerv, "bucket[{i}].total_nerv");
        assert_eq!(gen.share_permille, live.share_permille, "bucket[{i}].share_permille");
        assert_eq!(gen.account, live.account, "bucket[{i}].account");
        assert_eq!(gen.term_days, live.term_days, "bucket[{i}].term_days");
        assert_eq!(gen.cliff_days, live.cliff_days, "bucket[{i}].cliff_days");
        assert_eq!(gen.linear_days, live.linear_days, "bucket[{i}].linear_days");
        assert_eq!(gen.year_one_days, live.year_one_days, "bucket[{i}].year_one_days");
        assert_eq!(gen.quarters, live.quarters, "bucket[{i}].quarters");
        assert_eq!(gen.quarter_days, live.quarter_days, "bucket[{i}].quarter_days");
        assert_eq!(gen.ratio_num, live.ratio_num, "bucket[{i}].ratio_num");
        assert_eq!(gen.ratio_den, live.ratio_den, "bucket[{i}].ratio_den");
        assert_eq!(gen.window_days, live.window_days, "bucket[{i}].window_days");
        assert_eq!(gen.burn_unclaimed, live.burn_unclaimed, "bucket[{i}].burn_unclaimed");
    }
}

/// `errata` → the generated `ERRATA` structured array.
#[test]
fn errata_match_live_spec() {
    let loaded = load_spec();
    let live = &loaded.spec.errata;

    assert_eq!(P::ERRATA_COUNT, live.len(), "erratum count drift");
    assert_eq!(P::ERRATA.len(), live.len(), "erratum array len drift");

    for (i, (gen, live)) in P::ERRATA.iter().zip(live.iter()).enumerate() {
        assert_eq!(gen.id, live.id, "errata[{i}].id");
        assert_eq!(gen.refs, live.refs, "errata[{i}].refs");
        assert_eq!(gen.issue, live.issue, "errata[{i}].issue");
        assert_eq!(gen.resolution, live.resolution, "errata[{i}].resolution");
    }
}

/// The load-bearing named `pub const`s other crates import. This is a
/// representative pin (not exhaustive — `zero_drift.rs` covers every leaf);
/// it specifically re-pins the families that historically drifted (the
/// `PROOFS_FRI_POSEIDON2_*` infix bug of Session 10) and the constants
/// that drive executor / seal / mixnet correctness.
#[test]
fn named_consts_match_live_spec() {
    let loaded = load_spec();
    let s = &loaded.spec;

    // protocol
    assert_eq!(P::PROTOCOL_SUPPLY_NERV, s.protocol.supply_nerv);
    assert_eq!(P::PROTOCOL_NANO_PER_NERV, s.protocol.nano_per_nerv);
    assert_eq!(P::PROTOCOL_SHARD_COUNT_GENESIS, s.protocol.shard_count_genesis);
    assert_eq!(P::PROTOCOL_SHARD_ID_BITS_GENESIS, s.protocol.shard_id_bits_genesis);
    assert_eq!(P::PROTOCOL_SHARD_COUNT_MAX, s.protocol.shard_count_max);

    // timing
    assert_eq!(P::TIMING_BEACON_INTERVAL_SECS, s.timing.beacon_interval_secs);
    assert_eq!(P::TIMING_EPOCH_SECS, s.timing.epoch_secs);

    // consensus (executor / QC algebra)
    assert_eq!(P::CONSENSUS_SHARD_COMMITTEE_SIZE, s.consensus.shard_committee_size);
    assert_eq!(P::CONSENSUS_SHARD_QUORUM, s.consensus.shard_quorum);
    assert_eq!(P::CONSENSUS_BEACON_COMMITTEE_SIZE, s.consensus.beacon_committee_size);
    assert_eq!(P::CONSENSUS_BEACON_QUORUM, s.consensus.beacon_quorum);
    assert_eq!(P::CONSENSUS_ATTESTATION_SIGNERS, s.consensus.attestation_signers);
    assert_eq!(P::CONSENSUS_ATTESTATION_QUORUM, s.consensus.attestation_quorum);

    // custody
    assert_eq!(P::CUSTODY_NCT_DEPTH, s.custody.nct_depth);
    assert_eq!(P::CUSTODY_NULLIFIER_TREE_DEPTH, s.custody.nullifier_tree_depth);
    assert_eq!(P::CUSTODY_VALUE_MAX_NANO, s.custody.value_max_nano);
    assert_eq!(P::CUSTODY_MAX_LEGS_PER_TX, s.custody.tx_caps.max_legs_per_tx);

    // proofs — the Session-10 drift family, re-pinned.
    assert_eq!(P::PROOFS_POSEIDON2_T, s.proofs.poseidon2_t);
    assert_eq!(P::PROOFS_POSEIDON2_FULL_ROUNDS, s.proofs.poseidon2_full_rounds);
    assert_eq!(P::PROOFS_POSEIDON2_PARTIAL_ROUNDS, s.proofs.poseidon2_partial_rounds);
    assert_eq!(P::PROOFS_POSEIDON2_SBOX_DEGREE, s.proofs.poseidon2_sbox_degree);
    assert_eq!(P::PROOFS_FIELD_MODULUS, s.proofs.field_modulus);
    assert_eq!(P::PROOFS_FRI_SECURITY_BITS, s.proofs.fri_security_bits);
    assert_eq!(P::PROOFS_FRI_NUM_QUERIES, s.proofs.fri.num_queries);
    assert_eq!(P::PROOFS_FRI_LOG_BLOWUP, s.proofs.fri.log_blowup);

    // seal
    assert_eq!(P::SEAL_Q, s.seal.q);
    assert_eq!(P::SEAL_RING_DEGREE, s.seal.ring_degree);
    assert_eq!(P::SEAL_MODULE_RANK, s.seal.module_rank);
    assert_eq!(P::SEAL_DIGIT_BITS, s.seal.digit_bits);
    assert_eq!(P::SEAL_CHUNK_MIN, s.seal.chunk_min);
    assert_eq!(P::SEAL_CHUNK_MAX, s.seal.chunk_max);
    assert_eq!(P::SEAL_COMMITTEE_N, s.seal.committee_n);
    assert_eq!(P::SEAL_THRESHOLD_T, s.seal.threshold_t);

    // mixnet
    assert_eq!(P::MIXNET_PATH_RELAYS, s.mixnet.path_relays);
    assert_eq!(P::MIXNET_PACKET_BYTES, s.mixnet.packet_bytes);
    assert_eq!(P::MIXNET_SUBMISSION_FANOUT, s.mixnet.submission_fanout);

    // sharding (executor block cap / witness depth)
    assert_eq!(P::SHARDING_BLOCK_LEG_CAP, s.sharding.block_leg_cap);
    assert_eq!(P::WITNESS_LEG_TREE_DEPTH_MAX, s.witness.leg_tree_depth_max);

    // fees (the 40/30/20/10 split)
    assert_eq!(P::FEES_SPLIT_PRODUCER_PERMILLE, s.fees.split_producer_permille);
    assert_eq!(P::FEES_SPLIT_PROVER_PERMILLE, s.fees.split_prover_permille);
    assert_eq!(P::FEES_SPLIT_DA_PERMILLE, s.fees.split_da_permille);
    assert_eq!(P::FEES_SPLIT_RELAY_PERMILLE, s.fees.split_relay_permille);
}
