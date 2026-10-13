#![allow(clippy::unwrap_used, clippy::expect_used)]

use nerv_conformance::spec::LoadedSpec;
use nerv_conformance::util::locate;

fn real_text() -> String {
    let p = locate("specs/params.toml").expect("spec file");
    std::fs::read_to_string(p).expect("read spec")
}

fn mutate(find: &str, replace: &str) -> String {
    let t = real_text();
    assert!(t.matches(find).count() == 1, "mutation target not unique: {find}");
    t.replacen(find, replace, 1)
}

fn rejects(find: &str, replace: &str) {
    let text = mutate(find, replace);
    assert!(
        LoadedSpec::from_str(&text).is_err(),
        "spec must be rejected: {find} -> {replace}"
    );
}

#[test]
fn real_spec_validates() {
    let s = LoadedSpec::from_str(&real_text()).unwrap();
    assert_eq!(s.spec.errata.len(), 9);
    let summary = s.spec.committee_summary();
    assert_eq!(summary.len(), 4);
    assert_eq!(summary[0], ("shard", 21, 15, 6));
    assert_eq!(summary[1], ("beacon", 31, 21, 10));
    assert_eq!(summary[2], ("registry", 21, 15, 6));
    assert_eq!(summary[3], ("attestation", 21, 15, 6));
    assert_ne!(s.hash, [0u8; 32]);
}

#[test]
fn even_quorum_rejected() {
    let e = LoadedSpec::from_str(&mutate("shard_quorum = 15", "shard_quorum = 14")).unwrap_err();
    let msg = format!("{e}");
    assert!(msg.contains("odd"), "{msg}");
}

#[test]
fn fee_split_must_sum() {
    rejects("split_prover_permille = 300", "split_prover_permille = 301");
}

#[test]
fn composite_seal_modulus_rejected() {
    // 4293918723 = 3 × 1431306241 (digit sum 48 → divisible by 3)
    rejects("q = 4293918721", "q = 4293918723");
}

#[test]
fn wrong_field_modulus_rejected() {
    rejects(
        "field_modulus = 18446744069414584321",
        "field_modulus = 18446744069414584323",
    );
}

#[test]
fn spark_bound_below_2k_plus_1_rejected() {
    rejects("spark_bound = 13", "spark_bound = 12");
}

#[test]
fn bucket_total_drift_rejected() {
    rejects("total_nerv = 2000000000", "total_nerv = 2000000001");
}

#[test]
fn vesting_arithmetic_must_close() {
    rejects("cliff_days = 180", "cliff_days = 181");
}

#[test]
fn zero_floor_multiplier_rejected() {
    rejects("m_max = 8", "m_max = 0");
}

#[test]
fn chunk_privacy_floor_pinned() {
    rejects("chunk_min = 128", "chunk_min = 64");
}

#[test]
fn encoder_shape_pinned() {
    rejects("w_cols = 256", "w_cols = 255");
}

#[test]
fn epoch_length_pinned() {
    rejects("epoch_secs = 86400", "epoch_secs = 3600");
}

#[test]
fn fri_security_floor_enforced() {
    rejects("fri_security_bits = 100", "fri_security_bits = 96");
}

#[test]
fn attestation_subset_rule_enforced() {
    rejects("attestation_signers = 21", "attestation_signers = 35");
}

#[test]
fn claim_window_must_burn() {
    rejects("burn_unclaimed = true", "burn_unclaimed = false");
}

#[test]
fn unknown_field_rejected() {
    rejects("[custody]", "[custody]\nbogus_field = 1");
}

#[test]
fn float_in_spec_rejected() {
    rejects("path_relays = 5", "path_relays = 5.0");
}

#[test]
fn goldilocks_prime_verified_and_sbox_coprime() {
    // T1-grade assurance for the prover field: primality by deterministic
    // Miller–Rabin; p−1 = 2^32·(2^32−1) with 7 ∤ (2^32−1) ⇒ x^7 bijective.
    assert!(nerv_conformance::primality::is_prime_u64(nerv_core::field::GOLDILOCKS_PRIME));
    assert_eq!(
        nerv_conformance::primality::two_adicity(nerv_core::field::GOLDILOCKS_PRIME),
        32
    );
    assert_ne!((nerv_core::field::GOLDILOCKS_PRIME - 1) % 7, 0);
}
