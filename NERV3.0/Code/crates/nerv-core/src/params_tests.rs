//! Structural sanity over the GENERATED params module — these guard the
//! generator itself. Semantic validation (quorum algebra, primality, noise
//! headroom, emission pacing) and the `PARAMS_MAP` ↔ `specs/params.toml`
//! zero-drift check live in nerv-conformance (CI), which depends on this
//! crate and re-parses the live file.

use crate::params::*;

#[test]
fn map_is_sorted_and_unique() {
    for w in PARAMS_MAP.windows(2) {
        assert!(
            w[0].0 < w[1].0,
            "PARAMS_MAP paths must be strictly ascending: `{}` !< `{}`",
            w[0].0,
            w[1].0
        );
    }
    assert_eq!(PARAMS_MAP.len(), PARAMS_LEAF_COUNT);
}

#[test]
fn statics_match_map_counts() {
    let bucket_names = PARAMS_MAP
        .iter()
        .filter(|(p, _)| p.starts_with("economy.buckets[") && p.ends_with(".name"))
        .count();
    assert_eq!(bucket_names, ECONOMY_BUCKET_COUNT);

    let errata_ids = PARAMS_MAP
        .iter()
        .filter(|(p, _)| p.starts_with("errata[") && p.ends_with(".id"))
        .count();
    assert_eq!(errata_ids, ERRATA_COUNT);
}

#[test]
fn emission_buckets_sum_to_supply() {
    // WP §12.2: seven buckets, 100% of supply, shares in permille.
    assert_eq!(ECONOMY_BUCKETS.len(), 7);
    let share: u64 = ECONOMY_BUCKETS.iter().map(|b| b.share_permille).sum();
    assert_eq!(share, 1000, "bucket shares must sum to 100%");
    let total: u64 = ECONOMY_BUCKETS.iter().map(|b| b.total_nerv).sum();
    assert_eq!(total, PROTOCOL_SUPPLY_NERV, "bucket totals must sum to supply");
}

#[test]
fn raw_file_digest_is_nonzero() {
    assert_ne!(PARAMS_TOML_DIGEST, [0u8; 32]);
}

#[test]
fn conformance_families_are_the_declared_eight() {
    // 6 whitepaper families + 2 registry-internal families (chunk 1).
    assert_eq!(CONFORMANCE_VECTOR_FAMILIES.len(), 8);
    assert!(CONFORMANCE_VECTOR_FAMILIES.contains(&"adam_replay"));
    assert!(CONFORMANCE_VECTOR_FAMILIES.contains(&"emission_schedule"));
    assert_eq!(CONFORMANCE_DETERMINISM_ARCHES.len(), 3);
}

#[test]
fn errata_register_present() {
    assert!(ERRATA_COUNT >= 9, "the nine recorded errata must be generated");
    let ids: Vec<&str> = ERRATA.iter().map(|e| e.id).collect();
    for w in ids.windows(2) {
        assert_ne!(w[0], w[1], "erratum ids must be unique");
    }
}
