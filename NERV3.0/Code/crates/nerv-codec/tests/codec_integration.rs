//! Cross-module integration for nerv-codec (chunk 6, part 1): the
//! Appendix-A-shaped legs, the admissibility gate, canonical roundtrips
//! across all three serialized types, and the embedding accumulator's group
//! law.

use nerv_codec::codec_w::{CodecW, Delta, WeightVersion, EMBEDDING_DIM};
use nerv_codec::features::{
    build_leg_features, slot_index, FeatureVector, LegKind, LegMovement, MAX_ACTIVE_ACCOUNT_SLOTS,
    RAIL_COUNT, RAIL_VOLUME, FEATURE_COUNT,
};


fn addr(k: u32) -> Vec<u8> {
    let mut a = vec![0xA5u8; 32];
    a[0..4].copy_from_slice(&k.to_le_bytes());
    a
}

/// Addresses with pairwise-distinct slots, generated adaptively so the
/// scenario is deterministic regardless of BLAKE3's (fixed) outputs.
fn distinct_slot_addrs(n: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(n);
    let mut slots = std::collections::HashSet::new();
    let mut k = 0u32;
    while out.len() < n {
        let a = addr(k);
        if slots.insert(slot_index(&a)) {
            out.push(a);
        }
        k += 1;
    }
    out
}

#[test]
fn appendix_a_legs_build_and_apply() {
    // WP Appendix A: Alice (shard 7) pays Bob (shard 40) 50 NERV and Carol
    // (shard 7) 25 NERV, fee 10^-3 NERV, from two notes (35 + 25 NERV).
    let a = distinct_slot_addrs(4);
    let (alice, carol, change, bob) =
        (a[0].clone(), a[1].clone(), a[2].clone(), a[3].clone());

    let leg7 = LegMovement {
        inputs: vec![(alice.clone(), 35_000_000_000), (alice.clone(), 25_000_000_000)],
        outputs: vec![(carol.clone(), 25_000_000_000), (change.clone(), 9_999_000_000)],
        fee_nano: 600_000,
        kind: LegKind::SingleShard,
        expiry_height: 7_000,
        epoch_length_blocks: 43_200,
    };
    let f7 = build_leg_features(&leg7).unwrap();
    assert_eq!(f7.value(slot_index(&alice) as usize), -60_000_000_000);
    assert_eq!(f7.value(slot_index(&carol) as usize), 25_000_000_000);
    assert_eq!(f7.value(slot_index(&change) as usize), 9_999_000_000);
    assert_eq!(f7.value(RAIL_VOLUME), 34_999_000_000);
    assert_eq!(f7.active_account_slot_count(), 3);
    assert!(f7.check_admissible().is_ok());

    let leg40 = LegMovement {
        inputs: vec![],
        outputs: vec![(bob.clone(), 50_000_000_000)],
        fee_nano: 400_000,
        kind: LegKind::CrossShardIssue,
        expiry_height: 7_000,
        epoch_length_blocks: 43_200,
    };
    let f40 = build_leg_features(&leg40).unwrap();
    assert_eq!(f40.value(slot_index(&bob) as usize), 50_000_000_000);
    assert!(f40.check_admissible().is_ok());

    // A sparse codec: row 0 = 0.5 on the volume rail, row 1 = 1 − 2^-15 on
    // the count rail, rows 2.. zero.
    let mut rows = [[0i16; FEATURE_COUNT]; EMBEDDING_DIM];
    rows[0][RAIL_VOLUME] = 16_384;
    rows[1][RAIL_COUNT] = i16::MAX;
    let w = CodecW::new(WeightVersion::GENESIS, rows);

    let d7 = w.apply(&f7);
    let d40 = w.apply(&f40);
    // Statement 8, client-side: both deltas are non-zero.
    assert!(!d7.is_zero());
    assert!(!d40.is_zero());
    // Row 0: δ = round½even(volume / 2) — exact here.
    assert_eq!(d7.0[0], 17_499_500_000);
    assert_eq!(d40.0[0], 25_000_000_000);
    // Row 1: δ = round½even(32767 / 32768) = 1.
    assert_eq!(d7.0[1], 1);
    assert_eq!(d40.0[1], 1);

    // Embedding accumulation is order-free group arithmetic (WP §7.5).
    assert_eq!(d7.wrapping_add(&d40), d40.wrapping_add(&d7));
}

#[test]
fn admissibility_gate_forces_leg_reshaping() {
    // A leg touching more than six distinct account slots builds fine but
    // is off the admissible support — the wallet must split it, exactly as
    // the circuit's statement 6 would reject it.
    let outputs: Vec<(Vec<u8>, u64)> =
        distinct_slot_addrs(24).into_iter().map(|a| (a, 1)).collect();
    let m = LegMovement {
        inputs: vec![],
        outputs,
        fee_nano: 1,
        kind: LegKind::CrossShardIssue,
        expiry_height: 5,
        epoch_length_blocks: 100,
    };
    let fv = build_leg_features(&m).unwrap();
    assert!(fv.active_account_slot_count() > MAX_ACTIVE_ACCOUNT_SLOTS);
    assert!(fv.check_admissible().is_err());
}

#[test]
fn canonical_roundtrips_across_the_identity_layer() {
    let a = distinct_slot_addrs(2);
    let m = LegMovement {
        inputs: vec![(a[0].clone(), 1_000_000)],
        outputs: vec![(a[1].clone(), 999_999)],
        fee_nano: 1,
        kind: LegKind::SingleShard,
        expiry_height: 42,
        epoch_length_blocks: 43_200,
    };
    let fv = build_leg_features(&m).unwrap();
    let fb = fv.canonical_bytes();
    assert_eq!(fb.len(), 2048);
    assert_eq!(nerv_codec::features::FeatureVector::from_canonical_bytes(&fb), fv);

    let w = CodecW::new(WeightVersion(3), [[7i16; FEATURE_COUNT]; EMBEDDING_DIM]);
    let wb = w.canonical_bytes();
    assert_eq!(wb.len(), 32_772);
    assert_eq!(CodecW::parse(&wb).unwrap(), w);

    let d = w.apply(&fv);
    let db = d.canonical_bytes();
    assert_eq!(db.len(), 512);
    assert_eq!(Delta::from_canonical_bytes(&db), d);
}
