//! Cross-module integration for the weight-generation surface (chunk 6,
//! part 2): the ceremony pipeline (expand → certify → verify), statement
//! 8's structural basis on certified weights, and the slot-block
//! injectivity demonstration on sampled admissible differences.

#![allow(clippy::unwrap_used)]


use nerv_codec::codec_w::{CodecW, WeightVersion, EMBEDDING_DIM};
use nerv_codec::features::{
    build_leg_features, LegKind, LegMovement, ACCOUNT_SLOT_COUNT, FEATURE_COUNT,
    MAX_ACTIVE_ACCOUNT_SLOTS,
};
use nerv_codec::weight_gen::{
    certify, expand, expand_and_certify, verify_expansion, BeaconRandomness, CertConfig,
    CertificationError,
};


fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rand_addr(state: &mut u64) -> Vec<u8> {
    let mut a = vec![0u8; 32];
    for chunk in a.chunks_mut(8) {
        chunk.copy_from_slice(&splitmix64(state).to_le_bytes());
    }
    a
}

fn rand_leg(state: &mut u64, n_in: usize, n_out: usize, kind: u8) -> LegMovement {
    let inputs: Vec<(Vec<u8>, u64)> = (0..n_in)
        .map(|_| (rand_addr(state), (splitmix64(state) % (1 << 40)) + 1))
        .collect();
    let outputs: Vec<(Vec<u8>, u64)> = (0..n_out)
        .map(|_| (rand_addr(state), (splitmix64(state) % (1 << 40)) + 1))
        .collect();
    LegMovement {
        inputs,
        outputs,
        fee_nano: splitmix64(state) % 100_000_000 + 1,
        kind: LegKind::from_code(kind % 5).unwrap(),
        expiry_height: splitmix64(state) % 1_000_000,
        epoch_length_blocks: 43_200,
    }
}

#[test]
fn ceremony_pipeline_expand_certify_verify() {
    let beacon = BeaconRandomness::from_bytes([0x42; 32]);
    let cfg = CertConfig { spark_samples_per_size: 64, column_rail_samples: 32 };
    let w = expand(&beacon, WeightVersion(1));
    let cert = certify(&w, cfg).unwrap();
    assert_eq!(cert.w_commitment, w.commitment());
    assert_eq!(cert.fp_rank, EMBEDDING_DIM as u8);
    assert!(verify_expansion(&beacon, WeightVersion(1), &w));
    assert_eq!(certify(&w, cfg).unwrap().canonical_bytes(), cert.canonical_bytes());
    assert_eq!(cert.verify(&w), Ok(true));
}

#[test]
fn certified_w_moves_every_admissible_leg() {
    // WP §5.1 statement 8 (δ ≠ 0), demonstrated client-side (App A.1:
    // "checked non-zero client-side") on a certified codec: every
    // admissible leg — ≤ 6 active account slots by construction (≤ 3
    // inputs + ≤ 3 outputs) — produces a non-zero delta.
    let w = expand(&BeaconRandomness::from_bytes([0x42; 32]), WeightVersion(1));
    certify(&w, CertConfig::default()).expect("fixture is certified");
    let mut s = 0x5EED_1EAF;
    for _ in 0..500 {
        let n_in = 1 + (splitmix64(&mut s) % 3) as usize;
        let n_out = (splitmix64(&mut s) % 4) as usize;
        let kind = (splitmix64(&mut s) % 5) as u8;
        let leg = rand_leg(&mut s, n_in, n_out, kind);
        let fv = build_leg_features(&leg).expect("inputs in range");
        fv.check_admissible().expect("≤ 6 active slots by construction");
        assert!(!w.apply(&fv).is_zero(), "admissible leg rounded to a zero delta");
    }
}

#[test]
fn slot_block_injectivity_on_admissible_differences() {
    // WP §7.4 index completeness, at the exact linear map (no rounding): a
    // nonzero vector supported on ≤ 12 slot columns — the slot support of
    // the difference of two admissible feature vectors — is never in W's
    // kernel. A kernel member would be an 𝔽_p dependence among ≤ 12 slot
    // columns, contradicting spark ≥ 13.
    let w = expand(&BeaconRandomness::from_bytes([0x42; 32]), WeightVersion(1));
    let weights = w.weights();
    let mut s = 0xBEEF_CAFE;
    for _ in 0..300 {
        let k = 1 + (splitmix64(&mut s) % (2 * MAX_ACTIVE_ACCOUNT_SLOTS) as u64) as usize;
        let mut cols: Vec<usize> = Vec::with_capacity(k);
        while cols.len() < k {
            let c = (splitmix64(&mut s) % ACCOUNT_SLOT_COUNT as u64) as usize;
            if !cols.contains(&c) {
                cols.push(c);
            }
        }
        let xs: Vec<i64> = (0..k)
            .map(|_| {
                let mag = 1 + (splitmix64(&mut s) % (1 << 20)) as i64;
                if splitmix64(&mut s) & 1 == 0 { mag } else { -mag }
            })
            .collect();
        let mut any_nonzero = false;
        for row in weights.iter() {
            let mut acc: i128 = 0;
            for (&c, &x) in cols.iter().zip(xs.iter()) {
                acc += i128::from(row[c]) * i128::from(x);
            }
            if acc != 0 {
                any_nonzero = true;
            }
        }
        assert!(any_nonzero, "sparse slot vector in W's kernel");
    }
}

#[test]
fn distinct_admissible_legs_move_differently() {
    // The rounded-delta companion to the kernel test above: value-scaled
    // legs differ in their feature vectors and in their deltas (empirical
    // demonstration; the structural guarantee is the spark argument).
    let w = expand(&BeaconRandomness::from_bytes([0x42; 32]), WeightVersion(1));
    let mut s = 0xD1CE_B0B5;
    let in_addr = rand_addr(&mut s);
    let out_addr = rand_addr(&mut s);
    for _ in 0..100 {
        let v = (splitmix64(&mut s) % (1 << 40)) + (1 << 20);
        let mk = |amount: u64| LegMovement {
            inputs: vec![(in_addr.clone(), amount)],
            outputs: vec![(out_addr.clone(), amount - 1)],
            fee_nano: 1,
            kind: LegKind::SingleShard,
            expiry_height: 7_000,
            epoch_length_blocks: 43_200,
        };
        let fv1 = build_leg_features(&mk(v)).unwrap();
        let fv2 = build_leg_features(&mk(2 * v)).unwrap();
        assert_ne!(fv1, fv2);
        assert_ne!(w.apply(&fv1), w.apply(&fv2));
    }
}

#[test]
fn tampering_breaks_certificate_and_provenance() {
    let beacon = BeaconRandomness::from_bytes([0x42; 32]);
    let w = expand(&beacon, WeightVersion(1));
    let cert = certify(&w, CertConfig { spark_samples_per_size: 4, column_rail_samples: 2 })
        .unwrap();
    assert_eq!(cert.verify(&w), Ok(true));

    let mut rows = *w.weights();
    rows[7][123] = rows[7][123].wrapping_add(1);
    let bad = CodecW::new(WeightVersion(1), rows);
    assert!(!verify_expansion(&beacon, WeightVersion(1), &bad));
    // Either a gate now fails (Err) or the record differs (Ok(false));
    // both refute the tampered matrix.
    assert!(!matches!(cert.verify(&bad), Ok(true)));
}

#[test]
fn expand_and_certify_produces_consistent_pair() {
    let beacon = BeaconRandomness::from_bytes([0x99; 32]);
    let (w, cert) = expand_and_certify(&beacon, WeightVersion(5), CertConfig::default()).unwrap();
    assert_eq!(cert.version, WeightVersion(5));
    assert_eq!(cert.w_commitment, w.commitment());
    assert!(verify_expansion(&beacon, WeightVersion(5), &w));
    assert_eq!(cert.verify(&w), Ok(true));
}

#[test]
fn independent_beacons_yield_distinct_certified_codecs() {
    let mut commitments = std::collections::HashSet::new();
    for k in 1..=6u8 {
        let beacon = BeaconRandomness::from_bytes([k; 32]);
        let (w, cert) = expand_and_certify(
            &beacon,
            WeightVersion(u64::from(k)),
            CertConfig { spark_samples_per_size: 8, column_rail_samples: 4 },
        )
        .unwrap();
        assert_eq!(cert.version, WeightVersion(u64::from(k)));
        assert!(commitments.insert(w.commitment()));
    }
    assert_eq!(commitments.len(), 6);
}

// -- planted degeneracy: matrices engineered to pass every statistical and
// exhaustive gate up to the sampled checks, demonstrating those layers
// fire. Both constructions are deterministic (fixed seeds). --------------

fn degenerate_groups_matrix(seed: u64) -> CodecW {
    // 28 groups of 8 slot columns, each group confined to a 2-dimensional
    // span: column = u_g + b·v_g with b = 1..=8. Within a group every pair
    // is independent (distinct b) but every triple is dependent. Magnitudes
    // are fleet-safe (max column deviation ≈ 2 MAD; rows ≈ 4 MAD); the
    // 28·2 group dimensions + 32 rail dimensions keep full 𝔽_p rank at 64.
    // A size-3 spark sample drawn from a single group (P ≈ 8.5·10⁻⁴ per
    // sample) is dependent, so the SparkSample gate fires.
    let mut s = seed;
    let mut rows = [[0i16; FEATURE_COUNT]; EMBEDDING_DIM];
    for g in 0..28usize {
        let mut u = [0i16; EMBEDDING_DIM];
        let mut v = [0i16; EMBEDDING_DIM];
        for x in u.iter_mut().chain(v.iter_mut()) {
            *x = (splitmix64(&mut s) % 3001) as i16 - 1500;
        }
        for b in 1i16..=8 {
            let col = 8 * g + b as usize - 1;
            for r in 0..EMBEDDING_DIM {
                rows[r][col] = u[r] + b * v[r];
            }
        }
    }
    for col in ACCOUNT_SLOT_COUNT..FEATURE_COUNT {
        for r in 0..EMBEDDING_DIM {
            rows[r][col] = (splitmix64(&mut s) % 16385) as i16 - 8192;
        }
    }
    CodecW::new(WeightVersion(1), rows)
}

#[test]
fn sampled_spark_gate_fires_on_planted_group_structure() {
    let w = degenerate_groups_matrix(0xC0FF_EE01);
    let cfg = CertConfig { spark_samples_per_size: 65_536, column_rail_samples: 1 };
    match certify(&w, cfg) {
        Err(CertificationError::SparkSample { size: 3, .. }) => {}
        other => panic!("expected SparkSample at size 3, got {other:?}"),
    }
}

fn rail_in_slot_span_matrix(seed: u64) -> CodecW {
    // Generic ±8192 matrix with one exact 3-term relation planted where
    // only the column–rail check looks: rail 224 (the volume rail) equals
    // slot column 0 + slot column 1, summands drawn at ±6144. The
    // triangular-distribution identity E|s1+s2| = 12288/3 = 4096 equals
    // E|uniform ±8192| exactly, so the planted rail is statistically
    // invisible to the fleet band; no pair is proportional; full rank
    // stays 64; slot-only spark subsets stay independent. A 6-slot sample
    // containing both summands has rank 6 < 7 (P ≈ 6·10⁻⁴ per sample) —
    // the ColumnRail gate fires at rail 224, the first rail examined.
    let mut s = seed;
    let mut rows = [[0i16; FEATURE_COUNT]; EMBEDDING_DIM];
    for col in 0..FEATURE_COUNT {
        for r in 0..EMBEDDING_DIM {
            rows[r][col] = (splitmix64(&mut s) % 16385) as i16 - 8192;
        }
    }
    for r in 0..EMBEDDING_DIM {
        let s1 = (splitmix64(&mut s) % 12289) as i16 - 6144;
        let s2 = (splitmix64(&mut s) % 12289) as i16 - 6144;
        rows[r][0] = s1;
        rows[r][1] = s2;
        rows[r][ACCOUNT_SLOT_COUNT] = s1 + s2;
    }
    CodecW::new(WeightVersion(1), rows)
}

#[test]
fn column_rail_gate_fires_on_planted_span_relation() {
    let w = rail_in_slot_span_matrix(0xFACE_B00C);
    let cfg = CertConfig { spark_samples_per_size: 2, column_rail_samples: 65_536 };
    match certify(&w, cfg) {
        Err(CertificationError::ColumnRail { rail, bound, .. }) => {
            assert_eq!(rail, ACCOUNT_SLOT_COUNT);
            assert_eq!(bound, MAX_ACTIVE_ACCOUNT_SLOTS + 1);
        }
        other => panic!("expected ColumnRail, got {other:?}"),
    }
}

