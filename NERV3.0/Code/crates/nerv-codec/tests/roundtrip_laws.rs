#![allow(clippy::unwrap_used, clippy::expect_used)]

//! D3: proptest round-trip law — `CodecW` canonical serialization is a
//! bijection on the space of 1.15 weight matrices: every arbitrary matrix
//! round-trips through `canonical_bytes` → `parse` and re-commits to the
//! same `WCommitment`.

use nerv_codec::codec_w::{CodecW, WeightVersion, EMBEDDING_DIM};
use proptest::prelude::*;

// FEATURE_COUNT = 256 (specs/params.toml `overlay.w_cols`).
const FEATURE_COUNT: usize = 256;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn codec_w_canonical_roundtrip(
        version in any::<u64>(),
        flat in proptest::collection::vec(any::<i16>(), EMBEDDING_DIM * FEATURE_COUNT),
    ) {
        let mut weights = [[0i16; FEATURE_COUNT]; EMBEDDING_DIM];
        for (r, row) in weights.iter_mut().enumerate() {
            row.copy_from_slice(&flat[r * FEATURE_COUNT..(r + 1) * FEATURE_COUNT]);
        }
        let c = CodecW::new(WeightVersion(version), weights);
        let parsed = CodecW::parse(&c.canonical_bytes()).unwrap();
        // Commitment first (borrows), then the equality (moves).
        prop_assert_eq!(c.commitment(), parsed.commitment());
        prop_assert_eq!(c, parsed);
    }
}
