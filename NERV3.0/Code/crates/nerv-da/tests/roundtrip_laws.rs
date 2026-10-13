#![allow(clippy::unwrap_used, clippy::expect_used)]

//! D3: proptest round-trip law — `Square` encode/decode preserves the data
//! region (erratum 140's two-dimensional Reed–Solomon layout).

use nerv_core::codec::{Decode, Encode};
use nerv_da::blobs::Square;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn square_encode_decode_roundtrip(
        data in proptest::collection::vec(any::<u8>(), 1..4096),
    ) {
        let sq = Square::build(&data).unwrap();
        let enc = sq.encode();
        let dec = Square::decode(&enc).unwrap();
        prop_assert_eq!(sq.data(data.len() as u32), dec.data(data.len() as u32));
    }
}
