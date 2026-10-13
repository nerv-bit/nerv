#![no_main]
use libfuzzer_sys::fuzz_target;
use nerv_core::codec::Encode;
use nerv_core::types::{Height, ShardId, ShardSet};
use nerv_da::blobs::{BlobSet, Square};

// nerv-da: BlobSet build → cell_auth → verify round-trip, plus the Square
// encode/decode data-region preservation (erratum 140/141 surface).
fuzz_target!(|data: &[u8]| {
    let active = ShardSet::genesis();
    let shard: ShardId = active.ids()[7];
    let height = Height::from_u64(1);

    // Bound the blob size to a small number of chunks (max k=64 → 2 MB).
    let chunks = (data.first().copied().unwrap_or(0) as usize % 6) + 1;
    let len = chunks * nerv_da::blobs::CHUNK_LEN;
    let mut payload = vec![0u8; len];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = data[i % data.len().max(1)];
    }

    let set = BlobSet::encode(shard, height, &payload).expect("BlobSet::encode");
    let commitment = set.commitment();
    let auth = set.cell_auth(0, 0, 0);
    assert!(auth.verify(&commitment), "honest cell_auth must verify");

    let mut bad = auth.clone();
    bad.chunk[0] ^= 0x01;
    assert!(!bad.verify(&commitment), "tampered cell must NOT verify");

    let sq = Square::build(&payload).expect("Square::build");
    let enc = sq.encode();
    let dec = Square::decode(&enc).expect("Square::decode");
    assert_eq!(sq.data(len as u32), dec.data(len as u32));
});
