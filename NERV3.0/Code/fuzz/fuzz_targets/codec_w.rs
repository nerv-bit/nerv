#![no_main]
use libfuzzer_sys::fuzz_target;
use nerv_codec::codec_w::{CodecW, WeightVersion, EMBEDDING_DIM};

// nerv-codec: CodecW canonical serialization round-trip. Arbitrary i16
// weights are valid 1.15 values by type, so every matrix round-trips and
// re-commits to the same WCommitment.
fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let mut version = [0u8; 8];
    version.copy_from_slice(&data[..8]);
    let mut weights = [[0i16; 256]; EMBEDDING_DIM];
    let mut at = 8usize;
    for row in weights.iter_mut() {
        for w in row.iter_mut() {
            let lo = data[at % data.len()] as u8;
            let hi = data[(at + 1) % data.len()] as u8;
            *w = i16::from_le_bytes([lo, hi]);
            at += 2;
        }
    }
    let c = CodecW::new(WeightVersion(u64::from_le_bytes(version)), weights);
    let parsed = CodecW::parse(&c.canonical_bytes()).expect("canonical round-trip");
    assert_eq!(c, parsed);
    assert_eq!(c.commitment(), parsed.commitment());
});
