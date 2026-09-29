//! Test-only deterministic randomness (SplitMix64) — no external crates, no
//! floats, byte-stable across platforms and runs. (Property testing with
//! proptest arrives with the chunk-1 testkit surfaces; core primitives get
//! deterministic PRNG coverage now, with zero new dependencies.)

#[derive(Clone)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> SplitMix64 {
        SplitMix64 { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(crate) fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    /// Deterministic byte string (each 8-byte chunk is a fresh LE word).
    pub(crate) fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n);
        while out.len() + 8 <= n {
            out.extend_from_slice(&self.next_u64().to_le_bytes());
        }
        if out.len() < n {
            let tail = self.next_u64().to_le_bytes();
            let take = n - out.len();
            out.extend_from_slice(&tail[..take]);
        }
        out
    }
}

/// Roundtrip + strictness for any codec type: exact length, decode-equality,
/// truncation and trailing bytes rejected (codec law L2).
pub(crate) fn codec_roundtrip<T>(v: T)
where
    T: crate::codec::Encode + crate::codec::Decode + PartialEq + std::fmt::Debug,
{
    let enc = v.encode();
    assert_eq!(enc.len(), v.encoded_len());
    let dec = T::decode(&enc).expect("roundtrip decode");
    assert_eq!(&dec, &v);
    if !enc.is_empty() {
        assert!(T::decode(&enc[..enc.len() - 1]).is_err(), "truncation must fail");
        let mut ext = enc.clone();
        ext.push(0);
        assert!(T::decode(&ext).is_err(), "trailing bytes must fail");
    }
}
