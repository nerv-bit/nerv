//! Internal RNG used by test scaffolding. The high-level test helpers
//! (`params`, `TAU0`, `leg`, `payout`, `shared_qc`, `TestWorld`,
//! `register_tau`, `seal`, `valid_block`, `build_chain`, etc.) live in
//! the public `crate::testkit` module — this file only contains the
//! raw RNG primitive that `testkit` builds on.

#[derive(Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> SplitMix64 {
        SplitMix64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    pub fn bytes32(&mut self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for chunk in out.chunks_exact_mut(8) {
            chunk.copy_from_slice(&self.next_u64().to_le_bytes());
        }
        out
    }

    #[allow(dead_code)]
    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n);
        while out.len() + 8 <= n {
            out.extend_from_slice(&self.next_u64().to_le_bytes());
        }
        if out.len() < n {
            let tail = self.next_u64().to_le_bytes();
            out.extend_from_slice(&tail[..n - out.len()]);
        }
        out
    }
}