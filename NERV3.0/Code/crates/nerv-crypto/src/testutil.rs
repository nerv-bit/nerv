//! Test-only deterministic RNG (no external test dependencies).
//!
//! Lives behind `#[cfg(test)] mod testutil;` in `lib.rs` so it never ships
//! in the production binary. Used by `mlkem.rs` / `mldsa.rs` (and any
//! future crypto test) to seed deterministic input without pulling in a
//! rand-stack crate purely for tests.

pub(crate) struct DetRng {
    state: u64,
}

impl DetRng {
    pub(crate) fn new(seed: u64) -> DetRng {
        DetRng { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(crate) fn bytes32(&mut self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for chunk in out.chunks_exact_mut(8) {
            chunk.copy_from_slice(&self.next_u64().to_le_bytes());
        }
        out
    }

    pub(crate) fn bytes64(&mut self) -> [u8; 64] {
        let mut out = [0u8; 64];
        for chunk in out.chunks_exact_mut(8) {
            chunk.copy_from_slice(&self.next_u64().to_le_bytes());
        }
        out
    }
}
