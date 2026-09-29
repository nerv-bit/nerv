//! The node's in-memory chain tracker (erratum 207): stores resolved
//! legs from applied blocks, serving the ChainSource for the executor's
//! escrow-recovery path (D.3 reversion and claim validation).

use std::collections::BTreeMap;

use nerv_core::types::{Height, LegKey};
use nerv_state::block::{ResolvedLeg, ShardBlock};
use nerv_state::ChainSource;

/// In-memory ChainSource: height → resolved legs from applied blocks.
/// Bounded to the last `CAPACITY` heights (the D.3 expiry window is
/// ≤ 24 h; at 1 s blocks that's ≤ 86,400 — we keep a generous margin).
pub struct MemoryChain {
    legs: BTreeMap<u64, Vec<ResolvedLeg>>,
    capacity: u64,
}

const DEFAULT_CAPACITY: u64 = 100_000;

impl MemoryChain {
    pub fn new() -> MemoryChain {
        MemoryChain { legs: BTreeMap::new(), capacity: DEFAULT_CAPACITY }
    }

    /// Store a block's resolved legs. Prunes beyond capacity.
    pub fn store_block(&mut self, height: u64, block: &ShardBlock) {
        if let Ok(resolved) = block.resolve_legs() {
            self.legs.insert(height, resolved);
            self.prune();
        }
    }

    /// The highest stored height.
    pub fn tip(&self) -> Option<u64> {
        self.legs.keys().next_back().copied()
    }

    /// The number of stored heights.
    pub fn len(&self) -> usize {
        self.legs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.legs.is_empty()
    }

    fn prune(&mut self) {
        while self.legs.len() as u64 > self.capacity {
            if let Some(lowest) = self.legs.keys().next().copied() {
                self.legs.remove(&lowest);
            } else {
                break;
            }
        }
    }
}

impl Default for MemoryChain {
    fn default() -> Self {
        MemoryChain::new()
    }
}

impl ChainSource for MemoryChain {
    fn settled_leg(&self, height: Height, key: &LegKey) -> Option<ResolvedLeg> {
        self.legs
            .get(&height.as_u64())?
            .iter()
            .find(|r| r.key == *key)
            .cloned()
    }
}
