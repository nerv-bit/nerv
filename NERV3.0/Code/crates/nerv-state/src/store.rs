//! The node-store seam (DSR-10; erratum 117): one `NodeStore` trait, two
//! real backends — the always-on BTreeMap store (the testkit's) and the
//! feature-gated RocksDB backend — plus the rebuild-from-DA machinery.

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

use nerv_core::codec::{Decode, Encode};
use nerv_core::hash::Hash256;
use nerv_core::types::{Height, LegKey, ShardId};

use crate::block::{ResolvedLeg, ShardBlock};
use crate::error::{RebuildError, StoreError};
use crate::executor::{apply_block, BeaconView, ChainSource, ShardState};
use crate::header::ShardHeader;

pub const SCHEMA_VERSION: u32 = 1;

const CF_META: &str = "meta";
const CF_HEADERS: &str = "headers";
const CF_BLOCKS: &str = "blocks";
const CF_SNAPSHOTS: &str = "snapshots";
const VERSION_KEY: &[u8] = b"schema_version";
/// The per-shard tip markers' reserved height key.
const TIP_MARKER_HEIGHT: u64 = u64::MAX;
const KEY_LEN: usize = 11;

/// shard(3B) ‖ height(8B LE) — the shared key layout of both backends.
fn shard_key(shard: &ShardId, height: Height) -> [u8; KEY_LEN] {
    let mut k = [0u8; KEY_LEN];
    k[0] = shard.bits();
    k[1..3].copy_from_slice(&shard.value().to_le_bytes());
    k[3..].copy_from_slice(&height.as_u64().to_le_bytes());
    k
}

fn decode_opt<T: Decode>(bytes: Option<&[u8]>, what: &'static str) -> Result<Option<T>, StoreError> {
    match bytes {
        None => Ok(None),
        Some(b) => T::decode(b).map(Some).map_err(|source| StoreError::Decode { what, source }),
    }
}

fn parse_height_marker(bytes: Option<&[u8]>) -> Option<Height> {
    let b = bytes?;
    if b.len() != 8 {
        return None;
    }
    let mut m = [0u8; 8];
    m.copy_from_slice(b);
    Some(Height::from_u64(u64::from_le_bytes(m)))
}

/// The node's persistent state: headers, blocks, and state snapshots per
/// shard. Puts follow a nondecreasing-height-per-shard discipline; the
/// tip markers are last-write-wins.
pub trait NodeStore {
    fn put_header(&self, shard: ShardId, header: &ShardHeader) -> Result<(), StoreError>;
    fn header(&self, shard: ShardId, height: Height) -> Result<Option<ShardHeader>, StoreError>;
    fn put_block(&self, block: &ShardBlock) -> Result<(), StoreError>;
    fn block(&self, shard: ShardId, height: Height) -> Result<Option<ShardBlock>, StoreError>;
    fn block_tip(&self, shard: ShardId) -> Result<Option<Height>, StoreError>;
    fn put_snapshot(&self, state: &ShardState) -> Result<(), StoreError>;
    fn snapshot_at(&self, shard: ShardId, height: Height) -> Result<Option<ShardState>, StoreError>;
    fn latest_snapshot(&self, shard: ShardId) -> Result<Option<ShardState>, StoreError>;
    fn schema_version(&self) -> Result<u32, StoreError>;
}

// ---------------------------------------------------------------------------
// The in-memory backend (the testkit's; the key layout's twin)
// ---------------------------------------------------------------------------

#[derive(Default)]
struct MemCfs {
    headers: BTreeMap<[u8; KEY_LEN], Vec<u8>>,
    blocks: BTreeMap<[u8; KEY_LEN], Vec<u8>>,
    snapshots: BTreeMap<[u8; KEY_LEN], Vec<u8>>,
}

/// The always-on store: BTreeMaps over the shared key layout.
#[derive(Default)]
pub struct MemStore {
    cfs: Mutex<MemCfs>,
    version: u32,
}

impl MemStore {
    pub fn new() -> MemStore {
        MemStore { cfs: Mutex::new(MemCfs::default()), version: SCHEMA_VERSION }
    }

    fn lock(&self) -> Result<MutexGuard<'_, MemCfs>, StoreError> {
        self.cfs.lock().map_err(|_| StoreError::LockPoisoned)
    }
}

impl NodeStore for MemStore {
    fn put_header(&self, shard: ShardId, header: &ShardHeader) -> Result<(), StoreError> {
        let mut cfs = self.lock()?;
        cfs.headers.insert(shard_key(&shard, header.height), header.encode());
        Ok(())
    }

    fn header(&self, shard: ShardId, height: Height) -> Result<Option<ShardHeader>, StoreError> {
        let cfs = self.lock()?;
        decode_opt(cfs.headers.get(&shard_key(&shard, height)).map(|v| v.as_slice()), "header")
    }

    fn put_block(&self, block: &ShardBlock) -> Result<(), StoreError> {
        let mut cfs = self.lock()?;
        cfs.blocks.insert(shard_key(&block.shard, block.header.height), block.encode());
        let tip = shard_key(&block.shard, Height::from_u64(TIP_MARKER_HEIGHT));
        cfs.blocks.insert(tip, block.header.height.as_u64().to_le_bytes().to_vec());
        Ok(())
    }

    fn block(&self, shard: ShardId, height: Height) -> Result<Option<ShardBlock>, StoreError> {
        let cfs = self.lock()?;
        decode_opt(cfs.blocks.get(&shard_key(&shard, height)).map(|v| v.as_slice()), "block")
    }

    fn block_tip(&self, shard: ShardId) -> Result<Option<Height>, StoreError> {
        let cfs = self.lock()?;
        let tip = shard_key(&shard, Height::from_u64(TIP_MARKER_HEIGHT));
        Ok(parse_height_marker(cfs.blocks.get(&tip).map(|v| v.as_slice())))
    }

    fn put_snapshot(&self, state: &ShardState) -> Result<(), StoreError> {
        if state.height().as_u64() == TIP_MARKER_HEIGHT {
            return Err(StoreError::ReservedHeight);
        }
        let mut cfs = self.lock()?;
        cfs.snapshots.insert(shard_key(&state.shard(), state.height()), state.encode());
        let tip = shard_key(&state.shard(), Height::from_u64(TIP_MARKER_HEIGHT));
        cfs.snapshots.insert(tip, state.height().as_u64().to_le_bytes().to_vec());
        Ok(())
    }

    fn snapshot_at(&self, shard: ShardId, height: Height) -> Result<Option<ShardState>, StoreError> {
        let cfs = self.lock()?;
        decode_opt(cfs.snapshots.get(&shard_key(&shard, height)).map(|v| v.as_slice()), "snapshot")
    }

    fn latest_snapshot(&self, shard: ShardId) -> Result<Option<ShardState>, StoreError> {
        let cfs = self.lock()?;
        let tip = shard_key(&shard, Height::from_u64(TIP_MARKER_HEIGHT));
        let Some(h) = parse_height_marker(cfs.snapshots.get(&tip).map(|v| v.as_slice())) else {
            return Ok(None);
        };
        drop(cfs);
        self.snapshot_at(shard, h)
    }

    fn schema_version(&self) -> Result<u32, StoreError> {
        Ok(self.version)
    }
}

// ---------------------------------------------------------------------------
// The RocksDB backend (feature-gated; erratum 117)
// ---------------------------------------------------------------------------

#[cfg(feature = "rocksdb")]
mod rocks_backend {
    use super::*;

    /// The C++ backend: meta/headers/blocks/snapshots column families,
    /// schema-version-gated at open.
    pub struct RocksStore {
        db: rocksdb::DB,
    }

    fn back(e: rocksdb::Error) -> StoreError {
        StoreError::Backend(e.to_string())
    }

    impl RocksStore {
        pub fn open(path: impl AsRef<std::path::Path>) -> Result<RocksStore, StoreError> {
            let mut opts = rocksdb::Options::default();
            opts.create_if_missing(true);
            opts.create_missing_column_families(true);
            let cfs: Vec<rocksdb::ColumnFamilyDescriptor> = [
                CF_META, CF_HEADERS, CF_BLOCKS, CF_SNAPSHOTS,
            ]
            .iter()
            .map(|&name| rocksdb::ColumnFamilyDescriptor::new(name, rocksdb::Options::default()))
            .collect();
            let db =
                rocksdb::DB::open_cf_descriptors(&opts, path, cfs).map_err(back)?;
            let store = RocksStore { db };
            let meta = store.cf(CF_META)?;
            let want = SCHEMA_VERSION.to_le_bytes();
            match store.db.get_cf(&meta, VERSION_KEY) {
                Ok(None) => store.db.put_cf(&meta, VERSION_KEY, &want).map_err(back)?,
                Ok(Some(v)) if v.as_slice() == want => {}
                Ok(Some(v)) => {
                    let mut b = [0u8; 4];
                    let found = if v.len() == 4 {
                        b.copy_from_slice(&v);
                        u32::from_le_bytes(b)
                    } else {
                        0
                    };
                    return Err(StoreError::Version { found, expected: SCHEMA_VERSION });
                }
                Err(e) => return Err(back(e)),
            }
            Ok(store)
        }

        fn cf(&self, name: &str) -> Result<rocksdb::ColumnFamily, StoreError> {
            self.db
                .cf_handle(name)
                .ok_or_else(|| StoreError::Backend(format!("column family `{name}` missing")))
        }

        fn get(&self, cf_name: &str, key: &[u8]) -> Result<Option<Vec<u8>>, StoreError> {
            let cf = self.cf(cf_name)?;
            self.db.get_cf(&cf, key).map_err(back)
        }

        fn put1(&self, cf_name: &str, key: &[u8], value: &[u8]) -> Result<(), StoreError> {
            let cf = self.cf(cf_name)?;
            self.db.put_cf(&cf, key, value).map_err(back)
        }

        /// Atomic two-key write (data + tip marker).
        fn put2(
            &self,
            cf_name: &str,
            k1: &[u8],
            v1: &[u8],
            k2: &[u8],
            v2: &[u8],
        ) -> Result<(), StoreError> {
            let cf = self.cf(cf_name)?;
            let mut batch = rocksdb::WriteBatch::default();
            batch.put_cf(&cf, k1, v1);
            batch.put_cf(&cf, k2, v2);
            self.db.write(&batch).map_err(back)
        }
    }

    impl NodeStore for RocksStore {
        fn put_header(&self, shard: ShardId, header: &ShardHeader) -> Result<(), StoreError> {
            self.put1(CF_HEADERS, &shard_key(&shard, header.height), &header.encode())
        }

        fn header(&self, shard: ShardId, height: Height) -> Result<Option<ShardHeader>, StoreError> {
            decode_opt(self.get(CF_HEADERS, &shard_key(&shard, height))?.as_deref(), "header")
        }

        fn put_block(&self, block: &ShardBlock) -> Result<(), StoreError> {
            let data = shard_key(&block.shard, block.header.height);
            let tip = shard_key(&block.shard, Height::from_u64(TIP_MARKER_HEIGHT));
            self.put2(
                CF_BLOCKS,
                &data,
                &block.encode(),
                &tip,
                &block.header.height.as_u64().to_le_bytes(),
            )
        }

        fn block(&self, shard: ShardId, height: Height) -> Result<Option<ShardBlock>, StoreError> {
            decode_opt(self.get(CF_BLOCKS, &shard_key(&shard, height))?.as_deref(), "block")
        }

        fn block_tip(&self, shard: ShardId) -> Result<Option<Height>, StoreError> {
            let tip = shard_key(&shard, Height::from_u64(TIP_MARKER_HEIGHT));
            Ok(parse_height_marker(self.get(CF_BLOCKS, &tip)?.as_deref()))
        }

        fn put_snapshot(&self, state: &ShardState) -> Result<(), StoreError> {
            if state.height().as_u64() == TIP_MARKER_HEIGHT {
                return Err(StoreError::ReservedHeight);
            }
            let data = shard_key(&state.shard(), state.height());
            let tip = shard_key(&state.shard(), Height::from_u64(TIP_MARKER_HEIGHT));
            self.put2(
                CF_SNAPSHOTS,
                &data,
                &state.encode(),
                &tip,
                &state.height().as_u64().to_le_bytes(),
            )
        }

        fn snapshot_at(&self, shard: ShardId, height: Height) -> Result<Option<ShardState>, StoreError> {
            decode_opt(self.get(CF_SNAPSHOTS, &shard_key(&shard, height))?.as_deref(), "snapshot")
        }

        fn latest_snapshot(&self, shard: ShardId) -> Result<Option<ShardState>, StoreError> {
            let tip = shard_key(&shard, Height::from_u64(TIP_MARKER_HEIGHT));
            let Some(h) = parse_height_marker(self.get(CF_SNAPSHOTS, &tip)?.as_deref()) else {
                return Ok(None);
            };
            self.snapshot_at(shard, h)
        }

        fn schema_version(&self) -> Result<u32, StoreError> {
            match self.get(CF_META, VERSION_KEY)? {
                Some(v) if v.len() == 4 => {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&v);
                    Ok(u32::from_le_bytes(b))
                }
                _ => Err(StoreError::Backend("schema version record absent".into())),
            }
        }
    }
}

#[cfg(feature = "rocksdb")]
pub use rocks_backend::RocksStore;

// ---------------------------------------------------------------------------
// Rebuild-from-DA
// ---------------------------------------------------------------------------

/// An ordered source of finalized blocks — the DA archive's seam (the
/// node's own store serves via [`StoreArchive`]). `None` = absent or
/// undecodable: a conservative miss callers treat as unavailability.
pub trait ShardArchive {
    fn block_at(&self, shard: ShardId, height: Height) -> Option<ShardBlock>;
    fn tip(&self, shard: ShardId) -> Option<Height>;
}

/// Any `NodeStore` as a [`ShardArchive`].
pub struct StoreArchive<'a> {
    store: &'a dyn NodeStore,
}

impl<'a> StoreArchive<'a> {
    pub fn new(store: &'a dyn NodeStore) -> StoreArchive<'a> {
        StoreArchive { store }
    }
}

impl ShardArchive for StoreArchive<'_> {
    fn block_at(&self, shard: ShardId, height: Height) -> Option<ShardBlock> {
        self.store.block(shard, height).ok().flatten()
    }
    fn tip(&self, shard: ShardId) -> Option<Height> {
        self.store.block_tip(shard).ok().flatten()
    }
}

/// The archive as the executor's [`ChainSource`]: legs are resolved from
/// the archived blocks on demand (escrow-shell recovery, D.3(d)).
pub struct ArchiveChain<'a> {
    archive: &'a dyn ShardArchive,
    shard: ShardId,
}

impl<'a> ArchiveChain<'a> {
    pub fn new(archive: &'a dyn ShardArchive, shard: ShardId) -> ArchiveChain<'a> {
        ArchiveChain { archive, shard }
    }
}

impl ChainSource for ArchiveChain<'_> {
    fn settled_leg(&self, height: Height, key: &LegKey) -> Option<ResolvedLeg> {
        let block = self.archive.block_at(self.shard, height)?;
        let resolved = block.resolve_legs().ok()?;
        resolved.into_iter().find(|r| r.key == *key)
    }
}

/// The full rebuild: apply the archive's blocks 1..=tip over genesis
/// (DSR-10's rebuild-from-DA mode; the erasure-coded reconstruction that
/// feeds the archive is nerv-da's, chunk 15).
pub fn rebuild_state(
    shard: ShardId,
    params_root: Hash256,
    archive: &dyn ShardArchive,
    view: &dyn BeaconView,
) -> Result<ShardState, RebuildError> {
    let tip = archive.tip(shard).map(|t| t.as_u64()).unwrap_or(0);
    let mut st = ShardState::genesis(shard, params_root);
    let chain = ArchiveChain::new(archive, shard);
    for hgt in 1..=tip {
        let block = archive
            .block_at(shard, Height::from_u64(hgt))
            .ok_or(RebuildError::Gap { height: hgt })?;
        st = apply_block(st, &block, view, &chain).map_err(RebuildError::Executor)?.0;
    }
    Ok(st)
}

/// The node's load path: prefer the latest snapshot and apply forward;
/// on any fast-path failure (stale snapshot, gap, invalid block) fall
/// back to the full rebuild — a persistent failure surfaces from it.
pub fn load_shard(
    store: &dyn NodeStore,
    shard: ShardId,
    params_root: Hash256,
    archive: &dyn ShardArchive,
    view: &dyn BeaconView,
) -> Result<ShardState, RebuildError> {
    let full = || rebuild_state(shard, params_root, archive, view);
    let snapshot = store.latest_snapshot(shard).map_err(RebuildError::Store)?;
    let Some(mut st) = snapshot else { return full() };
    let h0 = st.height().as_u64();
    let Some(tip) = archive.tip(shard) else { return Ok(st) };
    let tip = tip.as_u64();
    if tip <= h0 {
        return Ok(st);
    }
    let chain = ArchiveChain::new(archive, shard);
    for hgt in (h0 + 1)..=tip {
        let Some(block) = archive.block_at(shard, Height::from_u64(hgt)) else {
            return full();
        };
        match apply_block(st, &block, view, &chain) {
            Ok((s, _)) => st = s,
            Err(_) => return full(),
        }
    }
    Ok(st)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::harness::{build_chain, params};
    use nerv_core::types::ShardSet;

    #[test]
    fn mem_store_roundtrips_and_markers() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let (blocks, states, _) = build_chain(shard, 3, 0x57A);
        let store = MemStore::new();
        assert_eq!(store.schema_version().unwrap(), SCHEMA_VERSION);

        for b in &blocks {
            store.put_block(b).unwrap();
        }
        store.put_header(shard, &blocks[1].header).unwrap();

        assert_eq!(store.block(shard, Height::from_u64(1)).unwrap().as_ref(), Some(&blocks[0]));
        assert_eq!(store.block(shard, Height::from_u64(2)).unwrap().as_ref(), Some(&blocks[1]));
        assert_eq!(store.block(shard, Height::from_u64(3)).unwrap().as_ref(), Some(&blocks[2]));
        assert!(store.block(shard, Height::from_u64(4)).unwrap().is_none());
        assert_eq!(store.header(shard, Height::from_u64(2)).unwrap().as_ref(), Some(&blocks[1].header));
        assert!(store.header(shard, Height::from_u64(1)).unwrap().is_none());
        assert_eq!(store.block_tip(shard).unwrap(), Some(Height::from_u64(3)));

        let other = set.ids()[9];
        assert!(store.block(other, Height::from_u64(1)).unwrap().is_none());
        assert_eq!(store.block_tip(other).unwrap(), None);

        store.put_snapshot(&states[2]).unwrap();
        assert_eq!(store.snapshot_at(shard, Height::from_u64(2)).unwrap().as_ref(), Some(&states[2]));
        assert!(store.snapshot_at(shard, Height::from_u64(1)).unwrap().is_none());
        assert_eq!(store.latest_snapshot(shard).unwrap().as_ref(), Some(&states[2]));
        assert!(store.latest_snapshot(other).unwrap().is_none());
        store.put_snapshot(&states[3]).unwrap();
        assert_eq!(store.latest_snapshot(shard).unwrap().as_ref(), Some(&states[3]));

        // A snapshot at the reserved marker height is rejected — built by
        // patching the encoded height (a hostile-snapshot decode test).
        let mut bytes = states[3].encode();
        let n = bytes.len();
        bytes[n - 8..].copy_from_slice(&u64::MAX.to_le_bytes());
        let maxed = ShardState::decode(&bytes).unwrap();
        assert_eq!(maxed.height().as_u64(), u64::MAX);
        assert!(matches!(store.put_snapshot(&maxed), Err(StoreError::ReservedHeight)));
    }

    #[test]
    fn snapshot_codec_roundtrip() {
        let set = ShardSet::genesis();
        let (_, states, _) = build_chain(set.ids()[7], 4, 0x57B);
        let enc = states[4].encode();
        assert_eq!(enc.len(), states[4].encoded_len());
        assert_eq!(ShardState::decode(&enc).unwrap(), states[4]);
        for cut in 0..enc.len() {
            assert!(ShardState::decode(&enc[..cut]).is_err(), "cut {cut}");
        }
        let mut ext = enc.clone();
        ext.push(0);
        assert!(ShardState::decode(&ext).is_err());
    }

    #[test]
    fn rebuild_matches_live() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let (blocks, states, world) = build_chain(shard, 5, 0x57C);
        let store = MemStore::new();
        for b in &blocks {
            store.put_block(b).unwrap();
        }
        let archive = StoreArchive::new(&store);
        let rebuilt = rebuild_state(shard, params(), &archive, &world).unwrap();
        assert_eq!(rebuilt.state_commitment(), states[5].state_commitment());
        assert_eq!(rebuilt.encode(), states[5].encode());
        assert_eq!(rebuilt, states[5]);
    }

    #[test]
    fn load_shard_full_rebuild_and_fast_path() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let (blocks, states, world) = build_chain(shard, 5, 0x57D);

        // No snapshot: the full rebuild.
        let store = MemStore::new();
        for b in &blocks {
            store.put_block(b).unwrap();
        }
        let archive = StoreArchive::new(&store);
        let loaded = load_shard(&store, shard, params(), &archive, &world).unwrap();
        assert_eq!(loaded, states[5]);

        // A mid-chain snapshot: apply forward from it.
        let store = MemStore::new();
        for b in &blocks[..3] {
            store.put_block(b).unwrap();
        }
        store.put_snapshot(&states[3]).unwrap();
        for b in &blocks[3..] {
            store.put_block(b).unwrap();
        }
        let archive = StoreArchive::new(&store);
        let loaded = load_shard(&store, shard, params(), &archive, &world).unwrap();
        assert_eq!(loaded, states[5]);

        // No archive at all: the snapshot stands.
        let empty = MemStore::new();
        empty.put_snapshot(&states[2]).unwrap();
        let archive = StoreArchive::new(&empty);
        let loaded = load_shard(&empty, shard, params(), &archive, &world).unwrap();
        assert_eq!(loaded, states[2]);
    }

    #[test]
    fn load_shard_falls_back_on_stale_snapshot() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let (blocks, states, world) = build_chain(shard, 5, 0x57E);
        let (_, states_b, _) = build_chain(shard, 3, 0x99E); // a different chain
        let store = MemStore::new();
        store.put_snapshot(&states_b[3]).unwrap(); // C_3 of chain B
        for b in &blocks {
            store.put_block(b).unwrap();
        }
        let archive = StoreArchive::new(&store);
        let loaded = load_shard(&store, shard, params(), &archive, &world).unwrap();
        assert_eq!(loaded, states[5]);
    }

    #[test]
    fn gap_detected() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let (blocks, _, world) = build_chain(shard, 4, 0x57F);
        let store = MemStore::new();
        store.put_block(&blocks[0]).unwrap();
        store.put_block(&blocks[1]).unwrap();
        store.put_block(&blocks[3]).unwrap(); // 3 missing; tip = 4
        let archive = StoreArchive::new(&store);
        assert!(matches!(
            rebuild_state(shard, params(), &archive, &world),
            Err(RebuildError::Gap { height: 3 })
        ));
    }

    #[test]
    fn invalid_archived_block_surfaces() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let (blocks, _, world) = build_chain(shard, 5, 0x580);
        let mut bad3 = blocks[2].clone();
        bad3.header.nullifier_root = Hash256::from_bytes([0xEE; 32]);
        let store = MemStore::new();
        store.put_block(&blocks[0]).unwrap();
        store.put_block(&blocks[1]).unwrap();
        store.put_block(&bad3).unwrap();
        let archive = StoreArchive::new(&store);
        assert!(matches!(
            rebuild_state(shard, params(), &archive, &world),
            Err(RebuildError::Executor(ExecutorError::NullifierRootMismatch))
        ));
    }

    use crate::error::ExecutorError;
}

#[cfg(all(test, feature = "rocksdb"))]
mod rocks_tests {
    use super::*;
    use crate::error::ExecutorError;
    use crate::testutil::harness::{build_chain, params};
    use nerv_core::types::ShardSet;

    fn temp_path(tag: u32) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("nerv-state-rocks-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn open_version_roundtrip_and_rebuild() {
        let path = temp_path(1);
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let (blocks, states, world) = build_chain(shard, 3, 0x90C);
        {
            let store = RocksStore::open(&path).unwrap();
            assert_eq!(store.schema_version().unwrap(), SCHEMA_VERSION);
            for b in &blocks {
                store.put_block(b).unwrap();
            }
            store.put_snapshot(&states[2]).unwrap();
            assert_eq!(store.block(shard, Height::from_u64(2)).unwrap().as_ref(), Some(&blocks[1]));
            assert_eq!(
                store.header(shard, Height::from_u64(3)).unwrap().as_ref(),
                Some(&blocks[2].header)
            );
            assert_eq!(store.block_tip(shard).unwrap(), Some(Height::from_u64(3)));
            assert_eq!(store.latest_snapshot(shard).unwrap().as_ref(), Some(&states[2]));
            let archive = StoreArchive::new(&store);
            let rebuilt = rebuild_state(shard, params(), &archive, &world).unwrap();
            assert_eq!(rebuilt, states[3]);
        }
        // Reopen: version and data persist.
        {
            let store = RocksStore::open(&path).unwrap();
            assert_eq!(store.schema_version().unwrap(), SCHEMA_VERSION);
            assert_eq!(store.block_tip(shard).unwrap(), Some(Height::from_u64(3)));
            assert_eq!(store.latest_snapshot(shard).unwrap().as_ref(), Some(&states[2]));
        }
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn invalid_archived_block_surfaces() {
        let path = temp_path(2);
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let (blocks, _, world) = build_chain(shard, 3, 0x90D);
        let mut bad2 = blocks[1].clone();
        bad2.header.transit_root = Hash256::from_bytes([0xEE; 32]);
        let store = RocksStore::open(&path).unwrap();
        store.put_block(&blocks[0]).unwrap();
        store.put_block(&bad2).unwrap();
        let archive = StoreArchive::new(&store);
        assert!(matches!(
            rebuild_state(shard, params(), &archive, &world),
            Err(RebuildError::Executor(ExecutorError::TransitRootMismatch))
        ));
        let _ = std::fs::remove_dir_all(&path);
    }
}
