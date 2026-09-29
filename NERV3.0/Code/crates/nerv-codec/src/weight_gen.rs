//! Verifiable weight generation and the per-epoch machine certification of
//! the codec `W` (WP §7.6; App C.3): beacon-XOF expansion, weight-norm and
//! robust fleet checks, exhaustive column-pair independence and full-rank
//! checks over the prover field, sampled spark and column–rail rank checks,
//! and the public certificate record.
//!
//! What is sound, what is sampled, what is argued (P9, stated plainly):
//!
//! * Exhaustive and sound: every norm gate; row distinctness; every one of
//!   the C(256,2) column pairs is linearly independent over 𝔽_p (this
//!   subsumes exact ℚ-proportionality of integer columns: if `b = (p/q)·a`
//!   with gcd(p,q) = 1, then `q` divides every entry of `a`, hence
//!   `q ≤ 2^15 ≪ p_G`, and the relation survives reduction); and the full
//!   64×256 matrix has row rank 64 over 𝔽_p. Together: spark ≥ 3 exactly,
//!   full image dimension, no dead or duplicated columns.
//!
//! * The spark ≥ 13 threshold itself (WP §7.4: differences of admissible
//!   vectors touch ≤ 12 slot columns; App B "spark ≥ 13 of 224") admits no
//!   cheap sound certificate — exact spark is NP-hard, and the classical
//!   coherence bound `spark ≥ 1 + 1/μ` is vacuous at these dimensions
//!   (random 64×256 matrices have μ ≈ 0.5 while spark 65). The WP's own
//!   certification is provenance: verifiable random generation. For the
//!   XOF expansion, a fixed 64×k (k ≤ 13) submatrix is 𝔽_p-dependent with
//!   probability ≤ k·p^(−52) ≈ 2^(−3308); union over all ≤ 12-subsets of the
//!   224 slot columns (≈ 2^70 of them) still leaves ≈ 2^(−3230). On top,
//!   this module runs deterministic sampled rank checks as detection:
//!   k-subsets of the slot block for k = 3..=13, and sets of 6 sampled slot
//!   columns plus one rail column. Sample sets are XOF-derived from the
//!   matrix's own commitment under `nerv.w.spark`, so anyone recomputes the
//!   identical battery.
//!
//! * Bounded failure (WP §7.4/§7.7): a `W` that slipped every gate degrades
//!   an advisory public index. Statement 8 (δ ≠ 0) still rejects any actual
//!   all-zero delta at proof time, and custody never consults `W` at all.

use nerv_core::constants::{W_GEN, W_SPARK};
use nerv_core::hash::Xof;

use crate::codec_w::{CodecW, WCommitment, WeightVersion, EMBEDDING_DIM};
use crate::features::{ACCOUNT_SLOT_COUNT, FEATURE_COUNT, MAX_ACTIVE_ACCOUNT_SLOTS};

/// WP §7.4 / App B: the slot block requires spark ≥ 13 — no nonzero vector
/// supported on ≤ 12 slot columns lies in W's kernel (differences of two
/// admissible feature vectors touch ≤ 2 × 6 = 12 slots).
pub const SPARK_REQUIREMENT: usize = 13;

/// Per-row ℓ2² bounds (frozen [genesis-config]; governance-adjustable).
/// Uniform i16 expansion sits at ~2^36.4; the floor kills near-dead rows.
pub const ROW_L2SQ_MIN: u64 = 1 << 30;
pub const ROW_L2SQ_MAX: u64 = (FEATURE_COUNT as u64) << 30;
/// Per-column ℓ2² bounds. Uniform expansion sits at ~2^34.4.
pub const COL_L2SQ_MIN: u64 = 1 << 24;
pub const COL_L2SQ_MAX: u64 = (EMBEDDING_DIM as u64) << 30;

/// Robust fleet band: every row (column) ℓ1 must satisfy
/// |v − median| ≤ `FLEET_BAND_MADS`·MAD ≈ 6 robust standard deviations.
/// Honest uniform expansion sits at ~4.9·MAD; a dominating all-extreme row
/// lands at ~41·MAD and is rejected.
pub const FLEET_BAND_MADS: u64 = 9;

// ---------------------------------------------------------------------------
// Beacon provenance and expansion (WP §7.6)
// ---------------------------------------------------------------------------

/// The governance epoch's freshness-beacon output (WP §7.6): 32 opaque
/// bytes. Its construction (the hash chain of finalized interval
/// attestations, DSR-5) lives in nerv-consensus; the identity layer only
/// consumes it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BeaconRandomness([u8; 32]);

impl BeaconRandomness {
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        BeaconRandomness(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Verifiable expansion: `W = XOF("nerv.w.gen", beacon ‖ version)` read as
/// 16,384 uniform little-endian i16 weights, row-major. Pure: identical
/// inputs produce bit-identical matrices on every platform.
pub fn expand(beacon: &BeaconRandomness, version: WeightVersion) -> CodecW {
    let ver = version.0.to_le_bytes();
    let parts: [&[u8]; 2] = [beacon.as_bytes(), &ver];
    let mut xof = Xof::framed(&W_GEN, &parts);
    let mut rows = [[0i16; FEATURE_COUNT]; EMBEDDING_DIM];
    for row in rows.iter_mut() {
        for w in row.iter_mut() {
            *w = i16::from_le_bytes(xof.read_array::<2>());
        }
    }
    CodecW::new(version, rows)
}

/// Provenance check: does `w` equal the verifiable expansion of
/// (beacon, version)? Compares `Hash(W ‖ version)` commitments — the value
/// `params_root` commits at adoption.
pub fn verify_expansion(beacon: &BeaconRandomness, version: WeightVersion, w: &CodecW) -> bool {
    expand(beacon, version).commitment() == w.commitment()
}

/// The epoch ceremony's happy path: expand, then certify.
pub fn expand_and_certify(
    beacon: &BeaconRandomness,
    version: WeightVersion,
    cfg: CertConfig,
) -> Result<(CodecW, Certificate), CertificationError> {
    let w = expand(beacon, version);
    let certificate = certify(&w, cfg)?;
    Ok((w, certificate))
}

// ---------------------------------------------------------------------------
// Certification (WP App C.3 machine checks)
// ---------------------------------------------------------------------------

/// Sampling configuration. Default targets sub-second release-build runs;
/// `ceremony()` is the governance-epoch profile.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CertConfig {
    /// Sampled k-subsets (k = 3..=13) of the 224 slot columns, per size.
    pub spark_samples_per_size: u32,
    /// Sampled {6 slot + rail} sets, per rail column.
    pub column_rail_samples: u32,
}

impl Default for CertConfig {
    fn default() -> Self {
        CertConfig { spark_samples_per_size: 512, column_rail_samples: 256 }
    }
}

impl CertConfig {
    pub const fn ceremony() -> Self {
        CertConfig { spark_samples_per_size: 4096, column_rail_samples: 2048 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CertificationError {
    #[error("row {row} ℓ2² {l2sq} outside [{min}, {max}]")]
    RowNorm { row: usize, l2sq: u64, min: u64, max: u64 },
    #[error("column {col} ℓ2² {l2sq} outside [{min}, {max}]")]
    ColNorm { col: usize, l2sq: u64, min: u64, max: u64 },
    #[error("row {row} ℓ1 deviates by {dev} > {bound} (robust fleet band)")]
    RowFleet { row: usize, dev: u64, bound: u64 },
    #[error("column {col} ℓ1 deviates by {dev} > {bound} (robust fleet band)")]
    ColFleet { col: usize, dev: u64, bound: u64 },
    #[error("rows {a} and {b} are identical")]
    DuplicateRows { a: usize, b: usize },
    #[error("columns {a} and {b} are linearly dependent over 𝔽_p")]
    FpPairDependent { a: usize, b: usize },
    #[error("full-matrix rank over 𝔽_p is {rank}, expected {expected}")]
    FpRank { rank: usize, expected: usize },
    #[error("spark sample {sample} (size {size}) has rank {rank} < {size}")]
    SparkSample { size: usize, sample: u64, rank: usize },
    #[error("column–rail sample {sample} for rail {rail} has rank {rank} < {bound}")]
    ColumnRail { rail: usize, sample: u64, rank: usize, bound: usize },
    #[error("certificate has {len} bytes, expected {expected}")]
    BadCertificateLength { len: usize, expected: usize },
}

/// Runs the App C.3 machine-check battery on a candidate `W` and returns
/// the public certificate record on success. Deterministic and
/// reproducible; first failure returns immediately. Gate order: row norms,
/// column norms, row-fleet ℓ1 band, column-fleet ℓ1 band, row
/// distinctness, exhaustive 𝔽_p column-pair independence, full 𝔽_p row
/// rank, sampled spark (slots, sizes 3..=13), sampled column–rail.
pub fn certify(w: &CodecW, cfg: CertConfig) -> Result<Certificate, CertificationError> {
    let weights = w.weights();

    let mut cols_i16: Vec<[i16; EMBEDDING_DIM]> = vec![[0; EMBEDDING_DIM]; FEATURE_COUNT];
    let mut cols_fp: Vec<FpCol> = vec![[0; EMBEDDING_DIM]; FEATURE_COUNT];
    for (r, row) in weights.iter().enumerate() {
        for (c, &wv) in row.iter().enumerate() {
            cols_i16[c][r] = wv;
            cols_fp[c][r] = fp_of_weight(wv);
        }
    }

    // Row norms.
    let mut row_l2sq_min = u64::MAX;
    let mut row_l2sq_max = 0;
    for (r, row) in weights.iter().enumerate() {
        let l2sq = l2sq_i16(row.iter().copied());
        if !(ROW_L2SQ_MIN..=ROW_L2SQ_MAX).contains(&l2sq) {
            return Err(CertificationError::RowNorm {
                row: r,
                l2sq,
                min: ROW_L2SQ_MIN,
                max: ROW_L2SQ_MAX,
            });
        }
        row_l2sq_min = row_l2sq_min.min(l2sq);
        row_l2sq_max = row_l2sq_max.max(l2sq);
    }

    // Column norms.
    let mut col_l2sq_min = u64::MAX;
    let mut col_l2sq_max = 0;
    for (c, col) in cols_i16.iter().enumerate() {
        let l2sq = l2sq_i16(col.iter().copied());
        if !(COL_L2SQ_MIN..=COL_L2SQ_MAX).contains(&l2sq) {
            return Err(CertificationError::ColNorm {
                col: c,
                l2sq,
                min: COL_L2SQ_MIN,
                max: COL_L2SQ_MAX,
            });
        }
        col_l2sq_min = col_l2sq_min.min(l2sq);
        col_l2sq_max = col_l2sq_max.max(l2sq);
    }

    // Robust fleet bands.
    let row_l1: Vec<u64> = weights.iter().map(|row| l1_i16(row.iter().copied())).collect();
    let col_l1: Vec<u64> = cols_i16.iter().map(|col| l1_i16(col.iter().copied())).collect();
    let (med, mad, max_dev) = fleet_stats(&row_l1);
    let bound = FLEET_BAND_MADS.saturating_mul(mad);
    if max_dev > bound {
        let row = row_l1.iter().position(|&v| v.abs_diff(med) == max_dev).unwrap_or(0);
        return Err(CertificationError::RowFleet { row, dev: max_dev, bound });
    }
    let row_fleet_margin = (bound - max_dev) as i128;
    let (med, mad, max_dev) = fleet_stats(&col_l1);
    let bound = FLEET_BAND_MADS.saturating_mul(mad);
    if max_dev > bound {
        let col = col_l1.iter().position(|&v| v.abs_diff(med) == max_dev).unwrap_or(0);
        return Err(CertificationError::ColFleet { col, dev: max_dev, bound });
    }
    let col_fleet_margin = (bound - max_dev) as i128;

    // Row distinctness.
    for (a, ra) in weights.iter().enumerate() {
        for (b, rb) in weights.iter().enumerate().skip(a + 1) {
            if ra == rb {
                return Err(CertificationError::DuplicateRows { a, b });
            }
        }
    }

    // Exhaustive column-pair independence over 𝔽_p (subsumes exact
    // ℚ-proportionality — module docs).
    let mut fp_pairs_checked: u64 = 0;
    for (a, ca) in cols_fp.iter().enumerate() {
        for (b, cb) in cols_fp.iter().enumerate().skip(a + 1) {
            if fp_pair_dependent(ca, cb) {
                return Err(CertificationError::FpPairDependent { a, b });
            }
            fp_pairs_checked += 1;
        }
    }

    // Full row rank over 𝔽_p.
    let mut basis = FpBasis::new();
    for col in cols_fp.iter() {
        basis.insert(*col);
    }
    let fp_rank = basis.len();
    if fp_rank != EMBEDDING_DIM {
        return Err(CertificationError::FpRank { rank: fp_rank, expected: EMBEDDING_DIM });
    }

    // Sampled spark over the slot block (detection; provenance is the
    // sound argument — module docs).
    let commitment = w.commitment();
    let mut spark_samples_total: u64 = 0;
    for size in 3..=SPARK_REQUIREMENT {
        let mut xof = sample_stream(&commitment, FAMILY_SPARK, size);
        for s in 0..cfg.spark_samples_per_size {
            let subset = sample_subset(&mut xof, ACCOUNT_SLOT_COUNT, size);
            let rank = subset_rank(&cols_fp, &subset);
            if rank != size {
                return Err(CertificationError::SparkSample {
                    size,
                    sample: u64::from(s),
                    rank,
                });
            }
            spark_samples_total += 1;
        }
    }

    // Sampled column–rail independence: 6 sampled slot columns plus the
    // rail column must have full rank 7.
    let rail_bound = MAX_ACTIVE_ACCOUNT_SLOTS + 1;
    let mut column_rail_samples_total: u64 = 0;
    let mut xof = sample_stream(&commitment, FAMILY_RAIL, MAX_ACTIVE_ACCOUNT_SLOTS);
    for (rail, _) in cols_fp.iter().enumerate().skip(ACCOUNT_SLOT_COUNT) {
        for s in 0..cfg.column_rail_samples {
            let mut set = sample_subset(&mut xof, ACCOUNT_SLOT_COUNT, MAX_ACTIVE_ACCOUNT_SLOTS);
            set.push(rail as u16);
            let rank = subset_rank(&cols_fp, &set);
            if rank != rail_bound {
                return Err(CertificationError::ColumnRail {
                    rail,
                    sample: u64::from(s),
                    rank,
                    bound: rail_bound,
                });
            }
            column_rail_samples_total += 1;
        }
    }

    Ok(Certificate {
        version: w.version(),
        w_commitment: commitment,
        row_l2sq_min,
        row_l2sq_max,
        col_l2sq_min,
        col_l2sq_max,
        row_fleet_margin,
        col_fleet_margin,
        fp_rank: fp_rank as u8,
        fp_pairs_checked,
        spark_samples_total,
        column_rail_samples_total,
        spark_samples_per_size: cfg.spark_samples_per_size,
        column_rail_samples: cfg.column_rail_samples,
    })
}

/// The public record of a successful certification run (WP App C.3): every
/// machine check passed, with the observed bounds and the exact sampling
/// configuration — re-running `certify` with the recorded config on the
/// same W reproduces these bytes bit-for-bit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Certificate {
    pub version: WeightVersion,
    pub w_commitment: WCommitment,
    pub row_l2sq_min: u64,
    pub row_l2sq_max: u64,
    pub col_l2sq_min: u64,
    pub col_l2sq_max: u64,
    pub row_fleet_margin: i128,
    pub col_fleet_margin: i128,
    pub fp_rank: u8,
    pub fp_pairs_checked: u64,
    pub spark_samples_total: u64,
    pub column_rail_samples_total: u64,
    pub spark_samples_per_size: u32,
    pub column_rail_samples: u32,
}

/// Canonical serialization size (fixed layout, little-endian).
pub const CERT_SIZE: usize = 137;

impl Certificate {
    pub fn canonical_bytes(&self) -> [u8; CERT_SIZE] {
        let mut o = [0u8; CERT_SIZE];
        put_u64(&mut o, 0, self.version.0);
        o[8..40].copy_from_slice(&self.w_commitment.0);
        put_u64(&mut o, 40, self.row_l2sq_min);
        put_u64(&mut o, 48, self.row_l2sq_max);
        put_u64(&mut o, 56, self.col_l2sq_min);
        put_u64(&mut o, 64, self.col_l2sq_max);
        o[72..88].copy_from_slice(&self.row_fleet_margin.to_le_bytes());
        o[88..104].copy_from_slice(&self.col_fleet_margin.to_le_bytes());
        o[104] = self.fp_rank;
        put_u64(&mut o, 105, self.fp_pairs_checked);
        put_u64(&mut o, 113, self.spark_samples_total);
        put_u64(&mut o, 121, self.column_rail_samples_total);
        put_u32(&mut o, 129, self.spark_samples_per_size);
        put_u32(&mut o, 133, self.column_rail_samples);
        o
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, CertificationError> {
        if bytes.len() != CERT_SIZE {
            return Err(CertificationError::BadCertificateLength {
                len: bytes.len(),
                expected: CERT_SIZE,
            });
        }
        let mut cm = [0u8; 32];
        cm.copy_from_slice(&bytes[8..40]);
        let mut ma = [0u8; 16];
        ma.copy_from_slice(&bytes[72..88]);
        let mut mb = [0u8; 16];
        mb.copy_from_slice(&bytes[88..104]);
        Ok(Certificate {
            version: WeightVersion(get_u64(bytes, 0)),
            w_commitment: WCommitment(cm),
            row_l2sq_min: get_u64(bytes, 40),
            row_l2sq_max: get_u64(bytes, 48),
            col_l2sq_min: get_u64(bytes, 56),
            col_l2sq_max: get_u64(bytes, 64),
            row_fleet_margin: i128::from_le_bytes(ma),
            col_fleet_margin: i128::from_le_bytes(mb),
            fp_rank: bytes[104],
            fp_pairs_checked: get_u64(bytes, 105),
            spark_samples_total: get_u64(bytes, 113),
            column_rail_samples_total: get_u64(bytes, 121),
            spark_samples_per_size: get_u32(bytes, 129),
            column_rail_samples: get_u32(bytes, 133),
        })
    }

    /// Re-runs the battery with the recorded config; `Ok(true)` iff the
    /// record reproduces bit-for-bit. `Err` means the matrix now fails a
    /// gate.
    pub fn verify(&self, w: &CodecW) -> Result<bool, CertificationError> {
        let cfg = CertConfig {
            spark_samples_per_size: self.spark_samples_per_size,
            column_rail_samples: self.column_rail_samples,
        };
        Ok(certify(w, cfg)?.canonical_bytes() == self.canonical_bytes())
    }
}

fn put_u32(out: &mut [u8], at: usize, v: u32) {
    out[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_u64(out: &mut [u8], at: usize, v: u64) {
    out[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

fn get_u32(src: &[u8], at: usize) -> u32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&src[at..at + 4]);
    u32::from_le_bytes(b)
}

fn get_u64(src: &[u8], at: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&src[at..at + 8]);
    u64::from_le_bytes(b)
}

// ---------------------------------------------------------------------------
// Integer statistics (exact)
// ---------------------------------------------------------------------------

fn l2sq_i16(vals: impl Iterator<Item = i16>) -> u64 {
    vals.map(|x| (i64::from(x) * i64::from(x)) as u64).sum()
}

fn l1_i16(vals: impl Iterator<Item = i16>) -> u64 {
    vals.map(|x| u64::from(x.unsigned_abs())).sum()
}

/// Robust fleet statistics: (median, MAD, max |v − median|). Exact integers
/// throughout. A population-σ band was rejected deliberately: coordinated
/// outliers dilute their own σ (a single outlier self-limits at √N·σ),
/// while the median and MAD are immune to bounded contamination (P8).
fn fleet_stats(values: &[u64]) -> (u64, u64, u64) {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let med = sorted[(sorted.len() - 1) / 2];
    let mut devs: Vec<u64> = values.iter().map(|&v| v.abs_diff(med)).collect();
    devs.sort_unstable();
    let mad = devs[(devs.len() - 1) / 2];
    let max_dev = devs[devs.len() - 1];
    (med, mad, max_dev)
}

// ---------------------------------------------------------------------------
// 𝔽_p machinery (Goldilocks, the prover field — WP §5.3). Self-contained on
// purpose: certification is a once-per-epoch ceremony and this arithmetic
// is the auditable native reference for the rank checks; the
// performance-critical Goldilocks reduction lives with the proof system.
// ---------------------------------------------------------------------------

const P: u64 = nerv_core::params::PROOFS_FIELD_MODULUS;
const P128: u128 = P as u128;

type FpCol = [u64; EMBEDDING_DIM];

fn fp_add(a: u64, b: u64) -> u64 {
    ((a as u128 + b as u128) % P128) as u64
}

fn fp_sub(a: u64, b: u64) -> u64 {
    fp_add(a, P - b)
}

fn fp_mul(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % P128) as u64
}

fn fp_pow(mut base: u64, mut exp: u64) -> u64 {
    let mut acc = 1u64;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = fp_mul(acc, base);
        }
        base = fp_mul(base, base);
        exp >>= 1;
    }
    acc
}

fn fp_inv(a: u64) -> u64 {
    fp_pow(a, P - 2)
}

/// Canonical embedding ℤ → 𝔽_p of a 1.15 weight.
fn fp_of_weight(w: i16) -> u64 {
    if w >= 0 {
        // In the non-negative branch, `w as u64` is the unique canonical
        // embedding (the i16 value is non-negative, so sign-extension
        // does not occur). No `From<i16> for u64` exists in std, so we
        // route through the explicit cast.
        w as u64
    } else {
        P - i64::from(w).unsigned_abs()
    }
}

/// Pairwise dependence over 𝔽_p: `b = λ·a` for some λ. Zero columns count
/// as dependent. No inversions: `b = λ·a` ⟺ `b[j]·a[t] = b[t]·a[j]` at any
/// nonzero position t of a.
fn fp_pair_dependent(a: &FpCol, b: &FpCol) -> bool {
    let ta = match a.iter().position(|&x| x != 0) {
        None => return true,
        Some(t) => t,
    };
    if b.iter().all(|&x| x == 0) {
        return true;
    }
    let scale = b[ta];
    b.iter()
        .zip(a.iter())
        .all(|(&bv, &av)| fp_mul(bv, a[ta]) == fp_mul(scale, av))
}

/// Incremental row-echelon basis over 𝔽_p^64: `insert` reports whether the
/// vector raised the rank; inserted vectors are normalized to 1 at their
/// pivot, so reduction against the basis needs no inversions.
struct FpBasis {
    vecs: Vec<FpCol>,
    pivots: Vec<usize>,
}

impl FpBasis {
    fn new() -> Self {
        FpBasis { vecs: Vec::new(), pivots: Vec::new() }
    }

    fn len(&self) -> usize {
        self.vecs.len()
    }

    fn insert(&mut self, mut v: FpCol) -> bool {
        for (i, &p) in self.pivots.iter().enumerate() {
            if v[p] != 0 {
                let c = v[p];
                for (vj, bj) in v.iter_mut().zip(self.vecs[i].iter()) {
                    *vj = fp_sub(*vj, fp_mul(c, *bj));
                }
            }
        }
        match v.iter().position(|&x| x != 0) {
            None => false,
            Some(p) => {
                let inv = fp_inv(v[p]);
                for x in v.iter_mut() {
                    *x = fp_mul(*x, inv);
                }
                v[p] = 1;
                self.vecs.push(v);
                self.pivots.push(p);
                true
            }
        }
    }
}

fn subset_rank(cols: &[FpCol], subset: &[u16]) -> usize {
    let mut basis = FpBasis::new();
    let mut rank = 0;
    for &c in subset {
        if basis.insert(cols[usize::from(c)]) {
            rank += 1;
        }
    }
    rank
}

// ---------------------------------------------------------------------------
// Reproducible sampling: streams XOF-derived from the matrix's commitment.
// ---------------------------------------------------------------------------

const FAMILY_SPARK: u32 = 0;
const FAMILY_RAIL: u32 = 1;

fn sample_stream(commitment: &WCommitment, family: u32, k: usize) -> Xof {
    debug_assert!(k < u32::MAX as usize);
    let fam = family.to_le_bytes();
    let kk = (k as u32).to_le_bytes();
    let parts: [&[u8]; 3] = [&commitment.0, &fam, &kk];
    Xof::framed(&W_SPARK, &parts)
}

fn sample_subset(xof: &mut Xof, n: usize, k: usize) -> Vec<u16> {
    debug_assert!(k <= n);
    let mut out: Vec<u16> = Vec::with_capacity(k);
    while out.len() < k {
        let idx = (xof.next_u64() % n as u64) as u16;
        if !out.contains(&idx) {
            out.push(idx);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn quick() -> CertConfig {
        CertConfig { spark_samples_per_size: 2, column_rail_samples: 1 }
    }

    fn tampered(w: &CodecW, f: impl FnOnce(&mut [[i16; FEATURE_COUNT]; EMBEDDING_DIM])) -> CodecW {
        let mut rows = *w.weights();
        f(&mut rows);
        CodecW::new(w.version(), rows)
    }

    fn certified_fixture() -> CodecW {
        expand(&BeaconRandomness::from_bytes([0x42; 32]), WeightVersion(1))
    }

    #[test]
    fn params_modulus_is_goldilocks() {
        assert_eq!(P, 18_446_744_069_414_584_321);
        assert_eq!(u128::from(P), (1u128 << 64) - (1u128 << 32) + 1);
    }

    #[test]
    fn fp_of_weight_exact_over_all_i16() {
        for w in i16::MIN..=i16::MIN + 512 {
            let expected = if w >= 0 {
                u64::from(w)
            } else {
                (P as i128 + i128::from(w)) as u64
            };
            assert_eq!(fp_of_weight(w), expected, "w = {w}");
            assert!(fp_of_weight(w) < P);
        }
        assert_eq!(fp_of_weight(i16::MAX), 32_767);
        assert_eq!(fp_of_weight(i16::MIN), P - 32_768);
        assert_eq!(fp_add(fp_of_weight(-1), 1), 0);
    }

    #[test]
    fn fp_pair_dependent_cases() {
        let mut a = [0u64; EMBEDDING_DIM];
        a[0] = 1;
        let mut b = [0u64; EMBEDDING_DIM];
        b[0] = 2;
        assert!(fp_pair_dependent(&a, &b));
        let mut c = [0u64; EMBEDDING_DIM];
        c[1] = 1;
        assert!(!fp_pair_dependent(&a, &c));
        assert!(!fp_pair_dependent(&c, &a));
        let zero = [0u64; EMBEDDING_DIM];
        assert!(fp_pair_dependent(&zero, &a));
        assert!(fp_pair_dependent(&a, &zero));
        let mut d = [0u64; EMBEDDING_DIM];
        d[0] = 5;
        d[1] = 5;
        let mut e = [0u64; EMBEDDING_DIM];
        e[0] = 15;
        e[1] = 15;
        assert!(fp_pair_dependent(&d, &e));
        let mut f = [0u64; EMBEDDING_DIM];
        f[0] = 15;
        f[1] = 16;
        assert!(!fp_pair_dependent(&d, &f));
        assert!(fp_pair_dependent(&a, &a));
    }

    #[test]
    fn basis_membership_and_rank() {
        fn unit(k: usize) -> FpCol {
            let mut v = [0u64; EMBEDDING_DIM];
            v[k] = 1;
            v
        }
        let mut basis = FpBasis::new();
        for k in 0..6 {
            assert!(basis.insert(unit(k)));
        }
        assert_eq!(basis.len(), 6);
        assert!(!basis.insert(unit(0)));
        assert!(!basis.insert(unit(5)));
        let mut sum01 = unit(0);
        sum01[1] = 1;
        assert!(!basis.insert(sum01));
        let mut sum06 = unit(0);
        sum06[6] = 1;
        assert!(basis.insert(sum06));
        assert_eq!(basis.len(), 7);

        let mut s = 0xDEAD_BEEFu64;
        let mut b2 = FpBasis::new();
        for _ in 0..EMBEDDING_DIM {
            let mut v = [0u64; EMBEDDING_DIM];
            for x in v.iter_mut() {
                *x = splitmix64(&mut s) % P;
            }
            b2.insert(v);
        }
        assert_eq!(b2.len(), EMBEDDING_DIM);
    }

    #[test]
    fn expand_deterministic_and_sensitive() {
        let b1 = BeaconRandomness::from_bytes([0x11; 32]);
        let b2 = BeaconRandomness::from_bytes([0x22; 32]);
        let w1 = expand(&b1, WeightVersion(1));
        let w2 = expand(&b1, WeightVersion(1));
        assert_eq!(w1, w2);
        assert_ne!(expand(&b1, WeightVersion(2)).commitment(), w1.commitment());
        assert_ne!(expand(&b2, WeightVersion(1)).commitment(), w1.commitment());
    }

    #[test]
    fn expand_matches_raw_blake3_framed_xof() {
        // Differential pin: the framed XOF construction must equal the
        // manual blake3 call with u32-LE length framing per part.
        let beacon = BeaconRandomness::from_bytes([0x5A; 32]);
        let version = WeightVersion(9);
        let w = expand(&beacon, version);
        let mut h = blake3::Hasher::new();
        let dom = W_GEN.as_bytes();
        h.update(&(dom.len() as u32).to_le_bytes());
        h.update(dom);
        h.update(&32u32.to_le_bytes());
        h.update(beacon.as_bytes());
        h.update(&8u32.to_le_bytes());
        h.update(&version.0.to_le_bytes());
        let mut reader = h.finalize_xof();
        for row in w.weights().iter() {
            for &wv in row.iter() {
                let mut b = [0u8; 2];
                reader.fill(&mut b);
                assert_eq!(wv, i16::from_le_bytes(b));
            }
        }
    }

    #[test]
    fn verify_expansion_pins_provenance() {
        let b = BeaconRandomness::from_bytes([0x42; 32]);
        let w = expand(&b, WeightVersion(1));
        assert!(verify_expansion(&b, WeightVersion(1), &w));
        assert!(!verify_expansion(&b, WeightVersion(2), &w));
        assert!(!verify_expansion(&BeaconRandomness::from_bytes([0x43; 32]), WeightVersion(1), &w));
        let bad = tampered(&w, |rows| {
            rows[0][0] = rows[0][0].wrapping_add(1);
        });
        assert!(!verify_expansion(&b, WeightVersion(1), &bad));
    }

    #[test]
    fn certify_passes_reports_reproduces() {
        let w = certified_fixture();
        let cfg = CertConfig { spark_samples_per_size: 4, column_rail_samples: 2 };
        let cert = certify(&w, cfg).unwrap();
        assert_eq!(cert.version, WeightVersion(1));
        assert_eq!(cert.w_commitment, w.commitment());
        assert_eq!(cert.fp_rank, EMBEDDING_DIM as u8);
        assert_eq!(cert.fp_pairs_checked, ((FEATURE_COUNT * (FEATURE_COUNT - 1)) / 2) as u64);
        assert!(cert.row_fleet_margin > 0);
        assert!(cert.col_fleet_margin > 0);
        assert!(cert.row_l2sq_min >= ROW_L2SQ_MIN);
        assert!(cert.row_l2sq_max <= ROW_L2SQ_MAX);
        assert!(cert.col_l2sq_min >= COL_L2SQ_MIN);
        assert!(cert.col_l2sq_max <= COL_L2SQ_MAX);
        let sizes = (SPARK_REQUIREMENT - 2) as u64;
        assert_eq!(cert.spark_samples_total, u64::from(cfg.spark_samples_per_size) * sizes);
        assert_eq!(
            cert.column_rail_samples_total,
            u64::from(cfg.column_rail_samples) * (FEATURE_COUNT - ACCOUNT_SLOT_COUNT) as u64
        );
        assert_eq!(cert.canonical_bytes(), certify(&w, cfg).unwrap().canonical_bytes());
        assert_eq!(cert.verify(&w), Ok(true));
    }

    #[test]
    fn certify_rejects_zero_row() {
        let bad = tampered(&certified_fixture(), |rows| {
            rows[3] = [0; FEATURE_COUNT];
        });
        assert!(matches!(
            certify(&bad, quick()),
            Err(CertificationError::RowNorm { row: 3, .. })
        ));
    }

    #[test]
    fn certify_rejects_zero_column() {
        let bad = tampered(&certified_fixture(), |rows| {
            for row in rows.iter_mut() {
                row[7] = 0;
            }
        });
        assert!(matches!(
            certify(&bad, quick()),
            Err(CertificationError::ColNorm { col: 7, .. })
        ));
    }

    #[test]
    fn certify_rejects_duplicate_rows() {
        let bad = tampered(&certified_fixture(), |rows| {
            rows[5] = rows[0];
        });
        assert!(matches!(
            certify(&bad, quick()),
            Err(CertificationError::DuplicateRows { a: 0, b: 5 })
        ));
    }

    #[test]
    fn certify_rejects_dominating_row() {
        // All-extreme weights: ~41 robust σ off the fleet.
        let bad = tampered(&certified_fixture(), |rows| {
            rows[3] = [i16::MIN; FEATURE_COUNT];
        });
        assert!(matches!(
            certify(&bad, quick()),
            Err(CertificationError::RowFleet { row: 3, .. })
        ));
    }

    #[test]
    fn certify_rejects_duplicate_columns() {
        let bad = tampered(&certified_fixture(), |rows| {
            for row in rows.iter_mut() {
                row[5] = row[4];
            }
        });
        assert!(matches!(
            certify(&bad, quick()),
            Err(CertificationError::FpPairDependent { a: 4, b: 5 })
        ));
    }

     #[test]
    fn certify_rejects_negated_column_pair() {
        // col1 = −col0 exactly: an 𝔽_p pair dependence (λ = −1) with ℓ1 and
        // ℓ2² identical to col0's — invisible to every statistical gate,
        // which is exactly the class the exhaustive pair check exists for.
        // (A |λ| ≠ 1 scaling cannot reach the pair gate: its ℓ1 mismatch
        // trips the robust fleet band first — verified by the fleet tests.)
        // wrapping_neg: an i16::MIN entry stays MIN, making the columns
        // identical — still pair-dependent, and overflow-free.
        let bad = tampered(&certified_fixture(), |rows| {
            for row in rows.iter_mut() {
                row[1] = row[0].wrapping_neg();
            }
        });
        assert!(matches!(
            certify(&bad, quick()),
            Err(CertificationError::FpPairDependent { a: 0, b: 1 })
        ));
    }


    #[test]
    fn certify_rejects_rail_built_from_slots() {
        // The volume rail becomes the exact sum of six Walsh-orthogonal
        // slot columns — a 7-column dependence, i.e. spark ≤ 7, precisely
        // the class the threshold exists to exclude. The distributional
        // gates reject it (robust fleet band: ~18 MAD off).
        let bad = tampered(&certified_fixture(), |rows| {
            for (r, row) in rows.iter_mut().enumerate() {
                let mut sum: i16 = 0;
                for j in 0..6 {
                    let v: i16 = if (r >> j) & 1 == 0 { 2_000 } else { -2_000 };
                    row[j] = v;
                    sum += v;
                }
                row[ACCOUNT_SLOT_COUNT] = sum;
            }
        });
        assert!(matches!(
            certify(&bad, quick()),
            Err(CertificationError::ColFleet { .. })
        ));
    }

    #[test]
    fn certificate_canonical_roundtrip() {
        let cert = certify(&certified_fixture(), quick()).unwrap();
        let bytes = cert.canonical_bytes();
        assert_eq!(bytes.len(), CERT_SIZE);
        assert_eq!(Certificate::from_canonical_bytes(&bytes).unwrap(), cert);
        assert!(matches!(
            Certificate::from_canonical_bytes(&bytes[..CERT_SIZE - 1]),
            Err(CertificationError::BadCertificateLength { .. })
        ));
    }

    #[test]
    fn configs_are_ordered() {
        let d = CertConfig::default();
        let c = CertConfig::ceremony();
        assert!(c.spark_samples_per_size > d.spark_samples_per_size);
        assert!(c.column_rail_samples > d.column_rail_samples);
        assert!(d.spark_samples_per_size > 0);
        assert!(d.column_rail_samples > 0);
    }

    proptest! {
        #[test]
        fn prop_fp_field_axioms(a in any::<u64>(), b in any::<u64>()) {
            let a = a % P;
            let b = b % P;
            prop_assert_eq!(fp_add(a, b), fp_add(b, a));
            prop_assert_eq!(fp_mul(a, b), fp_mul(b, a));
            prop_assert_eq!(fp_sub(a, a), 0);
            let c = fp_add(a, b);
            prop_assert_eq!(fp_mul(a, c), fp_add(fp_mul(a, b), fp_mul(a, a)));
            if a != 0 {
                prop_assert_eq!(fp_pow(a, P - 1), 1);
                prop_assert_eq!(fp_mul(fp_inv(a), a), 1);
            }
            prop_assert_eq!(fp_pow(a, 0), 1);
        }

        #[test]
        fn prop_certificate_roundtrip(
            version in any::<u64>(),
            rmin in any::<u64>(), rmax in any::<u64>(),
            cmin in any::<u64>(), cmax in any::<u64>(),
            rowm in any::<u128>(), colm in any::<u128>(),
            rank in any::<u8>(), pairs in any::<u64>(),
            sst in any::<u64>(), crst in any::<u64>(),
            sps in any::<u32>(), crs in any::<u32>(),
        ) {
            let mut cm = [0u8; 32];
            cm[..8].copy_from_slice(&version.to_le_bytes());
            let cert = Certificate {
                version: WeightVersion(version),
                w_commitment: WCommitment(cm),
                row_l2sq_min: rmin,
                row_l2sq_max: rmax,
                col_l2sq_min: cmin,
                col_l2sq_max: cmax,
                row_fleet_margin: rowm as i128,
                col_fleet_margin: colm as i128,
                fp_rank: rank,
                fp_pairs_checked: pairs,
                spark_samples_total: sst,
                column_rail_samples_total: crst,
                spark_samples_per_size: sps,
                column_rail_samples: crs,
            };
            prop_assert_eq!(
                Certificate::from_canonical_bytes(&cert.canonical_bytes()).unwrap(),
                cert
            );
        }
    }
}
