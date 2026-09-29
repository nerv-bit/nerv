//! The typed, validated, hashed protocol specification.
//!
//! `specs/params.toml` is the single source of truth; this module is its
//! enforcement: strict deserialization (unknown fields are errors), a P5
//! walk rejecting floats/datetimes anywhere in the tree, ~70 cross-field
//! invariants (including primality and NTT-order checks on the seal modulus
//! via deterministic Miller–Rabin, and derived BFT tolerances that exposed
//! erratum E-009), and the BLAKE3 spec-hash over the canonical encoding.

use std::fs;
use std::path::Path;

use blake3::Hasher;

use crate::encoding::Canonical;
use crate::error::SpecError;
use crate::primality::{is_prime_u64, two_adicity};

/// A fully-loaded, validated, hashed specification.
#[derive(Debug, Clone)]
pub struct LoadedSpec {
    /// The parsed raw tree (source of the canonical hash).
    pub raw: toml::Value,
    /// The strictly typed view (serde; deny_unknown_fields).
    pub spec: Spec,
    /// BLAKE3 spec-hash over the canonical encoding of `raw`.
    pub hash: [u8; 32],
}

impl LoadedSpec {
    pub fn load(path: &Path) -> Result<Self, SpecError> {
        let text = fs::read_to_string(path).map_err(|source| SpecError::Io { path: path.into(), source })?;
        Self::from_str(&text)
    }

    pub fn from_str(text: &str) -> Result<Self, SpecError> {
        let raw: toml::Value = toml::from_str(text).map_err(SpecError::Parse)?;
        let spec: Spec = toml::from_str(text).map_err(SpecError::Deserialize)?;

        // P5: the spec tree is integer-exact. Reject floats and datetimes
        // anywhere, with paths, before anything is hashed or trusted.
        let mut forbidden = Vec::new();
        walk_forbidden(&raw, "$", &mut forbidden);
        if !forbidden.is_empty() {
            return Err(SpecError::Forbidden { paths: forbidden });
        }

        spec.validate()?;

          let hash = spec_hash(&raw, &spec.meta.spec_hash_domain)?;
        Ok(LoadedSpec { raw, spec, hash })
    }
}

/// BLAKE3 over the canonical encoding of the whole validated tree,
/// domain-separated via blake3's derive_key mode. Infallible on a tree that
/// passed the P5 walk.
pub fn spec_hash(raw: &toml::Value, domain: &str) -> Result<[u8; 32], SpecError> {
    let canon = Canonical::from_toml(raw)
        .map_err(|_| SpecError::Internal("canonical conversion failed on a pre-validated tree"))?;
    let mut h = Hasher::new_derive_key(domain);
    h.update(&canon.encode());
    Ok(*h.finalize().as_bytes())
}

fn walk_forbidden(v: &toml::Value, path: &str, out: &mut Vec<String>) {
    match v {
        toml::Value::Float(_) => out.push(format!("{path} (float — P5)")),
        toml::Value::Datetime(_) => out.push(format!("{path} (datetime)")),
        toml::Value::Array(a) => {
            for (i, x) in a.iter().enumerate() {
                walk_forbidden(x, &format!("{path}[{i}]"), out);
            }
        }
        toml::Value::Table(t) => {
            for (k, x) in t.iter() {
                walk_forbidden(x, &format!("{path}.{k}"), out);
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Typed structures — mirror specs/params.toml exactly. deny_unknown_fields
// everywhere: a parameter that is not consumed by code is an error, not a
// silent drift.
// ---------------------------------------------------------------------------

macro_rules! strict {
    ($t:ty) => {
        impl $t {}
    };
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetaSpec {
    pub spec_name: String,
    pub spec_version: String,
    pub hash_algorithm: String,
    pub spec_hash_domain: String,
    pub source_documents: Vec<String>,
    pub freeze_milestone: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtocolSpec {
    pub supply_nerv: u64,
    pub nano_per_nerv: u64,
    pub shard_count_genesis: u64,
    pub shard_id_bits_genesis: u64,
    pub shard_count_max: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimingSpec {
    pub beacon_interval_secs: u64,
    pub shard_block_target_secs: u64,
    pub shard_block_max_secs: u64,
    pub epoch_secs: u64,
    pub governance_epoch_days: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsensusSpec {
    pub signature_scheme: String,
    pub emergency_signature_scheme: String,
    pub sortition: String,
    pub shard_committee_size: u64,
    pub shard_quorum: u64,
    pub beacon_committee_size: u64,
    pub beacon_quorum: u64,
    pub registry_committee_size: u64,
    pub registry_quorum: u64,
    pub attestation_signers: u64,
    pub attestation_quorum: u64,
    pub staggered_thirds: bool,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustodySpec {
    pub nct_depth: u64,
    pub nct_node_hash: String,
    pub nullifier_tree_depth: u64,
    pub nullifier_tree_hash: String,
    pub transit_log_hash: String,
    pub anchor_window_headers: u64,
    pub value_min_nano: u64,
    pub value_max_nano: u64,
    pub note_nonce_bits: u64,
    pub blinding_bits: u64,
    pub memo_max_bytes: u64,
    pub note_kem: String,
    pub note_aead: String,
    pub commitment_hash: String,
    pub nullifier_hash: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProofsSpec {
    pub system: String,
    pub field: String,
    pub field_modulus: u64,
    pub fri_security_bits: u64,
    pub fiat_shamir_hash: String,
    pub fiat_shamir_model: String,
    pub lookup_blake3_window_bits: u64,
    pub range_window_bits: u64,
    pub digit_decomposition_bits: u64,
    pub fs_binding_challenge: String,
    pub poseidon2_t: u64,
    pub poseidon2_full_rounds: u64,
    pub poseidon2_partial_rounds: u64,
    pub poseidon2_sbox_degree: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlaySpec {
    pub w_rows: u64,
    pub w_cols: u64,
    pub w_bits: u64,
    pub w_frac_bits: u64,
    pub embedding_dims: u64,
    pub embedding_bits_per_dim: u64,
    pub slot_count: u64,
    pub max_active_slots: u64,
    pub spark_bound: u64,
    pub volume_rail_index: u64,
    pub fee_rail_index: u64,
    pub count_rail_index: u64,
    pub log_rail_index: u64,
    pub type_time_rail_start: u64,
    pub forecaster_ar_window: u64,
    pub challenger_window_blocks: u64,
    pub challenger_min_coverage_permille: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealSpec {
    pub ring_degree: u64,
    pub q: u64,
    pub q_bits: u64,
    pub module_rank: u64,
    pub plaintext_rings: u64,
    pub digit_slots: u64,
    pub digits_used: u64,
    pub digit_bits: u64,
    pub digits_per_coordinate: u64,
    pub scale_log2: u64,
    pub noise_eta: u64,
    pub noise_in_circuit_bound: u64,
    pub noise_budget_bits: u64,
    pub committee_n: u64,
    pub threshold_t: u64,
    pub chunk_min: u64,
    pub chunk_max: u64,
    pub security_classical_bits: u64,
    pub security_quantum_bits: u64,
    pub pss: bool,
    pub pss_handoff_intervals: u64,
    pub committee_padding: bool,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MixnetSpec {
    pub path_relays: u64,
    pub packet_bytes: u64,
    pub jitter_distribution: String,
    pub jitter_mean_ms: u64,
    pub jitter_cap_ms: u64,
    pub fragmentation_max_packets: u64,
    pub fragmentation_classes: Vec<u64>,
    pub submission_fanout: u64,
    pub onion_kem: String,
    pub onion_aead: String,
    pub relay_selection: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistrySpec {
    pub bundle_min: u64,
    pub bundle_max: u64,
    pub dedup_rule: String,
    pub challenge_window_secs: u64,
    pub folding_grace_secs: u64,
    pub degraded_mode: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardingSpec {
    pub engineered_ceiling_legs_per_sec: u64,
    pub split_threshold_legs_per_sec: u64,
    pub split_window_days: u64,
    pub merge_threshold_legs_per_sec: u64,
    pub merge_window_days: u64,
    pub adoption_notice_days: u64,
    pub emergency_overload_multiple: u64,
    pub block_leg_cap: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicFloorSpec {
    pub enabled: bool,
    pub m_max: u64,
    pub decay_num: u64,
    pub decay_den: u64,
    pub base_floor_nano: u64,
    pub trigger_source: String,
    pub hysteresis: bool,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeesSpec {
    pub split_producer_permille: u64,
    pub split_prover_permille: u64,
    pub split_da_permille: u64,
    pub split_relay_permille: u64,
    pub prover_floor_nano: u64,
    pub mempool_ordering: String,
    pub dynamic_floor: DynamicFloorSpec,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BucketSpec {
    pub name: String,
    pub kind: String,
    pub total_nerv: u64,
    pub share_permille: u64,
    pub account: String,
    pub term_days: u64,
    pub cliff_days: Option<u64>,
    pub linear_days: Option<u64>,
    pub year_one_days: Option<u64>,
    pub quarters: Option<u64>,
    pub quarter_days: Option<u64>,
    pub ratio_num: Option<u64>,
    pub ratio_den: Option<u64>,
    pub window_days: Option<u64>,
    pub burn_unclaimed: Option<bool>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EconomySpec {
    pub supply_identity: String,
    pub buckets: Vec<BucketSpec>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DaSpec {
    pub erasure_scheme: String,
    pub b7_withhold_permille: u64,
    pub b7_detection_permille: u64,
    pub b7_window_secs: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WitnessSpec {
    pub leg_tree_depth_max: u64,
    pub light_anchor_max_bytes: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CiBudgets {
    pub whole_air_ceiling: u64,
    pub whole_air_target: u64,
    pub custody_module_ceiling: u64,
    pub delta_module_ceiling: u64,
    pub seal_chip_ceiling: u64,
    pub blake3_chip_ceiling: u64,
    pub wallet_proof_hard_bytes: u64,
    pub wallet_proof_target_bytes: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceBudgets {
    pub tx_proof_min_bytes: u64,
    pub tx_proof_max_bytes: u64,
    pub amortized_proof_min_bytes: u64,
    pub amortized_proof_max_bytes: u64,
    pub shell_bytes: u64,
    pub encrypted_notes_bytes: u64,
    pub seal_genesis_bytes: u64,
    pub seal_compressed_target_bytes: u64,
    pub transient_per_tx_bytes: u64,
    pub permanent_per_tx_bytes: u64,
    pub witness_standard_bytes: u64,
    pub witness_cold_bytes: u64,
    pub attestation_daily_bytes: u64,
    pub attestation_yearly_bytes: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetsSpec {
    pub ci: CiBudgets,
    pub reference: ReferenceBudgets,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConformanceSpec {
    pub determinism_arches: Vec<String>,
    pub vector_families: Vec<String>,
    pub fv_statements: Vec<String>,
    pub fv_gate: String,
    pub delete_test: bool,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Erratum {
    pub id: String,
    pub refs: String,
    pub issue: String,
    pub resolution: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub meta: MetaSpec,
    pub protocol: ProtocolSpec,
    pub timing: TimingSpec,
    pub consensus: ConsensusSpec,
    pub custody: CustodySpec,
    pub proofs: ProofsSpec,
    pub overlay: OverlaySpec,
    pub seal: SealSpec,
    pub mixnet: MixnetSpec,
    pub registry: RegistrySpec,
    pub sharding: ShardingSpec,
    pub fees: FeesSpec,
    pub economy: EconomySpec,
    pub da: DaSpec,
    pub witness: WitnessSpec,
    pub budgets: BudgetsSpec,
    pub conformance: ConformanceSpec,
    pub errata: Vec<Erratum>,
}

strict!(Spec);

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

macro_rules! check {
    ($v:expr, $cond:expr, $($arg:tt)*) => {
        if !$cond {
            $v.push(format!($($arg)*));
        }
    };
}

/// Effective Byzantine tolerance of a (size, quorum) pair:
/// safety  f ≤ 2q − n − 1  (quorum intersection contains ≥ f+1 honest),
/// liveness f ≤ n − q      (honest nodes alone can reach quorum).
pub fn byzantine_tolerance(n: u64, q: u64) -> i64 {
    let safety = (2 * q).saturating_sub(n).saturating_sub(1) as i64;
    let liveness = n.saturating_sub(q) as i64;
    safety.min(liveness)
}

impl Spec {
    /// Run every invariant. Collects ALL violations (diagnostics first).
    pub fn validate(&self) -> Result<(), SpecError> {
        let mut v: Vec<String> = Vec::new();
        let s = self;
   let goldilocks: u128 = (1u128 << 64) - (1u128 << 32) + 1;

        // -- meta -------------------------------------------------------------
        check!(v, !s.meta.spec_name.is_empty(), "meta.spec_name is empty");
        check!(v, s.meta.hash_algorithm == "blake3", "meta.hash_algorithm must be blake3 (the loader hashes with blake3)");
        check!(v, !s.meta.spec_hash_domain.is_empty(), "meta.spec_hash_domain is empty");
        check!(v, !s.meta.freeze_milestone.is_empty(), "meta.freeze_milestone is empty");

        // -- protocol ----------------------------------------------------------
        check!(v, s.protocol.supply_nerv > 0, "protocol.supply_nerv must be positive");
        check!(v, s.protocol.nano_per_nerv == 1_000_000_000, "protocol.nano_per_nerv must be 1e9");
        check!(v, (1u64 << s.protocol.shard_id_bits_genesis) == s.protocol.shard_count_genesis,
            "protocol: 2^shard_id_bits_genesis must equal shard_count_genesis (E-App B)");
        check!(v, s.protocol.shard_count_max >= s.protocol.shard_count_genesis,
            "protocol.shard_count_max < genesis count");

        // -- timing ------------------------------------------------------------
        check!(v, s.timing.beacon_interval_secs >= 1, "timing.beacon_interval_secs must be >= 1");
        check!(v, s.timing.epoch_secs % s.timing.beacon_interval_secs == 0,
            "timing: epoch must be a whole number of beacon intervals");
        check!(v, s.timing.epoch_secs == 86_400, "timing.epoch_secs must be 24h (App B)");
        check!(v, s.timing.shard_block_max_secs >= s.timing.shard_block_target_secs,
            "timing: block max < target");
        check!(v, s.timing.governance_epoch_days >= 1, "timing.governance_epoch_days must be >= 1");

        // -- consensus (derived tolerances; E-009) ------------------------------
        for (label, n, q) in [
            ("shard", s.consensus.shard_committee_size, s.consensus.shard_quorum),
            ("beacon", s.consensus.beacon_committee_size, s.consensus.beacon_quorum),
            ("registry", s.consensus.registry_committee_size, s.consensus.registry_quorum),
            ("attestation", s.consensus.attestation_signers, s.consensus.attestation_quorum),
        ] {
            check!(v, q >= 1 && q <= n, "consensus: {label} quorum {q} out of [1, {n}]");
            check!(v, byzantine_tolerance(n, q) >= 1,
                "consensus: {label} ({n}, {q}) has non-positive Byzantine tolerance — quorum algebra broken (E-009)");
            check!(v, q % 2 == 1, "consensus: {label} quorum {q} should be odd (2f+1 form)");
        }
        check!(v, s.consensus.attestation_signers <= s.consensus.beacon_committee_size,
            "consensus: attestation signer subset larger than the beacon committee (E-001)");
        check!(v, s.consensus.signature_scheme == "ml-dsa-65", "consensus.signature_scheme must be ml-dsa-65 (App B)");
        check!(v, s.consensus.emergency_signature_scheme == "slh-dsa", "consensus: SLH-DSA is the specified emergency fallback (§9.4)");
        check!(v, s.consensus.sortition == "beacon-seeded-hash-ranking", "consensus.sortition must match DSR-5 (E-007)");

        // -- custody ------------------------------------------------------------
        check!(v, s.custody.nct_depth == 32, "custody.nct_depth must be 32 (§3.3)");
        check!(v, s.custody.nullifier_tree_depth == 256, "custody.nullifier_tree_depth must be 256 (§3.4)");
        check!(v, s.custody.anchor_window_headers == 64, "custody.anchor_window_headers must be 64 (§4.3)");
        check!(v, s.custody.value_min_nano == 1, "custody.value_min_nano must be 1 (§3.2: 0 < v)");
        check!(v, s.custody.value_max_nano == 1u64 << 60, "custody.value_max_nano must be 2^60 (§3.2)");
        check!(v, s.custody.note_nonce_bits == 256 && s.custody.blinding_bits == 256,
            "custody: nonce and blinding must be 256-bit (§3.2/§3.3)");
        check!(v, s.custody.memo_max_bytes == 80, "custody.memo_max_bytes must be 80 (§3.2)");
        check!(v, s.custody.nct_node_hash == "poseidon2", "custody.nct_node_hash per E-002");
        check!(v, s.custody.nullifier_tree_hash == "blake3", "custody.nullifier_tree_hash per E-002");

        // -- proofs --------------------------------------------------------------
        check!(v, s.proofs.field_modulus == goldilocks as u64,
            "proofs.field_modulus must be the Goldilocks prime 2^64 - 2^32 + 1 (§5.3)");
        check!(v, s.proofs.fri_security_bits >= 100, "proofs.fri_security_bits: D.2 FRI floor is 100 bits");
        check!(v, s.proofs.fiat_shamir_hash == "blake3" && s.proofs.fiat_shamir_model == "qrom",
            "proofs: Fiat–Shamir is BLAKE3 in the QROM — the named assumption (§5.7)");
        check!(v, s.proofs.lookup_blake3_window_bits == 16, "proofs.lookup_blake3_window_bits must be 16 (D.2)");
        check!(v, s.proofs.range_window_bits == 60, "proofs.range_window_bits must be 60 (value ranges)");
        check!(v, s.proofs.digit_decomposition_bits == 10, "proofs.digit_decomposition_bits must be 10 (seal digits)");
           check!(v, s.proofs.poseidon2_t == 16, "proofs: Poseidon2 state width is t = 16 (§5.3)");
        check!(v, s.proofs.poseidon2_full_rounds >= 2 && s.proofs.poseidon2_full_rounds % 2 == 0,
            "proofs: Poseidon2 full rounds must be even (split half at each end)");
        check!(v, s.proofs.poseidon2_partial_rounds >= 1, "proofs: Poseidon2 partial rounds must be positive");
        check!(v, s.proofs.poseidon2_sbox_degree == 7,
            "proofs: Poseidon2 S-box is x^7 — bijective on Goldilocks (gcd(7, p−1) = 1, CI-verified)");


        // -- budgets (D.2 normative) ----------------------------------------------
        let b = &s.budgets.ci;
        check!(v, b.whole_air_target <= b.whole_air_ceiling, "budgets.ci: whole-AIR target exceeds ceiling");
        check!(v, b.wallet_proof_target_bytes <= b.wallet_proof_hard_bytes, "budgets.ci: wallet proof target exceeds hard ceiling");
        for (m, mv) in [
            ("custody", b.custody_module_ceiling),
            ("delta", b.delta_module_ceiling),
            ("seal_chip", b.seal_chip_ceiling),
            ("blake3", b.blake3_chip_ceiling),
        ] {
            check!(v, mv <= b.whole_air_ceiling,
                "budgets.ci: {m} module ceiling exceeds the whole-AIR ceiling (module maxima are independent; the whole-AIR ceiling binds their sum — D.2)");
        }

        // -- overlay ------------------------------------------------------------
        let o = &s.overlay;
        check!(v, o.w_rows == 64 && o.w_cols == 256, "overlay: W is 64×256 (§7.2)");
        check!(v, o.w_bits == 16 && o.w_frac_bits == 15, "overlay: W is 1.15 signed fixed-point (§7.2)");
        check!(v, o.embedding_dims == o.w_rows, "overlay: embedding dims must equal W rows");
        check!(v, o.embedding_bits_per_dim == 64, "overlay: 64 bits per dimension ((Z/2^64)^64)");
        check!(v, o.slot_count == 224, "overlay.slot_count must be 224 (§7.2)");
        check!(v, o.type_time_rail_start == o.slot_count + 4,
            "overlay: rails 224–227 then type/time from 228 (§7.2 layout)");
        check!(v, o.w_cols == o.slot_count + 32,
            "overlay: 224 slots + 4 named rails + 28 type/time rails = 256 columns (§7.2)");
        check!(v, o.max_active_slots == 6, "overlay.max_active_slots must be 6 (§7.4)");
        check!(v, o.spark_bound >= 2 * o.max_active_slots + 1,
            "overlay: spark bound must exceed 2·k_max (differences of ≤6-slot vectors involve ≤12 columns; spark ≥ 13 — §7.4)");
        check!(v, o.forecaster_ar_window == 1024, "overlay.forecaster_ar_window must be 1024 (§10.2)");
        check!(v, o.challenger_window_blocks == 2016, "overlay.challenger_window_blocks must be 2016 (§10.4)");
        check!(v, o.challenger_min_coverage_permille <= 1000, "overlay: challenger coverage permille > 1000");

        // -- seal ------------------------------------------------------------------
        let q = s.seal.q;
        check!(v, is_prime_u64(q), "seal.q = {q} is NOT prime — deterministic Miller–Rabin rejects it (§6.3.1)");
        check!(v, q < (1u64 << 32), "seal.q must be a 32-bit modulus (§6.3.8)");
        check!(v, s.seal.q_bits == 32, "seal.q_bits must be 32");
        check!(v, two_adicity(q) >= 9, "seal: 512 must divide q-1 for the degree-256 NTT (q ≡ 1 mod 512 — §6.3.1)");
        check!(v, s.seal.ring_degree == 256, "seal.ring_degree must be 256");
        check!(v, s.seal.plaintext_rings * s.seal.ring_degree == s.seal.digit_slots,
            "seal: 2 plaintext rings × 256 = 512 digit slots (§6.3.1)");
        check!(v, s.seal.digits_used == s.overlay.embedding_dims * s.seal.digits_per_coordinate,
            "seal: digits_used must equal 64 coordinates × 7 digits = 448 (§6.3.1)");
        check!(v, s.seal.digits_used <= s.seal.digit_slots, "seal: digits_used exceeds slots");
        check!(v, s.seal.digit_bits == 10, "seal.digit_bits must be 10 (§6.3.8)");
        // D.2/D.6 noise headroom (§6.3.8): worst-case digit sums + noise margin below q/2.
        let scale = 1u64 << s.seal.scale_log2;
        let digit_max = (1u64 << s.seal.digit_bits) - 1;
        let noise_bound = 1u64 << s.seal.noise_budget_bits;
        let headroom = scale
            .checked_mul(s.seal.chunk_max)
            .and_then(|x| x.checked_mul(digit_max))
            .and_then(|x| x.checked_add(noise_bound));
        check!(v, matches!(headroom, Some(h) if h < (q - 1) / 2),
            "seal: noise headroom violated — scale·chunk_max·(2^10-1) + 2^noise_budget must stay below (q-1)/2 (§6.3.8)");
        check!(v, s.seal.noise_eta == 2, "seal.noise_eta must be 2 (binomial, §6.3.8)");
        check!(v, s.seal.noise_in_circuit_bound == 3, "seal: 3σ in-circuit bound; CB(η=2) has σ=1 (§6.3.7)");
        check!(v, s.seal.threshold_t <= s.seal.committee_n, "seal: threshold exceeds committee (E-003)");
        check!(v, s.seal.chunk_min <= s.seal.chunk_max, "seal: chunk bounds inverted");
        check!(v, s.seal.chunk_min == 128, "seal.chunk_min must be 128 (B_min, §6.4)");
        check!(v, s.seal.chunk_max == 512, "seal.chunk_max must be 512 (noise ceiling)");
        check!(v, s.seal.security_classical_bits >= 128 && s.seal.security_quantum_bits >= 96,
            "seal: advisory-tier security is ≥128-bit classical / ≥96-bit quantum (§6.3.8)");
        check!(v, s.seal.pss, "seal: D.1 requires proactive secret sharing at epoch boundaries");
        check!(v, s.seal.pss_handoff_intervals == 2, "seal.pss_handoff_intervals = 2 (D.1/D.6)");

        // -- mixnet ------------------------------------------------------------------
        let m = &s.mixnet;
        check!(v, m.path_relays == 5, "mixnet.path_relays must be 5 (§6.2)");
        check!(v, m.packet_bytes == 20_000, "mixnet.packet_bytes must be 20000 (uniform, §6.2)");
        check!(v, m.jitter_cap_ms > m.jitter_mean_ms, "mixnet: jitter cap must exceed the mean (E-004)");
        check!(v, m.fragmentation_classes.windows(2).all(|w| w[0] < w[1]),
            "mixnet: fragmentation size classes must be strictly ascending (D.2)");
        check!(v, m.fragmentation_classes.iter().all(|c| c.is_power_of_two()),
            "mixnet: fragmentation size classes must be powers of two (D.2)");
        check!(v, *m.fragmentation_classes.last().unwrap_or(&0) == m.fragmentation_max_packets,
            "mixnet: the largest size class must equal K (D.2)");
        check!(v, *m.fragmentation_classes.first().unwrap_or(&0) == 1, "mixnet: the smallest size class must be 1");
        check!(v, m.fragmentation_max_packets == 16, "mixnet.fragmentation_max_packets = 16 (D.2/D.6)");
        check!(v, m.submission_fanout >= 2, "mixnet: submission fan-out must provide censorship insurance (§5.5)");

        // -- registry -------------------------------------------------------------------
        check!(v, s.registry.bundle_min <= s.registry.bundle_max, "registry: bundle bounds inverted");
        check!(v, s.registry.bundle_min == 512 && s.registry.bundle_max == 4096,
            "registry: bundles are 512–4,096 inner proofs (§5.5)");
        check!(v, s.registry.challenge_window_secs == 30, "registry.challenge_window_secs = 30 (§5.5)");
        check!(v, s.registry.dedup_rule == "txid-first-wins", "registry.dedup_rule must be txid-first-wins");
        check!(v, s.registry.folding_grace_secs >= 1, "registry: folding grace must be positive");

        // -- sharding --------------------------------------------------------------------
        let sh = &s.sharding;
        check!(v, sh.merge_threshold_legs_per_sec < sh.split_threshold_legs_per_sec,
            "sharding: merge threshold must sit below the split threshold (hysteresis)");
        check!(v, sh.split_threshold_legs_per_sec < sh.engineered_ceiling_legs_per_sec,
            "sharding: S_hi must sit below the engineered ceiling (E-005)");
        check!(v, sh.merge_window_days > sh.split_window_days,
            "sharding: merge window must be longer than the split window (hysteresis)");
        check!(v, sh.emergency_overload_multiple >= 2, "sharding: emergency overload factor ≥ 2 (§8.5)");
        check!(v, sh.block_leg_cap > (1u64 << 13) && sh.block_leg_cap <= (1u64 << s.witness.leg_tree_depth_max),
            "sharding: block leg cap must fit the witness leg-tree depth (§11.2: ≤ 10,000 legs, depth ≤ 14)");

        // -- fees ---------------------------------------------------------------------------
        let f = &s.fees;
        let split = f.split_producer_permille + f.split_prover_permille + f.split_da_permille + f.split_relay_permille;
        check!(v, split == 1000, "fees: 40/30/20/10 split must sum to 1000 permille, found {split}");
        check!(v, f.mempool_ordering == "fee-then-txid", "fees.mempool_ordering must be fee-then-txid (§4.3/D.4)");
        let dfl = &f.dynamic_floor;
           check!(v, dfl.m_max >= 1, "fees.dynamic_floor: m_max must be at least 1 (D.4)");
        check!(v, f.prover_floor_nano >= 1, "fees.prover_floor_nano must be a positive integer");
        if dfl.enabled {
            check!(v, dfl.decay_num >= 1 && dfl.decay_den > dfl.decay_num,
                "fees.dynamic_floor: decay must be a proper fraction (floor reverts toward 1)");
            check!(v, dfl.base_floor_nano >= 1, "fees.dynamic_floor: base_floor_nano must be positive");
            check!(v, dfl.trigger_source == "finalized-revealed-aggregates",
                "fees.dynamic_floor: trigger must be finalized revealed aggregates (D.4) — never forecaster residuals");
            check!(v, dfl.hysteresis, "fees.dynamic_floor: per-rail hysteresis is required (D.4)");
        }

        // -- economy (§12.2) ------------------------------------------------------
        let e = &s.economy;
        check!(v, e.supply_identity == "emitted-minus-burned-minus-abandoned",
            "economy: supply identity must be emitted-minus-burned-minus-abandoned (M1, §12.3)");
        check!(v, !e.buckets.is_empty(), "economy: at least one bucket");
        let supply_nano = (s.protocol.supply_nerv as u128) * (s.protocol.nano_per_nerv as u128);
        let mut names = std::collections::BTreeSet::new();
        let mut share_sum: u128 = 0;
        let mut total_sum: u128 = 0;
        for bk in &e.buckets {
            check!(v, !bk.name.is_empty() && names.insert(bk.name.clone()),
                "economy: duplicate or empty bucket name");
            check!(v, bk.total_nerv > 0, "economy: bucket `{}` total must be positive", bk.name);
            check!(v, bk.share_permille >= 1 && bk.share_permille <= 1000,
                "economy: bucket `{}` share_permille out of [1, 1000]", bk.name);
            check!(v, (bk.total_nerv as u128) * 1000 == (bk.share_permille as u128) * (s.protocol.supply_nerv as u128),
                "economy: bucket `{}` total must equal share × supply exactly", bk.name);
            check!(v, bk.term_days >= 1 && bk.term_days <= 36_500,
                "economy: bucket `{}` term_days out of [1, 36500]", bk.name);
            check!(v, matches!(bk.account.as_str(), "signed" | "producer-payout" | "commitment-note" | "signed-grant" | "challenger-market"),
                "economy: bucket `{}` unknown account kind `{}`", bk.name, bk.account);
            share_sum += bk.share_permille as u128;
            total_sum += (bk.total_nerv as u128) * (s.protocol.nano_per_nerv as u128);
            match bk.kind.as_str() {
                "linear-vesting" => {
                    let cliff = bk.cliff_days;
                    let linear = bk.linear_days;
                    check!(v, linear.is_some_and(|l| l >= 1),
                        "economy: bucket `{}` linear_days must be ≥ 1", bk.name);
                    check!(v, matches!(cliff.zip(linear), Some((c, l)) if c.checked_add(l) == Some(bk.term_days)),
                        "economy: bucket `{}` requires cliff_days + linear_days == term_days", bk.name);
                }
                "subsidy-decline" => {
                    check!(v, bk.year_one_days.is_some_and(|y| y >= 1 && y < bk.term_days),
                        "economy: bucket `{}` requires 1 ≤ year_one_days < term_days", bk.name);
                }
                "claim-window" => {
                    check!(v, bk.window_days.is_some_and(|w| w >= 1 && w <= bk.term_days),
                        "economy: bucket `{}` requires 1 ≤ window_days ≤ term_days", bk.name);
                    check!(v, bk.burn_unclaimed == Some(true),
                        "economy: claim-window `{}` must burn unclaimed amounts at close (§12.2)", bk.name);
                }
                "milestone-window" => {
                    check!(v, bk.window_days.is_some_and(|w| w >= 1 && w <= bk.term_days),
                        "economy: bucket `{}` requires 1 ≤ window_days ≤ term_days", bk.name);
                }
                "quarterly-geometric" => {
                    let q = bk.quarters;
                    let qd = bk.quarter_days;
                    check!(v, q.is_some_and(|q| q >= 1 && q <= 1000),
                        "economy: bucket `{}` quarters out of [1, 1000]", bk.name);
                    check!(v, matches!(q.zip(qd), Some((q, d)) if q.checked_mul(d) == Some(bk.term_days)),
                        "economy: bucket `{}` requires quarters × quarter_days == term_days", bk.name);
                    check!(v, bk.ratio_num.unwrap_or(0) >= 1 && bk.ratio_den.unwrap_or(0) > bk.ratio_num.unwrap_or(0),
                        "economy: bucket `{}` requires 0 < ratio_num < ratio_den", bk.name);
                }
                other => check!(v, false, "economy: bucket `{}` unknown kind `{other}`", bk.name),
            }
        }
        check!(v, share_sum == 1000, "economy: shares must sum to 1000 permille (found {share_sum})");
        check!(v, total_sum == supply_nano, "economy: totals must sum to the fixed supply exactly (§12.1)");

        // -- da (§8.7, B7) ----------------------------------------------------------
        check!(v, s.da.erasure_scheme == "2d-reed-solomon", "da.erasure_scheme must be 2d-reed-solomon (§8.7)");
        check!(v, s.da.b7_detection_permille >= 999, "da: B7 detection probability must be ≥ 0.999 (§13.5)");
        check!(v, s.da.b7_withhold_permille >= 1 && s.da.b7_window_secs >= 1, "da: B7 parameters must be positive");

        // -- witness (§11.2) -----------------------------------------------------------
        check!(v, s.witness.leg_tree_depth_max == 14, "witness: leg tree depth is 14 at the 10,000-leg cap (§11.2)");
        check!(v, s.witness.light_anchor_max_bytes <= 102_400, "witness: light anchor must stay under 100 KB (B6)");

        // -- budgets.reference (§5.4, §8.8, §11) ----------------------------------------
        let r = &s.budgets.reference;
        check!(v, r.tx_proof_min_bytes <= r.tx_proof_max_bytes, "budgets.reference: tx proof bounds inverted");
        check!(v, r.amortized_proof_min_bytes <= r.amortized_proof_max_bytes, "budgets.reference: amortized bounds inverted");
        check!(v, r.seal_compressed_target_bytes <= r.seal_genesis_bytes,
            "budgets.reference: R2 compression target must not exceed the genesis wire size");
        check!(v, r.transient_per_tx_bytes >= r.shell_bytes + r.encrypted_notes_bytes + r.seal_genesis_bytes + r.amortized_proof_max_bytes,
            "budgets.reference: §8.8 transient must carry shell + notes + seal + amortized proof");
        check!(v, r.witness_standard_bytes <= r.witness_cold_bytes, "budgets.reference: cold witness is the larger variant (§11.2)");

        // -- conformance & errata ---------------------------------------------------------
        let c = &s.conformance;
        let mut fams = std::collections::BTreeSet::new();
        for fam in &c.vector_families {
            check!(v, !fam.is_empty() && fams.insert(fam.clone()), "conformance: duplicate or empty vector family");
        }
        check!(v, c.determinism_arches.len() == 3, "conformance: determinism matrix is x86_64/aarch64/riscv64 (DSR-11)");
        check!(v, !c.fv_statements.is_empty(), "conformance: D.5 pins at least one FV statement");
        check!(v, c.fv_gate == "M5", "conformance: FV gate is M5 (D.5/D.6)");
        check!(v, c.delete_test, "conformance: the Axiom-3 delete test must be enabled (§13.4)");
        let mut ids = std::collections::BTreeSet::new();
        for er in &s.errata {
            check!(v, !er.id.is_empty() && ids.insert(er.id.clone()), "errata: duplicate or empty id");
            check!(v, !er.refs.is_empty() && !er.issue.is_empty() && !er.resolution.is_empty(),
                "errata: `{}` must carry refs, issue, and resolution", er.id);
        }

        if v.is_empty() {
            Ok(())
        } else {
            Err(SpecError::Validation { count: v.len(), report: v.join("\n  - ") })
        }
    }

impl Spec {
    /// (label, size, quorum, derived tolerance) for every committee.
    /// Tolerances are computed, never hand-copied (E-009).
    pub fn committee_summary(&self) -> Vec<(&'static str, u64, u64, i64)> {
        let c = &self.consensus;
        vec![
            ("shard", c.shard_committee_size, c.shard_quorum, byzantine_tolerance(c.shard_committee_size, c.shard_quorum)),
            ("beacon", c.beacon_committee_size, c.beacon_quorum, byzantine_tolerance(c.beacon_committee_size, c.beacon_quorum)),
            ("registry", c.registry_committee_size, c.registry_quorum, byzantine_tolerance(c.registry_committee_size, c.registry_quorum)),
            ("attestation", c.attestation_signers, c.attestation_quorum, byzantine_tolerance(c.attestation_signers, c.attestation_quorum)),
        ]
    }
}



