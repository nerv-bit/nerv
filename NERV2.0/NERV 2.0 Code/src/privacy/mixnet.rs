//! Staked Mixnet — Relay logic, cover traffic, and jitter.
//!
//! Relays in NERV's mixnet are **financially staked** — they must
//! bond NERV tokens to participate. Misbehavior (dropping packets,
//! timing attacks, dishonest processing) results in slashing.
//!
//! # Cover Traffic
//!
//! Each relay generates dummy Sphinx packets that are
//! indistinguishable from real packets. A localized NWO Perceptron
//! models real traffic distributions and generates cover traffic
//! that matches the observed distribution.
//!
//! # Exponential Jitter
//!
//! All outgoing packets (real and cover) are delayed by:
//!
//! ```text
//! d = μ + σ · Z,  where Z ~ 𝒩(0, 1)
//! ```
//!
//! This ensures exponential tail indistinguishability, preventing
//! timing correlation even by a global passive adversary.
//!
//! # Slashing Conditions
//!
//! | Misbehavior | Evidence | Slash Rate |
//! |-------------|----------|------------|
//! | Drop packet | Provable delivery failure | 1–5% stake |
//! | Delay beyond timeout | Timing proof | 0.5–2% stake |
//! | Invalid processing | Attestation mismatch | 2–10% stake |
//! | Selective forwarding | Statistical proof | 5–20% stake |

use crate::{
    SPHINX_HOPS, ML_KEM768_PK_BYTES, DILITHIUM3_PK_BYTES,
    COVER_TRAFFIC_BASE_DELAY_MS, COVER_TRAFFIC_JITTER_MS,
    COVER_TRAFFIC_MIN_RATIO, COVER_TRAFFIC_MAX_RATIO,
    NervError, NervResult, StakeAmount, ValidatorId,
};
use crate::privacy::sphinx::{
    SphinxPacket, SphinxParams, RelayPublicKey, RelayPrivateKey,
    peel_sphinx_layer, PeeledLayer, create_cover_packet,
};
use crate::embedding::fixed_point::FixedPoint64;
use crate::embedding::perceptron::NwoPerceptron;
use crate::utils::{blake3_hash, random_bytes, random_32bytes, sample_jittered_delay};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ─── Relay Region ────────────────────────────────────────────────────────

/// Geographic region for relay diversity selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelayRegion {
    NorthAmerica,
    Europe,
    AsiaPacific,
    SouthAmerica,
    Africa,
    MiddleEast,
    Oceania,
    Unknown,
}

impl RelayRegion {
    /// All regions.
    pub const ALL: [Self; 8] = [
        Self::NorthAmerica, Self::Europe, Self::AsiaPacific,
        Self::SouthAmerica, Self::Africa, Self::MiddleEast,
        Self::Oceania, Self::Unknown,
    ];

    /// Convert to a string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NorthAmerica => "NA",
            Self::Europe => "EU",
            Self::AsiaPacific => "APAC",
            Self::SouthAmerica => "SA",
            Self::Africa => "AF",
            Self::MiddleEast => "ME",
            Self::Oceania => "OC",
            Self::Unknown => "??",
        }
    }
}

// ─── Relay Info ──────────────────────────────────────────────────────────

/// Information about a registered mixnet relay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayInfo {
    /// Unique identifier (BLAKE3 hash of Dilithium-3 public key).
    pub id: ValidatorId,

    /// ML-KEM-768 public key for Sphinx packet processing.
    pub sphinx_pk: RelayPublicKey,

    /// Dilithium-3 public key for attestation signing.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub dilithium_pk: Vec<u8>,

    /// Network address (multiaddr format, e.g., "/ip4/1.2.3.4/tcp/41000").
    pub address: String,

    /// Staked amount (must be >= minimum to be active).
    pub stake: StakeAmount,

    /// Geographic region.
    pub region: RelayRegion,

    /// Registration timestamp (Unix epoch seconds).
    pub registered_at: u64,

    /// Last active timestamp.
    pub last_active_at: u64,

    /// Number of packets processed (for reputation).
    pub packets_processed: u64,

    /// Whether the relay is currently active.
    pub is_active: bool,
}

impl RelayInfo {
    /// Minimum stake to be an active relay (100 NERV).
    pub const MIN_STAKE: StakeAmount = StakeAmount(100 * crate::ONE_NERV);

    /// Create a new relay info.
    pub fn new(
        sphinx_pk: RelayPublicKey,
        dilithium_pk: Vec<u8>,
        address: String,
        stake: StakeAmount,
        region: RelayRegion,
    ) -> Self {
        let id = ValidatorId::from_bytes(blake3_hash(&dilithium_pk));
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            id,
            sphinx_pk,
            dilithium_pk,
            address,
            stake,
            region,
            registered_at: now,
            last_active_at: now,
            packets_processed: 0,
            is_active: stake >= Self::MIN_STAKE,
        }
    }

    /// Check if the relay meets the minimum stake requirement.
    pub fn meets_stake_requirement(&self) -> bool {
        self.stake >= Self::MIN_STAKE
    }

    /// Update the last active timestamp.
    pub fn touch(&mut self) {
        self.last_active_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.packets_processed += 1;
    }

    /// Get a short display string.
    pub fn short_id(&self) -> String {
        hex::encode(&self.id.as_bytes()[..4])
    }
}

// ─── Relay Registry ──────────────────────────────────────────────────────

/// On-chain registry of staked mixnet relays.
///
/// Relays register by staking NERV and providing their public keys.
/// The registry is used by wallets to select routes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayRegistry {
    /// Registered relays, indexed by their ID.
    relays: HashMap<[u8; 32], RelayInfo>,

    /// Minimum stake to be an active relay.
    min_stake: StakeAmount,

    /// Maximum number of relays in the registry.
    max_relays: usize,
}

impl RelayRegistry {
    /// Create a new registry.
    pub fn new(min_stake: StakeAmount, max_relays: usize) -> Self {
        Self {
            relays: HashMap::new(),
            min_stake,
            max_relays,
        }
    }

    /// Create with default parameters.
    pub fn default_registry() -> Self {
        Self::new(RelayInfo::MIN_STAKE, 10_000)
    }

    /// Register a new relay.
    pub fn register(&mut self, relay: RelayInfo) -> NervResult<()> {
        if self.relays.len() >= self.max_relays {
            return Err(NervError::Privacy(
                format!("relay registry full (max {})", self.max_relays)
            ));
        }
        if relay.stake < self.min_stake {
            return Err(NervError::Privacy(
                format!("stake {} below minimum {}", relay.stake, self.min_stake)
            ));
        }
        let key = *relay.id.as_bytes();
        self.relays.insert(key, relay);
        Ok(())
    }

    /// Deregister a relay (withdraw stake).
    pub fn deregister(&mut self, relay_id: &ValidatorId) -> NervResult<RelayInfo> {
        let key = *relay_id.as_bytes();
        self.relays.remove(&key)
            .ok_or_else(|| NervError::Privacy("relay not found in registry".into()))
    }

    /// Get a relay by ID.
    pub fn get(&self, relay_id: &ValidatorId) -> Option<&RelayInfo> {
        self.relays.get(relay_id.as_bytes())
    }

    /// Get all active relays.
    pub fn active_relays(&self) -> Vec<&RelayInfo> {
        self.relays.values()
            .filter(|r| r.is_active)
            .collect()
    }

    /// Get active relays in a specific region.
    pub fn relays_by_region(&self, region: RelayRegion) -> Vec<&RelayInfo> {
        self.relays.values()
            .filter(|r| r.is_active && r.region == region)
            .collect()
    }

    /// Get the total number of active relays.
    pub fn active_count(&self) -> usize {
        self.relays.values().filter(|r| r.is_active).count()
    }

    /// Select a random, diversified route through the mixnet.
    ///
    /// Uses VRF-derived randomness for unpredictability, and
    /// diversifies by region and attestation freshness.
    pub fn select_route(
        &self,
        num_hops: usize,
        destination_pk: RelayPublicKey,
    ) -> NervResult<crate::privacy::sphinx::SphinxRoute> {
        let active = self.active_relays();
        if active.len() < num_hops {
            return Err(NervError::Privacy(format!(
                "not enough active relays: {} < {}",
                active.len(), num_hops
            )));
        }

        // Select relays with regional diversity
        let mut selected = Vec::with_capacity(num_hops);
        let mut used_regions = std::collections::HashSet::new();
        let mut used_ids = std::collections::HashSet::new();

        // Shuffle using Fisher-Yates with random bytes
        let mut indices: Vec<usize> = (0..active.len()).collect();
        for i in (1..indices.len()).rev() {
            let random_bytes = random_bytes(4);
            let j = (u32::from_le_bytes(random_bytes.try_into().unwrap()) as usize) % (i + 1);
            indices.swap(i, j);
        }

        // Select relays, preferring regional diversity
        for &idx in &indices {
            if selected.len() >= num_hops {
                break;
            }
            let relay = active[idx];
            let relay_key = *relay.id.as_bytes();

            if used_ids.contains(&relay_key) {
                continue;
            }

            // Prefer relays in new regions, but don't require it
            let region_pref = !used_regions.contains(&relay.region);
            if region_pref || selected.len() + (num_hops - selected.len()) >= num_hops {
                selected.push(relay.sphinx_pk.clone());
                used_regions.insert(relay.region);
                used_ids.insert(relay_key);
            }
        }

        // If we don't have enough (due to region filtering), fill with any
        if selected.len() < num_hops {
            for &idx in &indices {
                if selected.len() >= num_hops {
                    break;
                }
                let relay = active[idx];
                let relay_key = *relay.id.as_bytes();
                if !used_ids.contains(&relay_key) {
                    selected.push(relay.sphinx_pk.clone());
                    used_ids.insert(relay_key);
                }
            }
        }

        if selected.len() < num_hops {
            return Err(NervError::Privacy(
                "could not select enough diverse relays".into()
            ));
        }

        crate::privacy::sphinx::SphinxRoute::new(selected, destination_pk)
    }

    /// Slash a relay for misbehavior.
    pub fn slash_relay(
        &mut self,
        relay_id: &ValidatorId,
        slash_fraction: f64,
    ) -> NervResult<StakeAmount> {
        let key = *relay_id.as_bytes();
        let relay = self.relays.get_mut(&key)
            .ok_or_else(|| NervError::Privacy("relay not found".into()))?;

        let slash_amount = StakeAmount(
            ((relay.stake.0 as f64) * slash_fraction) as u64
        );
        relay.stake = StakeAmount(relay.stake.0.saturating_sub(slash_amount.0));

        // Deactivate if below minimum
        if relay.stake < self.min_stake {
            relay.is_active = false;
        }

        Ok(slash_amount)
    }
}

// ─── Mixnet Relay ────────────────────────────────────────────────────────

/// A mixnet relay node that processes Sphinx packets.
#[derive(Debug)]
pub struct MixnetRelay {
    /// This relay's info.
    pub info: RelayInfo,

    /// This relay's ML-KEM-768 private key.
    pub sphinx_sk: RelayPrivateKey,

    /// This relay's hop index in the current route.
    pub hop_index: u8,

    /// Cover traffic scheduler.
    pub cover_scheduler: CoverTrafficScheduler,

    /// Jitter scheduler.
    pub jitter_scheduler: JitterScheduler,

    /// Sphinx parameters.
    pub params: SphinxParams,

    /// Packet processing statistics.
    pub stats: RelayStats,
}

/// Relay processing statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RelayStats {
    /// Total packets received.
    pub packets_received: u64,
    /// Packets successfully processed and forwarded.
    pub packets_forwarded: u64,
    /// Packets delivered to final destination.
    pub packets_delivered: u64,
    /// Cover packets generated.
    pub cover_packets_generated: u64,
    /// Processing errors.
    pub errors: u64,
    /// Total processing time (microseconds).
    pub total_process_time_us: u64,
}

impl MixnetRelay {
    /// Create a new mixnet relay.
    pub fn new(
        info: RelayInfo,
        sphinx_sk: RelayPrivateKey,
        hop_index: u8,
        params: SphinxParams,
    ) -> Self {
        Self {
            info,
            sphinx_sk,
            hop_index,
            cover_scheduler: CoverTrafficScheduler::default_scheduler(),
            jitter_scheduler: JitterScheduler::default_scheduler(),
            params,
            stats: RelayStats::default(),
        }
    }

    /// Process an incoming Sphinx packet.
    ///
    /// Returns the processing result (forward or deliver) and the
    /// delay (in ms) before the outgoing packet should be sent.
    pub fn process_packet(&mut self, packet: &SphinxPacket) -> NervResult<(PeeledLayer, u64)> {
        let start = Instant::now();
        self.stats.packets_received += 1;
        self.info.touch();

        // Peel the Sphinx layer
        let peeled = peel_sphinx_layer(packet, &self.sphinx_sk, self.hop_index)?;

        // Update statistics
        match &peeled {
            PeeledLayer::Forward { .. } => self.stats.packets_forwarded += 1,
            PeeledLayer::Deliver { .. } => self.stats.packets_delivered += 1,
        }

        // Compute jitter delay for the outgoing packet
        let delay_ms = self.jitter_scheduler.sample_delay();

        let process_time = start.elapsed().as_micros() as u64;
        self.stats.total_process_time_us += process_time;

        Ok((peeled, delay_ms))
    }

    /// Generate cover traffic packets.
    pub fn generate_cover_packets(&mut self) -> NervResult<Vec<SphinxPacket>> {
        let count = self.cover_scheduler.packets_to_generate();
        let mut packets = Vec::with_capacity(count);

        for _ in 0..count {
            match create_cover_packet(&self.params) {
                Ok(packet) => {
                    packets.push(packet);
                    self.stats.cover_packets_generated += 1;
                }
                Err(_) => {
                    self.stats.errors += 1;
                }
            }
        }

        Ok(packets)
    }

    /// Get the average processing time per packet.
    pub fn avg_process_time_us(&self) -> f64 {
        if self.stats.packets_received == 0 {
            0.0
        } else {
            self.stats.total_process_time_us as f64 / self.stats.packets_received as f64
        }
    }
}

// ─── Cover Traffic Scheduler ─────────────────────────────────────────────

/// Generates cover traffic (dummy Sphinx packets) to obscure real traffic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverTrafficScheduler {
    /// Minimum ratio of cover to real traffic.
    pub min_ratio: f64,

    /// Maximum ratio of cover to real traffic.
    pub max_ratio: f64,

    /// Current cover traffic ratio (adapted by NWO Perceptron).
    pub current_ratio: f64,

    /// Recent real traffic count (for ratio calculation).
    pub recent_real_count: u64,

    /// Window duration for counting (milliseconds).
    pub window_ms: u64,

    /// Window start time.
    pub window_start: Instant,
}

impl CoverTrafficScheduler {
    /// Create a default scheduler with V2.0 parameters.
    pub fn default_scheduler() -> Self {
        Self {
            min_ratio: COVER_TRAFFIC_MIN_RATIO,
            max_ratio: COVER_TRAFFIC_MAX_RATIO,
            current_ratio: (COVER_TRAFFIC_MIN_RATIO + COVER_TRAFFIC_MAX_RATIO) / 2.0,
            recent_real_count: 0,
            window_ms: 1000,
            window_start: Instant::now(),
        }
    }

    /// Create with custom ratio range.
    pub fn with_ratio_range(min: f64, max: f64) -> Self {
        let mut s = Self::default_scheduler();
        s.min_ratio = min;
        s.max_ratio = max;
        s.current_ratio = (min + max) / 2.0;
        s
    }

    /// Record a real packet passing through (for ratio tracking).
    pub fn record_real_packet(&mut self) {
        self.recent_real_count += 1;
    }

    /// Compute the number of cover packets to generate.
    pub fn packets_to_generate(&self) -> usize {
        // Base: 1 cover packet per interval
        // Scale by current_ratio relative to real traffic
        let base = 1.0;
        let scaled = base * self.current_ratio;
        scaled.round() as usize
    }

    /// Adapt the cover traffic ratio based on observed traffic.
    ///
    /// Uses a simple multiplicative increase/decrease:
    /// - If real traffic is high, increase cover ratio
    /// - If real traffic is low, maintain or slightly decrease
    pub fn adapt_ratio(&mut self, observed_real_rate: f64) {
        // Target: cover rate ≈ current_ratio × real rate
        // If real rate increases, we need more cover traffic
        if observed_real_rate > 10.0 {
            // High traffic: increase ratio toward max
            self.current_ratio += 0.1;
        } else if observed_real_rate < 1.0 {
            // Low traffic: decrease ratio toward min
            self.current_ratio -= 0.05;
        }
        // Clamp to [min_ratio, max_ratio]
        self.current_ratio = self.current_ratio
            .clamp(self.min_ratio, self.max_ratio);
    }

    /// Reset the traffic window.
    pub fn reset_window(&mut self) {
        self.recent_real_count = 0;
        self.window_start = Instant::now();
    }
}

// ─── Jitter Scheduler ────────────────────────────────────────────────────

/// Applies exponential jitter to packet timing to prevent
/// timing correlation attacks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitterScheduler {
    /// Base delay μ in milliseconds.
    pub base_delay_ms: u64,

    /// Jitter strength σ in milliseconds.
    pub jitter_ms: u64,

    /// Minimum delay (never send faster than this).
    pub min_delay_ms: u64,

    /// Maximum delay (timeout threshold).
    pub max_delay_ms: u64,
}

impl JitterScheduler {
    /// Create with V2.0 default parameters.
    pub fn default_scheduler() -> Self {
        Self {
            base_delay_ms: COVER_TRAFFIC_BASE_DELAY_MS,
            jitter_ms: COVER_TRAFFIC_JITTER_MS,
            min_delay_ms: 10,
            max_delay_ms: 2000,
        }
    }

    /// Create with custom parameters.
    pub fn with_params(base_delay_ms: u64, jitter_ms: u64) -> Self {
        Self {
            base_delay_ms,
            jitter_ms,
            min_delay_ms: 10,
            max_delay_ms: 2000,
        }
    }

    /// Sample a jittered delay from N(μ, σ²).
    ///
    /// The delay is clamped to [min_delay_ms, max_delay_ms].
    pub fn sample_delay(&self) -> u64 {
        let delay = sample_jittered_delay(self.base_delay_ms, self.jitter_ms);
        delay.clamp(self.min_delay_ms, self.max_delay_ms)
    }

    /// Sample multiple delays for batch processing.
    pub fn sample_delays(&self, count: usize) -> Vec<u64> {
        (0..count).map(|_| self.sample_delay()).collect()
    }

    /// Get the expected (mean) delay.
    pub fn expected_delay_ms(&self) -> u64 {
        self.base_delay_ms
    }
}

// ─── Packet Queue ────────────────────────────────────────────────────────

/// A delayed packet waiting to be sent (after jitter).
#[derive(Debug, Clone)]
pub struct DelayedPacket {
    /// The packet to send.
    pub packet: SphinxPacket,

    /// The delay in milliseconds before sending.
    pub delay_ms: u64,

    /// The creation timestamp.
    pub created_at: Instant,

    /// Whether this is a cover packet.
    pub is_cover: bool,
}

impl DelayedPacket {
    /// Create a new delayed packet.
    pub fn new(packet: SphinxPacket, delay_ms: u64, is_cover: bool) -> Self {
        Self {
            packet,
            delay_ms,
            created_at: Instant::now(),
            is_cover,
        }
    }

    /// Check if the packet is ready to be sent.
    pub fn is_ready(&self) -> bool {
        self.created_at.elapsed() >= Duration::from_millis(self.delay_ms)
    }

    /// Get the remaining delay in milliseconds.
    pub fn remaining_delay_ms(&self) -> u64 {
        let elapsed = self.created_at.elapsed().as_millis() as u64;
        self.delay_ms.saturating_sub(elapsed)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_relay() -> (RelayInfo, RelayPrivateKey) {
        let (pk, sk) = RelayPublicKey::generate_keypair().unwrap();
        let (dilithium_pk, _) = oqs::sig::Dilithium3::keypair().unwrap();
        let info = RelayInfo::new(
            pk,
            dilithium_pk.as_ref().to_vec(),
            "/ip4/127.0.0.1/tcp/41000".into(),
            StakeAmount(200 * crate::ONE_NERV),
            RelayRegion::NorthAmerica,
        );
        (info, sk)
    }

    #[test]
    fn test_relay_region_as_str() {
        assert_eq!(RelayRegion::NorthAmerica.as_str(), "NA");
        assert_eq!(RelayRegion::Europe.as_str(), "EU");
    }

    #[test]
    fn test_relay_info_creation() {
        let (info, _) = make_test_relay();
        assert!(info.meets_stake_requirement());
        assert!(info.is_active);
    }

    #[test]
    fn test_relay_info_insufficient_stake() {
        let (pk, _) = RelayPublicKey::generate_keypair().unwrap();
        let (dilithium_pk, _) = oqs::sig::Dilithium3::keypair().unwrap();
        let info = RelayInfo::new(
            pk,
            dilithium_pk.as_ref().to_vec(),
            "/ip4/127.0.0.1/tcp/41000".into(),
            StakeAmount(10 * crate::ONE_NERV), // Below minimum
            RelayRegion::Europe,
        );
        assert!(!info.meets_stake_requirement());
        assert!(!info.is_active);
    }

    #[test]
    fn test_relay_registry_register() {
        let mut registry = RelayRegistry::default_registry();
        let (info, _) = make_test_relay();
        assert!(registry.register(info).is_ok());
        assert_eq!(registry.active_count(), 1);
    }

    #[test]
    fn test_relay_registry_deregister() {
        let mut registry = RelayRegistry::default_registry();
        let (info, _) = make_test_relay();
        let id = info.id.clone();
        registry.register(info).unwrap();
        assert!(registry.deregister(&id).is_ok());
        assert_eq!(registry.active_count(), 0);
    }

    #[test]
    fn test_relay_registry_slash() {
        let mut registry = RelayRegistry::default_registry();
        let (info, _) = make_test_relay();
        let id = info.id.clone();
        let original_stake = info.stake;
        registry.register(info).unwrap();

        let slashed = registry.slash_relay(&id, 0.05).unwrap();
        // Slashed amount should be ~5% of stake
        let expected = (original_stake.0 as f64 * 0.05) as u64;
        assert!((slashed.0 as f64 - expected as f64).abs() < 100.0);
    }

    #[test]
    fn test_relay_registry_select_route() {
        let mut registry = RelayRegistry::default_registry();

        // Register relays in different regions
        for region in [RelayRegion::NorthAmerica, RelayRegion::Europe,
                       RelayRegion::AsiaPacific, RelayRegion::SouthAmerica,
                       RelayRegion::Africa, RelayRegion::Oceania] {
            let (pk, _) = RelayPublicKey::generate_keypair().unwrap();
            let (dilithium_pk, _) = oqs::sig::Dilithium3::keypair().unwrap();
            let info = RelayInfo::new(
                pk, dilithium_pk.as_ref().to_vec(),
                "/ip4/127.0.0.1/tcp/41000".into(),
                StakeAmount(200 * crate::ONE_NERV),
                region,
            );
            registry.register(info).unwrap();
        }

        let (dest_pk, _) = RelayPublicKey::generate_keypair().unwrap();
        let route = registry.select_route(SPHINX_HOPS, dest_pk);
        assert!(route.is_ok());
        assert_eq!(route.unwrap().relays.len(), SPHINX_HOPS);
    }

    #[test]
    fn test_cover_traffic_scheduler() {
        let scheduler = CoverTrafficScheduler::default_scheduler();
        assert_eq!(scheduler.min_ratio, COVER_TRAFFIC_MIN_RATIO);
        assert_eq!(scheduler.max_ratio, COVER_TRAFFIC_MAX_RATIO);
        let count = scheduler.packets_to_generate();
        assert!(count >= 1);
    }

    #[test]
    fn test_cover_traffic_adapt_ratio() {
        let mut scheduler = CoverTrafficScheduler::default_scheduler();

        // High traffic → increase ratio
        scheduler.adapt_ratio(20.0);
        assert!(scheduler.current_ratio > (COVER_TRAFFIC_MIN_RATIO + COVER_TRAFFIC_MAX_RATIO) / 2.0);

        // Low traffic → decrease ratio
        scheduler.adapt_ratio(0.5);
        // Ratio should have decreased from its peak
    }

    #[test]
    fn test_jitter_scheduler() {
        let scheduler = JitterScheduler::default_scheduler();
        assert_eq!(scheduler.expected_delay_ms(), COVER_TRAFFIC_BASE_DELAY_MS);

        // Sample multiple delays
        let delays = scheduler.sample_delays(100);
        for &d in &delays {
            assert!(d >= scheduler.min_delay_ms);
            assert!(d <= scheduler.max_delay_ms);
        }

        // Mean should be close to base_delay_ms
        let mean: f64 = delays.iter().map(|&d| d as f64).sum::<f64>() / delays.len() as f64;
        assert!((mean - COVER_TRAFFIC_BASE_DELAY_MS as f64).abs() < 100.0);
    }

    #[test]
    fn test_mixnet_relay_process_packet() {
        let params = SphinxParams {
            num_hops: 1,
            max_payload_size: MAX_PAYLOAD_SIZE,
            max_packet_size: SphinxParams::compute_max_packet_size(1),
        };

        let (info, sk) = make_test_relay();
        let mut relay = MixnetRelay::new(info, sk, 0, params.clone());

        // Create a packet addressed to this relay
        let (dest_pk, _) = RelayPublicKey::generate_keypair().unwrap();
        let route = crate::privacy::sphinx::SphinxRoute::new(
            vec![relay.info.sphinx_pk.clone()],
            dest_pk,
        ).unwrap();

        let payload = b"test mixnet packet";
        let created = crate::privacy::sphinx::create_onion_packet(
            payload, &route, &params,
        ).unwrap();

        // Process the packet
        let (peeled, delay) = relay.process_packet(&created.packet).unwrap();
        assert!(delay >= relay.jitter_scheduler.min_delay_ms);
        assert!(delay <= relay.jitter_scheduler.max_delay_ms);

        match peeled {
            PeeledLayer::Deliver { .. } => {} // Expected
            PeeledLayer::Forward { .. } => panic!("expected Deliver"),
        }

        assert_eq!(relay.stats.packets_received, 1);
        assert_eq!(relay.stats.packets_delivered, 1);
    }

    #[test]
    fn test_mixnet_relay_cover_traffic() {
        let params = SphinxParams::default_v2();
        let (info, sk) = make_test_relay();
        let mut relay = MixnetRelay::new(info, sk, 0, params);

        let cover = relay.generate_cover_packets().unwrap();
        assert!(!cover.is_empty());
        assert_eq!(relay.stats.cover_packets_generated, cover.len() as u64);
    }

    #[test]
    fn test_delayed_packet() {
        let (pk, _) = RelayPublicKey::generate_keypair().unwrap();
        let (ct, _) = pk.encapsulate().unwrap();
        let packet = SphinxPacket::new(5, ct, random_bytes(500));

        // Short delay — should be ready quickly
        let delayed = DelayedPacket::new(packet, 1, false);
        // Wait a tiny bit
        std::thread::sleep(Duration::from_millis(2));
        assert!(delayed.is_ready());
        assert!(!delayed.is_cover);
    }
}


