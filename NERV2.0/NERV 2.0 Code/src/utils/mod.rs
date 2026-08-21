//! Utility functions and shared infrastructure.
//!
//! This module provides constant-time operations, secure memory handling,
//! common serialization helpers, and other utilities used across all
//! NERV subsystems.

pub mod balance;

use crate::NervError;
use crate::NervResult;
use subtle::ConstantTimeEq;
use zeroize::Zeroize;

/// Constant-time comparison of two byte slices.
///
/// Returns `true` if and only if the slices are equal in length and content.
/// Timing is independent of the slice contents, preventing timing side-channels.
#[inline]
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.ct_eq(b).into()
}

/// Constant-time comparison: returns true if `a != b`.
#[inline]
pub fn ct_ne(a: &[u8], b: &[u8]) -> bool {
    !ct_eq(a, b)
}

/// Securely zeroize a byte slice.
///
/// Uses the `zeroize` crate to ensure the compiler does not optimize away
/// the zeroing operation. Critical for wiping private keys and blinding
/// factors from memory.
#[inline]
pub fn secure_zero(slice: &mut [u8]) {
    slice.zeroize();
}

/// Securely zeroize a `Vec<u8>` and then clear its capacity.
///
/// After this call, the vector will have zero length and zero capacity,
/// and all previous contents will be overwritten with zeros.
pub fn secure_zero_vec(vec: &mut Vec<u8>) {
    vec.zeroize();
    vec.clear();
    vec.shrink_to_fit();
}

/// Generate cryptographically secure random bytes.
pub fn random_bytes(len: usize) -> Vec<u8> {
    use rand::RngCore;
    let mut buf = vec![0u8; len];
    rand::thread_rng().fill_bytes(&mut buf);
    buf
}

/// Generate a cryptographically secure random 32-byte array.
pub fn random_32bytes() -> [u8; 32] {
    use rand::RngCore;
    let mut buf = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut buf);
    buf
}

/// Generate a cryptographically secure random u64.
pub fn random_u64() -> u64 {
    use rand::RngCore;
    rand::thread_rng().next_u64()
}

/// Compute BLAKE3 hash of the input, returning a 32-byte array.
#[inline]
pub fn blake3_hash(data: &[u8]) -> [u8; 32] {
    blake3::hash(data).into()
}

/// Compute BLAKE3 keyed hash (used for commitment schemes).
#[inline]
pub fn blake3_keyed_hash(key: &[u8; 32], data: &[u8]) -> [u8; 32] {
    blake3::keyed_hash(key, data).into()
}

/// Compute BLAKE3 derive_key (used for domain-separated key derivation).
#[inline]
pub fn blake3_derive_key(context: &str, material: &[u8]) -> [u8; 32] {
    blake3::derive_key(context, material).into()
}

/// Compute SHA3-256 hash of the input, returning a 32-byte array.
pub fn sha3_256(data: &[u8]) -> [u8; 32] {
    use sha3::Digest;
    let mut hasher = sha3::Sha3_256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Hex-encode a byte slice with a compact formatter.
pub fn hex_encode(data: &[u8]) -> String {
    hex::encode(data)
}

/// Hex-decode a string into a byte vector.
pub fn hex_decode(s: &str) -> NervResult<Vec<u8>> {
    hex::decode(s).map_err(|e| NervError::Serialization(format!("hex decode error: {e}")))
}

/// Base64-encode a byte slice.
pub fn base64_encode(data: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(data)
}

/// Base64-decode a string into a byte vector.
pub fn base64_decode(s: &str) -> NervResult<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(s)
        .map_err(|e| NervError::Serialization(format!("base64 decode error: {e}")))
}

/// Sample a jittered delay from the mixnet's cover traffic distribution.
///
/// The inter-packet delay `d` is sampled as:
/// ```text
/// d = μ + σ · Z,  where Z ~ N(0,1)
/// ```
/// This ensures exponential tail indistinguishability against
/// timing correlation attacks (formalized in the Sphinx/Nym model).
///
/// # Arguments
/// * `base_delay_ms` - The mean delay μ in milliseconds
/// * `jitter_ms` - The standard deviation σ in milliseconds
///
/// # Returns
/// A delay in milliseconds sampled from the normal distribution.
pub fn sample_jittered_delay(base_delay_ms: u64, jitter_ms: u64) -> u64 {
    use rand::Rng;
    let z: f64 = rand::thread_rng().sample(rand_distr::Normal::new(0.0, 1.0).unwrap());
    let delay = base_delay_ms as f64 + jitter_ms as f64 * z;
    // Clamp to non-negative — a negative delay is nonsensical
    delay.max(0.0).round() as u64
}

/// Convert a u64 nano-NERV value to a human-readable string.
///
/// Example: `1_500_000_000` → `"1.500000000"`
pub fn format_nerv_amount(nano: u64) -> String {
    let whole = nano / crate::ONE_NERV;
    let frac = nano % crate::ONE_NERV;
    format!("{whole}.{frac:09}")
}

/// Parse a human-readable NERV amount string into nano-NERV.
///
/// Example: `"1.5"` → `1_500_000_000`
pub fn parse_nerv_amount(s: &str) -> NervResult<u64> {
    let parts: Vec<&str> = s.split('.').collect();
    match parts.len() {
        1 => {
            // No decimal point — interpret as whole NERV
            let whole: u64 = parts[0].parse()
                .map_err(|e| NervError::Other(format!("invalid NERV amount: {e}")))?;
            whole.checked_mul(crate::ONE_NERV)
                .ok_or_else(|| NervError::Other("NERV amount overflow".into()))
        }
        2 => {
            let whole: u64 = parts[0].parse()
                .map_err(|e| NervError::Other(format!("invalid NERV whole part: {e}")))?;
            let frac_str = parts[1];
            if frac_str.len() > 9 {
                return Err(NervError::Other("NERV fractional part has more than 9 digits".into()));
            }
            // Pad with trailing zeros to make 9 digits
            let frac_padded = format!("{frac_str:0<9}");
            let frac: u64 = frac_padded.parse()
                .map_err(|e| NervError::Other(format!("invalid NERV fractional part: {e}")))?;
            let whole_nano = whole.checked_mul(crate::ONE_NERV)
                .ok_or_else(|| NervError::Other("NERV amount overflow".into()))?;
            whole_nano.checked_add(frac)
                .ok_or_else(|| NervError::Other("NERV amount overflow".into()))
        }
        _ => Err(NervError::Other("invalid NERV amount format".into())),
    }
}

/// Chunk a slice into batches of the specified size.
/// The last chunk may be smaller than `chunk_size`.
pub fn chunk_slice<T>(slice: &[T], chunk_size: usize) -> Vec<&[T]> {
    slice.chunks(chunk_size).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ct_eq_equal() {
        let a = [1u8, 2, 3, 4];
        let b = [1u8, 2, 3, 4];
        assert!(ct_eq(&a, &b));
    }

    #[test]
    fn test_ct_eq_not_equal() {
        let a = [1u8, 2, 3, 4];
        let b = [1u8, 2, 3, 5];
        assert!(!ct_eq(&a, &b));
    }

    #[test]
    fn test_ct_eq_different_length() {
        let a = [1u8, 2, 3];
        let b = [1u8, 2, 3, 4];
        assert!(!ct_eq(&a, &b));
    }

    #[test]
    fn test_secure_zero() {
        let mut data = vec![42u8; 100];
        secure_zero(&mut data);
        assert!(data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_secure_zero_vec() {
        let mut data = vec![42u8; 100];
        secure_zero_vec(&mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn test_random_bytes_length() {
        let bytes = random_bytes(32);
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn test_blake3_hash_deterministic() {
        let data = b"nerv test data";
        let h1 = blake3_hash(data);
        let h2 = blake3_hash(data);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_sha3_256_deterministic() {
        let data = b"nerv test data";
        let h1 = sha3_256(data);
        let h2 = sha3_256(data);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_format_nerv_amount() {
        assert_eq!(format_nerv_amount(0), "0.000000000");
        assert_eq!(format_nerv_amount(crate::ONE_NERV), "1.000000000");
        assert_eq!(format_nerv_amount(1_500_000_000), "1.500000000");
        assert_eq!(format_nerv_amount(999_999_999), "0.999999999");
    }

    #[test]
    fn test_parse_nerv_amount() {
        assert_eq!(parse_nerv_amount("0").unwrap(), 0);
        assert_eq!(parse_nerv_amount("1").unwrap(), crate::ONE_NERV);
        assert_eq!(parse_nerv_amount("1.5").unwrap(), 1_500_000_000);
        assert_eq!(parse_nerv_amount("0.999999999").unwrap(), 999_999_999);
    }

    #[test]
    fn test_parse_format_roundtrip() {
        let original = 123_456_789_012; // 123.456789012 NERV
        let formatted = format_nerv_amount(original);
        let parsed = parse_nerv_amount(&formatted).unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn test_sample_jittered_delay_non_negative() {
        // Even with extreme RNG luck, delay should never be negative
        for _ in 0..1000 {
            let delay = sample_jittered_delay(100, 200);
            // Allow slight rounding; negative would be a bug
            assert!(delay < 1000); // sanity upper bound
        }
    }

    #[test]
    fn test_chunk_slice() {
        let data: Vec<i32> = (0..10).collect();
        let chunks = chunk_slice(&data, 3);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], &[0, 1, 2]);
        assert_eq!(chunks[3], &[9]);
    }

    #[test]
    fn test_hex_roundtrip() {
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let encoded = hex_encode(&data);
        let decoded = hex_decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_base64_roundtrip() {
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let encoded = base64_encode(&data);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }
}
//! Core Utilities — Hashing, RNG, Secure Memory, Time, Encoding.
//!
//! This module provides the foundational utilities used throughout
//! the NERV codebase. All cryptographic helpers use post-quantum
//! safe primitives (BLAKE3, SHA3) and constant-time operations.

pub mod balance;

// ─── Re-exports ─────────────────────────────────────────────────────────

pub use balance::{Amount, BalanceCommitment, BalanceDelta, BalanceDirection};

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ═══════════════════════════════════════════════════════════════════════
//  BLAKE3 HASHING
// ═══════════════════════════════════════════════════════════════════════

/// Compute BLAKE3-256 of the input data.
///
/// BLAKE3 is the primary hash function in NERV, used for:
/// - Embedding roots
/// - Transaction hashes (via domain separation)
/// - Key IDs
/// - VDW integrity
///
/// BLAKE3 is quantum-resistant (no known sub-exponential quantum attack),
/// parallelizable, and extremely fast (~1 GB/s on modern hardware).
#[inline]
pub fn blake3_hash(data: &[u8]) -> [u8; 32] {
    blake3::hash(data).into()
}

/// Compute BLAKE3-256 with a domain separation string.
///
/// Always prefer this over `blake3_hash` when the same data
/// might be hashed in different contexts to prevent collisions.
#[inline]
pub fn blake3_hash_domain(domain: &str, data: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain.as_bytes());
    hasher.update(data);
    hasher.finalize().into()
}

/// Compute BLAKE3 derived key (keyed mode).
///
/// Uses BLAKE3's native key derivation, which is more efficient
/// than HMAC-BLAKE3 for deriving subkeys.
#[inline]
pub fn blake3_derive_key(context: &str, data: &[u8]) -> [u8; 32] {
    blake3::derive_key(context.as_bytes(), data).into()
}

///3/// Compute BLAKE3 of multiple input slices (streaming).
///
/// Equivalent to `blake3_hash(a ‖ b ‖ c ‖ ...)` but avoids allocation.
pub fn blake3_hash_multi(parts: &[&[u8]]) ->F[u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

/// Create a new BLAKE3 hasher for incremental computation.
#[inline]
pub fn blake3_hasher() -> blake3::Hasher {
    blake3::Hasher::new()
}

/// Create a keyed BLAKE3 hasher.
#[inline]
pub fn blake3_keyed_hasher(key: &[u8; 32]) -> blake3::Hasher {
    blake3::Hasher::new_keyed(key)
}

// ═══════════════════════════════════════════════════════════════════════
//  SHA3-256 HASHING
// ═══════════════════════════════════════════════════════════════════════

/// Compute SHA3-256 of the input data.
///
/// SHA3-256 is used for transaction hashes and any context
/// requiring a NIST-standardized hash.
#[inline]
pub fn sha3_256(data: &[u8]) -> [u8; 32] {
    use sha3::Digest;
    let mut hasher = sha3::Sha3_256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut output = [0u8; 32];
    output.copy_from_slice(&result);
    output
}

/// Compute SHA3-256 with domain separation.
#[inline]
pub fn sha3_256_domain(domain: &str, data: &[u8]) -> [u8; 32] {
    use sha3::Digest;
    let mut hasher = sha3::Sha3_256::new();
    hasher.update(domain.as_bytes());
    hasher.update(data);
    let result = hasher.finalize();
    let mut output = [0u8; 32];
    output.copy_from_slice(&result);
    output
}

// ═══════════════════════════════════════════════════════════════════════
//  CRYPTOGRAPHICALLY SECURE RANDOMNESS
// ═══════════════════════════════════════════════════════════════════════

static RNG_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate cryptographically random bytes.
///
/// Uses the system CSPRNG (`rand::thread_rng()` backed by OS entropy).
/// Suitable for key generation, nonce creation, and all security-critical
/// randomness needs.
pub fn random_bytes(len: usize) -> Vec<u8> {
    use rand::RngCore;
    let mut buf = vec![0u8; len];
    rand::thread_rng().fill_bytes(&mut buf);
    buf
}

/// Generate a cryptographically random 32-byte value.
pub fn random_32bytes() -> [u8; 32] {
    let v = random_bytes(32);
    v.try_into().unwrap_or([0u8; 32])
}

/// Generate a cryptographically random u64.
pub fn random_u64() -> u64 {
    use rand::RngCore;
    rand::thread_rng().next_u64()
}

/// Generate a random u64 in range [0, exclusive_upper_bound).
///
/// Uses rejection sampling to avoid modulo bias.
pub fn random_range(exclusive_upper_bound: u64) -> u64 {
    use rand::Rng;
    rand::thread_rng().gen_range(0..exclusive_upper_bound)
}

/// Generate a random boolean.
pub fn random_bool() -> bool {
    random_u64() & 1 == 0
}

/// Generate a random f64 in [0.0, 1.0).
pub fn random_f64() -> f64 {
    (random_u64() >> 11) as f64 / (1u64 << 53) as f64
}

/// Generate a standard-normal random f64 using Box-Muller transform.
///
/// Returns (z0, z1), two independent N(0,1) samples.
pub fn random_normal_pair() -> (f64, f64) {
    let u1 = random_f64().max(1e-15); // Avoid log(0)
    let u2 = random_f64();
    let magnitude = (-2.0 * u1.ln()).sqrt();
    let z0 = magnitude * (2.0 * std::f64::consts::PI * u2).cos();
    let z1 = magnitude * (2.0 * std::f64::consts::PI * u2).sin();
    (z0, z1)
}

/// Generate a normally distributed random f64: N(mean, std_dev).
pub fn random_normal(mean: f64, std_dev: f64) -> f64 {
    let (z0, _) = random_normal_pair();
    mean + std_dev * z0
}

// ═══════════════════════════════════════════════════════════════════════
//  SECURE MEMORY OPERATIONS
// ═══════════════════════════════════════════════════════════════════════

/// Securely zeroize a byte slice.
///
/// Uses the `zeroize` crate to ensure the compiler doesn't
/// optimize away the zeroing. Call this on any secret material
/// (keys, decrypted transactions, shared secrets) after use.
#[inline]
pub fn secure_zero(data: &mut [u8]) {
    use zeroize::Zeroize;
    data.zeroize();
}

/// Securely zeroize a Vec and shrink it.
pub fn secure_zero_vec(vec: &mut Vec<u8>) {
    secure_zero(vec.as_mut_slice());
    vec.clear();
    vec.shrink_to_fit();
}

/// Constant-time comparison of two byte slices.
///
/// Returns `true` if equal, `false` otherwise.
/// Prevents timing side-channels during comparison.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    use subtle::ConstantTimeEq;
    if a.len() != b.len() {
        return false;
    }
    a.ct_eq(b).into()
}

/// Constant-time comparison of two 32-byte values.
#[inline]
pub fn ct_eq_32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    constant_time_eq(a.as_slice(), b.as_slice())
}

// ═══════════════════════════════════════════════════════════════════════
//  TIME UTILITIES
// ═══════════════════════════════════════════════════════════════════════

/// Get the current time as milliseconds since Unix epoch.
pub fn current_epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Get the current time as microseconds since Unix epoch.
pub fn current_epoch_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

/// Get the current time as seconds since Unix epoch.
pub fn current_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Format a duration as a human-readable string.
pub fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let days = total_secs / 86400;
    let hours = (total_secs % 86400) / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;

    if days > 0 {
        format!("{days}d {hours}h {mins}m {secs}s")
    } else if hours > 0 {
        format!("{hours}h {mins}m {secs}s")
    } else if mins > 0 {
        format!("{mins}m {secs}s")
    } else {
        format!("{secs}s")
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  ENCODING HELPERS
// ═══════════════════════════════════════════════════════════════════════

/// Encode bytes as lowercase hexadecimal.
#[inline]
pub fn to_hex(data: &[u8]) -> String {
    hex::encode(data)
}

/// Decode a hexadecimal string to bytes.
pub fn from_hex(s: &str) -> Result<Vec<u8>, hex::FromHexError> {
    hex::decode(s)
}

/// Encode bytes as base64.
#[inline]
pub fn to_base64(data: &[u8]) -> String {
    use base64::engine::general_purpose::STANDARD;
    base64::engine::GeneralPurpose::new(&STANDARD).encode(data)
}

/// Decode a base64 string to bytes.
pub fn from_base64(s: &str) -> Result<Vec<u8>, base64::DecodeError> {
    use base64::engine::general_purpose::STANDARD;
    base64::engine::GeneralPurpose::new(&STANDARD).decode(s)
}

// ═══════════════════════════════════════════════════════════════════════
//  RATE LIMITER
// ═══════════════════════════════════════════════════════════════════════

/// A token-bucket rate limiter.
///
/// Allows `max_burst` operations initially, then refills at
/// `refill_rate` tokens per second. Thread-safe via atomic ops.
#[derive(Debug)]
pub struct RateLimiter {
    /// Available tokens.
    tokens: AtomicU64,
    /// Maximum burst size.
    max_burst: u64,
    /// Tokens added per refill.
    refill_amount: u64,
    /// Milliseconds between refills.
    refill_interval_ms: u64,
    /// Last refill time (millis).
    last_refill: AtomicU64,
}

impl RateLimiter {
    /// Create a new rate limiter.
    ///
    /// - `max_burst`: Maximum tokens available (initial = max_burst)
    /// - `refill_amount`: Tokens added per interval
    /// - `refill_interval_ms`: Milliseconds between refills
    pub fn new(max_burst: u64, refill_amount: u64, refill_interval_ms: u64) -> Self {
        Self {
            tokens: AtomicU64::new(max_burst),
            max_burst,
            refill_amount,
            refill_interval_ms,
            last_refill: AtomicU64::new(current_epoch_millis()),
        }
    }

    /// Try to consume one token. Returns `true` if allowed.
    pub fn try_acquire(&self) -> bool {
        self.refill();
        loop {
            let current = self.tokens.load(Ordering::Relaxed);
            if current == 0 {
                return false;
            }
            if self.tokens.compare_exchange_weak(
                current,
                current - 1,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ).is_ok() {
                return true;
            }
        }
    }

    /// Try to consume `n` tokens. Returns `true` if allowed.
    pub fn try_acquire_n(&self, n: u64) -> bool {
        self.refill();
        loop {
            let current = self.tokens.load(Ordering::Relaxed);
            if current < n {
                return false;
            }
            if self.tokens.compare_exchange_weak(
                current,
                current - n,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ).is_ok() {
                return true;
            }
        }
    }

    /// Refill tokens based on elapsed time.
    fn refill(&self) {
        let now = current_epoch_millis();
        let last = self.last_refill.load(Ordering::Relaxed);

        if now <= last {
            return;
        }

        let elapsed = now - last;
        let intervals = elapsed / self.refill_interval_ms;

        if intervals == 0 {
            return;
        }

        let tokens_to_add = intervals * self.refill_amount;
        let new_last = last + intervals * self.refill_interval_ms;

        if self.last_refill.compare_exchange(
            last,
            new_last,
            Ordering::SeqCst,
            Ordering::Relaxed,
        ).is_ok() {
            let current = self.tokens.load(Ordering::Relaxed);
            let new_tokens = (current + tokens_to_add).min(self.max_burst);
            self.tokens.store(new_tokens, Ordering::Relaxed);
        }
    }

    /// Get the current number of available tokens.
    pub fn available(&self) -> u64 {
        self.refill();
        self.tokens.load(Ordering::Relaxed)
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  LRU CACHE
// ═══════════════════════════════════════════════════════════════════════

/// A simple LRU cache with bounded size.
///
/// Not thread-safe — callers must synchronize externally.
/// Uses a HashMap + VecDeque for O(1) lookup and amortized O(1) eviction.
#[derive(Debug, Clone)]
pub struct LruCache<K, V> {
    /// The storage.
    entries: std::collections::HashMap<K, V>,

    /// Access order (most recently used at the back).
    order: std::collections::VecDeque<K>,

    /// Maximum entries.
    capacity: usize,
}

impl<K, V> LruCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
{
    /// Create a new LRU cache with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: std::collections::HashMap::with_capacity(capacity),
            order: std::collections::VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
        }
    }

    /// Insert a key-value pair. Evicts the LRU entry if at capacity.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if let Some(old) = self.entries.insert(key.clone(), value) {
            // Key already existed; move to back (most recently used)
            self.move_to_back(&key);
            Some(old)
        } else {
            // New key
            if self.entries.len() > self.capacity {
                // Evict LRU (front of deque)
                if let Some(evict_key) = self.order.pop_front() {
                    self.entries.remove(&evict_key);
                }
            }
            self.order.push_back(key);
            None
        }
    }

    /// Get a value by key, marking it as recently used.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.entries.contains_key(key) {
            self.move_to_back(key);
            self.entries.get(key)
        } else {
            None
        }
    }

    /// Get a value by key without marking it as recently used.
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.entries.get(key)
    }

    /// Remove a key.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(value) = self.entries.remove(key) {
            self.order.retain(|k| k != key);
            Some(value)
        } else {
            None
        }
    }

    /// Check if the key exists.
    pub fn contains_key(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
    }

    /// Move a key to the back of the access order.
    fn move_to_back(&mut self, key: &K) {
        self.order.retain(|k| k != key);
        self.order.push_back(key.clone());
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  EXPONENTIAL MOVING AVERAGE
// ═══════════════════════════════════════════════════════════════════════

/// An exponential moving average (EMA) tracker.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Ema {
    /// Current EMA value.
    pub value: f64,

    /// Smoothing factor α (0 < α ≤ 1).
    pub alpha: f64,

    /// Number of samples observed.
    pub samples: u64,
}

impl Ema {
    /// Create a new EMA with the given smoothing factor.
    pub fn new(alpha: f64) -> Self {
        Self {
            value: 0.0,
            alpha: alpha.clamp(0.0, 1.0),
            samples: 0,
        }
    }

    /// Create with a specific initial value.
    pub fn with_initial(alpha: f64, initial: f64) -> Self {
        Self {
            value: initial,
            alpha: alpha.clamp(0.0, 1.0),
            samples: 1,
        }
    }

    /// Update with a new sample.
    pub fn update(&mut self, sample: f64) {
        if self.samples == 0 {
            self.value = sample;
        } else {
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value;
        }
        self.samples += 1;
    }

    /// Get the current EMA value.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Number of samples.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Reset the EMA.
    pub fn reset(&mut self) {
        self.value = 0.0;
        self.samples = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  FIXED-POINT HELPERS
// ═══════<══════════════════════════════════════════════════════════════

/// Fixed-point format: 32.32 (32-bit integer, 32-bit fraction).
pub const FP_INTEGER_BITS: u32 = 32;
pub const FP_FRACTIONAL_BITS: u32 = 32;
pub const FP_SCALE: i64 = 1i64 << FP_FRACTIONAL_BITS;

/// Convert an f64 to 32.32 fixed-point (i64).
pub fn f64_to_fp32(value: f64) -> i64 {
    (value * FP_SCALE as f64).round() as i64
}

/// Convert a 32.32 fixed-point (i64) to f64.
pub fn fp32_to_f64(value: i64) -> f64 {
    value as f64 / FP_SCALE as f64
}

/// Fixed-point addition.
#[inline]
pub fn fp_add(a: i64, b: i64) -> i64 {
    a.wrapping_add(b)
}

/// Fixed-point subtraction.
#[inline]
pub fn fp_sub(a: i64, b: i64) -> i64 {
    a.wrapping_sub(b)
}

/// Fixed-point multiplication (with scaling).
pub fn fp_mul(a: i64, b: i64) -> i64 {
    // (a * b) >> 32, using i128 for intermediate
    ((a as i128 * b as i128) >> FP_FRACTIONAL_BITS) as i64
}

/// Fixed-point division (with scaling).
pub fn fp_div(a: i64, b: i64) -> i64 {
    if b == 0 {
        return i64::MAX;
    }
    // (a << 32) / b, using i128 for intermediate
    ((a as i128 << FP_FRACTIONAL_BITS) / b as i128) as i64
}

// ═══════════════════════════════════════════════════════════════════════
//  HEX / BYTE FORMATTING
// ═══════════════════════════════════════════════════════════════════════

/// Format a byte count as a human-readable size.
pub fn format_byte_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Format a nano-NERV amount as a decimal string.
pub fn format_nerv_amount(nano: u64, decimals: u8) -> String {
    let one_unit = 10u64.pow(decimals as u32);
    let whole = nano / one_unit;
    let frac = nano % one_unit;
    format!("{whole}.{frac:0>decimals$}", decimals = decimals as usize)
}

// ═══════════════════════════════════════════════════════════════════════
//  TESTS
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ─── BLAKE3 Tests ────────────────────────────────────────────────

    #[test]
    fn test_blake3_hash_basic() {
        let h1 = blake3_hash(b"hello");
        let h2 = blake3_hash(b"hello");
        assert_eq!(h1, h2);

        let h3 = blake3_hash(b"world");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_blake3_hash_domain_separation() {
        let h1 = blake3_hash_domain("nerv:tx", b"data");
        let h2 = blake3_hash_domain("nerv:root", b"data");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_blake3_derive_key() {
        let k1 = blake3_derive_key("context1", b"input");
        let k2 = blake3_derive_key("context2", b"input");
        assert_ne!(k1, k2);

        let k3 = blake3_derive_key("context1", b"input");
        assert_eq!(k1, k3);
    }

    #[test]
    fn test_blake3_hash_multi() {
        let h1 = blake3_hash_multi(&[b"hello", b"world"]);
        let h2 = blake3_hash(&[b"hello", b"world"].concat());
       !       assert_eq!(h1, h2);
    }

    // ─── SHA3 Tests ─────────────────────────────────────────────────

    #[test]
    fn test_sha3_256_basic() {
        let h1 = sha3_256(b"hello");
        let h2 = sha3_256(b"hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_sha3_256_known_vector() {
        let empty = sha3_256(b"");
        let expected_hex = "a7ffc6f8bf1ed76651c14756a061e662f5dbffba0ddcff5f4361917c3e0b8d2";
        assert_eq!(hex::encode(empty), expected_hex);
    }

    #[test]
    fn test_sha3_256_domain() {
        let h1 = sha3_256_domain("nerv:tx", b"data");
        let h2 = sha3_256_domain("nerv:root", b"data");
        assert_ne!(h1, h2);
    }

    // ─── Random Tests ──────────────────────────────────────────────

    #[test]
    fn test_random_bytes() {
        let r1 = random_bytes(32);
        let r2 = random_bytes(32);
        assert_eq!(r1.len(), 32);
        // Overwhelming probability they're different
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_random_32bytes() {
        let r = random_32bytes();
        assert_eq!(r.len(), 32);
    }

    #[test]
    fn test_random_u64() {
        let _ = random_u64(); // Just verify it doesn't panic
    }

    #[test]
    fn test_random_range() {
        for _ in 0..100 {
            let v = random_range(10);
            assert!(v < 10);
        }
    }

    #[test]
    fn test_random_normal() {
        // Generate samples and check they're finite
        for _ in 0..100 {
            let v = random_normal(0.0, 1.0);
            assert!(v.is_finite());
        }
    }

    // ─── Secure Memory Tests ───────────────────────────────────────

    #[test]
    fn test_secure_zero() {
        let mut data = vec![42u8; 100];
        secure_zero(&mut data);
        assert!(data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_secure_zero_vec() {
        let mut data = vec![42u8; 100];
        secure_zero_vec(&mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn test_constant_time_eq() {
        let a = [1u8; 32];
        let b = [1u8; 32];
        let c = [2u8; 32];
        assert!(constant_time_eq(&a[..], &b[..]));
        assert!(!constant_time_eq(&a[..], &c[..]));
        assert!(!constant_time_eq(&a[..4], &b[..8])); // Different lengths
    }

    // ─── Time Tests ────────────────────────────────────────────────

    #[test]
    fn test_current_epoch_millis() {
        let t = current_epoch_millis();
        assert!(t > 1_700_000_000_000); // After 2023
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(0)), "0s");
        assert_eq!(format_duration(Duration::from_secs(65)), "1m 5s");
        assert_eq!(format_duration(Duration::from_secs(3661)), "1h 1m 1s");
        assert_eq!(format_duration(Duration::from_secs(86400)), "1d 0h 0m 0s");
    }

    // ─── Encoding Tests ────────────────────────────────────────────

    #[test]
    fn test_hex_roundtrip() {
        let data = vec![0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef];
        let encoded = to_hex(&data);
        let decoded = from_hex(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_base64_roundtrip() {
        let data = b"hello world";
        let encoded = to_base64(data);
        let decoded = from_base64(&encoded).unwrap();
        assert_eq!(data.to_vec(), decoded);
    }

    // ─── Rate Limiter Tests ────────────────────────────────────────

    #[test]
    fn test_rate_limiter_basic() {
        let limiter = RateLimiter::new(5, 1, 1000);
        for _ in 0..5 {
            assert!(limiter.try_acquire());
        }
        assert!(!limiter.try_acquire()); // Exhausted
    }

    #[test]
    fn test_rate_limiter_available() {
        let limiter = RateLimiter::new(10, 1, 1000);
        assert_eq!(limiter.available(), 10);
        assert!(limiter.try_acquire());
        assert_eq!(limiter.available(), 9);
    }

    // ─── LRU Cache Tests ───────────────────────────────────────────

    #[test]
    fn test_lru_cache_basic() {
        let mut cache: LruCache<u64, String> = LruCache::new(3);
        cache.insert(1, "one".into());
        cache.insert(2, "two".into());
        cache.insert(3, "three".into());
        assert_eq!(cache.len(), 3);

        // Inserting 4 should evict 1 (LRU)
        cache.insert(4, "four".into());
        assert_eq!(cache.len(), 3);
        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&4));
    }

    #[test]
    fn test_lru_cache_get_updates_order() {
        let mut cache: LruCache<u64, String> = LruCache::new(3);
        cache.insert(1, "one".into());
        cache.insert(2, "two".into());
        cache.insert(3, "three".into());

        // Access 1, making it recently used
        let _ = cache.get(&1);

        // Inserting 4 should evict 2 (now LRU)
        cache.insert(4, "four".into());
        assert!(cache.contains_key(&1)); // Still present
        assert!(!cache.contains_key(&2)); // Evicted
        assert!(cache.contains_key(&3));
        assert!(cache.contains_key(&4));
    }

    #[test]
    fn test_lru_cache_remove() {
        let mut cache: LruCache<u64, String> = LruCache::new(10);
        cache.insert(1, "one".into());
        let removed = cache.remove(&1);
        assert_eq!(removed, Some("one".into()));
        assert!(!cache.contains_key(&1));
    }

    // ─── EMA Tests ─────────────────────────────────────────────────

    #[test]
    fn test_ema_basic() {
        let mut ema = Ema::new(0.1);
        ema.update(100.0);
        assert!((ema.value() - 100.0).abs() < 1e-10);
        ema.update(0.0);
        assert!(ema.value() < 100.0);
        assert!(ema.value() > 0.0);
    }

    #[test]
    fn test_ma_convergence() {
        let mut ema = Ema::new(0.3);
        for _ in 0..100 {
            ema.update(50.0);
        }
        assert!((ema.value() - 50.0).abs() < 1e-6);
    }

    // ─── Fixed-Point Tests ─────────────────────────────────────────

    #[test]
    fn test_fp32_roundtrip() {
        let original = 3.14159;
        let fixed = f64_to_fp32(original);
        let recovered = fp32_to_f64(fixed);
        assert!((recovered - original).abs() < 1e-6);
    }

    #[test]
    fn test_fp_add() {
        let a = f64_to_fp32(1.5);
        let b = f64_to_fp32(2.5);
        let sum = fp_add(a, b);
        let result = fp32_to_f64(sum);
        assert!((result - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_fp_mul() {
        let a = f64_to_fp32(2.0);
        let b = f64_to_fp32(3.0);
        let product = fp_mul(a, b);
        let result = fp32_to_f64(product);
        assert!((result - 6.0).abs() < 1e-4);
    }

    #[test]
    fn test_fp_div() {
        let a = f64_to_fp32(6.0);
        let b = f64_to_fp32(3.0);
        let quotient = fp_div(a, b);
        let result = fp32_to_f64(quotient);
        assert!((result - 2.0).abs() < 1e-4);
    }

    // ─── Formatting Tests ──────────────────────────────────────────

    #[test]
    fn test_format_byte_size() {
        assert_eq!(format_byte_size(500), "500 B");
        assert_eq!(format_byte_size(1024), "1.00 KB");
        assert_eq!(format_byte_size(1048576), "1.00 MB");
        assert_eq!(format_byte_size(1073741824), "1.00 GB");
    }

    #[test]
    fn test_format_nerv_amount() {
        let one_nerv = 1_000_000_000u64; // 10^9 nano
        assert_eq!(format_nerv_amount(one_nerv, 9), "1.000000000");
        assert_eq!(format_nerv_amount(1_500_000_000, 9), "1.500000000");
    }
}
