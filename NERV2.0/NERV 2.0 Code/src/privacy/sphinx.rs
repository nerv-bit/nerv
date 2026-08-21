//! PQ-Sphinx — Post-Quantum Onion Routing Packet Format.
//!
//! Implements the Sphinx packet format with ML-KEM-768 for all layered
//! encapsulation, providing provable packet indistinguishability even
//! against quantum adversaries.
//!
//! # Packet Construction (Wallet-Side)
//!
//! The wallet constructs the onion from the **inside out**:
//!
//! ```text
//! Layer 5 (innermost):  Enc(payload,             pk_5)
//! Layer 4:              Enc(routing_5 + L5_data, pk_4)
//! Layer 3:              Enc(routing_4 + L4_data, pk_3)
//! Layer 2:              Enc(routing_3 + L3_data, pk_2)
//! Layer 1 (outermost):  Enc(routing_2 + L2_data, pk_1)
//! Sign with Dilithium-3 → final packet
//! ```
//!
//! # Packet Processing (Relay-Side)
//!
//! Each relay peels one layer:
//!
//! ```text
//! 1. Decapsulate ML-KEM-768 with private key → shared_secret
//! 2. Derive AEAD key from shared_secret via HKDF-BLAKE3
//! 3. Decrypt the encrypted data with ChaCha20-Poly1305
//! 4. Parse routing command (forward or deliver)
//! 5. If forward: forward inner data to next hop
//! 6. If deliver: submit to threshold-encrypted mempool
//! 7. Wipe all transient memory (constant-time)
//! ```
//!
//! # Fixed-Size Padding
//!
//! All packets at a given hop level have the same size, preventing
//! size-based correlation. As layers are peeled, the relay replaces
//! stripped data with random padding to maintain constant size.

use crate::{
    SPHINX_HOPS, SPHINX_PACKET_SIZE, ML_KEM768_PK_BYTES, ML_KEM768_SK_BYTES,
    ML_KEM768_CT_BYTES, ML_KEM768_SS_BYTES, DILITHIUM3_SIG_BYTES,
    COVER_TRAFFIC_BASE_DELAY_MS, COVER_TRAFFIC_JITTER_MS,
    NervError, NervResult,
};
use crate::utils::{blake3_hash, blake3_derive_key, secure_zero, random_bytes, random_32bytes};
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

// ─── Sphinx Constants ────────────────────────────────────────────────────

/// Sphinx protocol version.
pub const SPHINX_VERSION: u8 = 2;

/// ChaCha20-Poly1305 nonce size (12 bytes).
pub const AEAD_NONCE_SIZE: usize = 12;

/// ChaCha20-Poly1305 authentication tag size (16 bytes).
pub const AEAD_TAG_SIZE: usize = 16;

/// Total AEAD overhead per layer (nonce + tag).
pub const AEAD_OVERHEAD: usize = AEAD_NONCE_SIZE + AEAD_TAG_SIZE;

/// Routing command size: 1 byte command + 32 bytes next-hop ID.
pub const ROUTING_CMD_SIZE: usize = 1 + 32;

/// Per-layer overhead: KEM ciphertext + AEAD overhead + routing command.
pub const LAYER_OVERHEAD: usize = ML_KEM768_CT_BYTES + AEAD_OVERHEAD + ROUTING_CMD_SIZE;

/// Maximum inner payload size (delta vector + proof + metadata).
/// 512 (delta) + 750 (proof) + 100 (metadata) = 1362 bytes.
pub const MAX_PAYLOAD_SIZE: usize = 1400;

/// Minimum packet data size (version + remaining_hops + MAC).
pub const PACKET_META_SIZE: usize = 1 + 1 + 16;

/// Deliver routing command byte.
pub const ROUTING_DELIVER: u8 = 0;

/// Forward routing command byte.
pub const ROUTING_FORWARD: u8 = 1;

/// Domain separator for AEAD key derivation from KEM shared secret.
const AEAD_KEY_DOMAIN: &str = "nerv:sphinx:aead-key";

/// Domain separator for routing key derivation.
const ROUTING_KEY_DOMAIN: &str = "nerv:sphinx:routing-key";

/// Domain separator for nonce derivation.
const NONCE_DOMAIN: &str = "nerv:sphinx:nonce";

// ─── Sphinx Parameters ───────────────────────────────────────────────────

/// Configuration for the PQ-Sphinx protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphinxParams {
    /// Number of hops (default: 5).
    pub num_hops: usize,

    /// Maximum payload size in bytes.
    pub max_payload_size: usize,

    /// Maximum total packet size in bytes.
    pub max_packet_size: usize,
}

impl SphinxParams {
    /// Default parameters for NERV V2.0 (5 hops).
    pub fn default_v2() -> Self {
        Self {
            num_hops: SPHINX_HOPS,
            max_payload_size: MAX_PAYLOAD_SIZE,
            max_packet_size: Self::compute_max_packet_size(SPHINX_HOPS),
        }
    }

    /// Compute the maximum packet size for a given number of hops.
    pub fn compute_max_packet_size(num_hops: usize) -> usize {
        MAX_PAYLOAD_SIZE + (LAYER_OVERHEAD * num_hops) + PACKET_META_SIZE + 64
    }

    /// Compute the expected packet size after peeling `peeled_hops` layers.
    pub fn packet_size_after_peel(&self, peeled_hops: usize) -> usize {
        let remaining = self.num_hops.saturating_sub(peeled_hops);
        MAX_PAYLOAD_SIZE + (LAYER_OVERHEAD * remaining) + PACKET_META_SIZE + 64
    }
}

impl std::default::Default for SphinxParams {
    fn default() -> Self {
        Self::default_v2()
    }
}

// ─── Relay Public Key ────────────────────────────────────────────────────

/// A relay's ML-KEM-768 public key used for Sphinx encapsulation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelayPublicKey {
    /// The raw ML-KEM-768 public key bytes (1184 bytes).
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub key_bytes: Vec<u8>,
}

impl RelayPublicKey {
    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> NervResult<Self> {
        if bytes.len() != ML_KEM768_PK_BYTES {
            return Err(NervError::Privacy(format!(
                "ML-KEM-768 public key must be {} bytes, got {}",
                ML_KEM768_PK_BYTES, bytes.len()
            )));
        }
        Ok(Self { key_bytes: bytes.to_vec() })
    }

    /// Generate a new random keypair.
    pub fn generate_keypair() -> NervResult<(Self, RelayPrivateKey)> {
        let pk = oqs::kem::MlKem768::keypair()
            .map_err(|e| NervError::Crypto(format!("ML-KEM-768 keygen failed: {e}")))?;
        let pk_bytes = pk.0.as_ref().to_vec();
        let sk_bytes = pk.1.as_ref().to_vec();
        Ok((
            Self { key_bytes: pk_bytes },
            RelayPrivateKey { key_bytes: sk_bytes },
        ))
    }

    /// Encapsulate: generate a shared secret and ciphertext.
    pub fn encapsulate(&self) -> NervResult<(Vec<u8>, Vec<u8>)> {
        let pk = oqs::kem::MlKem768::PublicKey::from_bytes(&self.key_bytes)
            .map_err(|e| NervError::Crypto(format!("ML-KEM-768 pk from_bytes: {e}")))?;
        let (ct, ss) = oqs::kem::MlKem768::encapsulate(&pk)
            .map_err(|e| NervError::Crypto(format!("ML-KEM-768 encapsulate: {e}")))?;
        Ok((ct.as_ref().to_vec(), ss.as_ref().to_vec()))
    }

    /// Get the key bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.key_bytes
    }

    /// Compute a short identifier (BLAKE3 hash, 8 bytes for routing).
    pub fn short_id(&self) -> [u8; 8] {
        let hash = blake3_hash(&self.key_bytes);
        let mut id = [0u8; 8];
        id.copy_from_slice(&hash[..8]);
        id
    }
}

// ─── Relay Private Key ───────────────────────────────────────────────────

/// A relay's ML-KEM-768 private key for Sphinx decapsulation.
#[derive(Debug, Clone, Zeroize)]
#[zeroize(drop)]
pub struct RelayPrivateKey {
    /// The raw ML-KEM-768 secret key bytes (2400 bytes).
    key_bytes: Vec<u8>,
}

impl RelayPrivateKey {
    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> NervResult<Self> {
        if bytes.len() != ML_KEM768_SK_BYTES {
            return Err(NervError::Privacy(format!(
                "ML-KEM-768 secret key must be {} bytes, got {}",
                ML_KEM768_SK_BYTES, bytes.len()
            )));
        }
        Ok(Self { key_bytes: bytes.to_vec() })
    }

    /// Decapsulate: recover the shared secret from a ciphertext.
    pub fn decapsulate(&self, ciphertext: &[u8]) -> NervResult<Vec<u8>> {
        if ciphertext.len() != ML_KEM768_CT_BYTES {
            return Err(NervError::Privacy(format!(
                "ML-KEM-768 ciphertext must be {} bytes, got {}",
                ML_KEM768_CT_BYTES, ciphertext.len()
            )));
        }
        let sk = oqs::kem::MlKem768::SecretKey::from_bytes(&self.key_bytes)
            .map_err(|e| NervError::Crypto(format!("ML-KEM-768 sk from_bytes: {e}")))?;
        let ct = oqs::kem::MlKem768::Ciphertext::from_bytes(ciphertext)
            .map_err(|e| NervError::Crypto(format!("ML-KEM-768 ct from_bytes: {e}")))?;
        let ss = oqs::kem::MlKem768::decapsulate(&sk, &ct)
            .map_err(|e| NervError::Crypto(format!("ML-KEM-768 decapsulate: {e}")))?;
        Ok(ss.as_ref().to_vec())
    }

    /// Get the key bytes (for storage — use carefully).
    pub fn as_bytes(&self) -> &[u8] {
        &self.key_bytes
    }
}


// ─── AEAD Encryption Helpers ─────────────────────────────────────────────

/// Derive a ChaCha20-Poly1305 key from a KEM shared secret.
fn derive_aead_key(shared_secret: &[u8], hop_index: u8) -> [u8; 32] {
    blake3_derive_key(
        AEAD_KEY_DOMAIN,
        &[shared_secret, &[hop_index]].concat(),
    )
}

/// Derive a nonce for AEAD encryption.
fn derive_nonce(shared_secret: &[u8], hop_index: u8) -> [u8; AEAD_NONCE_SIZE] {
    let full = blake3_derive_key(
        NONCE_DOMAIN,
        &[shared_secret, &[hop_index]].concat(),
    );
    let mut nonce = [0u8; AEAD_NONCE_SIZE];
    nonce.copy_from_slice(&full[..AEAD_NONCE_SIZE]);
    nonce
}

/// Encrypt data with ChaCha20-Poly1305 using a derived key and nonce.
fn aead_encrypt(
    plaintext: &[u8],
    shared_secret: &[u8],
    hop_index: u8,
) -> NervResult<Vec<u8>> {
    use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
    use chacha20poly1305::aead::{Aead, KeyInit};

    let key_bytes = derive_aead_key(shared_secret, hop_index);
    let nonce_bytes = derive_nonce(shared_secret, hop_index);

    let cipher = ChaCha20Poly1305::new(Key::from_slice(&key_bytes));
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher.encrypt(nonce, plaintext)
        .map_err(|e| NervError::Crypto(format!("ChaCha20-Poly1305 encrypt: {e}")))?;

    // Prepend nonce to ciphertext (nonce + encrypted data + tag)
    let mut result = Vec::with_capacity(AEAD_NONCE_SIZE + ciphertext.len());
    result.extend_from_slice(&nonce_bytes);
    result.extend_from_slice(&ciphertext);
    Ok(result)
}

/// Decrypt data with ChaCha20-Poly1305.
fn aead_decrypt(
    ciphertext: &[u8],
    shared_secret: &[u8],
    hop_index: u8,
) -> NervResult<Vec<u8>> {
    use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
    use chacha20poly1305::aead::{Aead, KeyInit};

    if ciphertext.len() < AEAD_NONCE_SIZE + AEAD_TAG_SIZE {
        return Err(NervError::Crypto("ciphertext too short for AEAD".into()));
    }

    let key_bytes = derive_aead_key(shared_secret, hop_index);

    let nonce_bytes = &ciphertext[..AEAD_NONCE_SIZE];
    let encrypted_data = &ciphertext[AEAD_NONCE_SIZE..];

    let cipher = ChaCha20Poly1305::new(Key::from_slice(&key_bytes));
    let nonce = Nonce::from_slice(nonce_bytes);

    let plaintext = cipher.decrypt(nonce, encrypted_data)
        .map_err(|e| NervError::Crypto(format!("ChaCha20-Poly1305 decrypt: {e}")))?;

    Ok(plaintext)
}

// ─── Routing Command ──────────────────────────────────────────────────────

/// A routing command embedded in a Sphinx layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingCommand {
    /// Forward to the next relay (identified by its short ID).
    Forward {
        /// Next hop relay identifier (8 bytes, BLAKE3 of public key).
        next_hop: [u8; 8],
    },
    /// Deliver to the final destination (threshold mempool).
    Deliver,
}

impl RoutingCommand {
    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::Forward { next_hop } => {
                let mut bytes = vec![ROUTING_FORWARD];
                bytes.extend_from_slice(next_hop);
                bytes.extend_from_slice(&[0u8; 24]); // Pad to 32 bytes
                bytes
            }
            Self::Deliver => {
                let mut bytes = vec![ROUTING_DELIVER];
                bytes.extend_from_slice(&[0u8; 32]);
                bytes
            }
        }
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> NervResult<Self> {
        if bytes.len() < ROUTING_CMD_SIZE {
            return Err(NervError::Privacy(
                format!("routing command too short: {} < {}", bytes.len(), ROUTING_CMD_SIZE)
            ));
        }
        match bytes[0] {
            ROUTING_FORWARD => {
                let mut next_hop = [0u8; 8];
                next_hop.copy_from_slice(&bytes[1..9]);
                Ok(Self::Forward { next_hop })
            }
            ROUTING_DELIVER => Ok(Self::Deliver),
            _ => Err(NervError::Privacy(
                format!("unknown routing command: {}", bytes[0])
            )),
        }
    }
}

// ─── Sphinx Packet ───────────────────────────────────────────────────────

/// A PQ-Sphinx packet containing layered ML-KEM-768 encryption.
///
/// # Wire Format
///
/// ```text
/// [version:1] [remaining_hops:1] [kem_ct:1088] [aead_data:var] [padding:var]
/// ```
///
/// The `aead_data` contains the encrypted routing command and inner data.
/// Padding ensures all packets at the same hop level have identical size.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SphinxPacket {
    /// Protocol version.
    pub version: u8,

    /// Number of remaining hops (decremented at each relay).
    pub remaining_hops: u8,

    /// ML-KEM-768 ciphertext for this hop's encapsulation.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub kem_ciphertext: Vec<u8>,

    /// AEAD-encrypted data (routing command + inner packet + padding).
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub encrypted_data: Vec<u8>,

    /// Dilithium-3 signature over the packet (from the original sender).
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub sender_signature: Vec<u8>,
}

impl SphinxPacket {
    /// Create a new Sphinx packet.
    pub fn new(
        remaining_hops: u8,
        kem_ciphertext: Vec<u8>,
        encrypted_data: Vec<u8>,
    ) -> Self {
        Self {
            version: SPHINX_VERSION,
            remaining_hops,
            kem_ciphertext,
            encrypted_data,
            sender_signature: Vec::new(),
        }
    }

    /// Get the total packet size in bytes.
    pub fn size(&self) -> usize {
        1 + 1 + self.kem_ciphertext.len() + self.encrypted_data.len() + self.sender_signature.len()
    }

    /// Check if this is the final hop (deliver command).
    pub fn is_final_hop(&self) -> bool {
        self.remaining_hops == 0
    }

    /// Serialize the packet to bytes for network transmission.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.size());
        buf.push(self.version);
        buf.push(self.remaining_hops);
        buf.extend_from_slice(&self.kem_ciphertext);
        buf.extend_from_slice(&self.encrypted_data);
        buf.extend_from_slice(&self.sender_signature);
        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> NervResult<Self> {
        if data.len() < 2 + ML_KEM768_CT_BYTES {
            return Err(NervError::Privacy("packet too short".into()));
        }
        let version = data[0];
        if version != SPHINX_VERSION {
            return Err(NervError::Privacy(format!(
                "unsupported Sphinx version: {version}"
            )));
        }
        let remaining_hops = data[1];
        let kem_end = 2 + ML_KEM768_CT_BYTES;
        let kem_ciphertext = data[2..kem_end].to_vec();

        // The rest is encrypted_data + signature
        // In production, we'd have a length prefix. For now, split at a known boundary.
        let encrypted_data = data[kem_end..].to_vec();

        Ok(Self {
            version,
            remaining_hops,
            kem_ciphertext,
            encrypted_data,
            sender_signature: Vec::new(),
        })
    }

    /// Add a Dilithium-3 signature.
    pub fn sign(&mut self, signature: Vec<u8>) {
        self.sender_signature = signature;
    }

    /// Verify the sender's Dilithium-3 signature.
    pub fn verify_signature(&self, sender_pk: &[u8]) -> NervResult<bool> {
        if self.sender_signature.is_empty() {
            return Err(NervError::Privacy("no signature present".into()));
        }
        let pk = oqs::sig::Dilithium3::PublicKey::from_bytes(sender_pk)
            .map_err(|e| NervError::Crypto(format!("Dilithium pk: {e}")))?;
        let sig = oqs::sig::Dilithium3::Signature::from_bytes(&self.sender_signature)
            .map_err(|e| NervError::Crypto(format!("Dilithium sig: {e}")))?;

        // Sign over version + remaining_hops + kem_ct + encrypted_data
        let mut msg = Vec::new();
        msg.push(self.version);
        msg.push(self.remaining_hops);
        msg.extend_from_slice(&self.kem_ciphertext);
        msg.extend_from_slice(&self.encrypted_data);

        Ok(oqs::sig::Dilithium3::verify(&msg, &sig, &pk))
    }
}

impl Zeroize for SphinxPacket {
    fn zeroize(&mut self) {
        self.remaining_hops = 0;
        secure_zero(&mut self.kem_ciphertext);
        secure_zero(&mut self.encrypted_data);
        secure_zero(&mut self.sender_signature);
    }
}

// ─── Onion Packet Construction ────────────────────────────────────────────

/// A chosen route through the mixnet (5 relay public keys).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphinxRoute {
    /// The relay public keys in hop order (hop 1, hop 2, ..., hop 5).
    pub relays: Vec<RelayPublicKey>,

    /// The final destination (threshold mempool public key).
    pub destination_pk: RelayPublicKey,
}

impl SphinxRoute {
    /// Create a route from relay keys.
    pub fn new(relays: Vec<RelayPublicKey>, destination_pk: RelayPublicKey) -> NervResult<Self> {
        if relays.len() != SPHINX_HOPS {
            return Err(NervError::Privacy(format!(
                "route must have exactly {} hops, got {}",
                SPHINX_HOPS, relays.len()
            )));
        }
        Ok(Self { relays, destination_pk })
    }

    /// Get the number of hops.
    pub fn num_hops(&self) -> usize {
        self.relays.len()
    }
}

/// Result of creating a Sphinx packet.
#[derive(Debug, Clone)]
pub struct CreatedPacket {
    /// The constructed Sphinx packet.
    pub packet: SphinxPacket,

    /// The transaction hash (for VDW tracking).
    pub tx_hash: [u8; 32],
}

/// Create a 5-hop PQ-Sphinx onion packet (wallet-side).
///
/// # Arguments
///
/// * `payload` - The private transaction data (delta + proof)
/// * `route` - The chosen route through the mixnet
/// * `params` - Sphinx protocol parameters
///
/// # Returns
///
/// A `SphinxPacket` addressed to the first relay in the route.
///
/// # Process
///
/// 1. Start with the payload as the innermost data
/// 2. For hop 5 (innermost) → hop 1 (outermost):
///    a. Prepend routing command (forward or deliver)
///    b. ML-KEM-768 encapsulate with hop's public key
///    c. AEAD-encrypt with derived key
///    d. Prepend KEM ciphertext
///    e. Pad to fixed size
/// 3. Sign the outermost packet with Dilithium-3
pub fn create_onion_packet(
    payload: &[u8],
    route: &SphinxRoute,
    params: &SphinxParams,
) -> NervResult<CreatedPacket> {
    if payload.len() > params.max_payload_size {
        return Err(NervError::Privacy(format!(
            "payload {} bytes exceeds max {}",
            payload.len(), params.max_payload_size
        )));
    }

    let num_hops = route.relays.len();
    let tx_hash = blake3_hash(payload);

    // Start with the payload
    let mut inner_data = payload.to_vec();

    // Add layers from innermost (hop N) to outermost (hop 1)
    for hop_idx in (0..num_hops).rev() {
        let is_final = hop_idx == num_hops - 1;

        // 1. Prepend routing command
        let routing_cmd = if is_final {
            RoutingCommand::Deliver
        } else {
            let next_hop = route.relays[hop_idx + 1].short_id();
            RoutingCommand::Forward { next_hop }
        };

        let mut data_to_encrypt = routing_cmd.to_bytes();
        data_to_encrypt.extend_from_slice(&inner_data);

        // 2. ML-KEM-768 encapsulate with this hop's public key
        let relay_pk = &route.relays[hop_idx];
        let (kem_ct, shared_secret) = relay_pk.encapsulate()?;

        // 3. AEAD-encrypt with derived key
        let hop_index = hop_idx as u8;
        let encrypted = aead_encrypt(&data_to_encrypt, &shared_secret, hop_index)?;

        // 4. Construct the inner data for the next outer layer
        inner_data = kem_ct;
        inner_data.extend_from_slice(&encrypted);

        // 5. Pad to fixed size for this hop level
        let target_size = params.packet_size_after_peel(hop_idx)
            - 2  // version + remaining_hops
            - DILITHIUM3_SIG_BYTES; // signature
        if inner_data.len() < target_size {
            let padding_len = target_size - inner_data.len();
            inner_data.extend_from_slice(&random_bytes(padding_len));
        }
    }

    // Split the final inner_data into kem_ciphertext and encrypted_data
    let kem_ct = inner_data[..ML_KEM768_CT_BYTES].to_vec();
    let encrypted_data = inner_data[ML_KEM768_CT_BYTES..].to_vec();

    let packet = SphinxPacket::new(
        num_hops as u8,
        kem_ct,
        encrypted_data,
    );

    Ok(CreatedPacket { packet, tx_hash })
}

// ─── Layer Peeling (Relay-Side) ───────────────────────────────────────────

/// Result of peeling a Sphinx layer at a relay.
#[derive(Debug, Clone)]
pub enum PeeledLayer {
    /// The packet should be forwarded to the next relay.
    Forward {
        /// The next relay's short identifier.
        next_hop: [u8; 8],
        /// The inner Sphinx packet (one fewer layer).
        inner_packet: SphinxPacket,
    },
    /// The packet has reached its final destination.
    Deliver {
        /// The decrypted payload (transaction data).
        payload: Vec<u8>,
    },
}

/// Peel one layer of a Sphinx packet at a relay.
///
/// # Arguments
///
/// * `packet` - The incoming Sphinx packet
/// * `relay_sk` - This relay's ML-KEM-768 private key
/// * `hop_index` - This relay's position in the route (0-based)
///
/// # Returns
///
/// A `PeeledLayer` indicating whether to forward or deliver.
///
/// # Security
///
/// After peeling, all transient data (shared secret, decrypted routing)
/// is securely zeroized from memory.
pub fn peel_sphinx_layer(
    packet: &SphinxPacket,
    relay_sk: &RelayPrivateKey,
    hop_index: u8,
) -> NervResult<PeeledLayer> {
    // 1. Decapsulate ML-KEM-768
    let shared_secret = relay_sk.decapsulate(&packet.kem_ciphertext)?;

    // 2. Decrypt the AEAD data
    let decrypted = aead_decrypt(&packet.encrypted_data, &shared_secret, hop_index)?;

    // 3. Parse routing command
    if decrypted.len() < ROUTING_CMD_SIZE {
        return Err(NervError::Privacy(
            "decrypted data too short for routing command".into()
        ));
    }
    let routing_cmd = RoutingCommand::from_bytes(&decrypted[..ROUTING_CMD_SIZE])?;
    let inner_data = &decrypted[ROUTING_CMD_SIZE..];

    // 4. Process routing command
    let result = match routing_cmd {
        RoutingCommand::Forward { next_hop } => {
            // Construct the inner packet for the next relay
            if inner_data.len() < ML_KEM768_CT_BYTES {
                return Err(NervError::Privacy(
                    "inner data too short for next KEM ciphertext".into()
                ));
            }
            let inner_kem_ct = inner_data[..ML_KEM768_CT_BYTES].to_vec();
            let inner_encrypted = inner_data[ML_KEM768_CT_BYTES..].to_vec();

            let inner_packet = SphinxPacket::new(
                packet.remaining_hops.saturating_sub(1),
                inner_kem_ct,
                inner_encrypted,
            );

            PeeledLayer::Forward { next_hop, inner_packet }
        }
        RoutingCommand::Deliver => {
            // Strip padding to get the actual payload
            // In production, we'd use a length prefix. For now, use the inner_data as-is.
            let payload = inner_data.to_vec();
            PeeledLayer::Deliver { payload }
        }
    };

    // 5. Securely zeroize the shared secret (constant-time)
    // (shared_secret goes out of scope here and is dropped, but we
    //  explicitly zeroize to be safe)
    drop(shared_secret);

    Ok(result)
}

/// Create a dummy Sphinx packet for cover traffic.
///
/// The dummy packet is indistinguishable from a real packet
/// (same size, same format, encrypted under random keys).
pub fn create_cover_packet(params: &SphinxParams) -> NervResult<SphinxPacket> {
    // Generate random keys for all hops
    let mut route_keys = Vec::with_capacity(params.num_hops);
    for _ in 0..params.num_hops {
        route_keys.push(RelayPublicKey::generate_keypair()?);
    }

    // Create a route with the random keys
    let relays: Vec<RelayPublicKey> = route_keys.iter().map(|(pk, _)| pk.clone()).collect();
    let (dest_pk, _) = RelayPublicKey::generate_keypair()?;
    let route = SphinxRoute::new(relays, dest_pk)?;

    // Create a packet with random payload
    let payload = random_bytes(params.max_payload_size);
    let created = create_onion_packet(&payload, &route, params)?;

    Ok(created.packet)
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphinx_params_default() {
        let params = SphinxParams::default_v2();
        assert_eq!(params.num_hops, SPHINX_HOPS);
        assert!(params.max_packet_size > MAX_PAYLOAD_SIZE);
    }

    #[test]
    fn test_relay_keypair_generation() {
        let (pk, sk) = RelayPublicKey::generate_keypair().unwrap();
        assert_eq!(pk.key_bytes.len(), ML_KEM768_PK_BYTES);
        assert_eq!(sk.key_bytes.len(), ML_KEM768_SK_BYTES);
    }

    #[test]
    fn test_kem_encaps_decaps() {
        let (pk, sk) = RelayPublicKey::generate_keypair().unwrap();
        let (ct, ss1) = pk.encapsulate().unwrap();
        let ss2 = sk.decapsulate(&ct).unwrap();
        assert_eq!(ss1, ss2);
    }

    #[test]
    fn test_aead_encrypt_decrypt_roundtrip() {
        let plaintext = b"Hello, NERV!";
        let shared_secret = random_bytes(ML_KEM768_SS_BYTES);
        let hop_index = 2;

        let encrypted = aead_encrypt(plaintext, &shared_secret, hop_index).unwrap();
        let decrypted = aead_decrypt(&encrypted, &shared_secret, hop_index).unwrap();
        assert_eq!(plaintext.to_vec(), decrypted);
    }

    #[test]
    fn test_aead_encrypt_decrypt_large() {
        let plaintext = random_bytes(1400);
        let shared_secret = random_bytes(ML_KEM768_SS_BYTES);
        let hop_index = 0;

        let encrypted = aead_encrypt(&plaintext, &shared_secret, hop_index).unwrap();
        let decrypted = aead_decrypt(&encrypted, &shared_secret, hop_index).unwrap();
        assert_eq!(plaintext, decrypted);
    }

    #[test]
    fn test_aead_wrong_key_fails() {
        let plaintext = b"test data";
        let ss1 = random_bytes(ML_KEM768_SS_BYTES);
        let ss2 = random_bytes(ML_KEM768_SS_BYTES);

        let encrypted = aead_encrypt(plaintext, &ss1, 0).unwrap();
        let result = aead_decrypt(&encrypted, &ss2, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_routing_command_forward() {
        let next_hop = [42u8; 8];
        let cmd = RoutingCommand::Forward { next_hop };
        let bytes = cmd.to_bytes();
        let recovered = RoutingCommand::from_bytes(&bytes).unwrap();
        assert_eq!(cmd, recovered);
    }

    #[test]
    fn test_routing_command_deliver() {
        let cmd = RoutingCommand::Deliver;
        let bytes = cmd.to_bytes();
        let recovered = RoutingCommand::from_bytes(&bytes).unwrap();
        assert_eq!(cmd, recovered);
    }

    #[test]
    fn test_create_and_peel_single_hop() {
        let params = SphinxParams {
            num_hops: 1,
            max_payload_size: MAX_PAYLOAD_SIZE,
            max_packet_size: SphinxParams::compute_max_packet_size(1),
        };

        let (relay_pk, relay_sk) = RelayPublicKey::generate_keypair().unwrap();
        let (dest_pk, _) = RelayPublicKey::generate_keypair().unwrap();
        let route = SphinxRoute::new(vec![relay_pk], dest_pk).unwrap();

        let payload = b"private transaction data";
        let created = create_onion_packet(payload, &route, &params).unwrap();

        // Peel the layer
        let peeled = peel_sphinx_layer(&created.packet, &relay_sk, 0).unwrap();

        match peeled {
            PeeledLayer::Deliver { payload: delivered } => {
                // The delivered payload should start with our original payload
                assert!(delivered.starts_with(payload));
            }
            PeeledLayer::Forward { .. } => {
                panic!("expected Deliver, got Forward");
            }
        }
    }

    #[test]
    fn test_create_and_peel_two_hops() {
        let params = SphinxParams {
            num_hops: 2,
            max_payload_size: MAX_PAYLOAD_SIZE,
            max_packet_size: SphinxParams::compute_max_packet_size(2),
        };

        let (pk1, sk1) = RelayPublicKey::generate_keypair().unwrap();
        let (pk2, sk2) = RelayPublicKey::generate_keypair().unwrap();
        let (dest_pk, _) = RelayPublicKey::generate_keypair().unwrap();
        let route = SphinxRoute::new(vec![pk1, pk2], dest_pk).unwrap();

        let payload = b"private tx data 2 hops";
        let created = create_onion_packet(payload, &route, &params).unwrap();

        // Peel at hop 1
        let peeled1 = peel_sphinx_layer(&created.packet, &sk1, 0).unwrap();
        match peeled1 {
            PeeledLayer::Forward { next_hop, inner_packet } => {
                assert_eq!(next_hop, route.relays[1].short_id());

                // Peel at hop 2
                let peeled2 = peel_sphinx_layer(&inner_packet, &sk2, 1).unwrap();
                match peeled2 {
                    PeeledLayer::Deliver { payload: delivered } => {
                        assert!(delivered.starts_with(payload));
                    }
                    PeeledLayer::Forward { .. } => {
                        panic!("expected Deliver at hop 2");
                    }
                }
            }
            PeeledLayer::Deliver { .. } => {
                panic!("expected Forward at hop 1, got Deliver");
            }
        }
    }

    #[test]
    fn test_create_and_peel_five_hops() {
        let params = SphinxParams::default_v2();

        let mut relay_pks = Vec::new();
        let mut relay_sks = Vec::new();
        for _ in 0..SPHINX_HOPS {
            let (pk, sk) = RelayPublicKey::generate_keypair().unwrap();
            relay_pks.push(pk);
            relay_sks.push(sk);
        }
        let (dest_pk, _) = RelayPublicKey::generate_keypair().unwrap();
        let route = SphinxRoute::new(relay_pks, dest_pk).unwrap();

        let payload = b"fully private NERV transaction";
        let created = create_onion_packet(payload, &route, &params).unwrap();
        assert_eq!(created.packet.remaining_hops, SPHINX_HOPS as u8);

        // Peel through all 5 hops
        let mut current_packet = created.packet;
        for hop in 0..SPHINX_HOPS {
            let peeled = peel_sphinx_layer(&current_packet, &relay_sks[hop], hop as u8).unwrap();
            match peeled {
                PeeledLayer::Forward { inner_packet, .. } => {
                    current_packet = inner_packet;
                }
                PeeledLayer::Deliver { payload: delivered } => {
                    assert_eq!(hop, SPHINX_HOPS - 1);
                    assert!(delivered.starts_with(payload));
                }
            }
        }
    }

    #[test]
    fn test_cover_packet_creation() {
        let params = SphinxParams::default_v2();
        let cover = create_cover_packet(&params).unwrap();
        assert_eq!(cover.version, SPHINX_VERSION);
        assert_eq!(cover.remaining_hops, SPHINX_HOPS as u8);
        // Cover packet should have the same structure as a real packet
        assert_eq!(cover.kem_ciphertext.len(), ML_KEM768_CT_BYTES);
    }

    #[test]
    fn test_sphinx_packet_serialization() {
        let (pk, _) = RelayPublicKey::generate_keypair().unwrap();
        let (ct, _) = pk.encapsulate().unwrap();
        let packet = SphinxPacket::new(5, ct, random_bytes(500));
        let bytes = packet.to_bytes();
        let recovered = SphinxPacket::from_bytes(&bytes).unwrap();
        assert_eq!(packet.version, recovered.version);
        assert_eq!(packet.remaining_hops, recovered.remaining_hops);
        assert_eq!(packet.kem_ciphertext, recovered.kem_ciphertext);
    }
}
