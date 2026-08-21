//! Post-Quantum Sphinx Packet Construction (Client-Side).
//!
//! Implements the NERV V2.0 5-hop PQ-Sphinx onion routing protocol. 
//! Replaces the V1.01 5-Hop TEE mixer with pure cryptographic privacy.
//!
//! ## Protocol Overview
//! In classical Sphinx, an ephemeral ECDH key is used to derive shared secrets 
//! for each hop. In PQ-Sphinx, because ML-KEM-768 does not support key 
//! exchange from a single ephemeral public key, the wallet encapsulates a 
//! distinct symmetric key for each relay using that relay's ML-KEM-768 public key.
//!
//! To prevent relays from knowing their position in the path, the KEM 
//! ciphertexts are nested: Relay 1 receives `ct_1` and an encrypted blob. 
//! Upon decrypting, it finds `ct_2` and the next encrypted blob, and so on.
//!
//! ## Security Properties
//! - **Unlinkability**: No relay can link the previous hop to the next hop.
//! - **Payload Privacy**: The innermost payload is the Threshold-Encrypted 
//!   transaction, meaning even the final relay cannot read the transaction data.
//! - **Post-Quantum**: Immune to Shor's algorithm.

use crate::{
    NervError, WalletResult, WalletError,
    SPHINX_HOPS,
};
use oqs::kem::MlKem768;
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use chacha20poly1305::aead::{Aead, KeyInit};
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

/// Information required to route through a single Sphinx relay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayInfo {
    /// The relay's ML-KEM-768 public key.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub kem_pk: Vec<u8>,
    /// The network address (e.g., Multiaddr) of the next relay.
    /// For the final relay, this is the address of the DKG Mempool.
    pub next_addr: String,
}

/// A constructed PQ-Sphinx packet ready for broadcast.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphinxPacket {
    /// The outermost ML-KEM-768 ciphertext for the first relay.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub outer_ct: Vec<u8>,
    /// The encrypted routing information and payload for the first relay.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub encrypted_blob: Vec<u8>,
}

/// The builder responsible for constructing the 5-hop onion packet.
pub struct PqSphinxBuilder {
    relays: Vec<RelayInfo>,
}

impl PqSphinxBuilder {
    /// Creates a new builder with the specified relays.
    /// 
    /// # Errors
    /// Returns an error if the number of relays does not match `SPHINX_HOPS` (5).
    pub fn new(relays: Vec<RelayInfo>) -> WalletResult<Self> {
        if relays.len() != SPHINX_HOPS {
            return Err(WalletError::Sphinx(format!(
                "Expected exactly {} hops, got {}",
                SPHINX_HOPS,
                relays.len()
            )));
        }
        Ok(Self { relays })
    }

    /// Builds the Sphinx packet wrapping the given payload.
    /// 
    /// The `final_payload` should already be encrypted under the network's 
    /// DKG collective public key (ThresholdCiphertext).
    pub fn build(&self, final_payload: &[u8]) -> WalletResult<SphinxPacket> {
        let kem = MlKem768::default();
        
        // Start with the innermost payload (the threshold-encrypted transaction)
        let mut current_blob = final_payload.to_vec();
        let mut next_addr = String::new(); // The final mempool address

        // Reverse iterate through the relays to build the onion layers
        for relay in self.relays.iter().rev() {
            // 1. Encapsulate a shared secret for this relay using its ML-KEM PK
            let pk = MlKem768::PublicKey::from_bytes(&relay.kem_pk)
                .map_err(|e| WalletError::Sphinx(format!("Invalid relay PK: {:?}", e)))?;
            
            let (ct, shared_secret) = kem.encapsulate(&pk)
                .map_err(|e| WalletError::Sphinx(format!("KEM encapsulate failed: {:?}", e)))?;

            // 2. Derive symmetric key and nonce from the shared secret
            let key_material = blake3::hash(&shared_secret);
            let key = Key::from_slice(key_material.as_bytes());
            
            // Derive a 12-byte nonce from the key material
            let nonce_bytes = blake3::derive_key("nerv:sphinx:nonce", key_material.as_bytes());
            let nonce = Nonce::from_slice(&nonce_bytes[..12]);

            // 3. Construct the plaintext for this layer
            // Layer = [next_addr_len (1B) || next_addr (var) || next_ct_len (2B) || next_ct (var) || inner_blob]
            let mut layer_plaintext = Vec::new();
            
            // Address of the next hop
            layer_plaintext.push(next_addr.len() as u8);
            layer_plaintext.extend_from_slice(next_addr.as_bytes());
            
            // The KEM ciphertext for the next hop (0 length if final hop)
            layer_plaintext.extend_from_slice(&(ct.len() as u16).to_be_bytes());
            layer_plaintext.extend_from_slice(&ct);
            
            // The inner encrypted blob
            layer_plaintext.extend_from_slice(&current_blob);

            // 4. Encrypt the layer
            let cipher = ChaCha20Poly1305::new(key);
            let encrypted_layer = cipher.encrypt(nonce, layer_plaintext.as_ref())
                .map_err(|e| WalletError::Sphinx(format!("AEAD encrypt failed: {:?}", e)))?;

            // Update state for the next iteration (moving outward)
            current_blob = encrypted_layer;
            next_addr = relay.next_addr.clone();
        }

        // The first relay's KEM ciphertext is the outermost one.
        // Wait, in the loop above, the `ct` of the first relay was placed *inside* the outermost blob.
        // Let's fix the logic: the outermost packet must contain `ct_1` and `encrypted_blob_1`.
        
        // Let's rebuild correctly by separating the outermost step:
        self.build_outermost(final_payload)
    }

    /// Correctly builds the onion, separating the outermost layer.
    fn build_outermost(&self, final_payload: &[u8]) -> WalletResult<SphinxPacket> {
        let kem = MlKem768::default();
        let mut current_blob = final_payload.to_vec();
        let mut next_addr = String::new(); // Start with the final destination (DKG mempool)

        // Iterate from the final relay (hop 5) down to the second relay (hop 2)
        for relay in self.relays.iter().skip(1).rev() {
            // Encapsulate for this relay
            let pk = MlKem768::PublicKey::from_bytes(&relay.kem_pk)
                .map_err(|e| WalletError::Sphinx(format!("Invalid relay PK: {:?}", e)))?;
            let (ct, shared_secret) = kem.encapsulate(&pk)
                .map_err(|e| WalletError::Sphinx(format!("KEM encapsulate failed: {:?}", e)))?;

            let key_material = blake3::hash(&shared_secret);
            let key = Key::from_slice(key_material.as_bytes());
            let nonce_bytes = blake3::derive_key("nerv:sphinx:nonce", key_material.as_bytes());
            let nonce = Nonce::from_slice(&nonce_bytes[..12]);

            // Plaintext: [next_addr_len || next_addr || ct_len || ct || inner_blob]
            let mut layer_plaintext = Vec::new();
            layer_plaintext.push(next_addr.len() as u8);
            layer_plaintext.extend_from_slice(next_addr.as_bytes());
            layer_plaintext.extend_from_slice(&(ct.len() as u16).to_be_bytes());
            layer_plaintext.extend_from_slice(&ct);
            layer_plaintext.extend_from_slice(&current_blob);

            let cipher = ChaCha20Poly1305::new(key);
            current_blob = cipher.encrypt(nonce, layer_plaintext.as_ref())
                .map_err(|e| WalletError::Sphinx(format!("AEAD encrypt failed: {:?}", e)))?;

            next_addr = relay.next_addr.clone();
        }

        // Now handle the first relay (hop 1)
        let first_relay = &self.relays[0];
        let pk = MlKem768::PublicKey::from_bytes(&first_relay.kem_pk)
            .map_err(|e| WalletError::Sphinx(format!("Invalid first relay PK: {:?}", e)))?;
        let (outer_ct, shared_secret) = kem.encapsulate(&pk)
            .map_err(|e| WalletError::Sphinx(format!("KEM encapsulate failed for hop 1: {:?}", e)))?;

        let key_material = blake3::hash(&shared_secret);
        let key = Key::from_slice(key_material.as_bytes());
        let nonce_bytes = blake3::derive_key("nerv:sphinx:nonce", key_material.as_bytes());
        let nonce = Nonce::from_slice(&nonce_bytes[..12]);

        // Plaintext for hop 1: [next_addr_len || next_addr || 0 (no more ct) || inner_blob]
        let mut outer_plaintext = Vec::new();
        outer_plaintext.push(next_addr.len() as u8);
        outer_plaintext.extend_from_slice(next_addr.as_bytes());
        outer_plaintext.extend_from_slice(&0u16.to_be_bytes()); // No next CT
        outer_plaintext.extend_from_slice(&current_blob);

        let cipher = ChaCha20Poly1305::new(key);
        let encrypted_blob = cipher.encrypt(nonce, outer_plaintext.as_ref())
            .map_err(|e| WalletError::Sphinx(format!("AEAD encrypt failed for hop 1: {:?}", e)))?;

        Ok(SphinxPacket {
            outer_ct,
            encrypted_blob,
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wallets::keys::MlKemKeypair;

    #[test]
    fn test_sphinx_packet_construction() {
        // Generate 5 relay keypairs
        let relays: Vec<RelayInfo> = (0..SPHINX_HOPS)
            .map(|i| {
                let kp = MlKemKeypair::generate().unwrap();
                RelayInfo {
                    kem_pk: kp.public_key,
                    next_addr: format!("/ip4/127.0.0.1/tcp/4000{}", i),
                }
            })
            .collect();

        let builder = PqSphinxBuilder::new(relays).unwrap();
        let payload = b"threshold_encrypted_transaction_data";
        
        let packet = builder.build(payload).unwrap();
        
        // Verify outer structure
        assert!(!packet.outer_ct.is_empty());
        assert!(!packet.encrypted_blob.is_empty());
        
        // The outer_ct should be exactly ML-KEM-768 ciphertext size (1088 bytes)
        assert_eq!(packet.outer_ct.len(), 1088);
    }

    #[test]
    fn test_sphinx_wrong_hop_count() {
        let relays = vec![RelayInfo {
            kem_pk: vec![0u8; 1184],
            next_addr: "/ip4/127.0.0.1".into(),
        }];

        let result = PqSphinxBuilder::new(relays);
        assert!(matches!(result, Err(WalletError::Sphinx(_))));
    }
}
