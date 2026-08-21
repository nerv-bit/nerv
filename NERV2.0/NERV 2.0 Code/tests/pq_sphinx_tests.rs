//! Unit and Integration tests for the PQ-Sphinx Mixnet.
//!
//! Validates the NERV V2.0 5-hop Post-Quantum Sphinx packet format.
//! Ensures that the onion packet is constructed correctly, that the
//! layered ML-KEM-768 encapsulation holds, and that simulated relays
//! can successfully "peel" their specific cryptographic layer to route
//! the packet to the final DKG mempool without gaining any knowledge
//! of the payload or the final destination until the final hop.

use nerv::wallets::sphinx_builder::{PqSphinxBuilder, RelayInfo, SphinxPacket};
use nerv::wallets::keys::MlKemKeypair;
use nerv::{SPHINX_HOPS, ML_KEM768_CT_BYTES};
use oqs::kem::MlKem768;
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use chacha20poly1305::aead::{Aead, KeyInit};

/// Simulates a single relay peeling one layer of the Sphinx onion packet.
/// 
/// Returns the next address, the next KEM ciphertext (if any), and the
/// remaining encrypted blob for the next hop.
fn simulate_relay_peel(
    ct: &[u8],
    encrypted_blob: &[u8],
    relay_sk: &[u8],
) -> (String, Vec<u8>, Vec<u8>) {
    let kem = MlKem768::default();
    let sk = MlKem768::SecretKey::from_bytes(relay_sk).unwrap();
    let ct_obj = MlKem768::Ciphertext::from_bytes(ct).unwrap();
    
    // 1. Decapsulate to get the shared secret
    let shared_secret = kem.decapsulate(&sk, &ct_obj).unwrap();
    
    // 2. Derive symmetric key and nonce
    let key_material = blake3::hash(&shared_secret);
    let key = Key::from_slice(key_material.as_bytes());
    let nonce_bytes = blake3::derive_key("nerv:sphinx:nonce", key_material.as_bytes());
    let nonce = Nonce::from_slice(&nonce_bytes[..12]);
    
    // 3. Decrypt the layer
    let cipher = ChaCha20Poly1305::new(key);
    let plaintext = cipher.decrypt(nonce, encrypted_blob.as_ref()).unwrap();
    
    // 4. Parse the peeled plaintext: [next_addr_len (1B) || next_addr || next_ct_len (2B) || next_ct || inner_blob]
    let mut idx = 0;
    let addr_len = plaintext[idx] as usize;
    idx += 1;
    
    let next_addr = String::from_utf8(plaintext[idx..idx+addr_len].to_vec()).unwrap();
    idx += addr_len;
    
    let next_ct_len = u16::from_be_bytes([plaintext[idx], plaintext[idx+1]]) as usize;
    idx += 2;
    
    let next_ct = plaintext[idx..idx+next_ct_len].to_vec();
    idx += next_ct_len;
    
    let inner_blob = plaintext[idx..].to_vec();
    
    (next_addr, next_ct, inner_blob)
}

#[test]
fn test_sphinx_packet_construction_5_hops() {
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
    
    // Build the 5-hop onion packet
    let packet = builder.build(payload).unwrap();
    
    // The outermost KEM ciphertext must be exactly ML-KEM-768 size (1088 bytes)
    assert_eq!(packet.outer_ct.len(), ML_KEM768_CT_BYTES, "Outer CT must match ML-KEM-768 size");
    assert!(!packet.encrypted_blob.is_empty(), "Encrypted blob must not be empty");
}

#[test]
fn test_sphinx_builder_rejects_wrong_hop_count() {
    // Provide only 3 relays instead of 5
    let relays = vec![
        RelayInfo { kem_pk: vec![0u8; 1184], next_addr: "/ip4/1".into() },
        RelayInfo { kem_pk: vec![0u8; 1184], next_addr: "/ip4/2".into() },
        RelayInfo { kem_pk: vec![0u8; 1184], next_addr: "/ip4/3".into() },
    ];

    let result = PqSphinxBuilder::new(relays);
    
    // Must fail validation because NERV V2.0 mandates exactly 5 hops
    assert!(result.is_err(), "Builder must reject incorrect hop counts");
}

#[test]
fn test_sphinx_5_hop_routing_and_peeling() {
    // Generate 5 relay keypairs and retain their secret keys for peeling
    let mut relay_infos = Vec::new();
    let mut relay_sks = Vec::new();
    
    for i in 0..SPHINX_HOPS {
        let kp = MlKemKeypair::generate().unwrap();
        relay_sks.push(kp.secret_key.clone());
        relay_infos.push(RelayInfo {
            kem_pk: kp.public_key,
            next_addr: format!("/ip4/127.0.0.1/tcp/4000{}", i),
        });
    }

    let builder = PqSphinxBuilder::new(relay_infos.clone()).unwrap();
    let final_payload = b"DKG_ENCRYPTED_TX_PAYLOAD_12345";
    
    // 1. Construct the packet
    let packet = builder.build(final_payload).unwrap();
    
    let mut current_ct = packet.outer_ct.clone();
    let mut current_blob = packet.encrypted_blob.clone();
    
    // 2. Simulate the 5 relays peeling the packet
    for i in 0..SPHINX_HOPS {
        let (next_addr, next_ct, inner_blob) = simulate_relay_peel(&current_ct, &current_blob, &relay_sks[i]);
        
        // Verify the routing address matches what we configured
        if i < SPHINX_HOPS - 1 {
            assert_eq!(next_addr, relay_infos[i + 1].next_addr, "Relay must learn the correct next address");
            assert!(!next_ct.is_empty(), "Intermediate hops must contain a next CT");
        } else {
            // The final hop should point to the DKG mempool (simulated as empty string in builder)
            assert_eq!(next_addr, "", "Final hop should point to the final destination");
            assert!(next_ct.is_empty(), "Final hop should have no next CT");
        }
        
        current_ct = next_ct;
        current_blob = inner_blob;
    }
    
    // 3. Verify the final payload extracted by the last relay
    assert_eq!(current_blob, final_payload.to_vec(), "Final payload must exactly match the original input");
}

#[test]
fn test_sphinx_tampered_outer_ct_fails() {
    let mut relay_infos = Vec::new();
    let mut relay_sks = Vec::new();
    
    for i in 0..SPHINX_HOPS {
        let kp = MlKemKeypair::generate().unwrap();
        relay_sks.push(kp.secret_key.clone());
        relay_infos.push(RelayInfo {
            kem_pk: kp.public_key,
            next_addr: format!("/ip4/127.0.0.1/tcp/4000{}", i),
        });
    }

    let builder = PqSphinxBuilder::new(relay_infos).unwrap();
    let packet = builder.build(b"payload").unwrap();
    
    // Tamper with the outermost ciphertext
    let mut tampered_ct = packet.outer_ct.clone();
    tampered_ct[0] ^= 0xFF;
    
    // Relay 1 attempts to peel the tampered packet
    let kem = MlKem768::default();
    let sk = MlKem768::SecretKey::from_bytes(&relay_sks[0]).unwrap();
    let ct_obj = MlKem768::Ciphertext::from_bytes(&tampered_ct).unwrap();
    
    // Decapsulation must fail (or produce a wrong key causing AEAD failure)
    // In PQ-Sphinx, a tampered KEM CT will either fail decapsulation or fail AEAD decryption.
    let result = kem.decapsulate(&sk, &ct_obj);
    
    // Depending on the KEM implementation, decapsulation might "succeed" but yield the wrong secret,
    // causing the subsequent AEAD decryption to fail. We test the AEAD failure here.
    if let Ok(shared_secret) = result {
        let key_material = blake3::hash(&shared_secret);
        let key = Key::from_slice(key_material.as_bytes());
        let nonce_bytes = blake3::derive_key("nerv:sphinx:nonce", key_material.as_bytes());
        let nonce = Nonce::from_slice(&nonce_bytes[..12]);
        
        let cipher = ChaCha20Poly1305::new(key);
        let decrypt_result = cipher.decrypt(nonce, packet.encrypted_blob.as_ref());
        
        assert!(decrypt_result.is_err(), "AEAD decryption must fail for tampered KEM ciphertext");
    }
}

