// src/types.rs
//! Common types for wallet

use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Note {
    pub amount: u128,
    pub memo: String,
    pub nullifier: [u8; 32],
    pub received_height: u64,
    pub account_index: u32,
}

#[derive(Clone, Debug)]
pub struct Output {
    pub ct: Vec<u8>, // ML-KEM ciphertext
    pub encrypted_payload: Vec<u8>,
    pub nonce: [u8; 12],
    pub height: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transaction {
    pub inputs: Vec<[u8; 32]>, // Nullifiers
    pub outputs: Vec<Output>,
    pub delta: Vec<u8>, // 512-byte homomorphic delta
    pub signature: Vec<u8>,
    pub fee_proof: Vec<u8>,
}

#[derive(Clone, Debug)]
pub enum HistoryKind {
    Received,
    Sent,
}

#[derive(Clone, Debug)]
pub struct TransactionHistoryEntry {
    pub height: u64,
    pub timestamp: DateTime<Utc>,
    pub kind: HistoryKind,
    pub amount: u128,
    pub memos: Vec<String>,
    pub category: String,
    pub confirmed: bool,
}



pub type ReceivingAddress = String;
