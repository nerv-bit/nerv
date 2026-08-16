// src/balance.rs
// Epic 2: Private Note Detection and Local Balance Computation
// Complete implementation with real-time balance tracking and scanning

use crate::keys::AccountKeys;
use crate::types::{Note, Output, Nullifier};
use nerv::crypto::ml_kem::MlKem768;
use nerv::embedding::{HomomorphicDelta, EmbeddingHash};
use hkdf::Hkdf;
use sha2::Sha256;
use aes_gcm::{Aes256Gcm, KeyInit, aead::Aead};
use zeroize::Zeroize;
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use thiserror::Error;
use tokio::sync::RwLock;
use std::sync::Arc;
use rayon::prelude::*;

// Superior UI Flow Description:
// 1. Balance display: Large, elegant typography with subtle glow effect
// 2. Real-time updates: Smooth counting animation when balance changes
// 3. Transaction list: Card-based design with expandable details
// 4. Search: Instant search with highlight animation
// 5. Filters: Beautiful tag-based filtering system
// 6. Export: Elegant PDF/CSV generation with custom branding

#[derive(Error, Debug)]
pub enum BalanceError {
    #[error("Note decryption failed")]
    DecryptionFailed,
    #[error("Invalid note format")]
    InvalidNote,
    #[error("Balance computation error")]
    ComputationError,
    #[error("Note scanning timeout")]
    ScanTimeout,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BalanceState {
    pub total: u128,
    pub spendable: u128,
    pub pending: u128,
    pub locked: u128,
    pub by_account: HashMap<u32, AccountBalance>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AccountBalance {
    pub account_index: u32,
    pub total: u128,
    pub unspent_notes: Vec<Note>,
    pub spent_nullifiers: HashSet<Nullifier>,
}

pub struct BalanceTracker {
    state: Arc<RwLock<BalanceState>>,
    notes: Arc<RwLock<HashMap<Nullifier, Note>>>,
    spent_nullifiers: Arc<RwLock<HashSet<Nullifier>>>,
    scan_progress: Arc<RwLock<ScanProgress>>,
}

#[derive(Clone)]
pub struct ScanProgress {
    pub total_outputs: usize,
    pub scanned: usize,
    pub current_height: u64,
    pub target_height: u64,
    pub estimated_time_remaining: std::time::Duration,
    pub active_scans: usize,
}

impl BalanceTracker {
    /// Create new balance tracker with empty state
    /// UI: Empty state with "Welcome" animation
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(BalanceState {
                total: 0,
                spendable: 0,
                pending: 0,
                locked: 0,
                by_account: HashMap::new(),
                last_updated: chrono::Utc::now(),
            })),
            notes: Arc::new(RwLock::new(HashMap::new())),
            spent_nullifiers: Arc::new(RwLock::new(HashSet::new())),
            scan_progress: Arc::new(RwLock::new(ScanProgress {
                total_outputs: 0,
                scanned: 0,
                current_height: 0,
                target_height: 0,
                estimated_time_remaining: std::time::Duration::from_secs(0),
                active_scans: 0,
            })),
        }
    }
    
    /// Scan outputs in parallel with progress tracking
    /// UI: Beautiful progress bar with gradient animation
    pub async fn scan_outputs(
        &self,
        outputs: &[Output],
        account_keys: &[&AccountKeys],
        height: u64,
    ) -> Result<Vec<Note>, BalanceError> {
        let mut scan_progress = self.scan_progress.write().await;
        scan_progress.active_scans += 1;
        scan_progress.total_outputs = outputs.len();
        scan_progress.current_height = height;
        drop(scan_progress);
        
        // Parallel scanning for performance
        let new_notes: Vec<Note> = outputs
            .par_iter()
            .filter_map(|output| {
                self.try_decrypt_output(output, account_keys).ok()
            })
            .collect();
        
        // Update state with new notes
        let mut notes = self.notes.write().await;
        let mut spent_nullifiers = self.spent_nullifiers.write().await;
        let mut state = self.state.write().await;
        
        for note in &new_notes {
            // Check if already spent
            if spent_nullifiers.contains(&note.nullifier) {
                continue;
            }
            
            notes.insert(note.nullifier, note.clone());
            
            // Update account balance
            let account_balance = state.by_account
                .entry(note.account_index)
                .or_insert_with(|| AccountBalance {
                    account_index: note.account_index,
                    total: 0,
                    unspent_notes: vec![],
                    spent_nullifiers: HashSet::new(),
                });
            
            account_balance.total += note.amount;
            account_balance.unspent_notes.push(note.clone());
            state.total += note.amount;
            state.spendable += note.amount;
        }
        
        state.last_updated = chrono::Utc::now();
        
        // Update scan progress
        let mut scan_progress = self.scan_progress.write().await;
        scan_progress.scanned += new_notes.len();
        scan_progress.active_scans -= 1;
        
        if scan_progress.active_scans == 0 {
            scan_progress.total_outputs = 0;
            scan_progress.scanned = 0;
        }
        
        Ok(new_notes)
    }
    
    /// Attempt to decrypt output with all account keys
    fn try_decrypt_output(
        &self,
        output: &Output,
        account_keys: &[&AccountKeys],
    ) -> Result<Note, BalanceError> {
        for keys in account_keys {
            if let Some(note_payload) = output.try_decrypt(&keys.enc_sk) {
                return Ok(Note {
                    amount: note_payload.amount,
                    memo: note_payload.memo,
                    nullifier: output.compute_nullifier(&keys.spending_sk),
                    received_height: output.height,
                    account_index: keys.index,
                    commitment: keys.commitment,
                    encrypted_data: output.encrypted_payload.clone(),
                    created_at: chrono::Utc::now(),
                    confirmation_status: ConfirmationStatus::Confirmed,
                });
            }
        }
        
        Err(BalanceError::DecryptionFailed)
    }
    
    /// Get current balance state
    /// UI: Balance card with animated number transitions
    pub async fn get_balance(&self) -> BalanceState {
        self.state.read().await.clone()
    }
    
    /// Get spendable notes for transaction construction
    /// UI: Note selection interface with visual balance indicators
    pub async fn get_spendable_notes(
        &self,
        min_amount: u128,
        account_index: Option<u32>,
    ) -> Vec<Note> {
        let state = self.state.read().await;
        let spent_nullifiers = self.spent_nullifiers.read().await;
        
        let mut spendable = Vec::new();
        let mut total = 0;
        
        for (_, account_balance) in state.by_account.iter() {
            if let Some(idx) = account_index {
                if account_balance.account_index != idx {
                    continue;
                }
            }
            
            for note in &account_balance.unspent_notes {
                if spent_nullifiers.contains(&note.nullifier) {
                    continue;
                }
                
                spendable.push(note.clone());
                total += note.amount;
                
                if total >= min_amount {
                    break;
                }
            }
            
            if total >= min_amount {
                break;
            }
        }
        
        spendable
    }
    
    /// Mark notes as spent (when broadcasting transaction)
    /// UI: Visual confirmation with note fading animation
    pub async fn spend_notes(&self, nullifiers: &[Nullifier]) {
        let mut spent_nullifiers = self.spent_nullifiers.write().await;
        let mut state = self.state.write().await;
        let mut notes = self.notes.write().await;
        
        for nullifier in nullifiers {
            if spent_nullifiers.insert(*nullifier) {
                if let Some(note) = notes.get(nullifier) {
                    // Update account balance
                    if let Some(account_balance) = state.by_account.get_mut(&note.account_index) {
                        account_balance.total = account_balance.total.saturating_sub(note.amount);
                        account_balance.unspent_notes.retain(|n| n.nullifier != *nullifier);
                        account_balance.spent_nullifiers.insert(*nullifier);
                    }
                    
                    state.total = state.total.saturating_sub(note.amount);
                    state.spendable = state.spendable.saturating_sub(note.amount);
                }
            }
        }
        
        state.last_updated = chrono::Utc::now();
    }
    
    /// Perform full rescan from specific height
    /// UI: Full-screen rescan with detailed progress and ETA
    pub async fn full_rescan(
        &self,
        start_height: u64,
        end_height: u64,
        account_keys: &[&AccountKeys],
        output_fetcher: impl Fn(u64) -> Vec<Output>,
    ) -> Result<ScanProgress, BalanceError> {
        let mut scan_progress = self.scan_progress.write().await;
        scan_progress.current_height = start_height;
        scan_progress.target_height = end_height;
        scan_progress.active_scans = 1;
        
        // Estimate total outputs
        let outputs_per_block = 100; // Conservative estimate
        let total_blocks = end_height.saturating_sub(start_height) as usize;
        scan_progress.total_outputs = total_blocks * outputs_per_block;
        
        drop(scan_progress);
        
        // Scan blocks in batches
        let batch_size = 100;
        for batch_start in (start_height..=end_height).step_by(batch_size) {
            let batch_end = std::cmp::min(batch_start + batch_size as u64, end_height);
            
            let mut batch_outputs = Vec::new();
            for height in batch_start..=batch_end {
                batch_outputs.extend(output_fetcher(height));
            }
            
            self.scan_outputs(&batch_outputs, account_keys, batch_end).await?;
            
            // Update progress
            let mut scan_progress = self.scan_progress.write().await;
            let scanned_percent = (batch_end - start_height) as f64 / (end_height - start_height) as f64;
            let elapsed = scan_progress.estimated_time_remaining;
            let remaining = std::time::Duration::from_secs_f64(
                elapsed.as_secs_f64() * (1.0 - scanned_percent) / scanned_percent.max(0.01)
            );
            scan_progress.estimated_time_remaining = remaining;
            scan_progress.current_height = batch_end;
        }
        
        let scan_progress = self.scan_progress.read().await.clone();
        Ok(scan_progress)
    }
    
    /// Get transaction history from notes
    /// UI: Timeline view with expandable transaction cards
    pub async fn get_history(
        &self,
        limit: Option<usize>,
        account_filter: Option<u32>,
    ) -> Vec<Note> {
        let notes = self.notes.read().await;
        let mut history: Vec<Note> = notes.values().cloned().collect();
        
        // Filter by account if specified
        if let Some(account_index) = account_filter {
            history.retain(|note| note.account_index == account_index);
        }
        
        // Sort by height (newest first)
        history.sort_by(|a, b| b.received_height.cmp(&a.received_height));
        
        // Apply limit
        if let Some(limit) = limit {
            history.truncate(limit);
        }
        
        history
    }
    
    /// Search notes by memo content
    /// UI: Instant search with highlight and animation
    pub async fn search_notes(&self, query: &str) -> Vec<Note> {
        let notes = self.notes.read().await;
        notes.values()
            .filter(|note| note.memo.to_lowercase().contains(&query.to_lowercase()))
            .cloned()
            .collect()
    }
    
    /// Export balance for tax/audit purposes
    /// UI: Beautiful export interface with format options
    pub async fn export_balance_report(
        &self,
        start_date: chrono::DateTime<chrono::Utc>,
        end_date: chrono::DateTime<chrono::Utc>,
    ) -> Result<BalanceReport, BalanceError> {
        let notes = self.notes.read().await;
        let state = self.state.read().await;
        
        let mut report = BalanceReport {
            period_start: start_date,
            period_end: end_date,
            generated_at: chrono::Utc::now(),
            total_balance: state.total,
            accounts: vec![],
            transactions: vec![],
        };
        
        // Filter notes by date
        for note in notes.values() {
            if note.created_at >= start_date && note.created_at <= end_date {
                report.transactions.push(note.clone());
            }
        }
        
        // Add account summaries
        for (_, account_balance) in &state.by_account {
            report.accounts.push(account_balance.clone());
        }
        
        Ok(report)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BalanceReport {
    pub period_start: chrono::DateTime<chrono::Utc>,
    pub period_end: chrono::DateTime<chrono::Utc>,
    pub generated_at: chrono::DateTime<chrono::Utc>,
    pub total_balance: u128,
    pub accounts: Vec<AccountBalance>,
    pub transactions: Vec<Note>,
}

impl Output {
    /// Attempt decryption with ML-KEM private key
    fn try_decrypt(&self, enc_sk: &[u8]) -> Option<NotePayload> {
        // Decapsulate shared secret
        let ss = MlKem768::decapsulate(enc_sk, &self.ciphertext).ok()?;
        
        // Derive symmetric key
        let mut sym_key = [0u8; 32];
        Hkdf::<Sha256>::new(None, &ss)
            .expand(b"nerv-note-symmetric-v2", &mut sym_key)
            .ok()?;
        
        // Decrypt payload
        let cipher = Aes256Gcm::new_from_slice(&sym_key).ok()?;
        let nonce = aes_gcm::Nonce::from_slice(&self.nonce);
        let plaintext = cipher.decrypt(nonce, self.encrypted_payload.as_ref()).ok()?;
        
        // Deserialize
        let payload: NotePayload = bincode::deserialize(&plaintext).ok()?;
        
        // Zeroize sensitive data
        sym_key.zeroize();
        
        Some(payload)
    }
    
    /// Compute nullifier for spending
    fn compute_nullifier(&self, spending_sk: &[u8]) -> Nullifier {
        let mut hasher = Blake3::new();
        hasher.update(spending_sk);
        hasher.update(&self.ciphertext);
        hasher.update(&self.nonce);
        hasher.finalize().into()
    }
}

#[derive(Serialize, Deserialize)]
struct NotePayload {
    amount: u128,
    memo: String,
    sender_commitment: [u8; 32],
    timestamp: chrono::DateTime<chrono::Utc>,
}
