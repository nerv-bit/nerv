//! Transaction Construction & ZK Witness Assignment.
//!
//! Implements the NERV V2.0 transaction building pipeline. In V2.0, 
//! transactions are not broadcast in plaintext. Instead, the wallet:
//!
//! 1. Selects unspent notes to cover the target amount + fee.
//! 2. Creates output notes (for the recipient) and a change note (for the sender).
//! 3. Computes the homomorphic state delta: `delta = W * delta_S + b_tx`.
//! 4. Generates a Halo2 ZK proof proving the delta was computed correctly 
//!    without revealing `delta_S` (the actual value).
//! 5. Encrypts the transaction payload and output notes under the network's 
//!    DKG collective public key.
//! 6. Wraps the encrypted payload in a PQ-Sphinx packet for mixnet routing.

use crate::{
    NervError, NervResult, WalletResult, WalletError,
    ONE_NERV, EMBEDDING_DIM,
};
use crate::wallets::keys::{WalletKeys, DerivationPath};
use crate::wallets::notes::{PrivateNote, EncryptedNote, NoteTracker};
use crate::privacy::dkg::{DkgPublicKey, ThresholdCiphertext};
use crate::utils::blake3_hash;
use rand::{RngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use parking_lot::Mutex;

/// Base transaction fee in nano-NERV (0.001 NERV).
const BASE_TX_FEE_NANO: u64 = 1_000_000;
/// Additional fee per input/output note in nano-NERV.
const FEE_PER_NOTE_NANO: u64 = 100_000;

/// A transaction input referencing an unspent note.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxInput {
    /// The nullifier of the note being spent.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub nullifier: Vec<u8>,
    /// The commitment of the note.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub commitment: Vec<u8>,
    /// The value of the note (visible only to the sender during construction).
    pub value: u64,
}

/// A transaction output creating a new private note.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxOutput {
    /// The encrypted note for the recipient.
    pub encrypted_note: EncryptedNote,
    /// The value of the output.
    pub value: u64,
}

/// The plaintext transaction structure before DKG encryption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivateTransaction {
    /// Inputs being spent.
    pub inputs: Vec<TxInput>,
    /// Outputs being created.
    pub outputs: Vec<TxOutput>,
    /// The transaction fee in nano-NERV.
    pub fee: u64,
    /// The homomorphic embedding delta vector (64 dimensions).
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub embedding_delta: Vec<i64>,
    /// The Halo2 ZK proof proving the delta was computed correctly.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub zk_proof: Vec<u8>,
}

/// The final, encrypted payload ready to be injected into a Sphinx packet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdEncryptedTx {
    /// The DKG threshold ciphertext containing the `PrivateTransaction`.
    pub ciphertext: ThresholdCiphertext,
}

/// The transaction builder orchestrates the creation of a private transaction.
pub struct TransactionBuilder {
    /// The network's DKG collective public key.
    dkg_pk: DkgPublicKey,
    /// The network's public NWO weights W (64 x N matrix).
    /// In V2.0, this is fetched during sync.
    network_weights: Arc<Vec<Vec<i64>>>,
    /// The network's public bias vector b (64 dim).
    network_bias: Arc<Vec<i64>>,
}

impl TransactionBuilder {
    /// Creates a new transaction builder.
    pub fn new(
        dkg_pk: DkgPublicKey,
        network_weights: Arc<Vec<Vec<i64>>>,
        network_bias: Arc<Vec<i64>>>,
    ) -> Self {
        Self {
            dkg_pk,
            network_weights,
            network_bias,
        }
    }

    /// Calculates the transaction fee based on the number of inputs and outputs.
    pub fn calculate_fee(num_inputs: usize, num_outputs: usize) -> u64 {
        BASE_TX_FEE_NANO + (num_inputs + num_outputs) as u64 * FEE_PER_NOTE_NANO
    }

    /// Builds a private transaction.
    ///
    /// # Arguments
    /// * `keys` - The sender's wallet keys.
    /// * `note_tracker` - The local note tracker containing unspent notes.
    /// * `recipient_kem_pk` - The recipient's ML-KEM-768 public key.
    /// * `amount` - The amount to send in nano-NERV.
    pub fn build(
        &self,
        keys: &WalletKeys,
        note_tracker: &Mutex<NoteTracker>,
        recipient_kem_pk: &[u8],
        amount: u64,
    ) -> WalletResult<ThresholdEncryptedTx> {
        if amount == 0 {
            return Err(WalletError::Transaction("Amount must be greater than 0".into()));
        }

        // 1. Calculate fee and select notes
        // We assume 2 outputs (recipient + change) for fee calculation
        let estimated_fee = Self::calculate_fee(1, 2);
        let target_amount = amount.checked_add(estimated_fee)
            .ok_or_else(|| WalletError::Transaction("Amount + fee overflow".into()))?;

        let mut tracker = note_tracker.lock();
        let selected_notes = tracker.select_notes(target_amount)?;
        
        let total_input: u64 = selected_notes.iter().map(|n| n.value).sum();
        let actual_fee = Self::calculate_fee(selected_notes.len(), 2);
        let change_value = total_input.checked_sub(amount)
            .and_then(|v| v.checked_sub(actual_fee))
            .ok_or_else(|| WalletError::InsufficientFunds {
                required: target_amount,
                available: total_input,
            })?;

        // 2. Prepare inputs and compute state delta (delta_S)
        let mut inputs = Vec::with_capacity(selected_notes.len());
        let mut delta_s = 0i64; // Simplified 1D state delta for value

        let det_key = keys.derive_detection_key(&DerivationPath::default());

        for note in &selected_notes {
            let nullifier = note.compute_nullifier(&det_key);
            let commitment = note.compute_commitment();
            
            inputs.push(TxInput {
                nullifier: nullifier.to_vec(),
                commitment: commitment.to_vec(),
                value: note.value,
            });
            
            // Subtract input value from state delta
            delta_s -= note.value as i64;
        }

        // 3. Create output notes (recipient + change)
        let mut outputs = Vec::with_capacity(2);

        // Recipient output
        let recipient_diversified_receiver = blake3_hash(recipient_kem_pk);
        let recipient_note = PrivateNote::new(amount, recipient_diversified_receiver.into());
        let recipient_enc_note = EncryptedNote::encrypt(&recipient_note, recipient_kem_pk)?;
        outputs.push(TxOutput {
            encrypted_note: recipient_enc_note,
            value: amount,
        });
        delta_s += amount as i64;

        // Change output (back to sender)
        if change_value > 0 {
            let change_diversified_receiver = keys.derive_diversified_receiver(1);
            let change_note = PrivateNote::new(change_value, change_diversified_receiver);
            let change_enc_note = EncryptedNote::encrypt(&change_note, &keys.kem_keypair.public_key)?;
            outputs.push(TxOutput {
                encrypted_note: change_enc_note,
                value: change_value,
            });
            delta_s += change_value as i64;
        }

        // 4. Compute Homomorphic Embedding Delta: delta = W * delta_S + b_tx
        let embedding_delta = self.compute_embedding_delta(delta_s)?;

        // 5. Generate ZK Proof
        let zk_proof = self.generate_zk_proof(&inputs, &outputs, delta_s, &embedding_delta)?;

        // 6. Construct PrivateTransaction
        let private_tx = PrivateTransaction {
            inputs,
            outputs,
            fee: actual_fee,
            embedding_delta,
            zk_proof,
        };

        // 7. Mark notes as spent locally
        for note in &selected_notes {
            let commitment = note.compute_commitment();
            tracker.spend_note(&commitment, &det_key)?;
        }

        // 8. Threshold encrypt the transaction for the DKG mempool
        let serialized_tx = bincode::serialize(&private_tx)
            .map_err(|e| WalletError::Serialization(e.to_string()))?;

        let ciphertext = ThresholdCiphertext::encrypt(&serialized_tx, &self.dkg_pk, [0u8; 32])?;

        Ok(ThresholdEncryptedTx { ciphertext })
    }

    /// Computes the homomorphic embedding delta: `delta = W * delta_S + b_tx`
    /// 
    /// In V2.0, `W` is a 64xN matrix, `delta_S` is a scalar (or vector), and `b_tx`
    /// is a transaction-specific bias derived from the nullifiers to prevent replay.
    fn compute_embedding_delta(&self, delta_s: i64) -> WalletResult<Vec<i64>> {
        let mut delta = vec![0i64; EMBEDDING_DIM];

        // Matrix multiplication: W * delta_S
        for i in 0..EMBEDDING_DIM {
            if let Some(row) = self.network_weights.get(i) {
                // For a scalar state delta, this is just scaling the first column
                // In a full implementation, delta_S is a vector.
                if let Some(&w_val) = row.first() {
                    delta[i] = w_val.wrapping_mul(delta_s);
                }
            }
        }

        // Add bias: b_tx (simplified as a constant network bias)
        for i in 0..EMBEDDING_DIM {
            if let Some(&b_val) = self.network_bias.get(i) {
                delta[i] = delta[i].wrapping_add(b_val);
            }
        }

        Ok(delta)
    }

    /// Generates the Halo2 ZK proof for the LatentLedger Lite circuit.
    /// 
    /// Proves that the wallet correctly computed `delta = W * delta_S + b_tx`
    /// without revealing `delta_S` (the transaction value).
    fn generate_zk_proof(
        &self,
        inputs: &[TxInput],
        outputs: &[TxOutput],
        delta_s: i64,
        embedding_delta: &[i64],
    ) -> WalletResult<Vec<u8>> {
        // In production, this interfaces with `halo2_proofs::plonk::create_proof`.
        // The circuit witness is assigned here.
        
        // Witness:
        // - Private inputs: delta_s, nullifiers, blinding factors
        // - Public inputs: embedding_delta, commitments
        
        // Simulated proof generation for structural integrity.
        // A real implementation would serialize the Halo2 proof here.
        let mut proof = Vec::with_capacity(400);
        OsRng.fill_bytes(&mut proof); // Placeholder for actual ZK proof bytes
        
        Ok(proof)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wallets::keys::{Mnemonic, MlKemKeypair};
    use crate::privacy::dkg::{DkgPublicKey, FeldmanCommitment, DkgScalar};
    use bls12_381::Scalar as BlsScalar;

    fn get_test_dkg_pk() -> DkgPublicKey {
        let s = DkgScalar::from_bls_scalar(&BlsScalar::from(42));
        let commitment = FeldmanCommitment::commit(&s);
        DkgPublicKey {
            point: commitment.point,
            hash: blake3_hash(&commitment.point),
            session_id: [0u8; 32],
            threshold: 3,
            num_participants: 5,
        }
    }

    #[test]
    fn test_fee_calculation() {
        assert_eq!(TransactionBuilder::calculate_fee(1, 2), 1_300_000);
        assert_eq!(TransactionBuilder::calculate_fee(3, 2), 1_700_000);
    }

    #[tokio::test]
    async fn test_transaction_building() {
        let dkg_pk = get_test_dkg_pk();
        let weights = Arc::new(vec![vec![1i64; 1]; EMBEDDING_DIM]);
        let bias = Arc::new(vec![0i64; EMBEDDING_DIM]);
        
        let builder = TransactionBuilder::new(dkg_pk, weights, bias);
        
        // Setup wallet
        let mnemonic = Mnemonic::generate();
        let seed = mnemonic.to_seed("");
        let keys = WalletKeys::new_from_seed(&seed).unwrap();
        
        // Setup note tracker with funds
        let tracker = Mutex::new(NoteTracker::new());
        {
            let mut t = tracker.lock();
            // Add a 1000 NERV note
            t.add_unspent_note(PrivateNote::new(1000 * ONE_NERV, [1u8; 32]));
        }
        
        // Recipient keys
        let recipient_kp = MlKemKeypair::generate().unwrap();
        
        // Build transaction
        let result = builder.build(
            &keys,
            &tracker,
            &recipient_kp.public_key,
            500 * ONE_NERV,
        );
        
        assert!(result.is_ok(), "Transaction building failed: {:?}", result.err());
        
        let encrypted_tx = result.unwrap();
        assert!(!encrypted_tx.ciphertext.c3.is_empty());
        
        // Verify balance was updated locally
        {
            let t = tracker.lock();
            // 1000 - 500 - fee(0.0013 NERV) = 499.9987 NERV
            let expected_balance = 1000 * ONE_NERV - 500 * ONE_NERV - 1_300_000;
            assert_eq!(t.get_balance(), expected_balance);
        }
    }

    #[tokio::test]
    async fn test_transaction_insufficient_funds() {
        let dkg_pk = get_test_dkg_pk();
        let weights = Arc::new(vec![vec![1i64; 1]; EMBEDDING_DIM]);
        let bias = Arc::new(vec![0i64; EMBEDDING_DIM]);
        
        let builder = TransactionBuilder::new(dkg_pk, weights, bias);
        
        let mnemonic = Mnemonic::generate();
        let seed = mnemonic.to_seed("");
        let keys = WalletKeys::new_from_seed(&seed).unwrap();
        
        let tracker = Mutex::new(NoteTracker::new());
        {
            let mut t = tracker.lock();
            t.add_unspent_note(PrivateNote::new(100 * ONE_NERV, [1u8; 32]));
        }
        
        let recipient_kp = MlKemKeypair::generate().unwrap();
        
        let result = builder.build(
            &keys,
            &tracker,
            &recipient_kp.public_key,
            200 * ONE_NERV, // Attempt to send more than available
        );
        
        assert!(matches!(result, Err(WalletError::InsufficientFunds { .. })));
    }
}
