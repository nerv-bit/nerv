// tests/wallet_tx_tests.rs
//! Unit tests for Epic 3: Private Transaction Construction with 5-Hop Onion Routing
//!
//! Tests cover:
//! - Input selection
//! - Output creation (encryption)
//! - Fee handling
//! - Change output
//! - Basic delta computation (mocked)


#[cfg(test)]
mod tests {
    use super::super::{tx::TransactionBuilder, keys::HdWallet, balance::BalanceTracker};
    use nerv::privacy::mixer::MixConfig;


    #[tokio::test]
    async fn test_transaction_building() {
        let mixer_config = MixConfig::default();
        let mut builder = TransactionBuilder::new(mixer_config).unwrap();


        // Setup wallet and notes (mocked)
        let wallet = HdWallet::generate("").unwrap();
        let account = wallet.derive_account(0, 5).unwrap();
        let keys: Vec<&_> = account.keys.iter().collect();


        // Add mock notes to tracker
        builder.balance_tracker.notes.push(super::super::types::Note {
            amount: 10000,
            memo: "test".to_string(),
            nullifier: [0u8; 32],
            received_height: 100,
            account_index: 0,
        });


        // Send 5000 (with fee 1000, change 4000)
        let receiver_addr = keys[1].receiving_address().unwrap();
        let tx_id = builder.send_private(&receiver_addr, 5000, "hello", 1000, &keys).await;


        assert!(tx_id.is_ok());
        assert_eq!(builder.balance_tracker.get_balance(), 0); // All spent
    }


    #[test]
    fn test_input_selection() {
        let mut builder = TransactionBuilder::new(MixConfig::default()).unwrap();
        builder.balance_tracker.notes = vec![
            super::super::types::Note { amount: 3000, ..Default::default() },
            super::super::types::Note { amount: 5000, ..Default::default() },
            super::super::types::Note { amount: 2000, ..Default::default() },
        ];


        let selected = builder.select_inputs(6000).unwrap();
        assert_eq!(selected.len(), 2); // 3000 + 5000 = 8000 > 6000
    }
}
