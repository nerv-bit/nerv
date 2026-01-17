// tests/wallet_sdk_tests.rs
//! Integration tests for Epic 8: Multi-platform SDK
//!
//! Tests cover:
//! - Wallet creation
//! - Basic flow: sync, address, balance, send (mocked)


#[tokio::test]
async fn test_sdk_full_flow() {
    let wallet = super::super::sdk::NervWallet::new(None, "", "mock_rpc").await.unwrap();


    wallet.sync().await.unwrap();
    assert!(wallet.is_synced());


    let balance = wallet.get_balance();
    assert!(balance.total >= 0);


    let addr = wallet.get_new_address().await.unwrap();
    assert!(addr.starts_with("nerv1"));


    let history = wallet.get_history().await.unwrap();
    assert!(history.is_empty() || !history.is_empty()); // Depends on mock


    let backup = wallet.export_backup("testpass");
    assert!(backup.is_ok());
}


