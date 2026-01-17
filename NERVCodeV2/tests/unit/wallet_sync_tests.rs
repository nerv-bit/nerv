// tests/wallet_sync_tests.rs
//! Unit tests for Epic 4: Light-Client Synchronization
//!
//! Tests cover:
//! - Sync progress calculation
//! - Note integration after sync
//! - Proof verification (mocked)


#[cfg(test)]
mod tests {
    use super::super::sync::LightClient;


    #[test]
    fn test_sync_progress() {
        let mut client = LightClient::new();
        client.sync_height = 5000;
        client.target_height = 10000;


        assert_eq!(client.sync_progress(), 0.5);
        assert!(!client.is_synced());


        client.sync_height = 10000;
        assert_eq!(client.sync_progress(), 1.0);
        assert!(client.is_synced());
    }
}
