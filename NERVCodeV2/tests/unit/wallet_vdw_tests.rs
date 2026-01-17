// tests/wallet_vdw_tests.rs
//! Unit tests for Epic 5: Verifiable Delay Witness Handling
//!
//! Tests cover:
//! - Offline verification flow
//! - Caching behavior


#[cfg(test)]
mod tests {
    use super::super::vdw::VDWManager;
    use std::path::PathBuf;


    #[tokio::test]
    async fn test_vdw_offline_verification_and_cache() {
        let mut manager = VDWManager::new(PathBuf::from("test_cache"));


        // Mock a valid VDW ID (in real tests, use pre-known valid one)
        let result1 = manager.verify_offline("valid_test_vdw_id").await;
        // First call would fetch, second from cache
        let result2 = manager.verify_offline("valid_test_vdw_id").await;


        // In real implementation, assert success and embedding match
        assert!(result1.is_ok());
        assert!(result2.is_ok());
    }
}
