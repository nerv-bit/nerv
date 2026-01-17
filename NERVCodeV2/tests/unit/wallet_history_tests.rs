// tests/wallet_history_tests.rs
//! Unit tests for Epic 6: Private Transaction History
//!
//! Tests cover:
//! - History rebuilding
//! - Search functionality
//! - Categorization


#[cfg(test)]
mod tests {
    use super::super::{history::HistoryManager, types::Note};
    use chrono::{Utc, TimeZone};


    #[test]
    fn test_history_rebuild_and_search() {
        let mut manager = HistoryManager::new();


        let notes = vec![
            Note { amount: 5000, memo: "coffee shop".to_string(), received_height: 100, ..Default::default() },
            Note { amount: 10000, memo: "salary payment".to_string(), received_height: 200, ..Default::default() },
        ];


        manager.rebuild_from_notes(&notes, &Default::default());


        assert_eq!(manager.get_entries().len(), 2);


        let search_results = manager.search("coffee");
        assert_eq!(search_results.len(), 1);
        assert!(search_results[0].category.contains("Coffee"));
    }
}
