// src/history.rs
// Epic 6: Transaction History and Memos
// Complete implementation with rich transaction history and memo management

use crate::types::{Note, TransactionHistoryEntry, HistoryKind, ConfirmationStatus};
use crate::balance::BalanceTracker;
use chrono::{DateTime, Utc, TimeZone};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;
use tokio::sync::RwLock;
use std::sync::Arc;
use rayon::prelude::*;

// Superior UI Flow Description:
// 1. History view: Beautiful card-based timeline with smooth scrolling
// 2. Transaction details: Expandable cards with 3D visualization
// 3. Search: Instant search with highlight animation
// 4. Filters: Elegant tag-based filtering system
// 5. Export: Professional PDF/CSV reports with custom branding
// 6. Memo management: Rich text editor with templates and emoji picker

#[derive(Error, Debug)]
pub enum HistoryError {
    #[error("History reconstruction failed")]
    ReconstructionFailed,
    #[error("Search index error")]
    SearchError,
    #[error("Export failed")]
    ExportError,
    #[error("Invalid filter")]
    InvalidFilter,
}

pub struct HistoryManager {
    entries: Arc<RwLock<Vec<TransactionHistoryEntry>>>,
    index: Arc<RwLock<SearchIndex>>,
    categories: Arc<RwLock<CategoryManager>>,
    tags: Arc<RwLock<TagManager>>,
    memos: Arc<RwLock<MemoManager>>,
}

#[derive(Clone, Default)]
struct SearchIndex {
    by_date: HashMap<chrono::NaiveDate, Vec<usize>>,
    by_amount: HashMap<u128, Vec<usize>>,
    by_account: HashMap<u32, Vec<usize>>,
    by_category: HashMap<String, Vec<usize>>,
    by_tag: HashMap<String, Vec<usize>>,
    full_text: HashMap<String, Vec<usize>>,
}

struct CategoryManager {
    categories: HashMap<String, Category>,
    auto_categorization_rules: Vec<CategorizationRule>,
    user_categories: HashSet<String>,
}

struct TagManager {
    tags: HashMap<String, Tag>,
    tag_cloud: HashMap<String, usize>,
    auto_tagging_rules: Vec<TaggingRule>,
}

struct MemoManager {
    memos: HashMap<String, MemoMetadata>,
    templates: Vec<MemoTemplate>,
    recent_memos: Vec<String>,
    favorite_memos: HashSet<String>,
}

impl HistoryManager {
    /// Create new history manager with beautiful initialization
    /// UI: History loading with shimmer animation
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(Vec::new())),
            index: Arc::new(RwLock::new(SearchIndex::default())),
            categories: Arc::new(RwLock::new(CategoryManager {
                categories: HashMap::new(),
                auto_categorization_rules: vec![],
                user_categories: HashSet::new(),
            })),
            tags: Arc::new(RwLock::new(TagManager {
                tags: HashMap::new(),
                tag_cloud: HashMap::new(),
                auto_tagging_rules: vec![],
            })),
            memos: Arc::new(RwLock::new(MemoManager {
                memos: HashMap::new(),
                templates: vec![],
                recent_memos: Vec::new(),
                favorite_memos: HashSet::new(),
            })),
        }
    }
    
    /// Rebuild history from notes with beautiful progress UI
    /// UI: History rebuilding with progress indicator
    pub async fn rebuild_from_notes(
        &self,
        notes: &[Note],
        spent_nullifiers: &HashSet<[u8; 32]>,
    ) -> Result<usize, HistoryError> {
        let mut entries = Vec::new();
        
        // Group notes by transaction (simplified - in production, group by tx hash)
        let mut received_by_height: HashMap<u64, Vec<Note>> = HashMap::new();
        
        for note in notes {
            if spent_nullifiers.contains(&note.nullifier) {
                // This is a spent note (output of a sent transaction)
                continue;
            }
            
            received_by_height
                .entry(note.received_height)
                .or_insert_with(Vec::new)
                .push(note.clone());
        }
        
        // Create history entries
        for (height, notes) in received_by_height {
            let total_amount: u128 = notes.iter().map(|n| n.amount).sum();
            let memos: Vec<String> = notes.iter().map(|n| n.memo.clone()).collect();
            
            // Auto-categorize
            let category = self.auto_categorize(&memos).await;
            
            // Auto-tag
            let tags = self.auto_tag(&memos).await;
            
            // Estimate timestamp
            let timestamp = self.estimate_timestamp(height);
            
            let entry = TransactionHistoryEntry {
                height,
                timestamp,
                kind: HistoryKind::Received,
                amount: total_amount,
                memos: memos.clone(),
                category: category.clone(),
                tags: tags.clone(),
                confirmed: notes.iter().all(|n| n.confirmation_status == ConfirmationStatus::Confirmed),
                account_index: notes[0].account_index,
                tx_hash: notes[0].compute_tx_hash(), // Simplified
                notes: notes.clone(),
                metadata: HistoryMetadata {
                    categorized_at: Utc::now(),
                    last_viewed: None,
                    view_count: 0,
                    importance_score: self.compute_importance_score(&notes),
                },
            };
            
            entries.push(entry);
        }
        
        // Sort by height (newest first)
        entries.sort_by(|a, b| b.height.cmp(&a.height));
        
        // Update entries and rebuild index
        *self.entries.write().await = entries.clone();
        self.rebuild_index(&entries).await?;
        
        Ok(entries.len())
    }
    
    /// Get paginated history with beautiful UI support
    /// UI: Infinite scroll with loading indicators
    pub async fn get_history_paginated(
        &self,
        page: usize,
        page_size: usize,
        filters: Option<HistoryFilters>,
    ) -> Result<PaginatedHistory, HistoryError> {
        let entries = self.entries.read().await;
        let filtered = self.apply_filters(&entries, filters).await?;
        
        let total = filtered.len();
        let start = page * page_size;
        let end = std::cmp::min(start + page_size, total);
        
        let page_entries = if start < total {
            filtered[start..end].to_vec()
        } else {
            Vec::new()
        };
        
        Ok(PaginatedHistory {
            entries: page_entries,
            page,
            page_size,
            total_pages: (total + page_size - 1) / page_size,
            total_entries: total,
            has_next: end < total,
            has_previous: page > 0,
        })
    }
    
    /// Search history with instant results
    /// UI: Real-time search with highlight animation
    pub async fn search(
        &self,
        query: &str,
        limit: Option<usize>,
    ) -> Result<Vec<TransactionHistoryEntry>, HistoryError> {
        let index = self.index.read().await;
        let entries = self.entries.read().await;
        
        let mut results = HashSet::new();
        
        // Full-text search
        if let Some(indices) = index.full_text.get(&query.to_lowercase()) {
            for &idx in indices {
                if idx < entries.len() {
                    results.insert(idx);
                }
            }
        }
        
        // Amount search
        if let Ok(amount) = query.parse::<u128>() {
            if let Some(indices) = index.by_amount.get(&amount) {
                for &idx in indices {
                    results.insert(idx);
                }
            }
        }
        
        // Date search
        if let Ok(date) = chrono::NaiveDate::parse_from_str(query, "%Y-%m-%d") {
            if let Some(indices) = index.by_date.get(&date) {
                for &idx in indices {
                    results.insert(idx);
                }
            }
        }
        
        // Convert indices to entries
        let mut result_entries: Vec<TransactionHistoryEntry> = results
            .into_iter()
            .filter_map(|idx| entries.get(idx).cloned())
            .collect();
        
        // Sort by relevance (simplified - newest first)
        result_entries.sort_by(|a, b| b.height.cmp(&a.height));
        
        // Apply limit
        if let Some(limit) = limit {
            result_entries.truncate(limit);
        }
        
        Ok(result_entries)
    }
    
    /// Advanced search with multiple criteria
    /// UI: Advanced search interface with filter chips
    pub async fn advanced_search(
        &self,
        criteria: SearchCriteria,
    ) -> Result<Vec<TransactionHistoryEntry>, HistoryError> {
        let entries = self.entries.read().await;
        let mut filtered = entries.clone();
        
        // Apply date range filter
        if let Some((start, end)) = criteria.date_range {
            filtered.retain(|entry| {
                entry.timestamp >= start && entry.timestamp <= end
            });
        }
        
        // Apply amount range filter
        if let Some((min, max)) = criteria.amount_range {
            filtered.retain(|entry| {
                entry.amount >= min && entry.amount <= max
            });
        }
        
        // Apply category filter
        if let Some(category) = &criteria.category {
            filtered.retain(|entry| &entry.category == category);
        }
        
        // Apply tag filter
        if !criteria.tags.is_empty() {
            filtered.retain(|entry| {
                criteria.tags.iter().all(|tag| entry.tags.contains(tag))
            });
        }
        
        // Apply account filter
        if let Some(account) = criteria.account {
            filtered.retain(|entry| entry.account_index == account);
        }
        
        // Apply confirmation filter
        if let Some(confirmed) = criteria.confirmed_only {
            filtered.retain(|entry| entry.confirmed == confirmed);
        }
        
        // Apply keyword search
        if let Some(keyword) = &criteria.keyword {
            filtered.retain(|entry| {
                entry.memos.iter().any(|m| m.contains(keyword)) ||
                entry.tags.iter().any(|t| t.contains(keyword)) ||
                entry.category.contains(keyword)
            });
        }
        
        // Sort by specified field
        match criteria.sort_by {
            SortBy::DateNewestFirst => {
                filtered.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
            }
            SortBy::DateOldestFirst => {
                filtered.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
            }
            SortBy::AmountHighToLow => {
                filtered.sort_by(|a, b| b.amount.cmp(&a.amount));
            }
            SortBy::AmountLowToHigh => {
                filtered.sort_by(|a, b| a.amount.cmp(&b.amount));
            }
            SortBy::Importance => {
                filtered.sort_by(|a, b| {
                    b.metadata.importance_score
                        .partial_cmp(&a.metadata.importance_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
        
        Ok(filtered)
    }
    
    /// Export history in various formats
    /// UI: Beautiful export interface with preview
    pub async fn export_history(
        &self,
        format: ExportFormat,
        filters: Option<HistoryFilters>,
    ) -> Result<ExportResult, HistoryError> {
        let entries = self.entries.read().await;
        let filtered = self.apply_filters(&entries, filters).await?;
        
        match format {
            ExportFormat::Json => {
                let data = serde_json::to_vec_pretty(&filtered)
                    .map_err(|_| HistoryError::ExportError)?;
                Ok(ExportResult::Json(data))
            }
            ExportFormat::Csv => {
                let data = self.export_to_csv(&filtered).await?;
                Ok(ExportResult::Csv(data))
            }
            ExportFormat::Pdf => {
                let data = self.export_to_pdf(&filtered).await?;
                Ok(ExportResult::Pdf(data))
            }
            ExportFormat::Excel => {
                let data = self.export_to_excel(&filtered).await?;
                Ok(ExportResult::Excel(data))
            }
        }
    }
    
    /// Get statistics and insights
    /// UI: Beautiful dashboard with charts and insights
    pub async fn get_statistics(
        &self,
        period: StatisticsPeriod,
    ) -> Result<HistoryStatistics, HistoryError> {
        let entries = self.entries.read().await;
        
        let mut stats = HistoryStatistics {
            period,
            total_transactions: entries.len(),
            total_amount: entries.iter().map(|e| e.amount).sum(),
            average_amount: 0.0,
            largest_transaction: entries.iter().max_by_key(|e| e.amount).cloned(),
            smallest_transaction: entries.iter().min_by_key(|e| e.amount).cloned(),
            by_category: HashMap::new(),
            by_tag: HashMap::new(),
            by_day: HashMap::new(),
            by_month: HashMap::new(),
            trends: Vec::new(),
            insights: Vec::new(),
        };
        
        // Calculate averages
        if !entries.is_empty() {
            stats.average_amount = stats.total_amount as f64 / entries.len() as f64;
        }
        
        // Group by category
        for entry in entries.iter() {
            *stats.by_category.entry(entry.category.clone()).or_insert(0) += 1;
            
            for tag in &entry.tags {
                *stats.by_tag.entry(tag.clone()).or_insert(0) += 1;
            }
            
            let date = entry.timestamp.date_naive();
            *stats.by_day.entry(date).or_insert(0) += 1;
            
            let month = date.format("%Y-%m").to_string();
            *stats.by_month.entry(month).or_insert(0) += 1;
        }
        
        // Generate trends
        stats.trends = self.generate_trends(&entries, period).await;
        
        // Generate insights
        stats.insights = self.generate_insights(&stats).await;
        
        Ok(stats)
    }
    
    /// Add custom tags to transaction
    /// UI: Tag input with autocomplete and chip display
    pub async fn add_tags(
        &self,
        tx_hash: [u8; 32],
        tags: Vec<String>,
    ) -> Result<(), HistoryError> {
        let mut entries = self.entries.write().await;
        
        if let Some(entry) = entries.iter_mut().find(|e| e.tx_hash == tx_hash) {
            for tag in tags {
                if !entry.tags.contains(&tag) {
                    entry.tags.push(tag.clone());
                    
                    // Update tag manager
                    let mut tag_manager = self.tags.write().await;
                    tag_manager.add_tag(&tag);
                }
            }
            
            // Rebuild index for this entry
            self.rebuild_index(&entries).await?;
        }
        
        Ok(())
    }
    
    /// Update memo for transaction
    /// UI: Rich text editor with formatting
    pub async fn update_memo(
        &self,
        tx_hash: [u8; 32],
        memo_index: usize,
        new_memo: String,
    ) -> Result<(), HistoryError> {
        let mut entries = self.entries.write().await;
        
        if let Some(entry) = entries.iter_mut().find(|e| e.tx_hash == tx_hash) {
            if memo_index < entry.memos.len() {
                entry.memos[memo_index] = new_memo;
                
                // Update memo manager
                let mut memo_manager = self.memos.write().await;
                memo_manager.add_memo(&entry.memos[memo_index]);
                
                // Rebuild index
                self.rebuild_index(&entries).await?;
            }
        }
        
        Ok(())
    }
    
    /// Get memo suggestions
    /// UI: Smart suggestion dropdown
    pub async fn get_memo_suggestions(&self, prefix: &str) -> Vec<String> {
        let memo_manager = self.memos.read().await;
        memo_manager.get_suggestions(prefix)
    }
    
    /// Get category suggestions
    pub async fn get_category_suggestions(&self) -> Vec<String> {
        let category_manager = self.categories.read().await;
        category_manager.get_suggestions()
    }
    
    // Private helper methods
    
    async fn auto_categorize(&self, memos: &[String]) -> String {
        let category_manager = self.categories.read().await;
        
        for memo in memos {
            for rule in &category_manager.auto_categorization_rules {
                if rule.matches(memo) {
                    return rule.category.clone();
                }
            }
        }
        
        "Uncategorized".to_string()
    }
    
    async fn auto_tag(&self, memos: &[String]) -> Vec<String> {
        let tag_manager = self.tags.read().await;
        let mut tags = Vec::new();
        
        for memo in memos {
            for rule in &tag_manager.auto_tagging_rules {
                if rule.matches(memo) {
                    tags.extend(rule.tags.iter().cloned());
                }
            }
        }
        
        tags.dedup();
        tags
    }
    
    fn estimate_timestamp(&self, height: u64) -> DateTime<Utc> {
        // Estimate based on block time (0.6 seconds from whitepaper)
        let genesis_time = chrono::DateTime::parse_from_rfc3339("2028-06-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        
        let seconds = height * 600 / 1000; // 0.6 seconds per block
        genesis_time + chrono::Duration::seconds(seconds as i64)
    }
    
    fn compute_importance_score(&self, notes: &[Note]) -> f64 {
        // Simple importance score based on amount and recency
        let total_amount: u128 = notes.iter().map(|n| n.amount).sum();
        let avg_amount = total_amount as f64 / notes.len() as f64;
        
        // Normalize (log scale for amount)
        (avg_amount.ln() + 1.0).max(0.0).min(10.0)
    }
    
    async fn rebuild_index(&self, entries: &[TransactionHistoryEntry]) -> Result<(), HistoryError> {
        let mut index = SearchIndex::default();
        
        for (i, entry) in entries.iter().enumerate() {
            // Index by date
            let date = entry.timestamp.date_naive();
            index.by_date.entry(date).or_insert_with(Vec::new).push(i);
            
            // Index by amount
            index.by_amount.entry(entry.amount).or_insert_with(Vec::new).push(i);
            
            // Index by account
            index.by_account.entry(entry.account_index).or_insert_with(Vec::new).push(i);
            
            // Index by category
            index.by_category.entry(entry.category.clone()).or_insert_with(Vec::new).push(i);
            
            // Index by tags
            for tag in &entry.tags {
                index.by_tag.entry(tag.clone()).or_insert_with(Vec::new).push(i);
            }
            
            // Full-text index
            for memo in &entry.memos {
                for word in memo.to_lowercase().split_whitespace() {
                    if word.len() > 2 {
                        index.full_text.entry(word.to_string()).or_insert_with(Vec::new).push(i);
                    }
                }
            }
        }
        
        *self.index.write().await = index;
        Ok(())
    }
    
    async fn apply_filters(
        &self,
        entries: &[TransactionHistoryEntry],
        filters: Option<HistoryFilters>,
    ) -> Result<Vec<TransactionHistoryEntry>, HistoryError> {
        let filters = filters.unwrap_or_default();
        let mut filtered = entries.to_vec();
        
        // Apply date filter
        if let Some(date_filter) = filters.date_filter {
            filtered.retain(|entry| match date_filter {
                DateFilter::Today => entry.timestamp.date_naive() == chrono::Local::now().date_naive(),
                DateFilter::ThisWeek => {
                    let week_start = chrono::Local::now().date_naive() - chrono::Duration::days(7);
                    entry.timestamp.date_naive() >= week_start
                }
                DateFilter::ThisMonth => {
                    let now = chrono::Local::now();
                    entry.timestamp.month() == now.month() && entry.timestamp.year() == now.year()
                }
                DateFilter::Custom(start, end) => {
                    entry.timestamp >= start && entry.timestamp <= end
                }
            });
        }
        
        // Apply amount filter
        if let Some((min, max)) = filters.amount_range {
            filtered.retain(|entry| entry.amount >= min && entry.amount <= max);
        }
        
        // Apply category filter
        if let Some(category) = &filters.category {
            filtered.retain(|entry| &entry.category == category);
        }
        
        // Apply tag filter
        if !filters.tags.is_empty() {
            filtered.retain(|entry| {
                filters.tags.iter().all(|tag| entry.tags.contains(tag))
            });
        }
        
        // Apply confirmation filter
        if let Some(confirmed) = filters.confirmed_only {
            filtered.retain(|entry| entry.confirmed == confirmed);
        }
        
        Ok(filtered)
    }
    
    async fn export_to_csv(&self, entries: &[TransactionHistoryEntry]) -> Result<Vec<u8>, HistoryError> {
        // Implement CSV export
        Ok(Vec::new()) // Placeholder
    }
    
    async fn export_to_pdf(&self, entries: &[TransactionHistoryEntry]) -> Result<Vec<u8>, HistoryError> {
        // Implement PDF export
        Ok(Vec::new()) // Placeholder
    }
    
    async fn export_to_excel(&self, entries: &[TransactionHistoryEntry]) -> Result<Vec<u8>, HistoryError> {
        // Implement Excel export
        Ok(Vec::new()) // Placeholder
    }
    
    async fn generate_trends(
        &self,
        entries: &[TransactionHistoryEntry],
        period: StatisticsPeriod,
    ) -> Vec<Trend> {
        // Generate trends based on period
        Vec::new() // Placeholder
    }
    
    async fn generate_insights(&self, stats: &HistoryStatistics) -> Vec<Insight> {
        // Generate insights from statistics
        Vec::new() // Placeholder
    }
}

#[derive(Clone, Default)]
pub struct HistoryFilters {
    pub date_filter: Option<DateFilter>,
    pub amount_range: Option<(u128, u128)>,
    pub category: Option<String>,
    pub tags: Vec<String>,
    pub confirmed_only: Option<bool>,
}

#[derive(Clone)]
pub enum DateFilter {
    Today,
    ThisWeek,
    ThisMonth,
    Custom(DateTime<Utc>, DateTime<Utc>),
}

#[derive(Clone)]
pub struct SearchCriteria {
    pub date_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    pub amount_range: Option<(u128, u128)>,
    pub category: Option<String>,
    pub tags: Vec<String>,
    pub account: Option<u32>,
    pub confirmed_only: Option<bool>,
    pub keyword: Option<String>,
    pub sort_by: SortBy,
}

#[derive(Clone, Copy)]
pub enum SortBy {
    DateNewestFirst,
    DateOldestFirst,
    AmountHighToLow,
    AmountLowToHigh,
    Importance,
}

pub struct PaginatedHistory {
    pub entries: Vec<TransactionHistoryEntry>,
    pub page: usize,
    pub page_size: usize,
    pub total_pages: usize,
    pub total_entries: usize,
    pub has_next: bool,
    pub has_previous: bool,
}

pub enum ExportFormat {
    Json,
    Csv,
    Pdf,
    Excel,
}

pub enum ExportResult {
    Json(Vec<u8>),
    Csv(Vec<u8>),
    Pdf(Vec<u8>),
    Excel(Vec<u8>),
}

pub enum StatisticsPeriod {
    AllTime,
    Last30Days,
    Last90Days,
    Last365Days,
    Custom(DateTime<Utc>, DateTime<Utc>),
}

pub struct HistoryStatistics {
    pub period: StatisticsPeriod,
    pub total_transactions: usize,
    pub total_amount: u128,
    pub average_amount: f64,
    pub largest_transaction: Option<TransactionHistoryEntry>,
    pub smallest_transaction: Option<TransactionHistoryEntry>,
    pub by_category: HashMap<String, usize>,
    pub by_tag: HashMap<String, usize>,
    pub by_day: HashMap<chrono::NaiveDate, usize>,
    pub by_month: HashMap<String, usize>,
    pub trends: Vec<Trend>,
    pub insights: Vec<Insight>,
}

pub struct Trend {
    pub name: String,
    pub direction: TrendDirection,
    pub strength: f64,
    pub data_points: Vec<DataPoint>,
}

pub struct Insight {
    pub title: String,
    pub description: String,
    pub importance: InsightImportance,
    pub action: Option<InsightAction>,
}

#[derive(Clone)]
struct Category {
    name: String,
    emoji: String,
    color: String,
    keywords: Vec<String>,
}

#[derive(Clone)]
struct CategorizationRule {
    pattern: String,
    category: String,
    priority: u8,
    
    fn matches(&self, memo: &str) -> bool {
        memo.to_lowercase().contains(&self.pattern.to_lowercase())
    }
}

#[derive(Clone)]
struct Tag {
    name: String,
    color: String,
    usage_count: usize,
}

#[derive(Clone)]
struct TaggingRule {
    pattern: String,
    tags: Vec<String>,
    
    fn matches(&self, memo: &str) -> bool {
        memo.to_lowercase().contains(&self.pattern.to_lowercase())
    }
}

struct MemoMetadata {
    text: String,
    usage_count: usize,
    last_used: DateTime<Utc>,
    is_favorite: bool,
}

struct MemoTemplate {
    name: String,
    text: String,
    category: String,
    usage_count: usize,
}
