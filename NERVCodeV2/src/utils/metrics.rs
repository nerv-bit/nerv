// src/utils/metrics.rs
// ============================================================================
// METRICS COLLECTION AND REPORTING
// ============================================================================
// This module provides comprehensive metrics collection for monitoring and
// observability of the NERV blockchain. We collect metrics for:
// 1. Performance (TPS, latency, throughput)
// 2. System health (CPU, memory, disk, network)
// 3. Blockchain state (shards, validators, transactions)
// 4. Privacy and security (TEE attestations, anonymity set)
// 5. Economic indicators (rewards, stake distribution)
// ============================================================================


use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tokio::time::interval;
use thiserror::Error;
use prometheus::{Registry, Counter, Gauge, Histogram, HistogramOpts, CounterVec, GaugeVec, HistogramVec};
use prometheus_exporter::PrometheusExporter;
use tracing::{debug, info, warn};


// ============================================================================
// ERROR DEFINITIONS
// ============================================================================


#[derive(Error, Debug)]
pub enum MetricsError {
    #[error("Failed to register metric: {0}")]
    Registration(String),
    
    #[error("Metric not found: {0}")]
    NotFound(String),
    
    #[error("Invalid metric value: {0}")]
    InvalidValue(String),
    
    #[error("Prometheus error: {0}")]
    Prometheus(#[from] prometheus::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}


// ============================================================================
// METRIC TYPES AND CONSTANTS
// ============================================================================


/// Supported metric types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
}


/// Metric labels for categorization
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct MetricLabels {
    pub shard_id: Option<u64>,
    pub node_type: Option<String>,
    pub tee_type: Option<String>,
    pub consensus_phase: Option<String>,
    pub operation: Option<String>,
}


impl MetricLabels {
    pub fn new() -> Self {
        Self {
            shard_id: None,
            node_type: None,
            tee_type: None,
            consensus_phase: None,
            operation: None,
        }
    }
    
    pub fn with_shard(mut self, shard_id: u64) -> Self {
        self.shard_id = Some(shard_id);
        self
    }
    
    pub fn with_node_type(mut self, node_type: &str) -> Self {
        self.node_type = Some(node_type.to_string());
        self
    }
    
    pub fn with_tee_type(mut self, tee_type: &str) -> Self {
        self.tee_type = Some(tee_type.to_string());
        self
    }
    
    pub fn to_prometheus_labels(&self) -> Vec<(&'static str, String)> {
        let mut labels = Vec::new();
        
        if let Some(shard_id) = self.shard_id {
            labels.push(("shard", shard_id.to_string()));
        }
        
        if let Some(node_type) = &self.node_type {
            labels.push(("node_type", node_type.clone()));
        }
        
        if let Some(tee_type) = &self.tee_type {
            labels.push(("tee_type", tee_type.clone()));
        }
        
        if let Some(consensus_phase) = &self.consensus_phase {
            labels.push(("consensus_phase", consensus_phase.clone()));
        }
        
        if let Some(operation) = &self.operation {
            labels.push(("operation", operation.clone()));
        }
        
        labels
    }
}


// ============================================================================
// METRICS REGISTRY AND COLLECTOR
// ============================================================================


/// Main metrics collector for NERV
pub struct MetricsCollector {
    registry: Registry,
    exporter: Option<PrometheusExporter>,
    metrics: Arc<RwLock<HashMap<String, RegisteredMetric>>>,
    config: MetricsConfig,
}


/// Configuration for metrics collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable Prometheus exporter
    pub enable_prometheus: bool,
    
    /// Prometheus bind address
    pub prometheus_addr: String,
    
    /// Metrics collection interval in seconds
    pub collection_interval: u64,
    
    /// Enable system metrics collection
    pub enable_system_metrics: bool,
    
    /// Enable blockchain metrics collection
    pub enable_blockchain_metrics: bool,
    
    /// Enable privacy metrics collection
    pub enable_privacy_metrics: bool,
    
    /// Enable economic metrics collection
    pub enable_economic_metrics: bool,
    
    /// Metrics retention period in hours
    pub retention_hours: u64,
}


impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enable_prometheus: true,
            prometheus_addr: "0.0.0.0:9090".to_string(),
            collection_interval: 5,
            enable_system_metrics: true,
            enable_blockchain_metrics: true,
            enable_privacy_metrics: true,
            enable_economic_metrics: true,
            retention_hours: 24,
        }
    }
}


/// Enumeration of registered metrics
#[derive(Debug, Clone)]
pub enum RegisteredMetric {
    Counter(Counter),
    Gauge(Gauge),
    Histogram(Histogram),
    CounterVec(CounterVec),
    GaugeVec(GaugeVec),
    HistogramVec(HistogramVec),
}


impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new(config: MetricsConfig) -> Result<Self, MetricsError> {
        let registry = Registry::new_custom(
            Some("nerv".to_string()),
            Some(HashMap::new()),
        )?;
        
        let exporter = if config.enable_prometheus {
            let exporter = PrometheusExporter::new(
                registry.clone(),
                config.prometheus_addr.parse()
                    .map_err(|e| MetricsError::Registration(e.to_string()))?,
            );
            Some(exporter)
        } else {
            None
        };
        
        Ok(Self {
            registry,
            exporter,
            metrics: Arc::new(RwLock::new(HashMap::new())),
            config,
        })
    }
    
    /// Start the metrics collector
    pub async fn start(&self) -> Result<(), MetricsError> {
        info!("Starting metrics collector with config: {:?}", self.config);
        
        // Initialize all metrics
        self.initialize_metrics().await?;
        
        // Start Prometheus exporter if enabled
        if let Some(exporter) = &self.exporter {
            exporter.start().await
                .map_err(|e| MetricsError::Registration(e.to_string()))?;
            info!("Prometheus exporter started on {}", self.config.prometheus_addr);
        }
        
        // Start periodic collection
        if self.config.enable_system_metrics {
            let collector = self.clone();
            tokio::spawn(async move {
                collector.collect_system_metrics().await;
            });
        }
        
        Ok(())
    }
    
    /// Stop the metrics collector
    pub async fn stop(&self) -> Result<(), MetricsError> {
        if let Some(exporter) = &self.exporter {
            exporter.stop().await
                .map_err(|e| MetricsError::Registration(e.to_string()))?;
        }
        
        info!("Metrics collector stopped");
        Ok(())
    }
    
    /// Clone the collector
    pub fn clone(&self) -> Self {
        Self {
            registry: self.registry.clone(),
            exporter: None, // Exporter cannot be cloned
            metrics: self.metrics.clone(),
            config: self.config.clone(),
        }
    }
    
    /// Register a counter metric
    pub async fn register_counter(
        &self,
        name: &str,
        help: &str,
        labels: Option<Vec<&'static str>>,
    ) -> Result<(), MetricsError> {
        let metric = if let Some(label_names) = labels {
            let counter = CounterVec::new(
                prometheus::Opts::new(name, help),
                label_names,
            )?;
            RegisteredMetric::CounterVec(counter)
        } else {
            let counter = Counter::new(name, help)?;
            RegisteredMetric::Counter(counter)
        };
        
        self.registry.register(Box::new(match &metric {
            RegisteredMetric::Counter(c) => c.clone(),
            RegisteredMetric::CounterVec(c) => c.clone(),
            _ => unreachable!(),
        }))?;
        
        let mut metrics = self.metrics.write().await;
        metrics.insert(name.to_string(), metric);
        
        debug!("Registered counter metric: {}", name);
        Ok(())
    }
    
    /// Register a gauge metric
    pub async fn register_gauge(
        &self,
        name: &str,
        help: &str,
        labels: Option<Vec<&'static str>>,
    ) -> Result<(), MetricsError> {
        let metric = if let Some(label_names) = labels {
            let gauge = GaugeVec::new(
                prometheus::Opts::new(name, help),
                label_names,
            )?;
            RegisteredMetric::GaugeVec(gauge)
        } else {
            let gauge = Gauge::new(name, help)?;
            RegisteredMetric::Gauge(gauge)
        };
        
        self.registry.register(Box::new(match &metric {
            RegisteredMetric::Gauge(g) => g.clone(),
            RegisteredMetric::GaugeVec(g) => g.clone(),
            _ => unreachable!(),
        }))?;
        
        let mut metrics = self.metrics.write().await;
        metrics.insert(name.to_string(), metric);
        
        debug!("Registered gauge metric: {}", name);
        Ok(())
    }
    
    /// Register a histogram metric
    pub async fn register_histogram(
        &self,
        name: &str,
        help: &str,
        buckets: Vec<f64>,
        labels: Option<Vec<&'static str>>,
    ) -> Result<(), MetricsError> {
        let metric = if let Some(label_names) = labels {
            let histogram = HistogramVec::new(
                HistogramOpts::new(name, help).buckets(buckets),
                label_names,
            )?;
            RegisteredMetric::HistogramVec(histogram)
        } else {
            let histogram = Histogram::with_opts(
                HistogramOpts::new(name, help).buckets(buckets),
            )?;
            RegisteredMetric::Histogram(histogram)
        };
        
        self.registry.register(Box::new(match &metric {
            RegisteredMetric::Histogram(h) => h.clone(),
            RegisteredMetric::HistogramVec(h) => h.clone(),
            _ => unreachable!(),
        }))?;
        
        let mut metrics = self.metrics.write().await;
        metrics.insert(name.to_string(), metric);
        
        debug!("Registered histogram metric: {}", name);
        Ok(())
    }
    
    /// Increment a counter metric
    pub async fn increment_counter(
        &self,
        name: &str,
        value: f64,
        labels: Option<&MetricLabels>,
    ) -> Result<(), MetricsError> {
        let metrics = self.metrics.read().await;
        let metric = metrics.get(name)
            .ok_or_else(|| MetricsError::NotFound(name.to_string()))?;
        
        match metric {
            RegisteredMetric::Counter(counter) => {
                counter.inc_by(value);
            }
            RegisteredMetric::CounterVec(counter_vec) => {
                if let Some(labels) = labels {
                    let prom_labels: Vec<(&'static str, String)> = labels.to_prometheus_labels();
                    counter_vec.with_label_values(
                        &prom_labels.iter().map(|(_, v)| v.as_str()).collect::<Vec<_>>()
                    ).inc_by(value);
                } else {
                    return Err(MetricsError::InvalidValue(
                        format!("CounterVec {} requires labels", name)
                    ));
                }
            }
            _ => {
                return Err(MetricsError::InvalidValue(
                    format!("Metric {} is not a counter", name)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Set a gauge metric value
    pub async fn set_gauge(
        &self,
        name: &str,
        value: f64,
        labels: Option<&MetricLabels>,
    ) -> Result<(), MetricsError> {
        let metrics = self.metrics.read().await;
        let metric = metrics.get(name)
            .ok_or_else(|| MetricsError::NotFound(name.to_string()))?;
        
        match metric {
            RegisteredMetric::Gauge(gauge) => {
                gauge.set(value);
            }
            RegisteredMetric::GaugeVec(gauge_vec) => {
                if let Some(labels) = labels {
                    let prom_labels: Vec<(&'static str, String)> = labels.to_prometheus_labels();
                    gauge_vec.with_label_values(
                        &prom_labels.iter().map(|(_, v)| v.as_str()).collect::<Vec<_>>()
                    ).set(value);
                } else {
                    return Err(MetricsError::InvalidValue(
                        format!("GaugeVec {} requires labels", name)
                    ));
                }
            }
            _ => {
                return Err(MetricsError::InvalidValue(
                    format!("Metric {} is not a gauge", name)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Observe a histogram metric value
    pub async fn observe_histogram(
        &self,
        name: &str,
        value: f64,
        labels: Option<&MetricLabels>,
    ) -> Result<(), MetricsError> {
        let metrics = self.metrics.read().await;
        let metric = metrics.get(name)
            .ok_or_else(|| MetricsError::NotFound(name.to_string()))?;
        
        match metric {
            RegisteredMetric::Histogram(histogram) => {
                histogram.observe(value);
            }
            RegisteredMetric::HistogramVec(histogram_vec) => {
                if let Some(labels) = labels {
                    let prom_labels: Vec<(&'static str, String)> = labels.to_prometheus_labels();
                    histogram_vec.with_label_values(
                        &prom_labels.iter().map(|(_, v)| v.as_str()).collect::<Vec<_>>()
                    ).observe(value);
                } else {
                    return Err(MetricsError::InvalidValue(
                        format!("HistogramVec {} requires labels", name)
                    ));
                }
            }
            _ => {
                return Err(MetricsError::InvalidValue(
                    format!("Metric {} is not a histogram", name)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Record time taken for an operation
    pub async fn record_duration(
        &self,
        name: &str,
        duration: Duration,
        labels: Option<&MetricLabels>,
    ) -> Result<(), MetricsError> {
        self.observe_histogram(name, duration.as_secs_f64(), labels).await
    }
    
    /// Get all metrics as JSON
    pub async fn get_metrics_json(&self) -> Result<String, MetricsError> {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        
        let metrics_text = String::from_utf8_lossy(&buffer);
        
        // Convert to structured JSON
        let mut metrics_map = serde_json::Map::new();
        for family in metric_families {
            let mut family_map = serde_json::Map::new();
            family_map.insert("help".to_string(), family.get_help().into());
            family_map.insert("type".to_string(), format!("{:?}", family.get_field_type()).into());
            
            let mut metrics_vec = Vec::new();
            for metric in family.get_metric() {
                let mut metric_map = serde_json::Map::new();
                
                // Add labels
                let labels = metric.get_label();
                if !labels.is_empty() {
                    let mut labels_map = serde_json::Map::new();
                    for label in labels {
                        labels_map.insert(
                            label.get_name().to_string(),
                            label.get_value().to_string().into()
                        );
                    }
                    metric_map.insert("labels".to_string(), labels_map.into());
                }
                
                // Add value based on metric type
                match family.get_field_type() {
                    prometheus::proto::MetricType::COUNTER => {
                        metric_map.insert("value".to_string(), metric.get_counter().get_value().into());
                    }
                    prometheus::proto::MetricType::GAUGE => {
                        metric_map.insert("value".to_string(), metric.get_gauge().get_value().into());
                    }
                    prometheus::proto::MetricType::HISTOGRAM => {
                        let histogram = metric.get_histogram();
                        let mut histogram_map = serde_json::Map::new();
                        histogram_map.insert("sample_count".to_string(), histogram.get_sample_count().into());
                        histogram_map.insert("sample_sum".to_string(), histogram.get_sample_sum().into());
                        
                        let mut buckets_vec = Vec::new();
                        for bucket in histogram.get_bucket() {
                            let mut bucket_map = serde_json::Map::new();
                            bucket_map.insert("upper_bound".to_string(), bucket.get_upper_bound().into());
                            bucket_map.insert("cumulative_count".to_string(), bucket.get_cumulative_count().into());
                            buckets_vec.push(bucket_map);
                        }
                        histogram_map.insert("buckets".to_string(), buckets_vec.into());
                        
                        metric_map.insert("histogram".to_string(), histogram_map.into());
                    }
                    _ => {
                        // Skip other types for now
                        continue;
                    }
                }
                
                metrics_vec.push(metric_map);
            }
            
            family_map.insert("metrics".to_string(), metrics_vec.into());
            metrics_map.insert(family.get_name().to_string(), family_map.into());
        }
        
        serde_json::to_string_pretty(&metrics_map).map_err(Into::into)
    }
    
    /// Initialize all NERV-specific metrics
    async fn initialize_metrics(&self) -> Result<(), MetricsError> {
        // ========== PERFORMANCE METRICS ==========
        
        // Transaction processing
        self.register_counter(
            "nerv_transactions_processed_total",
            "Total number of transactions processed",
            Some(vec!["shard", "operation"]),
        ).await?;
        
        self.register_histogram(
            "nerv_transaction_latency_seconds",
            "Transaction processing latency",
            vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            Some(vec!["shard", "operation"]),
        ).await?;
        
        self.register_gauge(
            "nerv_transactions_per_second",
            "Current transactions per second",
            Some(vec!["shard"]),
        ).await?;
        
        // Batch processing
        self.register_histogram(
            "nerv_batch_size",
            "Number of transactions per batch",
            vec![1.0, 10.0, 50.0, 100.0, 200.0, 256.0],
            Some(vec!["shard"]),
        ).await?;
        
        self.register_histogram(
            "nerv_batch_processing_time_seconds",
            "Batch processing time",
            vec![0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            Some(vec!["shard"]),
        ).await?;
        
        // ========== SYSTEM METRICS ==========
        
        if self.config.enable_system_metrics {
            self.register_gauge(
                "nerv_cpu_usage_percent",
                "CPU usage percentage",
                None,
            ).await?;
            
            self.register_gauge(
                "nerv_memory_usage_bytes",
                "Memory usage in bytes",
                None,
            ).await?;
            
            self.register_gauge(
                "nerv_disk_usage_bytes",
                "Disk usage in bytes",
                None,
            ).await?;
            
            self.register_gauge(
                "nerv_network_bytes_sent_total",
                "Total network bytes sent",
                None,
            ).await?;
            
            self.register_gauge(
                "nerv_network_bytes_received_total",
                "Total network bytes received",
                None,
            ).await?;
        }
        
        // ========== BLOCKCHAIN STATE METRICS ==========
        
        if self.config.enable_blockchain_metrics {
            self.register_gauge(
                "nerv_shards_active",
                "Number of active shards",
                None,
            ).await?;
            
            self.register_counter(
                "nerv_shard_splits_total",
                "Total number of shard splits",
                None,
            ).await?;
            
            self.register_counter(
                "nerv_shard_merges_total",
                "Total number of shard merges",
                None,
            ).await?;
            
            self.register_gauge(
                "nerv_validators_active",
                "Number of active validators",
                None,
            ).await?;
            
            self.register_histogram(
                "nerv_consensus_latency_seconds",
                "Consensus decision latency",
                vec![0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
                Some(vec!["phase"]),
            ).await?;
            
            self.register_gauge(
                "nerv_embedding_size_bytes",
                "Neural embedding size in bytes",
                Some(vec!["shard"]),
            ).await?;
        }
        
        // ========== PRIVACY AND SECURITY METRICS ==========
        
        if self.config.enable_privacy_metrics {
            self.register_counter(
                "nerv_tee_attestations_total",
                "Total number of TEE attestations",
                Some(vec!["tee_type", "result"]),
            ).await?;
            
            self.register_histogram(
                "nerv_tee_attestation_time_seconds",
                "TEE attestation processing time",
                vec![0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
                Some(vec!["tee_type"]),
            ).await?;
            
            self.register_gauge(
                "nerv_anonymity_set_size",
                "Current anonymity set size for mixer",
                None,
            ).await?;
            
            self.register_counter(
                "nerv_onion_routing_hops_total",
                "Total number of onion routing hops processed",
                Some(vec!["hop"]),
            ).await?;
            
            self.register_histogram(
                "nerv_mixer_latency_seconds",
                "Mixer processing latency",
                vec![0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
                None,
            ).await?;
        }
        
        // ========== ECONOMIC METRICS ==========
        
        if self.config.enable_economic_metrics {
            self.register_gauge(
                "nerv_stake_total",
                "Total stake in the network",
                None,
            ).await?;
            
            self.register_gauge(
                "nerv_rewards_distributed_total",
                "Total rewards distributed",
                None,
            ).await?;
            
            self.register_histogram(
                "nerv_shapley_values",
                "Distribution of Shapley values for gradient contributions",
                vec![0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
                None,
            ).await?;
            
            self.register_counter(
                "nerv_gradient_contributions_total",
                "Total number of gradient contributions",
                Some(vec!["shard"]),
            ).await?;
        }
        
        // ========== ERROR AND FAILURE METRICS ==========
        
        self.register_counter(
            "nerv_errors_total",
            "Total number of errors",
            Some(vec!["type", "module"]),
        ).await?;
        
        self.register_counter(
            "nerv_consensus_disputes_total",
            "Total number of consensus disputes",
            Some(vec!["resolution"]),
        ).await?;
        
        self.register_counter(
            "nerv_tee_failures_total",
            "Total number of TEE operation failures",
            Some(vec!["tee_type", "operation"]),
        ).await?;
        
        info!("Metrics initialization completed");
        Ok(())
    }
    
    /// Collect system metrics periodically
    async fn collect_system_metrics(&self) {
        let mut interval = interval(Duration::from_secs(self.config.collection_interval));
        
        loop {
            interval.tick().await;
            
            // Collect CPU usage (simplified - in production, use system-specific APIs)
            #[cfg(target_os = "linux")]
            {
                if let Ok(stat) = std::fs::read_to_string("/proc/stat") {
                    if let Some(first_line) = stat.lines().next() {
                        let parts: Vec<&str> = first_line.split_whitespace().collect();
                        if parts.len() >= 5 {
                            let user: u64 = parts[1].parse().unwrap_or(0);
                            let nice: u64 = parts[2].parse().unwrap_or(0);
                            let system: u64 = parts[3].parse().unwrap_or(0);
                            let idle: u64 = parts[4].parse().unwrap_or(0);
                            
                            let total = user + nice + system + idle;
                            let usage = if total > 0 {
                                ((user + nice + system) as f64 / total as f64) * 100.0
                            } else {
                                0.0
                            };
                            
                            if let Err(e) = self.set_gauge("nerv_cpu_usage_percent", usage, None).await {
                                warn!("Failed to set CPU usage metric: {}", e);
                            }
                        }
                    }
                }
            }
            
            // Collect memory usage
            #[cfg(target_os = "linux")]
            {
                if let Ok(info) = sys_info::mem_info() {
                    let used_memory = (info.total - info.free) as f64 * 1024.0; // Convert to bytes
                    if let Err(e) = self.set_gauge("nerv_memory_usage_bytes", used_memory, None).await {
                        warn!("Failed to set memory usage metric: {}", e);
                    }
                }
            }
            
            // Collect disk usage
            if let Ok(current_dir) = std::env::current_dir() {
                if let Ok(fs) = fs2::available_space(&current_dir) {
                    if let Err(e) = self.set_gauge("nerv_disk_usage_bytes", fs as f64, None).await {
                        warn!("Failed to set disk usage metric: {}", e);
                    }
                }
            }
            
            debug!("System metrics collection completed");
        }
    }
    
    /// Record a transaction processing event
    pub async fn record_transaction(
        &self,
        shard_id: u64,
        operation: &str,
        latency: Duration,
        success: bool,
    ) -> Result<(), MetricsError> {
        let labels = MetricLabels::new()
            .with_shard(shard_id)
            .with_operation(operation);
        
        // Increment transaction counter
        self.increment_counter(
            "nerv_transactions_processed_total",
            1.0,
            Some(&labels),
        ).await?;
        
        // Record latency if successful
        if success {
            self.record_duration(
                "nerv_transaction_latency_seconds",
                latency,
                Some(&labels),
            ).await?;
        }
        
        // Record error if failed
        if !success {
            let error_labels = MetricLabels::new()
                .with_operation(operation)
                .with_node_type("validator");
            
            self.increment_counter(
                "nerv_errors_total",
                1.0,
                Some(&error_labels.with_operation("transaction_failed")),
            ).await?;
        }
        
        Ok(())
    }
    
    /// Record a batch processing event
    pub async fn record_batch(
        &self,
        shard_id: u64,
        batch_size: usize,
        processing_time: Duration,
    ) -> Result<(), MetricsError> {
        let labels = MetricLabels::new().with_shard(shard_id);
        
        // Record batch size
        self.observe_histogram(
            "nerv_batch_size",
            batch_size as f64,
            Some(&labels),
        ).await?;
        
        // Record processing time
        self.record_duration(
            "nerv_batch_processing_time_seconds",
            processing_time,
            Some(&labels),
        ).await?;
        
        Ok(())
    }
    
    /// Record a TEE attestation event
    pub async fn record_tee_attestation(
        &self,
        tee_type: &str,
        attestation_time: Duration,
        success: bool,
    ) -> Result<(), MetricsError> {
        let labels = MetricLabels::new().with_tee_type(tee_type);
        
        // Increment attestation counter
        self.increment_counter(
            "nerv_tee_attestations_total",
            1.0,
            Some(&labels.with_operation(if success { "success" } else { "failure" })),
        ).await?;
        
        // Record attestation time if successful
        if success {
            self.record_duration(
                "nerv_tee_attestation_time_seconds",
                attestation_time,
                Some(&labels),
            ).await?;
        } else {
            // Record TEE failure
            self.increment_counter(
                "nerv_tee_failures_total",
                1.0,
                Some(&labels.with_operation("attestation")),
            ).await?;
        }
        
        Ok(())
    }
    
    /// Record a consensus event
    pub async fn record_consensus_event(
        &self,
        phase: &str,
        latency: Duration,
        dispute: bool,
        dispute_resolution: Option<&str>,
    ) -> Result<(), MetricsError> {
        let labels = MetricLabels::new().with_operation(phase);
        
        // Record consensus latency
        self.record_duration(
            "nerv_consensus_latency_seconds",
            latency,
            Some(&labels),
        ).await?;
        
        // Record dispute if applicable
        if dispute {
            self.increment_counter(
                "nerv_consensus_disputes_total",
                1.0,
                Some(&MetricLabels::new().with_operation(
                    dispute_resolution.unwrap_or("unknown")
                )),
            ).await?;
        }
        
        Ok(())
    }
    
    /// Record a shard operation (split or merge)
    pub async fn record_shard_operation(
        &self,
        operation: &str, // "split" or "merge"
        parent_shard: u64,
        child_shards: &[u64],
        duration: Duration,
    ) -> Result<(), MetricsError> {
        // Update active shards count
        let current_shards = match operation {
            "split" => child_shards.len() as f64, // Parent becomes children
            "merge" => 1.0, // Children become one parent
            _ => return Err(MetricsError::InvalidValue(
                format!("Invalid shard operation: {}", operation)
            )),
        };
        
        self.set_gauge("nerv_shards_active", current_shards, None).await?;
        
        // Increment operation counter
        self.increment_counter(
            &format!("nerv_shard_{}s_total", operation),
            1.0,
            None,
        ).await?;
        
        // Record operation duration
        self.record_duration(
            &format!("nerv_shard_{}_duration_seconds", operation),
            duration,
            Some(&MetricLabels::new().with_shard(parent_shard)),
        ).await?;
        
        debug!("Recorded shard {}: parent={}, children={:?}", 
               operation, parent_shard, child_shards);
        
        Ok(())
    }
    
    /// Record economic metrics
    pub async fn record_economic_metrics(
        &self,
        total_stake: f64,
        rewards_distributed: f64,
        shapley_value: f64,
        gradient_contributions: u64,
        shard_id: Option<u64>,
    ) -> Result<(), MetricsError> {
        // Update stake
        self.set_gauge("nerv_stake_total", total_stake, None).await?;
        
        // Update rewards
        self.set_gauge("nerv_rewards_distributed_total", rewards_distributed, None).await?;
        // New/enhanced record methods for privacy module
    pub async fn record_tee_operation(&self, tee_type: &str, operation: &str, duration: Duration, success: bool) -> Result<()> {
        let status = if success { "success" } else { "failure" };
        self.tee_operations_total.with_label_values(&[tee_type, operation, status]).inc();
        self.tee_operation_latency_ms.with_label_values(&[tee_type, operation]).observe(duration.as_millis() as f64);
        Ok(())
    }
   pub async fn record_cover_traffic(&self, direction: &str, count: u64) -> Result<()> {
        self.cover_traffic_total.with_label_values(&[direction]).inc_by(count);
        Ok(())
    }
   pub async fn record_mixer_operation(&self, operation: &str, duration: Duration) -> Result<()> {
        self.mixer_latency_ms.with_label_values(&[operation]).observe(duration.as_millis() as f64);
        Ok(())
    }
   pub async fn record_vdw_operation(&self, operation: &str, duration: Duration, success: bool) -> Result<()> {
        let status = if success { "success" } else { "failure" };
        self.vdw_operations_total.with_label_values(&[operation, status]).inc();
        self.vdw_operation_latency_ms.with_label_values(&[operation]).observe(duration.as_millis() as f64);
        Ok(())
    }
   pub async fn set_anonymity_set_size(&self, shard_id: u64, size: usize) -> Result<()> {
        self.anonymity_set_size.with_label_values(&[&shard_id.to_string()]).set(size as f64);
        Ok(())
    }
   // Timer utility (preserved)
    pub fn start_timer(&self, histogram: &HistogramVec, labels: &[&str]) -> Timer {
        Timer {
            histogram: histogram.clone(),
            labels: labels.iter().map(|s| s.to_string()).collect(),
            start: Instant::now(),
        }
    }
}
pub struct Timer {
    histogram: HistogramVec,
    labels: Vec,
    start: Instant,
}
impl Timer {
    pub fn stop(self) {
        let duration = self.start.elapsed();
        let label_values: Vec<&str> = self.labels.iter().map(|s| s.as_str()).collect();
        self.histogram.with_label_values(&label_values).observe(duration.as_millis() as f64);
    }
}


        // Record Shapley value
        self.observe_histogram("nerv_shapley_values", shapley_value, None).await?;
        
        // Record gradient contributions
        let labels = if let Some(shard) = shard_id {
            Some(MetricLabels::new().with_shard(shard))
        } else {
            None
        };
        
        self.increment_counter(
            "nerv_gradient_contributions_total",
            gradient_contributions as f64,
            labels.as_ref(),
        ).await?;
        
        Ok(())
    }
    
    /// Record anonymity set metrics
    pub async fn record_privacy_metrics(
        &self,
        anonymity_set_size: u64,
        mixer_latency: Duration,
        hops_processed: u64,
    ) -> Result<(), MetricsError> {
        // Update anonymity set size
        self.set_gauge("nerv_anonymity_set_size", anonymity_set_size as f64, None).await?;
        
        // Record mixer latency
        self.record_duration("nerv_mixer_latency_seconds", mixer_latency, None).await?;
        
        // Record onion routing hops
        for hop in 1..=5 {
            self.increment_counter(
                "nerv_onion_routing_hops_total",
                hops_processed as f64,
                Some(&MetricLabels::new().with_operation(&hop.to_string())),
            ).await?;
        }
        
        Ok(())
    }
}


// ============================================================================
// METRICS HELPER FUNCTIONS
// ============================================================================


/// Create a timer for measuring operation duration
pub struct Timer {
    start: Instant,
    metric_name: String,
    labels: Option<MetricLabels>,
    collector: Arc<MetricsCollector>,
}


impl Timer {
    pub fn new(
        collector: Arc<MetricsCollector>,
        metric_name: &str,
        labels: Option<MetricLabels>,
    ) -> Self {
        Self {
            start: Instant::now(),
            metric_name: metric_name.to_string(),
            labels,
            collector,
        }
    }
    
    pub async fn record(self) -> Result<(), MetricsError> {
        let duration = self.start.elapsed();
        self.collector.record_duration(
            &self.metric_name,
            duration,
            self.labels.as_ref(),
        ).await
    }
}


/// Macro for timing a code block and recording the duration
#[macro_export]
macro_rules! timed_metric {
    ($collector:expr, $name:expr, $labels:expr, $code:block) => {{
        let timer = $crate::utils::metrics::Timer::new(
            $collector.clone(),
            $name,
            $labels,
        );
        let result = $code;
        if let Err(e) = timer.record().await {
            tracing::warn!("Failed to record metric {}: {}", $name, e);
        }
        result
    }};
}


// ============================================================================
// TESTS
// ============================================================================


#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;
    
    #[tokio::test]
    async fn test_metrics_collector_creation() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config).unwrap();
        
        // Test that we can register metrics
        collector.register_counter(
            "test_counter",
            "A test counter",
            None,
        ).await.unwrap();
        
        collector.register_gauge(
            "test_gauge",
            "A test gauge",
            None,
        ).await.unwrap();
        
        collector.register_histogram(
            "test_histogram",
            "A test histogram",
            vec![0.1, 0.5, 1.0],
            None,
        ).await.unwrap();
        
        // Test incrementing counter
        collector.increment_counter("test_counter", 1.0, None).await.unwrap();
        
        // Test setting gauge
        collector.set_gauge("test_gauge", 42.0, None).await.unwrap();
        
        // Test observing histogram
        collector.observe_histogram("test_histogram", 0.3, None).await.unwrap();
        
        // Test JSON export
        let json = collector.get_metrics_json().await.unwrap();
        assert!(json.contains("test_counter"));
        assert!(json.contains("test_gauge"));
        assert!(json.contains("test_histogram"));
    }
    
    #[tokio::test]
    async fn test_metrics_with_labels() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config).unwrap();
        
        // Register counter with labels
        collector.register_counter(
            "labeled_counter",
            "A counter with labels",
            Some(vec!["shard", "operation"]),
        ).await.unwrap();
        
        // Increment with labels
        let labels = MetricLabels::new()
            .with_shard(1)
            .with_operation("transfer");
        
        collector.increment_counter("labeled_counter", 1.0, Some(&labels)).await.unwrap();
        
        // Test JSON export contains labels
        let json = collector.get_metrics_json().await.unwrap();
        assert!(json.contains("labeled_counter"));
        assert!(json.contains("shard"));
        assert!(json.contains("transfer"));
    }
    
    #[tokio::test]
    async fn test_transaction_recording() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config).unwrap();
        
        // Initialize metrics first
        collector.initialize_metrics().await.unwrap();
        
        // Record a transaction
        let latency = Duration::from_millis(150);
        collector.record_transaction(
            1,
            "transfer",
            latency,
            true,
        ).await.unwrap();
        collector.record_tee_operation("sev", "seal", Duration::from_millis(150), true).await.unwrap();
        collector.record_cover_traffic("sent", 10).await.unwrap();
        collector.record_vdw_operation("generate", Duration::from_millis(3000), true).await.unwrap();
        collector.record_mixer_operation("mix", Duration::from_millis(80)).await.unwrap();
        collector.set_anonymity_set_size(0, 1_000_000).await.unwrap();
        // Check that metrics were recorded
        let json = collector.get_metrics_json().await.unwrap();
        assert!(json.contains("nerv_transactions_processed_total"));
        assert!(json.contains("nerv_transaction_latency_seconds"));
    }
    
    #[tokio::test]
    async fn test_tee_attestation_recording() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config).unwrap();
        
        // Initialize metrics first
        collector.initialize_metrics().await.unwrap();
        
        // Record successful attestation
        let attestation_time = Duration::from_millis(50);
        collector.record_tee_attestation(
            "sgx",
            attestation_time,
            true,
        ).await.unwrap();
        
        // Record failed attestation
        collector.record_tee_attestation(
            "sev",
            Duration::from_millis(10),
            false,
        ).await.unwrap();
        
        let json = collector.get_metrics_json().await.unwrap();
        assert!(json.contains("nerv_tee_attestations_total"));
        assert!(json.contains("nerv_tee_failures_total"));
    }
    
    #[tokio::test]
    async fn test_shard_operation_recording() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config).unwrap();
        
        // Initialize metrics first
        collector.initialize_metrics().await.unwrap();
        
        // Record a shard split
        collector.record_shard_operation(
            "split",
            1,
            &[2, 3],
            Duration::from_millis(3400),
        ).await.unwrap();
        
        // Record a shard merge
        collector.record_shard_operation(
            "merge",
            2,
            &[4],
            Duration::from_millis(2800),
        ).await.unwrap();
        
        let json = collector.get_metrics_json().await.unwrap();
        assert!(json.contains("nerv_shard_splits_total"));
        assert!(json.contains("nerv_shard_merges_total"));
        assert!(json.contains("nerv_shards_active"));
    }
    
    #[tokio::test]
    async fn test_timer_utility() {
        let config = MetricsConfig::default();
        let collector = Arc::new(MetricsCollector::new(config).unwrap());
        
        // Register a histogram for the timer
        collector.register_histogram(
            "test_timer_histogram",
            "Test timer histogram",
            vec![0.1, 0.5, 1.0],
            None,
        ).await.unwrap();
        
        // Use the timer
        let timer = Timer::new(
            collector.clone(),
            "test_timer_histogram",
            None,
        );
        
        sleep(Duration::from_millis(100)).await;
        timer.record().await.unwrap();
        
        let json = collector.get_metrics_json().await.unwrap();
        assert!(json.contains("test_timer_histogram"));
    }
    
    #[tokio::test]
    async fn test_economic_metrics_recording() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config).unwrap();
        
        // Initialize metrics first
        collector.initialize_metrics().await.unwrap();
        
        // Record economic metrics
        collector.record_economic_metrics(
            1000000.0, // total stake
            50000.0,   // rewards distributed
            0.75,      // Shapley value
            1000,      // gradient contributions
            Some(1),   // shard ID
        ).await.unwrap();
        
        let json = collector.get_metrics_json().await.unwrap();
        assert!(json.contains("nerv_stake_total"));
        assert!(json.contains("nerv_rewards_distributed_total"));
        assert!(json.contains("nerv_shapley_values"));
        assert!(json.contains("nerv_gradient_contributions_total"));
    }
}
