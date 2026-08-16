// src/utils/logging.rs
// ============================================================================
// LOGGING AND TRACING CONFIGURATION
// ============================================================================
// This module configures structured logging and distributed tracing for NERV.
// We use the `tracing` ecosystem for:
// 1. Structured logging with JSON format in production
// 2. Distributed tracing across TEE enclaves and network nodes
// 3. Performance monitoring and debugging
// ============================================================================

use std::path::PathBuf;
use tracing::{Level, Subscriber};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{fmt, prelude::*, EnvFilter, Registry};
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ============================================================================
// ERROR DEFINITIONS
// ============================================================================

#[derive(Error, Debug)]
pub enum LoggingError {
    #[error("Failed to initialize logging: {0}")]
    Initialization(String),
    
    #[error("Failed to create log directory: {0}")]
    DirectoryCreation(#[from] std::io::Error),
    
    #[error("Invalid log level: {0}")]
    InvalidLevel(String),
}

// ============================================================================
// CONFIGURATION STRUCTURES
// ============================================================================

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (error, warn, info, debug, trace)
    pub level: LogLevel,
    
    /// Log format (text, json, pretty)
    pub format: LogFormat,
    
    /// Log directory (if file logging is enabled)
    pub log_dir: Option<PathBuf>,
    
    /// Enable file rotation (daily)
    pub enable_file_logging: bool,
    
    /// Enable stdout logging
    pub enable_stdout: bool,
    
    /// Enable distributed tracing
    pub enable_tracing: bool,
    
    /// Tracing collector endpoint (e.g., Jaeger)
    pub tracing_endpoint: Option<String>,
    
    /// Service name for distributed tracing
    pub service_name: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            format: LogFormat::Text,
            log_dir: Some(PathBuf::from("./logs")),
            enable_file_logging: true,
            enable_stdout: true,
            enable_tracing: false,
            tracing_endpoint: None,
            service_name: "nerv-node".to_string(),
        }
    }
}

/// Log level enumeration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl From<LogLevel> for Level {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Error => Level::ERROR,
            LogLevel::Warn => Level::WARN,
            LogLevel::Info => Level::INFO,
            LogLevel::Debug => Level::DEBUG,
            LogLevel::Trace => Level::TRACE,
        }
    }
}

impl TryFrom<&str> for LogLevel {
    type Error = LoggingError;
    
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "error" => Ok(LogLevel::Error),
            "warn" => Ok(LogLevel::Warn),
            "info" => Ok(LogLevel::Info),
            "debug" => Ok(LogLevel::Debug),
            "trace" => Ok(LogLevel::Trace),
            _ => Err(LoggingError::InvalidLevel(s.to_string())),
        }
    }
}

/// Log format enumeration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LogFormat {
    Text,    // Human-readable text
    Json,    // Structured JSON
    Pretty,  // Colored, pretty-printed text
}

// ============================================================================
// LOGGING GUARD (keeps non-blocking writer alive)
// ============================================================================

/// Guard that keeps the non-blocking logging writer alive
/// Must be kept in scope for the duration of the program
pub struct LoggingGuard {
    _file_guard: Option<WorkerGuard>,
    _tracing_guard: Option<tracing_opentelemetry::OpenTelemetryLayerGuard>,
}

impl LoggingGuard {
    pub fn new(
        file_guard: Option<WorkerGuard>,
        tracing_guard: Option<tracing_opentelemetry::OpenTelemetryLayerGuard>,
    ) -> Self {
        Self {
            _file_guard: file_guard,
            _tracing_guard: tracing_guard,
        }
    }
}

// ============================================================================
// LOGGING INITIALIZATION
// ============================================================================

/// Initialize logging subsystem with given configuration
/// Returns a guard that must be kept alive
pub fn init_logging(config: &LoggingConfig) -> Result<LoggingGuard, LoggingError> {
    // Create log directory if needed
    if let Some(log_dir) = &config.log_dir {
        if config.enable_file_logging {
            std::fs::create_dir_all(log_dir).map_err(LoggingError::DirectoryCreation)?;
        }
    }
    
    // Build the subscriber layer by layer
    let mut layers = Vec::new();
    let mut file_guard = None;
    
    // 1. Console logging layer
    if config.enable_stdout {
        let console_layer = match config.format {
            LogFormat::Text => fmt::layer()
                .with_writer(std::io::stdout)
                .with_target(true)
                .with_level(true)
                .with_thread_ids(false)
                .with_thread_names(false)
                .boxed(),
            
            LogFormat::Json => fmt::layer()
                .json()
                .with_writer(std::io::stdout)
                .with_target(true)
                .with_level(true)
                .with_thread_ids(true)
                .with_thread_names(true)
                .with_current_span(true)
                .with_file(true)
                .with_line_number(true)
                .boxed(),
            
            LogFormat::Pretty => fmt::layer()
                .pretty()
                .with_writer(std::io::stdout)
                .with_target(true)
                .with_level(true)
                .with_thread_ids(true)
                .with_thread_names(true)
                .boxed(),
        };
        
        layers.push(console_layer);
    }
    
    // 2. File logging layer (with rotation)
    if config.enable_file_logging {
        if let Some(log_dir) = &config.log_dir {
            let file_appender = tracing_appender::rolling::daily(log_dir, "nerv.log");
            let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
            
            let file_layer = fmt::layer()
                .json() // Always JSON for file logs for structured analysis
                .with_writer(non_blocking)
                .with_target(true)
                .with_level(true)
                .with_thread_ids(true)
                .with_thread_names(true)
                .with_file(true)
                .with_line_number(true)
                .boxed();
            
            layers.push(file_layer);
            file_guard = Some(guard);
        }
    }
    
    // 3. Distributed tracing layer (OpenTelemetry)
    let mut tracing_guard = None;
    if config.enable_tracing {
        if let Some(endpoint) = &config.tracing_endpoint {
            // In production, we would set up OpenTelemetry here
            // For now, we'll just log that tracing is enabled
            tracing::info!("Distributed tracing enabled to {}", endpoint);
            
            // Example setup (commented out as it requires otel dependencies):
            /*
            let tracer = opentelemetry_jaeger::new_pipeline()
                .with_service_name(&config.service_name)
                .with_endpoint(endpoint)
                .install_batch(opentelemetry::runtime::Tokio)
                .map_err(|e| LoggingError::Initialization(e.to_string()))?;
            
            let telemetry_layer = tracing_opentelemetry::layer().with_tracer(tracer);
            layers.push(telemetry_layer.boxed());
            
            // Create a guard to ensure traces are flushed on shutdown
            let guard = tracing_opentelemetry::OpenTelemetryLayerGuard::new();
            tracing_guard = Some(guard);
            */
        }
    }
    
    // 4. Filter layer (based on log level)
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| {
            let level: Level = config.level.into();
            EnvFilter::new(format!("{}={}", config.service_name, level))
        });
    
    layers.push(env_filter.boxed());
    
    // 5. Install the subscriber
    let subscriber = Registry::default().with(layers);
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| LoggingError::Initialization(e.to_string()))?;
    
    // Log initialization message
    tracing::info!(
        "Logging initialized with level {:?}, format {:?}",
        config.level,
        config.format
    );
    tracing::debug!("Debug logging enabled");
    tracing::trace!("Trace logging enabled");
    
    Ok(LoggingGuard::new(file_guard, tracing_guard))
}

/// Initialize logging with default configuration
pub fn init_default_logging() -> Result<LoggingGuard, LoggingError> {
    let config = LoggingConfig::default();
    init_logging(&config)
}

/// Initialize logging from environment variables
pub fn init_logging_from_env() -> Result<LoggingGuard, LoggingError> {
    let config = LoggingConfig {
        level: std::env::var("NERV_LOG_LEVEL")
            .ok()
            .and_then(|s| LogLevel::try_from(s.as_str()).ok())
            .unwrap_or(LogLevel::Info),
        
        format: std::env::var("NERV_LOG_FORMAT")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "json" => Some(LogFormat::Json),
                "pretty" => Some(LogFormat::Pretty),
                "text" => Some(LogFormat::Text),
                _ => None,
            })
            .unwrap_or(LogFormat::Text),
        
        log_dir: std::env::var("NERV_LOG_DIR")
            .ok()
            .map(PathBuf::from),
        
        enable_file_logging: std::env::var("NERV_LOG_FILE")
            .map(|s| s.to_lowercase() == "true")
            .unwrap_or(true),
        
        enable_stdout: std::env::var("NERV_LOG_STDOUT")
            .map(|s| s.to_lowercase() == "true")
            .unwrap_or(true),
        
        enable_tracing: std::env::var("NERV_TRACING_ENABLED")
            .map(|s| s.to_lowercase() == "true")
            .unwrap_or(false),
        
        tracing_endpoint: std::env::var("NERV_TRACING_ENDPOINT").ok(),
        
        service_name: std::env::var("NERV_SERVICE_NAME")
            .unwrap_or_else(|_| "nerv-node".to_string()),
    };
    
    init_logging(&config)
}

// ============================================================================
// STRUCTURED LOGGING MACROS AND HELPERS
// ============================================================================

/// Log a transaction with structured fields
pub fn log_transaction(tx_hash: &[u8], shard_id: u64, action: &str) {
    tracing::info!(
        tx_hash = hex::encode(tx_hash),
        shard_id = shard_id,
        action = action,
        "Transaction processed"
    );
}

/// Log a batch update with embedding information
pub fn log_batch_update(batch_id: u64, batch_size: usize, embedding_hash: &[u8]) {
    tracing::debug!(
        batch_id = batch_id,
        batch_size = batch_size,
        embedding_hash = hex::encode(&embedding_hash[..8]), // First 8 bytes for brevity
        "Batch update applied"
    );
}

/// Log TEE attestation events
pub fn log_tee_attestation(tee_type: &str, enclave_hash: &[u8], success: bool) {
    if success {
        tracing::info!(
            tee_type = tee_type,
            enclave_hash = hex::encode(&enclave_hash[..8]),
            "TEE attestation successful"
        );
    } else {
        tracing::error!(
            tee_type = tee_type,
            enclave_hash = hex::encode(&enclave_hash[..8]),
            "TEE attestation failed"
        );
    }
}

/// Log consensus events
pub fn log_consensus_event(height: u64, round: u64, event_type: &str, details: &str) {
    tracing::info!(
        height = height,
        round = round,
        event_type = event_type,
        details = details,
        "Consensus event"
    );
}

/// Log shard split/merge events
pub fn log_shard_event(parent_shard: u64, child_shards: &[u64], event_type: &str, duration_ms: u64) {
    tracing::info!(
        parent_shard = parent_shard,
        child_shards = format!("{:?}", child_shards),
        event_type = event_type,
        duration_ms = duration_ms,
        "Shard event"
    );
}

/// Log performance metrics
pub fn log_performance_metric(metric_name: &str, value: f64, unit: &str) {
    tracing::debug!(
        metric = metric_name,
        value = value,
        unit = unit,
        "Performance metric"
    );
}

// ============================================================================
// TRACING SPANS FOR DISTRIBUTED CONTEXT PROPAGATION
// ============================================================================

/// Create a span for transaction processing
pub fn transaction_span(tx_hash: &[u8]) -> tracing::Span {
    tracing::info_span!(
        "transaction",
        tx_hash = hex::encode(tx_hash),
        otel.kind = "PRODUCER"
    )
}

/// Create a span for batch processing
pub fn batch_span(batch_id: u64) -> tracing::Span {
    tracing::info_span!(
        "batch",
        batch_id = batch_id,
        otel.kind = "INTERNAL"
    )
}

/// Create a span for consensus round
pub fn consensus_span(height: u64, round: u64) -> tracing::Span {
    tracing::info_span!(
        "consensus",
        height = height,
        round = round,
        otel.kind = "CONSUMER"
    )
}

/// Create a span for TEE operation
pub fn tee_span(operation: &str, enclave_id: &str) -> tracing::Span {
    tracing::info_span!(
        "tee_operation",
        operation = operation,
        enclave_id = enclave_id,
        otel.kind = "INTERNAL"
    )
}

// ============================================================================
// METRICS AND MONITORING HELPERS
// ============================================================================

/// Record a counter metric
pub fn record_counter(name: &str, value: u64, labels: &[(&str, &str)]) {
    let label_str = labels
        .iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join(",");
    
    tracing::debug!(
        metric_type = "counter",
        metric_name = name,
        value = value,
        labels = label_str,
        "Counter metric recorded"
    );
}

/// Record a gauge metric
pub fn record_gauge(name: &str, value: f64, labels: &[(&str, &str)]) {
    let label_str = labels
        .iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join(",");
    
    tracing::debug!(
        metric_type = "gauge",
        metric_name = name,
        value = value,
        labels = label_str,
        "Gauge metric recorded"
    );
}

/// Record a histogram metric
pub fn record_histogram(name: &str, value: f64, labels: &[(&str, &str)]) {
    let label_str = labels
        .iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join(",");
    
    tracing::debug!(
        metric_type = "histogram",
        metric_name = name,
        value = value,
        labels = label_str,
        "Histogram metric recorded"
    );
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_log_level_conversion() {
        assert!(matches!(LogLevel::try_from("info").unwrap(), LogLevel::Info));
        assert!(matches!(LogLevel::try_from("DEBUG").unwrap(), LogLevel::Debug));
        assert!(matches!(LogLevel::try_from("trace").unwrap(), LogLevel::Trace));
        
        assert!(LogLevel::try_from("invalid").is_err());
    }
    
    #[test]
    fn test_log_level_to_tracing_level() {
        let level: Level = LogLevel::Info.into();
        assert_eq!(level, Level::INFO);
        
        let level: Level = LogLevel::Error.into();
        assert_eq!(level, Level::ERROR);
    }
    
    #[test]
    fn test_structured_logging_functions() {
        // These functions should not panic
        log_transaction(&[1, 2, 3, 4], 1, "transfer");
        log_batch_update(1, 256, &[0u8; 32]);
        log_tee_attestation("sgx", &[0u8; 32], true);
        log_consensus_event(100, 1, "vote", "67% agreement");
        log_shard_event(1, &[2, 3], "split", 3400);
        log_performance_metric("tps", 1000.0, "tx/s");
    }
    
    #[test]
    fn test_metrics_recording() {
        // These functions should not panic
        record_counter("transactions_processed", 1000, &[("shard", "1")]);
        record_gauge("memory_usage", 1024.0, &[("node", "validator-1")]);
        record_histogram("response_time", 150.0, &[("endpoint", "/api/v1/tx")]);
    }
    
    #[test]
    fn test_config_defaults() {
        let config = LoggingConfig::default();
        assert!(matches!(config.level, LogLevel::Info));
        assert!(matches!(config.format, LogFormat::Text));
        assert!(config.enable_file_logging);
        assert!(config.enable_stdout);
        assert!(!config.enable_tracing);
    }
}
