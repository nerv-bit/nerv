// tests/unit/test_utils.rs
// ============================================================================
// UTILS MODULE UNIT TESTS
// ============================================================================


use nerv_bit::utils::serialization::*;
use nerv_bit::utils::metrics::*;
use nerv_bit::utils::logging::*;
use serde::{Serialize, Deserialize};


#[test]
fn test_protobuf_serialization() {
    // Create test VDW
    let vdw = VdwProto::new(
        [1u8; 32],            // tx_hash
        1,                    // shard_id
        100,                  // lattice_height
        vec![2u8; 500],       // delta_proof
        [3u8; 32],            // final_root
        vec![4u8; 100],       // attestation
        vec![5u8; 100],       // signature
        1234567890,           // timestamp
        1,                    // monotonic_counter
    );
    
    // Serialize
    let bytes = vdw.to_protobuf().unwrap();
    assert!(bytes.len() > 0);
    
    // Deserialize
    let restored = VdwProto::from_protobuf(&bytes).unwrap();
    
    assert_eq!(vdw.tx_hash, restored.tx_hash);
    assert_eq!(vdw.shard_id, restored.shard_id);
    assert_eq!(vdw.lattice_height, restored.lattice_height);
    assert_eq!(vdw.delta_proof, restored.delta_proof);
    assert_eq!(vdw.final_root, restored.final_root);
    assert_eq!(vdw.timestamp, restored.timestamp);
    assert_eq!(vdw.monotonic_counter, restored.monotonic_counter);
    
    // Test size validation
    let is_valid = vdw.validate_size();
    assert!(is_valid.is_ok(), "VDW size should be valid");
}


#[test]
fn test_fixed_bytes_serialization() {
    let original = FixedBytes::new([1, 2, 3, 4, 5]);
    
    // Convert to bytes
    let bytes: Vec<u8> = original.clone().into();
    assert_eq!(bytes, vec![1, 2, 3, 4, 5]);
    
    // Convert back
    let restored = FixedBytes::<5>::try_from(bytes).unwrap();
    assert_eq!(original, restored);
    
    // Test invalid size
    let wrong_bytes = vec![1, 2, 3];
    let result = FixedBytes::<5>::try_from(wrong_bytes);
    assert!(result.is_err(), "Should fail with wrong size");
}


#[test]
fn test_embedding_vec_operations() {
    let mut values1 = [0.0; 512];
    let mut values2 = [0.0; 512];
    
    for i in 0..512 {
        values1[i] = i as f32;
        values2[i] = (i * 2) as f32;
    }
    
    let e1 = EmbeddingVec::new(values1);
    let e2 = EmbeddingVec::new(values2);
    
    // Test addition
    let sum = e1.add(&e2);
    for i in 0..512 {
        assert!((sum.0[i] - (i as f32 + (i * 2) as f32)).abs() < 0.0001);
    }
    
    // Test subtraction
    let diff = sum.sub(&e2);
    assert!(diff.approx_eq(&e1, 0.0001));
    
    // Test from_slice
    let slice: Vec<f32> = (0..512).map(|i| i as f32).collect();
    let from_slice = EmbeddingVec::from_slice(&slice).unwrap();
    assert!(from_slice.approx_eq(&e1, 0.0001));
    
    // Test invalid slice size
    let invalid_slice = vec![1.0; 100];
    let result = EmbeddingVec::from_slice(&invalid_slice);
    assert!(result.is_err(), "Should fail with wrong size");
}


#[test]
fn test_varint_encoding() {
    let test_cases = vec![
        (0u64, vec![0x00]),
        (1, vec![0x01]),
        (127, vec![0x7F]),
        (128, vec![0x80, 0x01]),
        (300, vec![0xAC, 0x02]),
        (u64::MAX, vec![0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]),
    ];
    
    for (value, expected) in test_cases {
        let encoded = encode_varint(value);
        assert_eq!(encoded, expected, "Failed for value {}", value);
        
        let mut reader = &encoded[..];
        let decoded = decode_varint(&mut reader).unwrap();
        assert_eq!(decoded, value, "Failed for value {}", value);
    }
}


#[test]
fn test_limited_reader_writer() {
    // Test LimitedReader
    let data = vec![1, 2, 3, 4, 5];
    let mut reader = LimitedReader::new(&data[..], 3);
    
    let mut buf = [0u8; 2];
    let n = reader.read(&mut buf).unwrap();
    assert_eq!(n, 2);
    assert_eq!(buf, [1, 2]);
    
    // Try to read more than limit
    let n = reader.read(&mut buf).unwrap();
    assert_eq!(n, 1); // Only 1 byte left before hitting limit
    assert_eq!(buf[0], 3);
    
    // Should return 0 now (limit reached)
    let n = reader.read(&mut buf).unwrap();
    assert_eq!(n, 0);
    
    // Test LimitedWriter
    let mut writer = LimitedWriter::new(Vec::new(), 3);
    
    writer.write_all(&[1, 2]).unwrap();
    writer.write_all(&[3]).unwrap();
    
    // Should fail to write more
    let result = writer.write_all(&[4]);
    assert!(result.is_err(), "Should fail when limit exceeded");
}


#[test]
fn test_serialize_with_limit() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestStruct {
        id: u64,
        name: String,
        value: f64,
    }
    
    let data = TestStruct {
        id: 1,
        name: "test".to_string(),
        value: 3.14,
    };
    
    // Serialize with limit
    let bytes = serialize_with_limit(&data, 1000).unwrap();
    
    // Deserialize
    let restored: TestStruct = deserialize_with_limit(&bytes, 1000).unwrap();
    
    assert_eq!(data.id, restored.id);
    assert_eq!(data.name, restored.name);
    assert!((data.value - restored.value).abs() < 0.0001);
    
    // Test with too small limit
    let result = serialize_with_limit(&data, 10);
    assert!(result.is_err(), "Should fail with too small limit");
}


#[test]
fn test_metrics_collector_creation() {
    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(config).unwrap();
    
    // Test basic metric registration
    collector.register_counter(
        "test_counter",
        "A test counter",
        None,
    ).unwrap();
    
    collector.register_gauge(
        "test_gauge",
        "A test gauge",
        None,
    ).unwrap();
    
    // Test metric operations
    collector.increment_counter("test_counter", 1.0, None).unwrap();
    collector.set_gauge("test_gauge", 42.0, None).unwrap();
    
    // Test JSON export
    let json = collector.get_metrics_json().unwrap();
    assert!(json.contains("test_counter"));
    assert!(json.contains("test_gauge"));
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
    ).unwrap();
    
    // Increment with labels
    let labels = MetricLabels::new()
        .with_shard(1)
        .with_operation("transfer");
    
    collector.increment_counter("labeled_counter", 1.0, Some(&labels)).unwrap();
    
    // Test JSON export contains labels
    let json = collector.get_metrics_json().unwrap();
    assert!(json.contains("labeled_counter"));
    assert!(json.contains("shard"));
    assert!(json.contains("transfer"));
}


#[test]
fn test_logging_configuration() {
    let config = LogConfig {
        level: "debug".to_string(),
        file_path: Some("/tmp/test.log".to_string()),
        enable_json: true,
        max_file_size_mb: 10,
        max_files: 5,
    };
    
    let logger = Logger::new(config).unwrap();
    
    // Test logging methods
    logger.debug("Debug message").unwrap();
    logger.info("Info message").unwrap();
    logger.warn("Warning message").unwrap();
    logger.error("Error message").unwrap();
    
    // Test with context
    let context = vec![
        ("module".to_string(), "test".to_string()),
        ("shard".to_string(), "1".to_string()),
    ];
    
    logger.info_with_context("Message with context", &context).unwrap();
}


#[test]
fn test_error_handling() {
    use nerv_bit::utils::errors::*;
    
    // Test NervError
    let io_error = NervError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "file not found"));
    let serialization_error = NervError::Serialization("parse error".to_string());
    let crypto_error = NervError::Crypto("invalid key".to_string());
    let network_error = NervError::NeuralNetwork("training failed".to_string());
    
    // Test formatting
    assert!(format!("{}", io_error).contains("IO error"));
    assert!(format!("{}", serialization_error).contains("Serialization error"));
    assert!(format!("{}", crypto_error).contains("Cryptographic error"));
    assert!(format!("{}", network_error).contains("Neural network error"));
    
    // Test result type
    let result: Result<()> = Err(NervError::Io(std::io::Error::new(std::io::ErrorKind::Other, "test")));
    assert!(result.is_err());
}


#[test]
fn test_config_management() {
    use nerv_bit::utils::config::*;
    
    let mut config = Config::new();
    
    // Set values
    config.set("database.url", "postgres://localhost:5432/nerv").unwrap();
    config.set("network.port", "8080").unwrap();
    config.set("consensus.timeout", "5000").unwrap();
    
    // Get values
    let db_url = config.get("database.url").unwrap();
    let port: u16 = config.get_parsed("network.port").unwrap();
    let timeout: u64 = config.get_parsed("consensus.timeout").unwrap();
    
    assert_eq!(db_url, "postgres://localhost:5432/nerv");
    assert_eq!(port, 8080);
    assert_eq!(timeout, 5000);
    
    // Test non-existent key
    let result = config.get("nonexistent.key");
    assert!(result.is_err());
    
    // Test save/load (would need filesystem)
    let temp_file = std::env::temp_dir().join("test_config.json");
    
    config.save(&temp_file).unwrap();
    assert!(temp_file.exists());
    
    let loaded_config = Config::load(&temp_file).unwrap();
    let loaded_db_url = loaded_config.get("database.url").unwrap();
    assert_eq!(loaded_db_url, db_url);
    
    // Cleanup
    std::fs::remove_file(temp_file).ok();
}


#[test]
fn test_performance_monitoring() {
    use nerv_bit::utils::performance::*;
    
    let mut monitor = PerformanceMonitor::new();
    
    // Start measurement
    monitor.start_measurement("test_operation");
    
    // Simulate some work
    std::thread::sleep(std::time::Duration::from_millis(100));
    
    // Stop measurement
    let duration = monitor.stop_measurement("test_operation").unwrap();
    
    assert!(duration.as_millis() >= 100);
    
    // Get statistics
    let stats = monitor.get_statistics("test_operation").unwrap();
    assert_eq!(stats.count, 1);
    assert!(stats.average_ms >= 100.0);
    
    // Test multiple measurements
    for _ in 0..4 {
        monitor.start_measurement("test_operation");
        std::thread::sleep(std::time::Duration::from_millis(50));
        monitor.stop_measurement("test_operation").unwrap();
    }
    
    let stats = monitor.get_statistics("test_operation").unwrap();
    assert_eq!(stats.count, 5);
    assert!(stats.average_ms >= 70.0 && stats.average_ms <= 80.0);
}


#[test]
fn test_cache_operations() {
    use nerv_bit::utils::cache::*;
    
    let mut cache = LruCache::new(3); // Max 3 items
    
    // Insert items
    cache.insert("key1", "value1");
    cache.insert("key2", "value2");
    cache.insert("key3", "value3");
    
    assert_eq!(cache.len(), 3);
    
    // Access key1 to make it recently used
    let value = cache.get(&"key1").unwrap();
    assert_eq!(value, &"value1");
    
    // Insert another item (should evict key2 as least recently used)
    cache.insert("key4", "value4");
    
    assert_eq!(cache.len(), 3);
    assert!(cache.get(&"key2").is_none()); // Should be evicted
    assert!(cache.get(&"key1").is_some()); // Should still be there
    assert!(cache.get(&"key3").is_some());
    assert!(cache.get(&"key4").is_some());
    
    // Remove item
    cache.remove(&"key1");
    assert_eq!(cache.len(), 2);
    assert!(cache.get(&"key1").is_none());
    
    // Clear cache
    cache.clear();
    assert_eq!(cache.len(), 0);
}


// Placeholder implementations for utils module


#[derive(Debug)]
pub enum NervError {
    Io(std::io::Error),
    Serialization(String),
    Crypto(String),
    NeuralNetwork(String),
}


impl std::fmt::Display for NervError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            Self::Crypto(msg) => write!(f, "Cryptographic error: {}", msg),
            Self::NeuralNetwork(msg) => write!(f, "Neural network error: {}", msg),
        }
    }
}


impl std::error::Error for NervError {}


impl From<std::io::Error> for NervError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}


type Result<T> = std::result::Result<T, NervError>;


mod config {
    use super::*;
    
    pub struct Config {
        data: std::collections::HashMap<String, String>,
    }
    
    impl Config {
        pub fn new() -> Self {
            Self {
                data: std::collections::HashMap::new(),
            }
        }
        
        pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
            self.data.insert(key.to_string(), value.to_string());
            Ok(())
        }
        
        pub fn get(&self, key: &str) -> Result<String> {
            self.data
                .get(key)
                .cloned()
                .ok_or_else(|| NervError::Serialization(format!("Key not found: {}", key)))
        }
        
        pub fn get_parsed<T: std::str::FromStr>(&self, key: &str) -> Result<T> {
            let value = self.get(key)?;
            value.parse()
                .map_err(|_| NervError::Serialization(format!("Failed to parse value for key: {}", key)))
        }
        
        pub fn save(&self, path: &std::path::Path) -> Result<()> {
            let json = serde_json::to_string_pretty(&self.data)
                .map_err(|e| NervError::Serialization(e.to_string()))?;
            std::fs::write(path, json)?;
            Ok(())
        }
        
        pub fn load(path: &std::path::Path) -> Result<Self> {
            let json = std::fs::read_to_string(path)?;
            let data: std::collections::HashMap<String, String> = serde_json::from_str(&json)
                .map_err(|e| NervError::Serialization(e.to_string()))?;
            Ok(Self { data })
        }
    }
}


mod performance {
    use std::time::{Duration, Instant};
    
    #[derive(Debug, Clone)]
    pub struct PerformanceStats {
        pub count: u64,
        pub total_ms: f64,
        pub average_ms: f64,
        pub min_ms: f64,
        pub max_ms: f64,
    }
    
    pub struct PerformanceMonitor {
        measurements: std::collections::HashMap<String, Vec<Duration>>,
        active: std::collections::HashMap<String, Instant>,
    }
    
    impl PerformanceMonitor {
        pub fn new() -> Self {
            Self {
                measurements: std::collections::HashMap::new(),
                active: std::collections::HashMap::new(),
            }
        }
        
        pub fn start_measurement(&mut self, name: &str) {
            self.active.insert(name.to_string(), Instant::now());
        }
        
        pub fn stop_measurement(&mut self, name: &str) -> Option<Duration> {
            if let Some(start) = self.active.remove(name) {
                let duration = start.elapsed();
                self.measurements
                    .entry(name.to_string())
                    .or_insert_with(Vec::new)
                    .push(duration);
                Some(duration)
            } else {
                None
            }
        }
        
        pub fn get_statistics(&self, name: &str) -> Option<PerformanceStats> {
            self.measurements.get(name).map(|durations| {
                let count = durations.len() as u64;
                let total_ms: f64 = durations.iter().map(|d| d.as_secs_f64() * 1000.0).sum();
                let average_ms = total_ms / count as f64;
                let min_ms = durations.iter().map(|d| d.as_secs_f64() * 1000.0).fold(f64::INFINITY, f64::min);
                let max_ms = durations.iter().map(|d| d.as_secs_f64() * 1000.0).fold(f64::NEG_INFINITY, f64::max);
                
                PerformanceStats {
                    count,
                    total_ms,
                    average_ms,
                    min_ms,
                    max_ms,
                }
            })
        }
    }
}


mod cache {
    use std::collections::HashMap;
    use std::hash::Hash;
    
    pub struct LruCache<K, V> {
        capacity: usize,
        cache: HashMap<K, (V, usize)>,
        counter: usize,
    }
    
    impl<K: Eq + Hash + Clone, V> LruCache<K, V> {
        pub fn new(capacity: usize) -> Self {
            Self {
                capacity,
                cache: HashMap::new(),
                counter: 0,
            }
        }
        
        pub fn insert(&mut self, key: K, value: V) {
            self.counter += 1;
            self.cache.insert(key, (value, self.counter));
            
            if self.cache.len() > self.capacity {
                // Find and remove least recently used
                let lru_key = self.cache.iter()
                    .min_by_key(|(_, (_, timestamp))| timestamp)
                    .map(|(k, _)| k.clone());
                
                if let Some(key) = lru_key {
                    self.cache.remove(&key);
                }
            }
        }
        
        pub fn get(&mut self, key: &K) -> Option<&V> {
            if let Some((value, timestamp)) = self.cache.get_mut(key) {
                self.counter += 1;
                *timestamp = self.counter;
                Some(value)
            } else {
                None
            }
        }
        
        pub fn remove(&mut self, key: &K) -> Option<V> {
            self.cache.remove(key).map(|(value, _)| value)
        }
        
        pub fn len(&self) -> usize {
            self.cache.len()
        }
        
        pub fn clear(&mut self) {
            self.cache.clear();
            self.counter = 0;
        }
    }
}

