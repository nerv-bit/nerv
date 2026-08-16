// tests/unit/mod.rs
// ============================================================================
// UNIT TESTS - Individual module testing
// ============================================================================


mod test_embedding;
mod test_crypto;
mod test_privacy;
mod test_consensus;
mod test_sharding;
mod test_economy;
mod test_network;
mod test_utils;


// Re-export for easier access
pub use test_embedding::*;
pub use test_crypto::*;
pub use test_privacy::*;
pub use test_consensus::*;
pub use test_sharding::*;
pub use test_economy::*;
pub use test_network::*;
pub use test_utils::*;


/// Test runner configuration
#[derive(Debug, Clone)]
pub struct TestConfig {
    pub max_test_duration_secs: u64,
    pub log_level: String,
    pub enable_coverage: bool,
    pub parallel_tests: bool,
}


impl Default for TestConfig {
    fn default() -> Self {
        Self {
            max_test_duration_secs: 30,
            log_level: "debug".to_string(),
            enable_coverage: false,
            parallel_tests: true,
        }
    }
}


/// Test result aggregator
pub struct TestRunner {
    config: TestConfig,
    results: Vec<TestResult>,
    start_time: std::time::Instant,
}


impl TestRunner {
    pub fn new(config: TestConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
            start_time: std::time::Instant::now(),
        }
    }
    
    pub fn add_result(&mut self, result: TestResult) {
        self.results.push(result);
    }
    
    pub fn summary(&self) -> TestSummary {
        let total = self.results.len();
        let passed = self.results.iter().filter(|r| r.passed).count();
        let failed = total - passed;
        let duration = self.start_time.elapsed();
        
        TestSummary {
            total,
            passed,
            failed,
            duration,
            results: self.results.clone(),
        }
    }
}


/// Individual test result
#[derive(Debug, Clone)]
pub struct TestResult {
    pub name: String,
    pub module: String,
    pub passed: bool,
    pub duration: std::time::Duration,
    pub error: Option<String>,
    pub assertions: usize,
}


/// Test summary
#[derive(Debug, Clone)]
pub struct TestSummary {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub duration: std::time::Duration,
    pub results: Vec<TestResult>,
}


impl TestSummary {
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.passed as f64 / self.total as f64
    }
    
    pub fn print(&self) {
        println!("\n=== UNIT TEST SUMMARY ===");
        println!("Total tests: {}", self.total);
        println!("Passed: {} ({}%)", self.passed, (self.success_rate() * 100.0) as i32);
        println!("Failed: {}", self.failed);
        println!("Duration: {:.2?}", self.duration);
        
        if self.failed > 0 {
            println!("\nFailed tests:");
            for result in &self.results {
                if !result.passed {
                    println!("  - {} ({:?})", result.name, result.duration);
                    if let Some(err) = &result.error {
                        println!("    Error: {}", err);
                    }
                }
            }
        }
    }
}
