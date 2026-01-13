// tests/integration/mod.rs
// ============================================================================
// INTEGRATION TESTS - Multi-module interaction testing
// ============================================================================


mod test_blockchain_workflow;
mod test_privacy_protocol;
mod test_consensus_flow;
mod test_sharding_dynamics;
mod test_economy_ecosystem;
mod test_network_simulation;


// Re-export for easier access
pub use test_blockchain_workflow::*;
pub use test_privacy_protocol::*;
pub use test_consensus_flow::*;
pub use test_sharding_dynamics::*;
pub use test_economy_ecosystem::*;
pub use test_network_simulation::*;


/// Integration test configuration
#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    pub test_duration_secs: u64,
    pub node_count: usize,
    pub transaction_count: u64,
    pub enable_metrics: bool,
    pub enable_logging: bool,
    pub random_seed: u64,
}


impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            test_duration_secs: 60,
            node_count: 3,
            transaction_count: 100,
            enable_metrics: true,
            enable_logging: false,
            random_seed: 42,
        }
    }
}


/// Integration test environment
pub struct TestEnvironment {
    pub config: IntegrationConfig,
    pub nodes: Vec<TestNode>,
    pub transactions: Vec<TestTransaction>,
    pub metrics: Option<IntegrationMetrics>,
    pub start_time: std::time::Instant,
}


impl TestEnvironment {
    pub fn new(config: IntegrationConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            transactions: Vec::new(),
            metrics: None,
            start_time: std::time::Instant::now(),
        }
    }
    
    pub fn add_node(&mut self, node: TestNode) {
        self.nodes.push(node);
    }
    
    pub fn add_transaction(&mut self, tx: TestTransaction) {
        self.transactions.push(tx);
    }
    
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
    
    pub fn is_timeout(&self) -> bool {
        self.elapsed().as_secs() > self.config.test_duration_secs
    }
}


/// Test node representation
#[derive(Debug, Clone)]
pub struct TestNode {
    pub id: u64,
    pub address: String,
    pub stake: f64,
    pub is_validator: bool,
    pub is_tee_enabled: bool,
    pub state: NodeState,
}


impl TestNode {
    pub fn new(id: u64, address: &str, stake: f64, is_validator: bool) -> Self {
        Self {
            id,
            address: address.to_string(),
            stake,
            is_validator,
            is_tee_enabled: true,
            state: NodeState::Initializing,
        }
    }
}


/// Node state
#[derive(Debug, Clone, PartialEq)]
pub enum NodeState {
    Initializing,
    Syncing,
    Ready,
    Validating,
    Error(String),
}


/// Test transaction
#[derive(Debug, Clone)]
pub struct TestTransaction {
    pub id: [u8; 32],
    pub sender: [u8; 32],
    pub receiver: [u8; 32],
    pub amount: f64,
    pub fee: f64,
    pub timestamp: u64,
    pub status: TransactionStatus,
}


impl TestTransaction {
    pub fn new(sender: [u8; 32], receiver: [u8; 32], amount: f64, fee: f64) -> Self {
        let mut id = [0u8; 32];
        rand::thread_rng().fill(&mut id);
        
        Self {
            id,
            sender,
            receiver,
            amount,
            fee,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            status: TransactionStatus::Pending,
        }
    }
}


/// Transaction status
#[derive(Debug, Clone, PartialEq)]
pub enum TransactionStatus {
    Pending,
    Processing,
    Confirmed,
    Failed(String),
}


/// Integration test metrics
#[derive(Debug, Clone)]
pub struct IntegrationMetrics {
    pub transactions_processed: u64,
    pub blocks_produced: u64,
    pub consensus_rounds: u64,
    pub shard_operations: u64,
    pub tee_attestations: u64,
    pub avg_latency_ms: f64,
    pub success_rate: f64,
    pub error_count: u64,
}


impl Default for IntegrationMetrics {
    fn default() -> Self {
        Self {
            transactions_processed: 0,
            blocks_produced: 0,
            consensus_rounds: 0,
            shard_operations: 0,
            tee_attestations: 0,
            avg_latency_ms: 0.0,
            success_rate: 1.0,
            error_count: 0,
        }
    }
}


/// Integration test result
#[derive(Debug, Clone)]
pub struct IntegrationResult {
    pub test_name: String,
    pub duration: std::time::Duration,
    pub metrics: IntegrationMetrics,
    pub passed: bool,
    pub errors: Vec<String>,
}


impl IntegrationResult {
    pub fn summary(&self) -> String {
        format!(
            "Test: {} | Duration: {:.2?} | TPS: {:.1} | Success: {:.1}% | Errors: {}",
            self.test_name,
            self.duration,
            self.metrics.transactions_processed as f64 / self.duration.as_secs_f64().max(1.0),
            self.metrics.success_rate * 100.0,
            self.metrics.error_count
        )
    }
}


/// Integration test runner
pub struct IntegrationRunner {
    pub config: IntegrationConfig,
    pub results: Vec<IntegrationResult>,
}


impl IntegrationRunner {
    pub fn new(config: IntegrationConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }
    
    pub fn add_result(&mut self, result: IntegrationResult) {
        self.results.push(result);
    }
    
    pub fn print_summary(&self) {
        println!("\n=== INTEGRATION TEST SUMMARY ===");
        println!("Total tests: {}", self.results.len());
        
        let passed = self.results.iter().filter(|r| r.passed).count();
        let failed = self.results.len() - passed;
        
        println!("Passed: {} | Failed: {}", passed, failed);
        
        if failed > 0 {
            println!("\nFailed tests:");
            for result in &self.results {
                if !result.passed {
                    println!("  - {}: {:?}", result.test_name, result.errors);
                }
            }
        }
        
        // Print metrics summary
        println!("\nAggregate Metrics:");
        let total_tx: u64 = self.results.iter().map(|r| r.metrics.transactions_processed).sum();
        let total_duration: f64 = self.results.iter().map(|r| r.duration.as_secs_f64()).sum();
        let avg_tps = total_tx as f64 / total_duration.max(1.0);
        
        println!("  Total Transactions: {}", total_tx);
        println!("  Average TPS: {:.1}", avg_tps);
        println!("  Total Blocks: {}", self.results.iter().map(|r| r.metrics.blocks_produced).sum::<u64>());
        println!("  Total Errors: {}", self.results.iter().map(|r| r.metrics.error_count).sum::<u64>());
    }
}
