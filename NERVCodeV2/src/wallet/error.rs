// src/error.rs
// Comprehensive error handling for all wallet operations

use thiserror::Error;

#[derive(Error, Debug)]
pub enum WalletError {
    #[error("Wallet not initialized")]
    NotInitialized,
    
    #[error("Key error: {0}")]
    Key(String),
    
    #[error("Balance error: {0}")]
    Balance(String),
    
    #[error("Transaction error: {0}")]
    Tx(String),
    
    #[error("Sync error: {0}")]
    Sync(String),
    
    #[error("VDW error: {0}")]
    VDW(String),
    
    #[error("History error: {0}")]
    History(String),
    
    #[error("Backup error: {0}")]
    Backup(String),
    
    #[error("Export error: {0}")]
    Export(String),
    
    #[error("Network error: {0}")]
    Network(String),
    
    #[error("Storage error: {0}")]
    Storage(String),
    
    #[error("Platform error: {0}")]
    Platform(String),
    
    #[error("UI error: {0}")]
    UI(String),
}

impl WalletError {
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Network(_) | Self::Sync(_) => true,
            _ => false,
        }
    }
    
    pub fn should_retry(&self) -> bool {
        match self {
            Self::Network(_) | Self::Sync(_) | Self::Tx(_) => true,
            _ => false,
        }
    }
    
    pub fn user_friendly_message(&self) -> String {
        match self {
            Self::NotInitialized => "Please initialize your wallet first".to_string(),
            Self::Key(msg) => format!("Security error: {}", msg),
            Self::Balance(msg) => format!("Balance calculation failed: {}", msg),
            Self::Tx(msg) => format!("Transaction failed: {}", msg),
            Self::Sync(msg) => format!("Sync failed: {}", msg),
            Self::VDW(msg) => format!("Verification failed: {}", msg),
            Self::History(msg) => format!("History error: {}", msg),
            Self::Backup(msg) => format!("Backup failed: {}", msg),
            Self::Export(msg) => format!("Export failed: {}", msg),
            Self::Network(msg) => format!("Network error: {}", msg),
            Self::Storage(msg) => format!("Storage error: {}", msg),
            Self::Platform(msg) => format!("Platform error: {}", msg),
            Self::UI(msg) => format!("UI error: {}", msg),
        }
    }
}
