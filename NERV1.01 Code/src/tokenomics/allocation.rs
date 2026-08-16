//! Genesis allocation management
//! Implements NERV's fair launch distribution with no pre-mine
//! 
//! Genesis Allocation (10 billion NERV):
//! 1. 60% - Useful Work Rewards (emission over 10 years)
//! 2. 10% - Code & Research Contributors (4-year vesting)
//! 3. 5%  - Security Audit Bounties (2-year vesting)
//! 4. 5%  - Early Research Grants (2-year vesting)
//! 5. 5%  - Early Donors & Supporters (1-year cliff, 3-year vesting)
//! 6. 10% - Treasury & Ecosystem Fund (4-year linear vesting)
//! 7. 5%  - Visionary Allocation (2-year vesting)
//! 
//! All allocations are transparently recorded on-chain at genesis.


use crate::Result;
use crate::params::{
    GENESIS_USEFUL_WORK_PERCENT, GENESIS_CODE_CONTRIB_PERCENT,
    GENESIS_AUDIT_BOUNTY_PERCENT, GENESIS_RESEARCH_PERCENT,
    GENESIS_EARLY_DONOR_PERCENT, GENESIS_TREASURY_PERCENT,
    GENESIS_VISIONARY_PERCENT, TOTAL_SUPPLY,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;


/// Genesis allocation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationRecord {
    /// Unique allocation ID
    pub id: u64,
    
    /// Recipient address
    pub recipient: [u8; 20],
    
    /// Total allocated amount (wei)
    pub total_amount: u128,
    
    /// Amount already claimed/vested
    pub claimed_amount: u128,
    
    /// Vesting schedule for this allocation
    pub vesting_schedule: VestingSchedule,
    
    /// Allocation category
    pub category: AllocationCategory,
    
    /// Description/purpose of allocation
    pub description: String,
    
    /// Creation timestamp (seconds since Unix epoch)
    pub created_at: u64,
    
    /// Whether allocation is revocable (for misbehavior)
    pub revocable: bool,
    
    /// Revocation reason if applicable
    pub revocation_reason: Option<String>,
}


/// Allocation category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationCategory {
    /// Useful work rewards (60% - emission)
    UsefulWork,
    
    /// Code and research contributors (10%)
    CodeContribution,
    
    /// Security audit bounties (5%)
    SecurityAudit,
    
    /// Early research grants (5%)
    ResearchGrant,
    
    /// Early donors and supporters (5%)
    EarlyDonor,
    
    /// Treasury and ecosystem fund (10%)
    Treasury,
    
    /// Visionary allocation (5%)
    Visionary,
    
    /// Community airdrop (optional, not in initial allocation)
    Airdrop,
    
    /// Partnership allocation
    Partnership,
}


impl AllocationCategory {
    /// Get the percentage allocation for this category
    pub fn percentage(&self) -> f64 {
        match self {
            Self::UsefulWork => GENESIS_USEFUL_WORK_PERCENT,
            Self::CodeContribution => GENESIS_CODE_CONTRIB_PERCENT,
            Self::SecurityAudit => GENESIS_AUDIT_BOUNTY_PERCENT,
            Self::ResearchGrant => GENESIS_RESEARCH_PERCENT,
            Self::EarlyDonor => GENESIS_EARLY_DONOR_PERCENT,
            Self::Treasury => GENESIS_TREASURY_PERCENT,
            Self::Visionary => GENESIS_VISIONARY_PERCENT,
            Self::Airdrop => 0.0, // Not part of genesis
            Self::Partnership => 0.0, // Not part of genesis
        }
    }
    
    /// Get default vesting schedule for this category
    pub fn default_vesting_schedule(&self) -> VestingSchedule {
        match self {
            Self::UsefulWork => {
                // Emission over 10 years through block rewards
                VestingSchedule::new_emission(10 * 365 * 24 * 60 * 60) // 10 years in seconds
            }
            Self::CodeContribution => {
                // 4-year linear vesting with 1-year cliff
                VestingSchedule::new_linear_with_cliff(
                    4 * 365 * 24 * 60 * 60, // 4 years
                    1 * 365 * 24 * 60 * 60, // 1 year cliff
                )
            }
            Self::SecurityAudit => {
                // 2-year linear vesting, 6-month cliff
                VestingSchedule::new_linear_with_cliff(
                    2 * 365 * 24 * 60 * 60, // 2 years
                    6 * 30 * 24 * 60 * 60,  // 6 months in seconds
                )
            }
            Self::ResearchGrant => {
                // 2-year linear vesting, 3-month cliff
                VestingSchedule::new_linear_with_cliff(
                    2 * 365 * 24 * 60 * 60, // 2 years
                    3 * 30 * 24 * 60 * 60,  // 3 months
                )
            }
            Self::EarlyDonor => {
                // 1-year cliff, then 3-year linear vesting
                VestingSchedule::new_linear_with_cliff(
                    4 * 365 * 24 * 60 * 60, // 4 years total
                    1 * 365 * 24 * 60 * 60, // 1 year cliff
                )
            }
            Self::Treasury => {
                // 4-year linear vesting, quarterly unlocks
                VestingSchedule::new_quarterly(16) // 16 quarters = 4 years
            }
            Self::Visionary => {
                // 2-year linear vesting
                VestingSchedule::new_linear(2 * 365 * 24 * 60 * 60)
            }
            _ => VestingSchedule::new_immediate(),
        }
    }
    
    /// Check if this category allows early vesting acceleration
    pub fn allows_acceleration(&self) -> bool {
        matches!(self, 
            Self::CodeContribution | 
            Self::SecurityAudit | 
            Self::ResearchGrant | 
            Self::EarlyDonor
        )
    }
}


/// Vesting schedule for an allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VestingSchedule {
    /// Total vesting duration in seconds
    pub total_duration: u64,
    
    /// Cliff duration in seconds (no vesting before this)
    pub cliff_duration: u64,
    
    /// Start timestamp (seconds since Unix epoch)
    pub start_time: u64,
    
    /// Vesting type
    pub vesting_type: VestingType,
    
    /// Whether schedule has been accelerated
    pub accelerated: bool,
    
    /// Acceleration timestamp if accelerated
    pub acceleration_time: Option<u64>,
}


impl VestingSchedule {
    /// Create a new immediate vesting schedule
    pub fn new_immediate() -> Self {
        Self {
            total_duration: 0,
            cliff_duration: 0,
            start_time: 0,
            vesting_type: VestingType::Immediate,
            accelerated: false,
            acceleration_time: None,
        }
    }
    
    /// Create a new linear vesting schedule
    pub fn new_linear(total_duration: u64) -> Self {
        Self {
            total_duration,
            cliff_duration: 0,
            start_time: 0,
            vesting_type: VestingType::Linear,
            accelerated: false,
            acceleration_time: None,
        }
    }
    
    /// Create a new linear vesting schedule with cliff
    pub fn new_linear_with_cliff(total_duration: u64, cliff_duration: u64) -> Self {
        Self {
            total_duration,
            cliff_duration,
            start_time: 0,
            vesting_type: VestingType::Linear,
            accelerated: false,
            acceleration_time: None,
        }
    }
    
    /// Create a new emission schedule (special case for useful work)
    pub fn new_emission(total_duration: u64) -> Self {
        Self {
            total_duration,
            cliff_duration: 0,
            start_time: 0,
            vesting_type: VestingType::Emission,
            accelerated: false,
            acceleration_time: None,
        }
    }
    
    /// Create a new quarterly vesting schedule
    pub fn new_quarterly(quarters: u64) -> Self {
        Self {
            total_duration: quarters * 90 * 24 * 60 * 60, // Approx 90 days per quarter
            cliff_duration: 0,
            start_time: 0,
            vesting_type: VestingType::Quarterly(quarters),
            accelerated: false,
            acceleration_time: None,
        }
    }
    
    /// Calculate vested amount at a given timestamp
    pub fn vested_amount(&self, total_amount: u128, timestamp: u64) -> u128 {
        if timestamp < self.start_time + self.cliff_duration {
            // Still in cliff period
            return 0;
        }
        
        if timestamp >= self.start_time + self.total_duration {
            // Fully vested
            return total_amount;
        }
        
        let elapsed = timestamp - self.start_time - self.cliff_duration;
        let vesting_period = self.total_duration - self.cliff_duration;
        
        match self.vesting_type {
            VestingType::Immediate => total_amount,
            VestingType::Linear => {
                // Linear vesting: (elapsed / vesting_period) * total_amount
                let vested_ratio = elapsed as f64 / vesting_period as f64;
                (vested_ratio * total_amount as f64) as u128
            }
            VestingType::Quarterly(quarters) => {
                // Quarterly vesting: unlock 1/quarters each quarter
                let quarter_duration = self.total_duration / quarters;
                let quarters_passed = elapsed / quarter_duration;
                (total_amount * quarters_passed as u128) / quarters as u128
            }
            VestingType::Emission => {
                // Emission schedule: vest through block rewards, not direct allocation
                0
            }
        }
    }
    
    /// Check if vesting can be accelerated
    pub fn can_accelerate(&self, current_time: u64, category: AllocationCategory) -> bool {
        category.allows_acceleration() && 
        !self.accelerated && 
        current_time >= self.start_time + self.cliff_duration
    }
    
    /// Accelerate vesting schedule
    pub fn accelerate(&mut self, acceleration_time: u64) -> Result<()> {
        if self.accelerated {
            return Err(AllocationError::AlreadyAccelerated.into());
        }
        
        self.accelerated = true;
        self.acceleration_time = Some(acceleration_time);
        
        // Reduce remaining vesting period by 50% (standard acceleration)
        let remaining = self.total_duration.saturating_sub(acceleration_time - self.start_time);
        self.total_duration = acceleration_time + remaining / 2;
        
        Ok(())
    }
}


/// Vesting type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VestingType {
    /// Immediate vesting (no lockup)
    Immediate,
    
    /// Linear vesting over time
    Linear,
    
    /// Quarterly unlocks
    Quarterly(u64),
    
    /// Emission through block rewards (special case)
    Emission,
}


/// Genesis allocation manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenesisAllocation {
    /// Total genesis supply (10 billion NERV in wei)
    total_supply: u128,
    
    /// Allocation records by ID
    allocations: HashMap<u64, AllocationRecord>,
    
    /// Allocation IDs by recipient address
    recipient_allocations: HashMap<[u8; 20], Vec<u64>>,
    
    /// Allocation IDs by category
    category_allocations: HashMap<AllocationCategory, Vec<u64>>,
    
    /// Next allocation ID
    next_id: u64,
    
    /// Genesis timestamp
    genesis_timestamp: u64,
    
    /// Total allocated per category
    allocated_per_category: HashMap<AllocationCategory, u128>,
}


impl Default for GenesisAllocation {
    fn default() -> Self {
        Self::new(TOTAL_SUPPLY as u128 * 10u128.pow(18), 0)
    }
}


impl GenesisAllocation {
    /// Create a new genesis allocation manager
    pub fn new(total_supply: u128, genesis_timestamp: u64) -> Self {
        Self {
            total_supply,
            allocations: HashMap::new(),
            recipient_allocations: HashMap::new(),
            category_allocations: HashMap::new(),
            next_id: 1,
            genesis_timestamp,
            allocated_per_category: HashMap::new(),
        }
    }
    
    /// Add a new allocation
    pub fn add_allocation(
        &mut self,
        recipient: [u8; 20],
        amount: u128,
        category: AllocationCategory,
        description: String,
        custom_vesting: Option<VestingSchedule>,
    ) -> Result<u64> {
        // Check category allocation limit
        let category_percentage = category.percentage();
        let category_limit = (self.total_supply as f64 * category_percentage) as u128;
        
        let current_allocated = self.allocated_per_category
            .get(&category)
            .copied()
            .unwrap_or(0);
        
        if current_allocated + amount > category_limit {
            return Err(AllocationError::CategoryLimitExceeded(
                category_percentage,
                current_allocated as f64 / 10f64.powi(18),
                amount as f64 / 10f64.powi(18),
            ).into());
        }
        
        // Check total allocation
        let total_allocated: u128 = self.allocated_per_category.values().sum();
        if total_allocated + amount > self.total_supply {
            return Err(AllocationError::TotalSupplyExceeded.into());
        }
        
        // Create vesting schedule
        let vesting_schedule = custom_vesting.unwrap_or_else(|| {
            let mut schedule = category.default_vesting_schedule();
            schedule.start_time = self.genesis_timestamp;
            schedule
        });
        
        // Create allocation record
        let allocation = AllocationRecord {
            id: self.next_id,
            recipient,
            total_amount: amount,
            claimed_amount: 0,
            vesting_schedule,
            category,
            description,
            created_at: self.genesis_timestamp,
            revocable: category != AllocationCategory::UsefulWork, // Useful work not revocable
            revocation_reason: None,
        };
        
        // Update tracking structures
        self.allocations.insert(self.next_id, allocation);
        
        self.recipient_allocations
            .entry(recipient)
            .or_insert_with(Vec::new)
            .push(self.next_id);
        
        self.category_allocations
            .entry(category)
            .or_insert_with(Vec::new)
            .push(self.next_id);
        
        *self.allocated_per_category.entry(category).or_insert(0) += amount;
        
        let allocated_id = self.next_id;
        self.next_id += 1;
        
        Ok(allocated_id)
    }
    
    /// Claim vested tokens for an allocation
    pub fn claim_vested(&mut self, allocation_id: u64, current_time: u64) -> Result<u128> {
        let allocation = self.allocations
            .get_mut(&allocation_id)
            .ok_or(AllocationError::AllocationNotFound)?;
        
        if allocation.revocation_reason.is_some() {
            return Err(AllocationError::AllocationRevoked.into());
        }
        
        let vested_amount = allocation.vesting_schedule
            .vested_amount(allocation.total_amount, current_time);
        
        let claimable = vested_amount.saturating_sub(allocation.claimed_amount);
        
        if claimable > 0 {
            allocation.claimed_amount += claimable;
        }
        
        Ok(claimable)
    }
    
    /// Get total claimable amount for a recipient
    pub fn total_claimable(&self, recipient: [u8; 20], current_time: u64) -> u128 {
        let allocation_ids = self.recipient_allocations
            .get(&recipient)
            .unwrap_or(&Vec::new());
        
        let mut total = 0;
        for &id in allocation_ids {
            if let Some(allocation) = self.allocations.get(&id) {
                if allocation.revocation_reason.is_none() {
                    let vested = allocation.vesting_schedule
                        .vested_amount(allocation.total_amount, current_time);
                    total += vested.saturating_sub(allocation.claimed_amount);
                }
            }
        }
        
        total
    }
    
    /// Revoke an allocation (for misbehavior)
    pub fn revoke_allocation(
        &mut self,
        allocation_id: u64,
        reason: String,
    ) -> Result<u128> {
        let allocation = self.allocations
            .get_mut(&allocation_id)
            .ok_or(AllocationError::AllocationNotFound)?;
        
        if !allocation.revocable {
            return Err(AllocationError::NotRevocable.into());
        }
        
        if allocation.revocation_reason.is_some() {
            return Err(AllocationError::AlreadyRevoked.into());
        }
        
        allocation.revocation_reason = Some(reason);
        
        // Return unvested amount
        let unvested = allocation.total_amount - allocation.claimed_amount;
        Ok(unvested)
    }
    
    /// Accelerate vesting for an allocation
    pub fn accelerate_vesting(
        &mut self,
        allocation_id: u64,
        current_time: u64,
    ) -> Result<()> {
        let allocation = self.allocations
            .get_mut(&allocation_id)
            .ok_or(AllocationError::AllocationNotFound)?;
        
        if !allocation.vesting_schedule.can_accelerate(current_time, allocation.category) {
            return Err(AllocationError::CannotAccelerate.into());
        }
        
        allocation.vesting_schedule.accelerate(current_time)?;
        Ok(())
    }
    
    /// Get allocation by ID
    pub fn get_allocation(&self, allocation_id: u64) -> Option<&AllocationRecord> {
        self.allocations.get(&allocation_id)
    }
    
    /// Get allocations for a recipient
    pub fn get_recipient_allocations(&self, recipient: [u8; 20]) -> Vec<&AllocationRecord> {
        self.recipient_allocations
            .get(&recipient)
            .map(|ids| ids.iter().filter_map(|id| self.allocations.get(id)).collect())
            .unwrap_or_else(Vec::new)
    }
    
    /// Get allocations by category
    pub fn get_category_allocations(&self, category: AllocationCategory) -> Vec<&AllocationRecord> {
        self.category_allocations
            .get(&category)
            .map(|ids| ids.iter().filter_map(|id| self.allocations.get(id)).collect())
            .unwrap_or_else(Vec::new)
    }
    
    /// Get total allocated per category
    pub fn allocated_by_category(&self) -> &HashMap<AllocationCategory, u128> {
        &self.allocated_per_category
    }
    
    /// Get total allocated amount
    pub fn total_allocated(&self) -> u128 {
        self.allocated_per_category.values().sum()
    }
    
    /// Get remaining unallocated supply
    pub fn remaining_supply(&self) -> u128 {
        self.total_supply - self.total_allocated()
    }
    
    /// Get total vested amount across all allocations
    pub fn total_vested(&self, current_time: u64) -> u128 {
        self.allocations.values()
            .filter(|a| a.revocation_reason.is_none())
            .map(|a| a.vesting_schedule.vested_amount(a.total_amount, current_time))
            .sum()
    }
}


/// Allocation errors
#[derive(Debug, thiserror::Error)]
pub enum AllocationError {
    #[error("Allocation not found")]
    AllocationNotFound,
    
    #[error("Allocation already revoked")]
    AlreadyRevoked,
    
    #[error("Allocation is not revocable")]
    NotRevocable,
    
    #[error("Allocation already accelerated")]
    AlreadyAccelerated,
    
    #[error("Cannot accelerate vesting at this time")]
    CannotAccelerate,
    
    #[error("Category allocation limit exceeded: max {0}% ({1} NERV allocated, trying to add {2} NERV)")]
    CategoryLimitExceeded(f64, f64, f64),
    
    #[error("Total supply exceeded")]
    TotalSupplyExceeded,
    
    #[error("Allocation revoked: {0}")]
    AllocationRevoked,
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_allocation_creation() {
        let genesis_time = 1735689600; // Jan 1, 2025
        let mut manager = GenesisAllocation::new(
            10_000_000_000 * 10u128.pow(18), // 10B NERV
            genesis_time,
        );
        
        let recipient = [1u8; 20];
        let amount = 1_000_000 * 10u128.pow(18); // 1M NERV
        
        let id = manager.add_allocation(
            recipient,
            amount,
            AllocationCategory::CodeContribution,
            "Core protocol development".to_string(),
            None,
        ).unwrap();
        
        assert_eq!(id, 1);
        assert_eq!(manager.total_allocated(), amount);
        
        let allocation = manager.get_allocation(id).unwrap();
        assert_eq!(allocation.recipient, recipient);
        assert_eq!(allocation.total_amount, amount);
        assert_eq!(allocation.category, AllocationCategory::CodeContribution);
    }
    
    #[test]
    fn test_category_limits() {
        let genesis_time = 1735689600;
        let mut manager = GenesisAllocation::new(
            10_000_000_000 * 10u128.pow(18),
            genesis_time,
        );
        
        let recipient = [1u8; 20];
        let category = AllocationCategory::CodeContribution;
        let category_percent = category.percentage(); // 10%
        let category_limit = (10_000_000_000.0 * category_percent) as u128 * 10u128.pow(18);
        
        // Try to allocate more than category limit
        let result = manager.add_allocation(
            recipient,
            category_limit + 1,
            category,
            "Exceeds limit".to_string(),
            None,
        );
        
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().downcast::<AllocationError>().unwrap(),
            AllocationError::CategoryLimitExceeded(_, _, _)
        ));
    }
    
    #[test]
    fn test_vesting_calculation() {
        let genesis_time = 1735689600;
        let vesting_schedule = VestingSchedule::new_linear_with_cliff(
            2 * 365 * 24 * 60 * 60, // 2 years total
            6 * 30 * 24 * 60 * 60,   // 6 month cliff
        );
        
        let amount = 1_000_000 * 10u128.pow(18);
        
        // Before cliff
        let vested_before = vesting_schedule.vested_amount(amount, genesis_time + 5 * 30 * 24 * 60 * 60);
        assert_eq!(vested_before, 0);
        
        // At cliff
        let vested_at_cliff = vesting_schedule.vested_amount(amount, genesis_time + 6 * 30 * 24 * 60 * 60);
        assert_eq!(vested_at_cliff, 0); // Still 0 at exact cliff
        
        // Halfway through vesting
        let halfway = genesis_time + 6 * 30 * 24 * 60 * 60 + (18 * 30 * 24 * 60 * 60) / 2;
        let vested_halfway = vesting_schedule.vested_amount(amount, halfway);
        
        // Should be 50% of total (linear from cliff to end)
        assert_eq!(vested_halfway, amount / 2);
        
        // Fully vested
        let fully_vested = vesting_schedule.vested_amount(amount, genesis_time + 2 * 365 * 24 * 60 * 60);
        assert_eq!(fully_vested, amount);
    }
    
    #[test]
    fn test_claiming() {
        let genesis_time = 1735689600;
        let mut manager = GenesisAllocation::new(
            10_000_000_000 * 10u128.pow(18),
            genesis_time,
        );
        
        let recipient = [1u8; 20];
        let amount = 1_000_000 * 10u128.pow(18);
        
        let id = manager.add_allocation(
            recipient,
            amount,
            AllocationCategory::Visionary,
            "Visionary allocation".to_string(),
            None,
        ).unwrap();
        
        // Try to claim immediately (before any vesting)
        let claimable = manager.claim_vested(id, genesis_time).unwrap();
        assert_eq!(claimable, 0);
        
        // Claim after 1 year (should be 50% vested for 2-year linear)
        let one_year = genesis_time + 365 * 24 * 60 * 60;
        let claimable = manager.claim_vested(id, one_year).unwrap();
        assert_eq!(claimable, amount / 2);
        
        let allocation = manager.get_allocation(id).unwrap();
        assert_eq!(allocation.claimed_amount, amount / 2);
    }
}
