//! Emission schedule management
//! Implements NERV's 10-year emission schedule with 0.5% perpetual tail emission
//! 
//! Emission Schedule:
//! - Year 1-2: 38% of genesis allocation (highest to bootstrap network)
//! - Year 3-5: 18% annual (decelerating growth)
//! - Year 6-10: 8% annual (approaching steady state)
//! - Year 11+: 0.5% perpetual tail emission (security subsidy)
//! 
//! Total supply: 10 billion NERV with maximum 2x buffer for safety


use crate::Result;
use crate::params::{
    EMISSION_YEAR_1_2_PERCENT, EMISSION_YEAR_3_5_PERCENT,
    EMISSION_YEAR_6_10_PERCENT, TAIL_EMISSION_PERCENT,
    TOTAL_SUPPLY,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;


/// Emission schedule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmissionSchedule {
    /// Total supply in wei (10 billion NERV × 10^18)
    total_supply: u128,
    
    /// Annual emission rates by year
    annual_emission_rates: BTreeMap<u64, f64>, // year -> emission rate (percentage)
    
    /// Tail emission rate (0.5% after year 10)
    tail_emission_rate: f64,
    
    /// Current year in the emission schedule
    current_year: u64,
    
    /// Cumulative emitted tokens
    cumulative_emitted: u128,
    
    /// Blocks per year (calculated from 0.6 second block time)
    blocks_per_year: u64,
}


impl Default for EmissionSchedule {
    fn default() -> Self {
        let mut schedule = Self::new(
            TOTAL_SUPPLY as u128 * 10u128.pow(18), // Convert to wei
            0.6, // 0.6 second block time
        );
        
        // Set up the 10-year emission schedule
        schedule.setup_standard_schedule();
        schedule
    }
}


impl EmissionSchedule {
    /// Create a new emission schedule
    pub fn new(total_supply: u128, block_time_seconds: f64) -> Self {
        // Calculate blocks per year based on block time
        let seconds_per_year = 365.25 * 24.0 * 60.0 * 60.0;
        let blocks_per_year = (seconds_per_year / block_time_seconds) as u64;
        
        Self {
            total_supply,
            annual_emission_rates: BTreeMap::new(),
            tail_emission_rate: TAIL_EMISSION_PERCENT,
            current_year: 1,
            cumulative_emitted: 0,
            blocks_per_year,
        }
    }
    
    /// Set up the standard NERV emission schedule
    pub fn setup_standard_schedule(&mut self) {
        // Year 1-2: 38% each
        for year in 1..=2 {
            self.annual_emission_rates.insert(year, EMISSION_YEAR_1_2_PERCENT);
        }
        
        // Year 3-5: 18% each
        for year in 3..=5 {
            self.annual_emission_rates.insert(year, EMISSION_YEAR_3_5_PERCENT);
        }
        
        // Year 6-10: 8% each
        for year in 6..=10 {
            self.annual_emission_rates.insert(year, EMISSION_YEAR_6_10_PERCENT);
        }
        
        // Year 11+: handled by tail emission
    }
    
    /// Get emission for a specific year
    pub fn emission_for_year(&self, year: u64) -> u128 {
        if year <= 10 {
            // Look up in schedule
            if let Some(&rate) = self.annual_emission_rates.get(&year) {
                // Calculate emission as percentage of remaining supply
                let remaining_supply = self.total_supply - self.cumulative_emitted;
                (remaining_supply as f64 * rate) as u128
            } else {
                // If year not in schedule but ≤10, use last defined rate
                let last_rate = self.annual_emission_rates.values().last().copied().unwrap_or(0.0);
                let remaining_supply = self.total_supply - self.cumulative_emitted;
                (remaining_supply as f64 * last_rate) as u128
            }
        } else {
            // Tail emission: 0.5% of current supply
            let current_supply = self.cumulative_emitted;
            (current_supply as f64 * self.tail_emission_rate) as u128
        }
    }
    
    /// Get block reward for current block height
    pub fn block_reward(&self, block_height: u64) -> u128 {
        let year = self.year_from_block(block_height);
        let annual_emission = self.emission_for_year(year);
        
        // Convert annual emission to per-block emission
        annual_emission / self.blocks_per_year as u128
    }
    
    /// Calculate which emission year a block belongs to
    pub fn year_from_block(&self, block_height: u64) -> u64 {
        let blocks_per_year = self.blocks_per_year;
        let year = block_height / blocks_per_year + 1; // Year starts at 1
        year
    }
    
    /// Update cumulative emission (called after each block)
    pub fn update_cumulative_emission(&mut self, block_reward: u128) {
        self.cumulative_emitted += block_reward;
        
        // Check if we need to advance to next year
        let current_block = self.current_block();
        let new_year = self.year_from_block(current_block);
        if new_year > self.current_year {
            self.current_year = new_year;
            
            // Log year transition for monitoring
            tracing::info!(
                "Emission schedule advanced to year {}, cumulative emitted: {} NERV",
                new_year,
                self.cumulative_emitted as f64 / 10f64.powi(18)
            );
        }
    }
    
    /// Calculate current block number based on emission schedule
    pub fn current_block(&self) -> u64 {
        // In a real implementation, this would track actual block production
        // For now, we estimate based on elapsed time since genesis
        self.current_year * self.blocks_per_year
    }
    
    /// Get remaining emission for the current year
    pub fn remaining_emission_current_year(&self) -> u128 {
        let annual_emission = self.emission_for_year(self.current_year);
        let blocks_this_year = self.current_block() % self.blocks_per_year;
        let blocks_remaining = self.blocks_per_year - blocks_this_year;
        
        (annual_emission * blocks_remaining as u128) / self.blocks_per_year as u128
    }
    
    /// Get inflation rate for a specific year
    pub fn inflation_rate_for_year(&self, year: u64) -> f64 {
        let emission = self.emission_for_year(year);
        let supply_at_start = if year == 1 {
            // Year 1 starts with 0 supply (all emission)
            0
        } else {
            // Calculate cumulative emission up to previous year
            let mut cumulative = 0;
            for y in 1..year {
                cumulative += self.emission_for_year(y);
            }
            cumulative
        };
        
        if supply_at_start == 0 {
            emission as f64
        } else {
            emission as f64 / supply_at_start as f64
        }
    }
    
    /// Get projected supply at year end
    pub fn projected_supply_at_year(&self, year: u64) -> u128 {
        let mut supply = 0;
        for y in 1..=year {
            supply += self.emission_for_year(y);
        }
        supply.min(self.total_supply)
    }
    
    /// Check if we've reached maximum supply
    pub fn is_max_supply_reached(&self) -> bool {
        self.cumulative_emitted >= self.total_supply
    }
    
    /// Get total supply cap
    pub fn total_supply_cap(&self) -> u128 {
        self.total_supply
    }
    
    /// Get current cumulative emission
    pub fn cumulative_emitted(&self) -> u128 {
        self.cumulative_emitted
    }
    
    /// Get current year in emission schedule
    pub fn current_year(&self) -> u64 {
        self.current_year
    }
}


/// Emission validation error
#[derive(Debug, thiserror::Error)]
pub enum EmissionError {
    #[error("Emission exceeds total supply cap")]
    ExceedsSupplyCap,
    
    #[error("Invalid emission year: {0}")]
    InvalidYear(u64),
    
    #[error("Emission schedule not initialized")]
    NotInitialized,
    
    #[error("Block time too fast for stable emission")]
    InvalidBlockTime,
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_emission_schedule() {
        let schedule = EmissionSchedule::default();
        
        // Check year 1 emission (38% of 10B = 3.8B NERV)
        let year1_emission = schedule.emission_for_year(1);
        let expected_year1 = (TOTAL_SUPPLY as f64 * EMISSION_YEAR_1_2_PERCENT) as u128 * 10u128.pow(18);
        assert_eq!(year1_emission, expected_year1);
        
        // Check year 3 emission (18%)
        let year3_emission = schedule.emission_for_year(3);
        let remaining_after_year2 = TOTAL_SUPPLY as u128 * 10u128.pow(18) - year1_emission * 2;
        let expected_year3 = (remaining_after_year2 as f64 * EMISSION_YEAR_3_5_PERCENT) as u128;
        assert_eq!(year3_emission, expected_year3);
        
        // Check tail emission (0.5% of current supply)
        let year11_emission = schedule.emission_for_year(11);
        let supply_after_10_years = schedule.projected_supply_at_year(10);
        let expected_tail = (supply_after_10_years as f64 * TAIL_EMISSION_PERCENT) as u128;
        assert_eq!(year11_emission, expected_tail);
    }
    
    #[test]
    fn test_block_reward_calculation() {
        let schedule = EmissionSchedule::default();
        
        // Block reward should be annual emission divided by blocks per year
        let year1_emission = schedule.emission_for_year(1);
        let blocks_per_year = schedule.blocks_per_year;
        let expected_block_reward = year1_emission / blocks_per_year as u128;
        
        let block_reward = schedule.block_reward(0); // First block of year 1
        assert_eq!(block_reward, expected_block_reward);
    }
    
    #[test]
    fn test_year_calculation() {
        let schedule = EmissionSchedule::default();
        let blocks_per_year = schedule.blocks_per_year;
        
        // First block is year 1
        assert_eq!(schedule.year_from_block(0), 1);
        
        // Last block of year 1
        assert_eq!(schedule.year_from_block(blocks_per_year - 1), 1);
        
        // First block of year 2
        assert_eq!(schedule.year_from_block(blocks_per_year), 2);
        
        // Check transition to tail emission (year 11)
        assert_eq!(schedule.year_from_block(blocks_per_year * 10), 11);
    }
    
    #[test]
    fn test_inflation_rate() {
        let schedule = EmissionSchedule::default();
        
        // Year 1 inflation should be very high (38% of total supply from 0)
        let year1_inflation = schedule.inflation_rate_for_year(1);
        assert_eq!(year1_inflation, EMISSION_YEAR_1_2_PERCENT);
        
        // Year 11 inflation should be tail rate (0.5%)
        let year11_inflation = schedule.inflation_rate_for_year(11);
        assert!(year11_inflation - TAIL_EMISSION_PERCENT < 0.001); // Floating point tolerance
    }
}
