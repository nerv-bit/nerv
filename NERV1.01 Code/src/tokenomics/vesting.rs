//! Vesting schedule management
//! Handles all vesting logic for allocations, staking rewards, and team distributions
//! 
//! Vesting Types:
//! 1. Linear Vesting: Constant rate over time
//! 2. Cliff Vesting: No vesting until cliff, then linear
//! 3. Quarterly Vesting: Discrete unlocks every quarter
//! 4. Accelerated Vesting: Community-voted acceleration
//! 5. Performance Vesting: Tied to milestones/achievements


use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap};
use std::time::{SystemTime, UNIX_EPOCH};


/// Vesting schedule for token allocations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VestingSchedule {
    /// Unique schedule ID
    pub id: u64,
    
    /// Owner address
    pub owner: [u8; 20],
    
    /// Total amount (wei)
    pub total_amount: u128,
    
    /// Amount already vested/claimed
    pub vested_amount: u128,
    
    /// Start timestamp (seconds since Unix epoch)
    pub start_time: u64,
    
    /// Cliff duration in seconds
    pub cliff_duration: u64,
    
    /// Total vesting duration in seconds
    pub total_duration: u64,
    
    /// Vesting type
    pub vesting_type: VestingType,
    
    /// Whether schedule can be accelerated
    pub can_accelerate: bool,
    
    /// Whether schedule has been accelerated
    pub accelerated: bool,
    
    /// Acceleration timestamp if accelerated
    pub acceleration_time: Option<u64>,
    
    /// Last claim timestamp
    pub last_claim_time: u64,
    
    /// Schedule status
    pub status: VestingStatus,
    
    /// Custom vesting curve parameters
    pub curve_params: Option<VestingCurveParams>,
    
    /// Performance milestones (for performance-based vesting)
    pub milestones: Option<Vec<VestingMilestone>>,
}


/// Vesting type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VestingType {
    /// Immediate vesting (no lockup)
    Immediate,
    
    /// Linear vesting over time
    Linear,
    
    /// Cliff then linear vesting
    CliffLinear,
    
    /// Quarterly unlocks
    Quarterly(u8), // Number of quarters
    
    /// Exponential vesting (fast early, slow later)
    Exponential,
    
    /// Step function vesting (discrete unlocks)
    Step(Vec<u64>), // Timestamps for each step
    
    /// Performance-based vesting
    Performance,
    
    /// Community-voted vesting
    CommunityVoted,
}


/// Vesting status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VestingStatus {
    /// Active and vesting
    Active,
    
    /// Fully vested
    FullyVested,
    
    /// Revoked/canceled
    Revoked,
    
    /// Paused (temporarily stopped)
    Paused,
    
    /// Accelerated
    Accelerated,
}


/// Vesting curve parameters for custom vesting curves
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VestingCurveParams {
    /// Curve type
    pub curve_type: CurveType,
    
    /// Acceleration parameter (0-1)
    pub acceleration: f64,
    
    /// Deceleration parameter (0-1)
    pub deceleration: f64,
    
    /// Initial vested percentage (0-1)
    pub initial_percent: f64,
    
    /// Curve smoothing factor
    pub smoothing: f64,
}


/// Curve type for custom vesting
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CurveType {
    /// Sigmoid curve (S-shaped)
    Sigmoid,
    
    /// Quadratic curve
    Quadratic,
    
    /// Cubic curve
    Cubic,
    
    /// Logarithmic curve
    Logarithmic,
    
    /// Exponential curve
    Exponential,
}


/// Performance milestone for vesting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VestingMilestone {
    /// Milestone ID
    pub milestone_id: u64,
    
    /// Description
    pub description: String,
    
    /// Completion criteria
    pub criteria: String,
    
    /// Vesting percentage released on completion (0-100)
    pub vesting_percentage: u8,
    
    /// Completion timestamp
    pub completion_time: Option<u64>,
    
    /// Verification proof
    pub verification_proof: Option<[u8; 32]>,
    
    /// Verified by
    pub verified_by: Option<[u8; 20]>,
}


/// Vesting manager for handling multiple vesting schedules
#[derive(Debug, Clone)]
pub struct VestingManager {
    /// Vesting schedules by ID
    schedules: HashMap<u64, VestingSchedule>,
    
    /// Schedule IDs by owner
    owner_schedules: HashMap<[u8; 20], Vec<u64>>,
    
    /// Schedule IDs by status
    status_schedules: HashMap<VestingStatus, Vec<u64>>,
    
    /// Next schedule ID
    next_id: u64,
    
    /// Acceleration voting threshold
    acceleration_threshold: f64,
    
    /// Minimum vesting duration (seconds)
    min_vesting_duration: u64,
    
    /// Maximum vesting duration (seconds)
    max_vesting_duration: u64,
}


impl Default for VestingManager {
    fn default() -> Self {
        Self {
            schedules: HashMap::new(),
            owner_schedules: HashMap::new(),
            status_schedules: HashMap::new(),
            next_id: 1,
            acceleration_threshold: 0.67, // 67% community vote required
            min_vesting_duration: 30 * 24 * 60 * 60, // 30 days minimum
            max_vesting_duration: 10 * 365 * 24 * 60 * 60, // 10 years maximum
        }
    }
}


impl VestingManager {
    /// Create a new vesting manager
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Create a new vesting schedule
    pub fn create_schedule(
        &mut self,
        owner: [u8; 20],
        total_amount: u128,
        start_time: u64,
        vesting_type: VestingType,
        cliff_duration: Option<u64>,
        can_accelerate: bool,
        curve_params: Option<VestingCurveParams>,
        milestones: Option<Vec<VestingMilestone>>,
    ) -> Result<u64> {
        // Validate parameters
        let total_duration = self.calculate_total_duration(&vesting_type, &curve_params);
        
        if total_duration < self.min_vesting_duration {
            return Err(VestingError::DurationTooShort(total_duration, self.min_vesting_duration).into());
        }
        
        if total_duration > self.max_vesting_duration {
            return Err(VestingError::DurationTooLong(total_duration, self.max_vesting_duration).into());
        }
        
        let cliff_duration = cliff_duration.unwrap_or(0);
        if cliff_duration >= total_duration {
            return Err(VestingError::CliffExceedsDuration(cliff_duration, total_duration).into());
        }
        
        // Create schedule
        let schedule = VestingSchedule {
            id: self.next_id,
            owner,
            total_amount,
            vested_amount: 0,
            start_time,
            cliff_duration,
            total_duration,
            vesting_type,
            can_accelerate,
            accelerated: false,
            acceleration_time: None,
            last_claim_time: start_time,
            status: VestingStatus::Active,
            curve_params,
            milestones,
        };
        
        // Store schedule
        self.schedules.insert(self.next_id, schedule);
        
        self.owner_schedules
            .entry(owner)
            .or_insert_with(Vec::new)
            .push(self.next_id);
        
        self.status_schedules
            .entry(VestingStatus::Active)
            .or_insert_with(Vec::new)
            .push(self.next_id);
        
        let schedule_id = self.next_id;
        self.next_id += 1;
        
        Ok(schedule_id)
    }
    
    /// Calculate vested amount for a schedule at a given timestamp
    pub fn calculate_vested_amount(
        &self,
        schedule_id: u64,
        timestamp: u64,
    ) -> Result<u128> {
        let schedule = self.schedules
            .get(&schedule_id)
            .ok_or(VestingError::ScheduleNotFound)?;
        
        if schedule.status == VestingStatus::Revoked {
            return Ok(schedule.vested_amount); // No further vesting if revoked
        }
        
        if schedule.status == VestingStatus::Paused {
            // Use last claim time if paused
            return self.calculate_vested_for_schedule(schedule, schedule.last_claim_time);
        }
        
        self.calculate_vested_for_schedule(schedule, timestamp)
    }
    
    /// Claim vested tokens
    pub fn claim_vested(
        &mut self,
        schedule_id: u64,
        timestamp: u64,
    ) -> Result<u128> {
        let schedule = self.schedules
            .get_mut(&schedule_id)
            .ok_or(VestingError::ScheduleNotFound)?;
        
        if schedule.status == VestingStatus::Revoked {
            return Err(VestingError::ScheduleRevoked.into());
        }
        
        if schedule.status == VestingStatus::Paused {
            return Err(VestingError::SchedulePaused.into());
        }
        
        let newly_vested = self.calculate_vested_for_schedule(schedule, timestamp)?;
        let claimable = newly_vested.saturating_sub(schedule.vested_amount);
        
        if claimable > 0 {
            schedule.vested_amount = newly_vested;
            schedule.last_claim_time = timestamp;
            
            // Check if fully vested
            if schedule.vested_amount >= schedule.total_amount {
                schedule.status = VestingStatus::FullyVested;
                
                // Update status tracking
                self.update_schedule_status(schedule_id, VestingStatus::FullyVested);
            }
        }
        
        Ok(claimable)
    }
    
    /// Accelerate vesting schedule
    pub fn accelerate_schedule(
        &mut self,
        schedule_id: u64,
        acceleration_time: u64,
        vote_percentage: f64,
    ) -> Result<()> {
        let schedule = self.schedules
            .get_mut(&schedule_id)
            .ok_or(VestingError::ScheduleNotFound)?;
        
        if !schedule.can_accelerate {
            return Err(VestingError::CannotAccelerate.into());
        }
        
        if schedule.accelerated {
            return Err(VestingError::AlreadyAccelerated.into());
        }
        
        if vote_percentage < self.acceleration_threshold {
            return Err(VestingError::InsufficientVotes(vote_percentage, self.acceleration_threshold).into());
        }
        
        // Calculate acceleration
        let elapsed = acceleration_time.saturating_sub(schedule.start_time);
        let remaining = schedule.total_duration.saturating_sub(elapsed);
        
        // Accelerate by 50% (standard acceleration)
        let new_remaining = remaining / 2;
        schedule.total_duration = elapsed + new_remaining;
        schedule.accelerated = true;
        schedule.acceleration_time = Some(acceleration_time);
        schedule.status = VestingStatus::Accelerated;
        
        // Update status tracking
        self.update_schedule_status(schedule_id, VestingStatus::Accelerated);
        
        Ok(())
    }
    
    /// Complete a performance milestone
    pub fn complete_milestone(
        &mut self,
        schedule_id: u64,
        milestone_id: u64,
        completion_time: u64,
        verification_proof: [u8; 32],
        verified_by: [u8; 20],
    ) -> Result<()> {
        let schedule = self.schedules
            .get_mut(&schedule_id)
            .ok_or(VestingError::ScheduleNotFound)?;
        
        let milestones = schedule.milestones
            .as_mut()
            .ok_or(VestingError::NoMilestones)?;
        
        let milestone = milestones
            .iter_mut()
            .find(|m| m.milestone_id == milestone_id)
            .ok_or(VestingError::MilestoneNotFound)?;
        
        if milestone.completion_time.is_some() {
            return Err(VestingError::MilestoneAlreadyCompleted.into());
        }
        
        milestone.completion_time = Some(completion_time);
        milestone.verification_proof = Some(verification_proof);
        milestone.verified_by = Some(verified_by);
        
        Ok(())
    }
    
    /// Pause a vesting schedule (emergency only)
    pub fn pause_schedule(&mut self, schedule_id: u64) -> Result<()> {
        let schedule = self.schedules
            .get_mut(&schedule_id)
            .ok_or(VestingError::ScheduleNotFound)?;
        
        if schedule.status != VestingStatus::Active {
            return Err(VestingError::CannotPause.into());
        }
        
        schedule.status = VestingStatus::Paused;
        self.update_schedule_status(schedule_id, VestingStatus::Paused);
        
        Ok(())
    }
    
    /// Resume a paused vesting schedule
    pub fn resume_schedule(&mut self, schedule_id: u64) -> Result<()> {
        let schedule = self.schedules
            .get_mut(&schedule_id)
            .ok_or(VestingError::ScheduleNotFound)?;
        
        if schedule.status != VestingStatus::Paused {
            return Err(VestingError::NotPaused.into());
        }
        
        schedule.status = VestingStatus::Active;
        self.update_schedule_status(schedule_id, VestingStatus::Active);
        
        Ok(())
    }
    
    /// Revoke a vesting schedule (for misbehavior)
    pub fn revoke_schedule(&mut self, schedule_id: u64) -> Result<u128> {
        let schedule = self.schedules
            .get_mut(&schedule_id)
            .ok_or(VestingError::ScheduleNotFound)?;
        
        if schedule.status == VestingStatus::Revoked {
            return Err(VestingError::AlreadyRevoked.into());
        }
        
        // Calculate unvested amount to return
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        let vested = self.calculate_vested_for_schedule(schedule, current_time)?;
        let unvested = schedule.total_amount.saturating_sub(vested);
        
        schedule.status = VestingStatus::Revoked;
        self.update_schedule_status(schedule_id, VestingStatus::Revoked);
        
        Ok(unvested)
    }
    
    /// Get schedules for an owner
    pub fn get_owner_schedules(&self, owner: [u8; 20]) -> Vec<&VestingSchedule> {
        self.owner_schedules
            .get(&owner)
            .map(|ids| ids.iter().filter_map(|id| self.schedules.get(id)).collect())
            .unwrap_or_else(Vec::new)
    }
    
    /// Get schedules by status
    pub fn get_schedules_by_status(&self, status: VestingStatus) -> Vec<&VestingSchedule> {
        self.status_schedules
            .get(&status)
            .map(|ids| ids.iter().filter_map(|id| self.schedules.get(id)).collect())
            .unwrap_or_else(Vec::new)
    }
    
    /// Get total vested amount for an owner
    pub fn total_vested_for_owner(&self, owner: [u8; 20], timestamp: u64) -> u128 {
        self.get_owner_schedules(owner)
            .iter()
            .filter_map(|s| self.calculate_vested_for_schedule(s, timestamp).ok())
            .sum()
    }
    
    /// Get total unvested amount for an owner
    pub fn total_unvested_for_owner(&self, owner: [u8; 20], timestamp: u64) -> u128 {
        self.get_owner_schedules(owner)
            .iter()
            .map(|s| s.total_amount)
            .sum::<u128>()
            .saturating_sub(self.total_vested_for_owner(owner, timestamp))
    }
    
    // Helper methods
    
    fn calculate_vested_for_schedule(
        &self,
        schedule: &VestingSchedule,
        timestamp: u64,
    ) -> Result<u128> {
        if timestamp < schedule.start_time {
            return Ok(0);
        }
        
        // Check cliff
        if timestamp < schedule.start_time + schedule.cliff_duration {
            return Ok(0);
        }
        
        // Check if fully vested
        let effective_duration = if schedule.accelerated {
            schedule.acceleration_time.unwrap_or(schedule.start_time) + 
            (schedule.total_duration - schedule.cliff_duration) / 2
        } else {
            schedule.start_time + schedule.total_duration
        };
        
        if timestamp >= effective_duration {
            return Ok(schedule.total_amount);
        }
        
        // Calculate vested amount based on vesting type
        let elapsed = timestamp.saturating_sub(schedule.start_time + schedule.cliff_duration);
        let vesting_period = schedule.total_duration - schedule.cliff_duration;
        
        let vested_ratio = match &schedule.vesting_type {
            VestingType::Immediate => 1.0,
            VestingType::Linear => {
                elapsed as f64 / vesting_period as f64
            }
            VestingType::CliffLinear => {
                if elapsed == 0 {
                    0.0
                } else {
                    elapsed as f64 / vesting_period as f64
                }
            }
            VestingType::Quarterly(quarters) => {
                let quarter_duration = vesting_period / *quarters as u64;
                let quarters_passed = elapsed / quarter_duration;
                quarters_passed as f64 / *quarters as f64
            }
            VestingType::Exponential => {
                // Exponential curve: faster early, slower later
                1.0 - (-3.0 * elapsed as f64 / vesting_period as f64).exp()
            }
            VestingType::Step(steps) => {
                // Find last completed step
                let mut completed_steps = 0;
                for step in steps {
                    if timestamp >= schedule.start_time + step {
                        completed_steps += 1;
                    } else {
                        break;
                    }
                }
                completed_steps as f64 / steps.len() as f64
            }
            VestingType::Performance => {
                // Calculate based on completed milestones
                if let Some(milestones) = &schedule.milestones {
                    let completed: u8 = milestones
                        .iter()
                        .filter(|m| m.completion_time.is_some())
                        .map(|m| m.vesting_percentage)
                        .sum();
                    completed as f64 / 100.0
                } else {
                    0.0
                }
            }
            VestingType::CommunityVoted => {
                // Default to linear if no community votes
                elapsed as f64 / vesting_period as f64
            }
        };
        
        let vested_amount = (schedule.total_amount as f64 * vested_ratio.clamp(0.0, 1.0)) as u128;
        
        Ok(vested_amount)
    }
    
    fn calculate_total_duration(
        &self,
        vesting_type: &VestingType,
        curve_params: &Option<VestingCurveParams>,
    ) -> u64 {
        match vesting_type {
            VestingType::Immediate => 0,
            VestingType::Linear => 4 * 365 * 24 * 60 * 60, // 4 years default
            VestingType::CliffLinear => 4 * 365 * 24 * 60 * 60, // 4 years default
            VestingType::Quarterly(quarters) => *quarters as u64 * 90 * 24 * 60 * 60, // ~90 days per quarter
            VestingType::Exponential => 4 * 365 * 24 * 60 * 60, // 4 years default
            VestingType::Step(steps) => *steps.last().unwrap_or(&0),
            VestingType::Performance => 4 * 365 * 24 * 60 * 60, // 4 years default
            VestingType::CommunityVoted => 4 * 365 * 24 * 60 * 60, // 4 years default
        }
    }
    
    fn update_schedule_status(
        &mut self,
        schedule_id: u64,
        new_status: VestingStatus,
    ) {
        // Remove from old status
        for schedules in self.status_schedules.values_mut() {
            if let Some(pos) = schedules.iter().position(|&id| id == schedule_id) {
                schedules.remove(pos);
                break;
            }
        }
        
        // Add to new status
        self.status_schedules
            .entry(new_status)
            .or_insert_with(Vec::new)
            .push(schedule_id);
    }
}


/// Vesting errors
#[derive(Debug, thiserror::Error)]
pub enum VestingError {
    #[error("Vesting schedule not found")]
    ScheduleNotFound,
    
    #[error("Schedule already revoked")]
    AlreadyRevoked,
    
    #[error("Schedule is paused")]
    SchedulePaused,
    
    #[error("Schedule is revoked")]
    ScheduleRevoked,
    
    #[error("Cannot accelerate this schedule")]
    CannotAccelerate,
    
    #[error("Schedule already accelerated")]
    AlreadyAccelerated,
    
    #[error("Insufficient votes for acceleration: {0} < {1}")]
    InsufficientVotes(f64, f64),
    
    #[error("Cannot pause schedule in current state")]
    CannotPause,
    
    #[error("Schedule is not paused")]
    NotPaused,
    
    #[error("No milestones defined for this schedule")]
    NoMilestones,
    
    #[error("Milestone not found")]
    MilestoneNotFound,
    
    #[error("Milestone already completed")]
    MilestoneAlreadyCompleted,
    
    #[error("Vesting duration too short: {0} < {1}")]
    DurationTooShort(u64, u64),
    
    #[error("Vesting duration too long: {0} > {1}")]
    DurationTooLong(u64, u64),
    
    #[error("Cliff duration exceeds total duration: {0} >= {1}")]
    CliffExceedsDuration(u64, u64),
    
    #[error("Invalid vesting parameters")]
    InvalidParameters,
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_linear_vesting() {
        let mut manager = VestingManager::new();
        let owner = [1u8; 20];
        let total_amount = 1_000_000 * 10u128.pow(18);
        let start_time = 1735689600; // Jan 1, 2025
        let vesting_type = VestingType::Linear;
        
        let schedule_id = manager.create_schedule(
            owner,
            total_amount,
            start_time,
            vesting_type,
            Some(6 * 30 * 24 * 60 * 60), // 6 month cliff
            true,
            None,
            None,
        ).unwrap();
        
        // Before cliff
        let vested = manager.calculate_vested_amount(schedule_id, start_time + 5 * 30 * 24 * 60 * 60).unwrap();
        assert_eq!(vested, 0);
        
        // At cliff (no vesting yet at exact cliff)
        let vested = manager.calculate_vested_amount(schedule_id, start_time + 6 * 30 * 24 * 60 * 60).unwrap();
        assert_eq!(vested, 0);
        
        // Halfway through vesting (after 2 years total, 1.5 years after cliff)
        let halfway = start_time + 2 * 365 * 24 * 60 * 60;
        let vested = manager.calculate_vested_amount(schedule_id, halfway).unwrap();
        assert_eq!(vested, total_amount / 2);
        
        // Fully vested
        let fully_vested = start_time + 4 * 365 * 24 * 60 * 60;
        let vested = manager.calculate_vested_amount(schedule_id, fully_vested).unwrap();
        assert_eq!(vested, total_amount);
    }
    
    #[test]
    fn test_quarterly_vesting() {
        let mut manager = VestingManager::new();
        let owner = [2u8; 20];
        let total_amount = 1_000_000 * 10u128.pow(18);
        let start_time = 1735689600;
        let vesting_type = VestingType::Quarterly(8); // 2 years (8 quarters)
        
        let schedule_id = manager.create_schedule(
            owner,
            total_amount,
            start_time,
            vesting_type,
            None,
            false,
            None,
            None,
        ).unwrap();
        
        // After 1 quarter
        let after_quarter = start_time + 90 * 24 * 60 * 60;
        let vested = manager.calculate_vested_amount(schedule_id, after_quarter).unwrap();
        assert_eq!(vested, total_amount / 8);
        
        // After 4 quarters (1 year)
        let after_year = start_time + 4 * 90 * 24 * 60 * 60;
        let vested = manager.calculate_vested_amount(schedule_id, after_year).unwrap();
        assert_eq!(vested, total_amount / 2);
        
        // After 8 quarters (2 years)
        let after_2_years = start_time + 8 * 90 * 24 * 60 * 60;
        let vested = manager.calculate_vested_amount(schedule_id, after_2_years).unwrap();
        assert_eq!(vested, total_amount);
    }
    
    #[test]
    fn test_performance_vesting() {
        let mut manager = VestingManager::new();
        let owner = [3u8; 20];
        let total_amount = 1_000_000 * 10u128.pow(18);
        let start_time = 1735689600;
        
        let milestones = vec![
            VestingMilestone {
                milestone_id: 1,
                description: "Complete Phase 1".to_string(),
                criteria: "Deliver MVP".to_string(),
                vesting_percentage: 25,
                completion_time: None,
                verification_proof: None,
                verified_by: None,
            },
            VestingMilestone {
                milestone_id: 2,
                description: "Complete Phase 2".to_string(),
                criteria: "Launch mainnet".to_string(),
                vesting_percentage: 50,
                completion_time: None,
                verification_proof: None,
                verified_by: None,
            },
            VestingMilestone {
                milestone_id: 3,
                description: "Complete Phase 3".to_string(),
                criteria: "Reach 1M users".to_string(),
                vesting_percentage: 25,
                completion_time: None,
                verification_proof: None,
                verified_by: None,
            },
        ];
        
        let schedule_id = manager.create_schedule(
            owner,
            total_amount,
            start_time,
            VestingType::Performance,
            None,
            false,
            None,
            Some(milestones.clone()),
        ).unwrap();
        
        // No milestones completed
        let vested = manager.calculate_vested_amount(schedule_id, start_time + 365 * 24 * 60 * 60).unwrap();
        assert_eq!(vested, 0);
        
        // Complete first milestone
        manager.complete_milestone(
            schedule_id,
            1,
            start_time + 100 * 24 * 60 * 60,
            [1u8; 32],
            [4u8; 20],
        ).unwrap();
        
        let vested = manager.calculate_vested_amount(schedule_id, start_time + 200 * 24 * 60 * 60).unwrap();
        assert_eq!(vested, total_amount * 25 / 100);
        
        // Complete second milestone
        manager.complete_milestone(
            schedule_id,
            2,
            start_time + 300 * 24 * 60 * 60,
            [2u8; 32],
            [4u8; 20],
        ).unwrap();
        
        let vested = manager.calculate_vested_amount(schedule_id, start_time + 400 * 24 * 60 * 60).unwrap();
        assert_eq!(vested, total_amount * 75 / 100);
        
        // Complete third milestone
        manager.complete_milestone(
            schedule_id,
            3,
            start_time + 500 * 24 * 60 * 60,
            [3u8; 32],
            [4u8; 20],
        ).unwrap();
        
        let vested = manager.calculate_vested_amount(schedule_id, start_time + 600 * 24 * 60 * 60).unwrap();
        assert_eq!(vested, total_amount);
    }
    
    #[test]
    fn test_acceleration() {
        let mut manager = VestingManager::new();
        let owner = [4u8; 20];
        let total_amount = 1_000_000 * 10u128.pow(18);
        let start_time = 1735689600;
        
        let schedule_id = manager.create_schedule(
            owner,
            total_amount,
            start_time,
            VestingType::Linear,
            Some(6 * 30 * 24 * 60 * 60), // 6 month cliff
            true, // Can accelerate
            None,
            None,
        ).unwrap();
        
        // Try to accelerate before cliff (should fail)
        let result = manager.accelerate_schedule(
            schedule_id,
            start_time + 5 * 30 * 24 * 60 * 60,
            0.8, // 80% vote
        );
        assert!(result.is_err());
        
        // Accelerate after cliff
        let acceleration_time = start_time + 7 * 30 * 24 * 60 * 60;
        manager.accelerate_schedule(
            schedule_id,
            acceleration_time,
            0.8, // 80% vote > 67% threshold
        ).unwrap();
        
        // Check accelerated vesting
        let schedule = manager.schedules.get(&schedule_id).unwrap();
        assert!(schedule.accelerated);
        assert_eq!(schedule.status, VestingStatus::Accelerated);
        
        // Calculate vested amount after acceleration
        let vested_after = manager.calculate_vested_amount(
            schedule_id,
            acceleration_time + 365 * 24 * 60 * 60,
        ).unwrap();
        
        // Should be more than without acceleration
        let manager2 = VestingManager::new();
        let schedule_id2 = manager2.create_schedule(
            owner,
            total_amount,
            start_time,
            VestingType::Linear,
            Some(6 * 30 * 24 * 60 * 60),
            true,
            None,
            None,
        ).unwrap();
        
        let vested_without = manager2.calculate_vested_amount(
            schedule_id2,
            acceleration_time + 365 * 24 * 60 * 60,
        ).unwrap();
        
        assert!(vested_after > vested_without);
    }
    
    #[test]
    fn test_claiming() {
        let mut manager = VestingManager::new();
        let owner = [5u8; 20];
        let total_amount = 1_000_000 * 10u128.pow(18);
        let start_time = 1735689600;
        
        let schedule_id = manager.create_schedule(
            owner,
            total_amount,
            start_time,
            VestingType::Linear,
            None,
            false,
            None,
            None,
        ).unwrap();
        
        // Claim after 1 year (25% vested)
        let claim_time = start_time + 365 * 24 * 60 * 60;
        let claimable = manager.claim_vested(schedule_id, claim_time).unwrap();
        assert_eq!(claimable, total_amount / 4);
        
        // Try to claim again immediately (no new vesting)
        let claimable = manager.claim_vested(schedule_id, claim_time).unwrap();
        assert_eq!(claimable, 0);
        
        // Claim after 2 years (additional 25%)
        let claim_time2 = start_time + 2 * 365 * 24 * 60 * 60;
        let claimable = manager.claim_vested(schedule_id, claim_time2).unwrap();
        assert_eq!(claimable, total_amount / 4);
        
        let schedule = manager.schedules.get(&schedule_id).unwrap();
        assert_eq!(schedule.vested_amount, total_amount / 2);
    }
}
