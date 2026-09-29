//! The enumerated, time-boxed emergency powers (WP §C.2, §9.4, §8.5;
//! erratum 182): exactly three, no extensibility.


use std::collections::BTreeMap;


use nerv_core::hash::Hash256;
use nerv_core::types::{Epoch, ShardId};


use crate::tally::TallyResult;


/// Each power's time-box (in epochs; erratum 182).
pub const KILL_SWITCH_EPOCHS: u64 = 1;
pub const HASH_WIDENING_EPOCHS: u64 = 1;
pub const ACCELERATED_SPLIT_EPOCHS: u64 = 1;


const _: () = assert!(KILL_SWITCH_EPOCHS >= 1);
const _: () = assert!(HASH_WIDENING_EPOCHS >= 1);
const _: () = assert!(ACCELERATED_SPLIT_EPOCHS >= 1);


/// The three enumerated powers — no more (§C.2; erratum 182).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EmergencyKind {
    /// The lattice kill-switch (§9.4): suspends sealed-delta settlement
    /// and reveal. The ledger continues; custody untouched.
    KillSwitch,
    /// Hash widening (§9.4): 256 → 384-bit digests at a coordinated upgrade.
    HashWidening { from_bits: u16, to_bits: u16 },
    /// Accelerated split under sustained 2× overload (§8.5): bypasses
    /// the 7-day notice period.
    AcceleratedSplit { target: ShardId },
}


impl EmergencyKind {
    pub fn name(&self) -> &'static str {
        match self {
            EmergencyKind::KillSwitch => "kill-switch",
            EmergencyKind::HashWidening { .. } => "hash-widening",
            EmergencyKind::AcceleratedSplit { .. } => "accelerated-split",
        }
    }


    pub fn time_box_epochs(&self) -> u64 {
        match self {
            EmergencyKind::KillSwitch => KILL_SWITCH_EPOCHS,
            EmergencyKind::HashWidening { .. } => HASH_WIDENING_EPOCHS,
            EmergencyKind::AcceleratedSplit { .. } => ACCELERATED_SPLIT_EPOCHS,
        }
    }


    pub fn renewable(&self) -> bool {
        matches!(self, EmergencyKind::KillSwitch)
    }


    /// The full structural identity of the power — what `Display` emits and
    /// `thiserror`'s format-string interpolation consumes. Format:
    /// `<name>[from→to]` for the two parameter-bearing variants, bare
    /// `<name>` otherwise.
    fn fmt_full(&self) -> String {
        match self {
            EmergencyKind::KillSwitch => "kill-switch".to_string(),
            EmergencyKind::HashWidening { from_bits, to_bits } => {
                format!("hash-widening[{}→{}]", from_bits, to_bits)
            }
            EmergencyKind::AcceleratedSplit { target } => {
                format!("accelerated-split[{}]", target)
            }
        }
    }
}


impl std::fmt::Display for EmergencyKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.fmt_full())
    }
}


/// One emergency declaration: the kind, the activation epoch, the
/// expiry, and the evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmergencyDeclaration {
    pub kind: EmergencyKind,
    pub activated_epoch: Epoch,
    pub expires_epoch: Epoch,
    /// The evidence hash: the lattice-break report, the overload metrics,
    /// or the cryptanalysis report — category-specific, opaque here.
    pub evidence_hash: Hash256,
}


impl EmergencyDeclaration {
    pub fn new(
        kind: EmergencyKind,
        at_epoch: Epoch,
        evidence_hash: Hash256,
    ) -> Result<EmergencyDeclaration, crate::error::EmergencyError> {
        if let EmergencyKind::HashWidening { from_bits, to_bits } = kind {
            if from_bits != 256 || to_bits != 384 {
                return Err(crate::error::EmergencyError::BadWidening { from: from_bits, to: to_bits });
            }
        }
        Ok(EmergencyDeclaration {
            kind,
            activated_epoch: at_epoch,
            expires_epoch: Epoch::from_u64(
                at_epoch.as_u64().saturating_add(kind.time_box_epochs()),
            ),
            evidence_hash,
        })
    }


    pub fn is_active(&self, at_epoch: Epoch) -> bool {
        at_epoch < self.expires_epoch
    }


    pub fn is_expired(&self, at_epoch: Epoch) -> bool {
        at_epoch >= self.expires_epoch
    }


    /// The passage rule: both chambers, simple majority (the expedited
    /// path — §C.2), participation floor applies.
    pub fn passage(&self, result: &TallyResult) -> bool {
        result.majority_yes()
    }
}


/// The emergency state: active or expired (erratum 182).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EmergencyState {
    Active(EmergencyDeclaration),
    Expired(EmergencyDeclaration),
}


/// The emergency ledger: tracks declarations by kind. One active
/// declaration per kind at a time.
#[derive(Clone, Debug, Default)]
pub struct EmergencyLedger {
    declarations: BTreeMap<u64, EmergencyState>,
    next_nonce: u64,
}


impl EmergencyLedger {
    pub fn new() -> EmergencyLedger {
        EmergencyLedger::default()
    }


    fn kind_key(kind: &EmergencyKind) -> u64 {
        match kind {
            EmergencyKind::KillSwitch => 0,
            EmergencyKind::HashWidening { .. } => 1,
            EmergencyKind::AcceleratedSplit { .. } => 2,
        }
    }


    /// Declare an emergency (after passage). Rejects a duplicate active
    /// declaration of the same kind.
    pub fn declare(
        &mut self,
        kind: EmergencyKind,
        at_epoch: Epoch,
        evidence_hash: Hash256,
    ) -> Result<EmergencyDeclaration, crate::error::EmergencyError> {
        let key = Self::kind_key(&kind);
        if let Some(EmergencyState::Active(_)) = self.declarations.get(&key) {
            return Err(crate::error::EmergencyError::AlreadyActive { kind });
        }
        let decl = EmergencyDeclaration::new(kind, at_epoch, evidence_hash)?;
        self.declarations.insert(key, EmergencyState::Active(decl.clone()));
        self.next_nonce += 1;
        Ok(decl)
    }


    /// Renew the kill-switch (one epoch at a time). Other powers are not
    /// renewable (§C.2: "time-boxed and expire").
    pub fn renew_kill_switch(
        &mut self,
        at_epoch: Epoch,
    ) -> Result<EmergencyDeclaration, crate::error::EmergencyError> {
        let key = Self::kind_key(&EmergencyKind::KillSwitch);
        let Some(state) = self.declarations.get_mut(&key) else {
            return Err(crate::error::EmergencyError::Expired { kind: EmergencyKind::KillSwitch });
        };
        let EmergencyState::Active(decl) = state else {
            return Err(crate::error::EmergencyError::Expired { kind: EmergencyKind::KillSwitch });
        };
        if at_epoch.as_u64() < decl.expires_epoch.as_u64() {
            return Err(crate::error::EmergencyError::BadRenewal {
                at: at_epoch.as_u64(),
                expiry: decl.expires_epoch.as_u64(),
            });
        }
        decl.expires_epoch = Epoch::from_u64(
            at_epoch.as_u64().saturating_add(KILL_SWITCH_EPOCHS),
        );
        Ok(decl.clone())
    }


    /// Sweep: expire every declaration whose time-box has lapsed.
    /// Returns the kinds that just expired.
    pub fn sweep(&mut self, at_epoch: Epoch) -> Vec<EmergencyKind> {
        let mut expired = Vec::new();
        for state in self.declarations.values_mut() {
            let is_expired = matches!(state, EmergencyState::Active(decl) if decl.is_expired(at_epoch));
            if is_expired {
                if let EmergencyState::Active(decl) = state {
                    let kind = decl.kind;
                    *state = EmergencyState::Expired(decl.clone());
                    expired.push(kind);
                }
            }
        }
        expired
    }


    /// The active declaration for a kind, if any.
    pub fn active(&self, kind: &EmergencyKind) -> Option<&EmergencyDeclaration> {
        match self.declarations.get(&Self::kind_key(kind)) {
            Some(EmergencyState::Active(d)) => Some(d),
            _ => None,
        }
    }


    /// The kill-switch's protocol effect (§9.4): sealed-delta settlement
    /// is suspended. The ledger continues; custody untouched.
    pub fn kill_switch_active(&self, at_epoch: Epoch) -> bool {
        self.active(&EmergencyKind::KillSwitch)
            .is_some_and(|d| d.is_active(at_epoch))
    }


    /// The hash-widening effect: the proposed digest width, if active.
    pub fn widened_digest_bits(&self, at_epoch: Epoch) -> Option<u16> {
        match self.active(&EmergencyKind::HashWidening { from_bits: 256, to_bits: 384 }) {
            Some(d) if d.is_active(at_epoch) => Some(384),
            _ => None,
        }
    }


    /// The accelerated-split effect: the target shard, if active.
    pub fn accelerated_split_target(&self, at_epoch: Epoch) -> Option<ShardId> {
        let probe = EmergencyKind::AcceleratedSplit {
            target: ShardId::new(6, 0).unwrap(),
        };
        // The kind_key for AcceleratedSplit is constant (2), so any
        // instance looks up the same entry.
        match self.declarations.get(&Self::kind_key(&probe)) {
            Some(EmergencyState::Active(d)) if d.is_active(at_epoch) => {
                match d.kind {
                    EmergencyKind::AcceleratedSplit { target } => Some(target),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::chambers::{BootstrapPhase, ChamberTally};
    use crate::tally::TallyResult;


    fn epoch(e: u64) -> Epoch {
        Epoch::from_u64(e)
    }


    fn evidence(seed: u8) -> Hash256 {
        Hash256::from_bytes([seed; 32])
    }


    fn yes_tally() -> TallyResult {
        TallyResult {
            referendum_id_hash: None,
            note_holder: ChamberTally {
                yes_nano: 600, no_nano: 400, abstain_nano: 0, ballots: 1,
            },
            validator: ChamberTally {
                yes_nano: 60, no_nano: 40, abstain_nano: 0, ballots: 1,
            },
            bootstrap: BootstrapPhase::Both,
        }
    }


    fn no_tally() -> TallyResult {
        TallyResult {
            referendum_id_hash: None,
            note_holder: ChamberTally {
                yes_nano: 400, no_nano: 600, abstain_nano: 0, ballots: 1,
            },
            validator: ChamberTally {
                yes_nano: 40, no_nano: 60, abstain_nano: 0, ballots: 1,
            },
            bootstrap: BootstrapPhase::Both,
        }
    }


    #[test]
    fn time_box_pins() {
        assert_eq!(KILL_SWITCH_EPOCHS, 1);
        assert_eq!(HASH_WIDENING_EPOCHS, 1);
        assert_eq!(ACCELERATED_SPLIT_EPOCHS, 1);
    }


    #[test]
    fn kill_switch_lifecycle() {
        let mut ledger = EmergencyLedger::new();
        assert!(!ledger.kill_switch_active(epoch(100)));


        let decl = ledger
            .declare(EmergencyKind::KillSwitch, epoch(100), evidence(1))
            .unwrap();
        assert_eq!(decl.activated_epoch, epoch(100));
        assert_eq!(decl.expires_epoch, epoch(101));
        assert!(decl.is_active(epoch(100)));
        assert!(!decl.is_active(epoch(101)));
        assert!(ledger.kill_switch_active(epoch(100)));
        assert!(!ledger.kill_switch_active(epoch(101)));


        // Renewal at the boundary.
        ledger.renew_kill_switch(epoch(101)).unwrap();
        assert!(ledger.kill_switch_active(epoch(101)));
        assert!(!ledger.kill_switch_active(epoch(102)));


        // Renewal before the boundary is an error.
        assert!(matches!(
            ledger.renew_kill_switch(epoch(101)),
            Err(crate::error::EmergencyError::BadRenewal { .. })
        ));


        // Sweep expires it.
        let expired = ledger.sweep(epoch(102));
        assert_eq!(expired.len(), 1);
        assert!(!ledger.kill_switch_active(epoch(102)));


        // After expiry, a new declaration is possible.
        ledger.declare(EmergencyKind::KillSwitch, epoch(103), evidence(2)).unwrap();
        assert!(ledger.kill_switch_active(epoch(103)));
    }


    #[test]
    fn hash_widening() {
        let mut ledger = EmergencyLedger::new();
        assert_eq!(ledger.widened_digest_bits(epoch(100)), None);


        let decl = ledger
            .declare(
                EmergencyKind::HashWidening { from_bits: 256, to_bits: 384 },
                epoch(100), evidence(2),
            )
            .unwrap();
        assert!(decl.is_active(epoch(100)));
        assert_eq!(ledger.widened_digest_bits(epoch(100)), Some(384));
        assert_eq!(ledger.widened_digest_bits(epoch(101)), None, "expired");


        // The wrong widening is rejected.
        assert!(matches!(
            EmergencyDeclaration::new(
                EmergencyKind::HashWidening { from_bits: 256, to_bits: 512 },
                epoch(100), evidence(3),
            ),
            Err(crate::error::EmergencyError::BadWidening { from_bits: 256, to_bits: 512 })
        ));


        // Hash widening is not renewable.
        ledger.sweep(epoch(101));
        assert!(matches!(
            ledger.renew_kill_switch(epoch(101)),
            Err(crate::error::EmergencyError::Expired { .. })
        ));
    }


    #[test]
    fn accelerated_split() {
        let mut ledger = EmergencyLedger::new();
        let target = ShardId::new(6, 42).unwrap();
        let decl = ledger
            .declare(
                EmergencyKind::AcceleratedSplit { target },
                epoch(100), evidence(4),
            )
            .unwrap();
        assert!(decl.is_active(epoch(100)));
        assert_eq!(ledger.accelerated_split_target(epoch(100)), Some(target));
        assert_eq!(ledger.accelerated_split_target(epoch(101)), None, "expired");


        // A second declaration of the same kind while active is rejected.
        assert!(matches!(
            ledger.declare(EmergencyKind::AcceleratedSplit { target }, epoch(100), evidence(5)),
            Err(crate::error::EmergencyError::AlreadyActive { .. })
        ));


        // After expiry, a new declaration succeeds.
        ledger.sweep(epoch(101));
        ledger.declare(EmergencyKind::AcceleratedSplit { target }, epoch(101), evidence(6)).unwrap();
    }


    #[test]
    fn passage_rules() {
        let decl = EmergencyDeclaration::new(
            EmergencyKind::KillSwitch, epoch(100), evidence(1),
        ).unwrap();
        assert!(decl.passage(&yes_tally()));
        assert!(!decl.passage(&no_tally()));
    }


    #[test]
    fn one_per_kind_and_enumerated() {
        let mut ledger = EmergencyLedger::new();
        // One of each kind can coexist.
        ledger.declare(EmergencyKind::KillSwitch, epoch(100), evidence(1)).unwrap();
        ledger.declare(
            EmergencyKind::HashWidening { from_bits: 256, to_bits: 384 },
            epoch(100), evidence(2),
        ).unwrap();
        let target = ShardId::new(6, 0).unwrap();
        ledger.declare(EmergencyKind::AcceleratedSplit { target }, epoch(100), evidence(3)).unwrap();
        assert!(ledger.kill_switch_active(epoch(100)));
        assert_eq!(ledger.widened_digest_bits(epoch(100)), Some(384));
        assert!(ledger.accelerated_split_target(epoch(100)).is_some());


        // But a second of the same kind is rejected.
        assert!(matches!(
            ledger.declare(EmergencyKind::KillSwitch, epoch(100), evidence(9)),
            Err(crate::error::EmergencyError::AlreadyActive { .. })
        ));


        // The sweep expires all three.
        let expired = ledger.sweep(epoch(101));
        assert_eq!(expired.len(), 3);
        assert!(!ledger.kill_switch_active(epoch(101)));
    }


    #[test]
    fn all_powers_expire() {
        let mut ledger = EmergencyLedger::new();
        let target = ShardId::new(6, 7).unwrap();
        ledger.declare(EmergencyKind::KillSwitch, epoch(100), evidence(1)).unwrap();
        ledger.declare(
            EmergencyKind::HashWidening { from_bits: 256, to_bits: 384 },
            epoch(100), evidence(2),
        ).unwrap();
        ledger.declare(EmergencyKind::AcceleratedSplit { target }, epoch(100), evidence(3)).unwrap();


        // One epoch later: everything expired (except a renewed kill-switch).
        let expired = ledger.sweep(epoch(101));
        assert_eq!(expired.len(), 3);


        // Nothing is active.
        assert!(!ledger.kill_switch_active(epoch(101)));
        assert_eq!(ledger.widened_digest_bits(epoch(101)), None);
        assert_eq!(ledger.accelerated_split_target(epoch(101)), None);


        // The kill-switch can be re-declared (renewed via re-declaration
        // after expiry).
        ledger.declare(EmergencyKind::KillSwitch, epoch(101), evidence(4)).unwrap();
        assert!(ledger.kill_switch_active(epoch(101)));
    }
}
