//! The two chambers (WP §12.8, App C; erratum 179).


use nerv_core::hash::Hash256;
use nerv_crypto::mldsa::VerifyingKey;


/// Which chamber a vote belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Chamber {
    NoteHolder,
    Validator,
}


impl Chamber {
    pub fn name(self) -> &'static str {
        match self {
            Chamber::NoteHolder => "note-holder",
            Chamber::Validator => "validator",
        }
    }
}


/// The bootstrap phase (§12.8): at mainnet the note-holder chamber is
/// empty; parameter governance rests with the validator chamber under
/// constitutional constraints until circulation accumulates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BootstrapPhase {
    /// The note-holder chamber is empty: the validator chamber alone
    /// decides parameter-tier questions.
    #[default]
    Bootstrap,
    /// Both chambers are active.
    Both,
}


impl BootstrapPhase {
    /// The phase is Bootstrap iff the note-holder chamber has no
    /// participation (zero cast weight in the shielded tally).
    pub fn from_participation(note_holder_weight: u128) -> BootstrapPhase {
        if note_holder_weight == 0 {
            BootstrapPhase::Bootstrap
        } else {
            BootstrapPhase::Both
        }
    }
}


/// One chamber's cast vote (the abstract form both ballot types reduce
/// to for threshold evaluation).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChamberVote {
    pub voter_id: Hash256,
    pub choice: crate::ballot::Choice,
    pub weight_nano: u64,
}


/// One chamber's tally contribution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChamberTally {
    pub yes_nano: u128,
    pub no_nano: u128,
    pub abstain_nano: u128,
    pub ballots: usize,
}


impl ChamberTally {
    pub fn total_cast(&self) -> u128 {
        self.yes_nano + self.no_nano + self.abstain_nano
    }


    pub fn add(&mut self, choice: crate::ballot::Choice, weight: u64) {
        let w = u128::from(weight);
        match choice {
            crate::ballot::Choice::Yes => self.yes_nano += w,
            crate::ballot::Choice::No => self.no_nano += w,
            crate::ballot::Choice::Abstain => self.abstain_nano += w,
        }
        self.ballots += 1;
    }


    /// Simple majority of cast weight (≥, not >).
    pub fn majority_yes(&self) -> bool {
        self.yes_nano * 2 >= self.total_cast()
    }


    /// Supermajority (≥ 2/3 of cast weight).
    pub fn supermajority_yes(&self) -> bool {
        self.yes_nano * 3 >= self.total_cast() * 2
    }
}

