//! Entitlement claiming (WP §12.3; erratum 178/180): the wallet's
//! claim-leg construction for emission commitment-notes.

use nerv_core::hash::Hash256;
use nerv_core::types::TxId;
use nerv_economy::claim::ClaimLegParts;
use nerv_economy::emission::{
    derive_claim_key, eligibility_digest, note_commitment, claim_nullifier,
};

use crate::keys::WalletAddress;

/// The wallet's claim credentials: derived from the master seed.
#[derive(Clone, Debug)]
pub struct ClaimCredentials {
    pub ck: [u8; 32],
    pub bucket: &'static str,
    pub amount_nano: u64,
    pub blinding: [u8; 32],
}

impl ClaimCredentials {
    pub fn from_master_seed(
        seed: &[u8; 32],
        bucket: &'static str,
        amount_nano: u64,
        blinding: [u8; 32],
    ) -> ClaimCredentials {
        ClaimCredentials {
            ck: derive_claim_key(seed),
            bucket,
            amount_nano,
            blinding,
        }
    }

    pub fn commitment(&self) -> [u8; 32] {
        note_commitment(&self.ck, self.bucket, self.amount_nano, &self.blinding)
    }

    pub fn nullifier(&self) -> [u8; 32] {
        claim_nullifier(&self.ck, self.bucket)
    }

    pub fn eligibility(&self) -> [u8; 32] {
        eligibility_digest(&self.ck, self.bucket, self.amount_nano)
    }

    pub fn to_leg_parts(&self) -> ClaimLegParts {
        ClaimLegParts {
            ck: self.ck,
            bucket: self.bucket,
            amount_nano: self.amount_nano,
            blinding: self.blinding,
        }
    }
}

/// A claim request: what the wallet needs to claim its entitlement.
#[derive(Clone, Debug)]
pub struct ClaimRequest {
    pub credentials: ClaimCredentials,
    pub recipient: WalletAddress,
    pub fee_nano: u64,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn claim_credentials_derivation() {
        let seed = [7u8; 32];
        let c1 = ClaimCredentials::from_master_seed(&seed, "community", 100, [1u8; 32]);
        let c2 = ClaimCredentials::from_master_seed(&seed, "community", 100, [1u8; 32]);
        assert_eq!(c1.commitment(), c2.commitment());
        assert_eq!(c1.nullifier(), c2.nullifier());
        assert_eq!(c1.eligibility(), c2.eligibility());

        let c3 = ClaimCredentials::from_master_seed(&[8u8; 32], "community", 100, [1u8; 32]);
        assert_ne!(c1.commitment(), c3.commitment());
        assert_ne!(c1.nullifier(), c3.nullifier());

        let c4 = ClaimCredentials::from_master_seed(&seed, "ecosystem", 100, [1u8; 32]);
        assert_ne!(c1.commitment(), c4.commitment());
        assert_ne!(c1.eligibility(), c4.eligibility());
        // Same ck, different bucket → different commitment but the
        // nullifier differs (it's bucket-dependent too).
        assert_ne!(c1.nullifier(), c4.nullifier());

        let c5 = ClaimCredentials::from_master_seed(&seed, "community", 200, [1u8; 32]);
        assert_ne!(c1.commitment(), c5.commitment());
        assert_ne!(c1.eligibility(), c5.eligibility());
        assert_eq!(c1.nullifier(), c5.nullifier(), "the nullifier is amount-blind");
    }
}
