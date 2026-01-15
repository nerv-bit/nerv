// src/embedding/circuit/recursive.rs
// ============================================================================
// RECURSIVE PROOF FOLDING WITH NOVA
// ============================================================================
// Implements infinite recursion for unbounded history proofs using Nova.
// Folds new LatentLedger step proofs into a running IVC proof.
// Enables light-client verification of entire chain state from genesis.
// ============================================================================

use nova_scotia::{
    nova::{NovaCycleFoldPublicParams, Prover as NovaProver, Verifier as NovaVerifier},
    traits::{Engine, Group},
    FileLocation,
};
use nova_snark::{
    provider::{PastaEngine, PallasEngine},
    traits::snark::default_ck_hint,
};

use nova_snark::{
    traits::{Engine, Group},
    relaxed::{RelaxedR1CSInstance, RelaxedR1CSWitness},
    Proof, Prover,
};

use crate::{
    embedding::circuit::latent_ledger::LatentLedgerCircuit,
    Result, NervError,
};
use serde::{Serialize, Deserialize};

type E1 = PallasEngine; // Primary curve
type E2 = PastaEngine;  // Secondary for cycle folding

/// Recursive prover for NERV embedding updates with proper Nova IVC folding
pub struct RecursiveProver<E1, E2>
where
    E1: Engine,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
{
    /// Public parameters
    pp: NovaCycleFoldPublicParams<E1, E2>,

    /// Running relaxed instance (public, commited in state/embedding hash)
    running_instance: RelaxedR1CSInstance<E1::Scalar>,

    /// Running relaxed witness (private - maintained by full/archive nodes only)
    running_witness: RelaxedR1CSWitness<E1::Scalar>,
}

impl<E1, E2> RecursiveProver<E1, E2>
where
    E1: Engine,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
{
    /// Initialize from genesis (default relaxed state)
    pub fn new(pp: NovaCycleFoldPublicParams<E1, E2>) -> Self {
        let running_instance = RelaxedR1CSInstance::default(&pp);
        let running_witness = RelaxedR1CSWitness::default(&pp);

        Self {
            pp,
            running_instance,
            running_witness,
        }
    }

    /// Prove one embedding transition step and fold into running IVC state
    pub fn prove_step(
        &mut self,
        step_circuit: LatentLedgerCircuit<E1::Scalar>,
    ) -> Result<Proof<E1, E2>, NervError> {
        let prover = NovaProver::<E1, E2>::new(&self.pp);

        // Proper Nova folding: prove the step while folding previous relaxed state
        // This produces a succinct proof for the step and updates the running relaxed state
        let (new_instance, new_witness, step_proof) = prover
            .prove_relaxed(
                &self.pp,
                &step_circuit,
                &self.running_instance,
                &self.running_witness,
            )
            .map_err(|_| NervError::Circuit("Nova relaxed prove failed".into()))?;

        // Update running state for next fold
        self.running_instance = new_instance;
        self.running_witness = new_witness;

        // The step_proof is succinct and can be verified independently for this step
        // For full chain verification, use the final proof + final running_instance commitment
        Ok(step_proof)
    }

    /// Verify the final folded proof against the final embedding hash
    pub fn verify_final(
        &self,
        final_proof: &Proof<E1, E2>,
        final_embedding_hash: [u8; 32],
    ) -> Result<bool, NervError> {
        let verifier = NovaVerifier::<E1, E2>::new(&self.pp);

        // Verify the proof w.r.t. the final relaxed instance (whose commitment includes the hash)
        verifier
            .verify(final_proof, &self.running_instance)
            .map_err(|_| NervError::Circuit("Nova verify failed".into()))?;

        // Additional check: final instance's public IO commitment matches embedding hash
        let committed_hash = self.running_instance.commitment_to_io(); // Or equivalent API
        Ok(committed_hash == final_embedding_hash)
    }
}
