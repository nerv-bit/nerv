//! The epoch lifecycle end to end (chunk 9, part 3): two epochs of fresh
//! DKGs; a full chunk and a force-padded sub-minimum batch revealed under
//! the outgoing key; the handoff window's clean close; backup delivery
//! and stand-in verification; the ledger; and epoch independence.


#![allow(clippy::unwrap_used, clippy::expect_used)]


use nerv_core::constants::SEAL_NOISE;
use nerv_core::hash::{Hash256, Xof};
use nerv_seal::decrypt::{decode_chunk, ChunkReveal, CHUNK_MAX};
use nerv_seal::dkg::{
    share_commitment, CommitMatrix, MemberSecret, ShareSecret, COMMITTEE_SIZE, THRESHOLD,
};
use nerv_seal::digitize::{digitize, Plaintext, COORDS, SLOTS};
use nerv_seal::encrypt::Ciphertext;
use nerv_seal::epoch::{
    backup_payload, establish_epoch, open_backup_payload, pad_batch, verify_backup,
    BatchId, EpochError, EpochIndex, EpochKeyHandle, HandoffWindow, MissedReveal,
    PendingBatch, RevealLedger, RotationRecord, SAMPLES_PER_REVEAL, SealedBackup,
    BACKUP_PAYLOAD_SIZE, CHUNK_MIN,
};
use nerv_seal::error::RevealError;
use nerv_seal::ring::{Mat2x8Ntt, Mat8x8Ntt, Poly, Vec8, Q};
use nerv_seal::sampling::{ASeed, NoiseSeed};
use nerv_seal::vpd::{combine_partials, prove_partial};


fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}


fn seed_of(tag: u8, k: u64) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[0] = tag;
    b[24..32].copy_from_slice(&k.to_le_bytes());
    b
}


struct Epoch {
    handle: EpochKeyHandle,
    ac: CommitMatrix,
    secrets: Vec<MemberSecret>,
    a_ntt: Mat8x8Ntt,
    t_ntt: Mat2x8Ntt,
}


fn establish(tag: u8, epoch: u64) -> Epoch {
    let a_seed = ASeed::from_bytes(seed_of(tag, 0xE0));
    let members: Vec<(MemberSecret, [u8; 32])> = (1..=COMMITTEE_SIZE as u8)
        .map(|i| {
            (
                MemberSecret::generate(i, &seed_of(tag, u64::from(i)), COMMITTEE_SIZE, THRESHOLD)
                    .unwrap(),
                seed_of(tag, 0x2000 + u64::from(i)),
            )
        })
        .collect();
    let handle = establish_epoch(EpochIndex(epoch), &a_seed, &members).unwrap();
    let ac = CommitMatrix::expand(&a_seed).unwrap();
    Epoch {
        a_ntt: handle.public.expand_a().unwrap().ntt(),
        t_ntt: handle.public.t().ntt(),
        handle,
        ac,
        secrets: members.into_iter().map(|(s, _)| s).collect(),
    }
}


fn share_secret(e: &Epoch, j: u8) -> ShareSecret {
    let ji = j as usize - 1;
    let own = e.secrets[ji].fragment_for(j);
    let received: Vec<(u8, Vec8, Vec8)> = e
        .secrets
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != ji)
        .map(|(_, s)| {
            let (f, r) = s.fragment_for(j);
            (s.member, f, r)
        })
        .collect();
    ShareSecret::assemble(j, &own, &received, COMMITTEE_SIZE, THRESHOLD).unwrap()
}


fn build_legs(e: &Epoch, st: &mut u64, legs: u64, coords_out: &mut Vec<[u64; COORDS]>) -> Ciphertext {
    let mut agg = Ciphertext::zero();
    for _ in 0..legs {
        let mut c = [0u64; COORDS];
        for v in c.iter_mut() {
            *v = splitmix64(st);
        }
        coords_out.push(c);
        let m = digitize(&c);
        let mut nb = [0u8; 32];
        nb[..8].copy_from_slice(&splitmix64(st).to_le_bytes());
        agg = agg.add(&Ciphertext::encrypt_cached(&e.a_ntt, &e.t_ntt, &NoiseSeed::from_bytes(nb), &m).unwrap());
    }
    agg
}


fn expected_sums(coords: &[[u64; COORDS]]) -> [u64; SLOTS] {
    let mut sums = [0u64; SLOTS];
    for c in coords {
        for (s, d) in sums.iter_mut().zip(digitize(c).slot_values()) {
            *s += d;
        }
    }
    sums
}


fn reveal(e: &Epoch, ct: &Ciphertext, legs: u64) -> ChunkReveal {
    let subset: Vec<u8> = (1..=THRESHOLD as u8).collect();
    let mut partials = Vec::new();
    let mut w_js = Vec::new();
    for (k, &j) in subset.iter().enumerate() {
        let share = share_secret(e, j);
        partials.push(prove_partial(&share, &e.ac, &e.handle.public.a_seed(), ct.u(), &subset, COMMITTEE_SIZE, THRESHOLD, &[0xE0 + k as u8; 32]).unwrap());
        // Reconstruct W_j from the members' public fragments.
        let ji = j as usize - 1;
        let mut w = Vec8::zero();
        for s in e.secrets.iter() {
            let (f, r) = s.fragment_for(j);
            let mp_w = nerv_seal::dkg::commit(&e.ac, &f, &r);
            w = w.add(&mp_w);
        }
        let _ = ji;
        w_js.push(w);
    }
    let su = combine_partials(&partials, &e.ac, &e.handle.public.a_seed(), ct.u(), &subset, &w_js, COMMITTEE_SIZE, THRESHOLD).unwrap();
    decode_chunk(legs, &su, ct.v()).unwrap()
}


#[test]
fn rotation_ceremony_end_to_end() {
    let e1 = establish(0x61, 1);
    let e2 = establish(0x62, 2);
    assert_ne!(e1.handle.transcript_digest, e2.handle.transcript_digest);


    // A full chunk under the epoch-1 key.
    let mut st = 0xEF00_0001u64;
    let mut coords = Vec::new();
    let full = build_legs(&e1, &mut st, CHUNK_MAX, &mut coords);
    let sums = expected_sums(&coords);
    let r_full = reveal(&e1, &full, CHUNK_MAX);
    assert_eq!(*r_full.sums(), sums);


    // A sub-minimum batch, force-padded and revealed with REAL legs.
    let mut coords2 = Vec::new();
    let sub = build_legs(&e1, &mut st, 100, &mut coords2);
    let sums2 = expected_sums(&coords2);
    let padded = pad_batch(&e1.a_ntt, &e1.t_ntt, &sub, 100, &seed_of(0x61, 0xF00D)).unwrap();
    let r_sub = reveal(&e1, &padded, 100);
    assert_eq!(*r_sub.sums(), sums2);


    // Handoff window: both batches opened, clean close.
    let mut window = HandoffWindow::new(
        EpochIndex(1),
        1_000_000 + 2,
        vec![
            PendingBatch { id: BatchId(1), legs: CHUNK_MAX },
            PendingBatch { id: BatchId(2), legs: 100 },
        ],
    )
    .unwrap();
    window.record_reveal(BatchId(1)).unwrap();
    window.record_reveal(BatchId(2)).unwrap();
    assert!(window.close().is_empty());


    // The rotation record and ledger.
    let mut ledger = RevealLedger::new(None);
    ledger.record(&e1.handle.transcript_digest).unwrap();
    ledger.record(&e1.handle.transcript_digest).unwrap();
    assert_eq!(ledger.samples_exposed(&e1.handle.transcript_digest), 2 * SAMPLES_PER_REVEAL);
    let record = RotationRecord {
        epoch_out: EpochIndex(1),
        epoch_in: EpochIndex(2),
        out_transcript_digest: e1.handle.transcript_digest,
        in_transcript_digest: e2.handle.transcript_digest,
        padded_batches: 1,
        opened: 2,
        missed: window.close(),
    };
    assert_eq!(RotationRecord::from_bytes(&record.to_bytes()).unwrap(), record);
    assert_ne!(record.digest(), [0u8; 32]);


    // Epoch independence: the epoch-1 ciphertext does not decode under the
    // epoch-2 key (garbage decode lands outside the envelope).
    let subset: Vec<u8> = (1..=THRESHOLD as u8).collect();
    let mut partials = Vec::new();
    let mut w_js = Vec::new();
    for (k, &j) in subset.iter().enumerate() {
        let share = share_secret(&e2, j);
        partials.push(prove_partial(&share, &e2.ac, &e2.handle.public.a_seed(), full.u(), &subset, COMMITTEE_SIZE, THRESHOLD, &[0x90 + k as u8; 32]).unwrap());
        w_js.push(share_commitment_publics(&e2, j));
    }
    let su2 = combine_partials(&partials, &e2.ac, &e2.handle.public.a_seed(), full.u(), &subset, &w_js, COMMITTEE_SIZE, THRESHOLD).unwrap();
    assert!(matches!(
        decode_chunk(CHUNK_MAX, &su2, full.v()),
        Err(RevealError::NegativeDigit { .. }) | Err(RevealError::DigitSumAboveEnvelope { .. })
    ));
}


fn share_commitment_publics(e: &Epoch, j: u8) -> Vec8 {
    let mut w = Vec8::zero();
    for s in e.secrets.iter() {
        let (f, r) = s.fragment_for(j);
        w = w.add(&nerv_seal::dkg::commit(&e.ac, &f, &r));
    }
    w
}


#[test]
fn missed_reveals_and_window_discipline() {
    let e = EpochIndex(5);
    let mut w = HandoffWindow::new(
        e,
        42,
        vec![
            PendingBatch { id: BatchId(1), legs: 128 },
            PendingBatch { id: BatchId(2), legs: 128 },
            PendingBatch { id: BatchId(3), legs: 40 },
        ],
    )
    .unwrap();
    w.record_reveal(BatchId(2)).unwrap();
    let missed = w.close();
    assert_eq!(
        missed,
        vec![
            MissedReveal { epoch: e, batch: BatchId(1), legs: 128 },
            MissedReveal { epoch: e, batch: BatchId(3), legs: 40 },
        ]
    );
    assert!(matches!(
        w.record_reveal(BatchId(1)),
        Err(EpochError::WindowClosed)
    ));
    let mut capped = RevealLedger::new(Some(1));
    capped.record(&[1; 32]).unwrap();
    assert!(matches!(
        capped.record(&[1; 32]),
        Err(EpochError::OverRevealCap { cap: 1, .. })
    ));
}


#[test]
fn backup_delivery_and_stand_in_verification() {
    let e = establish(0x63, 9);
    let j = 4u8;
    let share = share_secret(&e, j);
    let w_j = share_commitment_publics(&e, j);


    // Payload roundtrip + verification (the verifiable-resharing check).
    let payload = backup_payload(&share);
    assert_eq!(payload.len(), BACKUP_PAYLOAD_SIZE);
    let opened = open_backup_payload(&payload).unwrap();
    assert_eq!(opened, share);
    assert!(verify_backup(&e.ac, &w_j, &opened));


    // The test-side reference seal (a real cipher: XOF keystream +
    // BLAKE3 tag — encrypt-and-MAC), standing in for the delivery
    // layer's ML-KEM-768 + ChaCha20-Poly1305 box (erratum 53).
    let seal = |key: &[u8; 32], pt: &[u8]| -> Vec<u8> {
        let mut ks = Xof::framed(&SEAL_NOISE, &[b"test.seal.ks", key, &(pt.len() as u64).to_le_bytes()]);
        let ct: Vec<u8> = pt.iter().map(|&b| b ^ ks.next_u64() as u8).collect();
        let tag = *Hash256::framed(&SEAL_NOISE, &[b"test.seal.mac", key, &ct]).as_bytes();
        let mut out = ct;
        out.extend_from_slice(&tag);
        out
    };
    let open = |key: &[u8; 32], boxed: &[u8]| -> Option<Vec<u8>> {
        if boxed.len() < 32 {
            return None;
        }
        let (ct, tag) = boxed.split_at(boxed.len() - 32);
        let expect = *Hash256::framed(&SEAL_NOISE, &[b"test.seal.mac", key, ct]).as_bytes();
        if tag != expect {
            return None;
        }
        let mut ks = Xof::framed(&SEAL_NOISE, &[b"test.seal.ks", key, &(ct.len() as u64).to_le_bytes()]);
        Some(ct.iter().map(|&b| b ^ ks.next_u64() as u8).collect())
    };
    let key = [0x5EED; 32];
    let mut envelope = SealedBackup {
        from: j,
        to: j,
        epoch: EpochIndex(9),
        sealed: seal(&key, &payload),
    };
    let wire = envelope.to_bytes();
    assert_eq!(SealedBackup::from_bytes(&wire).unwrap(), envelope);
    let recovered = open(&key, &envelope.sealed).unwrap();
    let stand_in = open_backup_payload(&recovered).unwrap();
    assert!(verify_backup(&e.ac, &w_j, &stand_in));


    // A tampered box fails the tag; a wrong key fails; garbage payload
    // fails bounds; a mismatched share fails the commitment check.
    envelope.sealed[0] ^= 0xFF;
    assert!(open(&key, &envelope.sealed).is_none());
    assert!(open(&[0; 32], &SealedBackup::from_bytes(&wire).unwrap().sealed).is_none());
    assert!(matches!(
        open_backup_payload(&[0u8; BACKUP_PAYLOAD_SIZE]),
        Err(EpochError::BadPayloadMember { member: 0, .. })
    ));
    let other = share_secret(&e, 5);
    assert!(!verify_backup(&e.ac, &w_j, &other));
}


#[test]
fn pad_injection_beyond_the_real_envelope_is_detected() {
    let e = establish(0x64, 12);
    // One real leg with minimal digits: delta = all ones.
    let delta = [1u64; COORDS];
    let m = digitize(&delta);
    let mut st = 0xEF00_0002u64;
    let mut nb = [0u8; 32];
    nb[..8].copy_from_slice(&splitmix64(&mut st).to_le_bytes());
    let real = Ciphertext::encrypt_cached(&e.a_ntt, &e.t_ntt, &NoiseSeed::from_bytes(nb), &m).unwrap();


    // Honest padding to CHUNK_MIN: decodes exactly at legs = 1.
    let padded = pad_batch(&e.a_ntt, &e.t_ntt, &real, 1, &seed_of(0x64, 0x1)).unwrap();
    let honest = reveal(&e, &padded, 1);
    assert_eq!(honest.sums()[0], 1);
    assert_eq!(honest.coords(), delta);


    // Malicious "pads": two encryptions of a 200-digit plaintext, added to
    // the real aggregate — +400/slot at slot 0, far past the legs-1
    // envelope of 255 → an invalid reveal, not silent corruption.
    let mut pad_vals = [0u64; 256];
    pad_vals.fill(200);
    let evil_m = Plaintext::from_pair(&nerv_seal::ring::Vec2::new([
        Poly::new(pad_vals),
        Poly::new(pad_vals),
    ]))
    .unwrap();
    let mut evil = real.clone();
    for i in 0..2u64 {
        let mut xof = Xof::framed(&SEAL_NOISE, &[b"test.evil.pad", &i.to_le_bytes()]);
        let sb = xof.read_array::<32>();
        evil = evil.add(&Ciphertext::encrypt_cached(&e.a_ntt, &e.t_ntt, &NoiseSeed::from_bytes(sb), &evil_m).unwrap());
    }
    let subset: Vec<u8> = (1..=THRESHOLD as u8).collect();
    let mut partials = Vec::new();
    let mut w_js = Vec::new();
    for (k, &j) in subset.iter().enumerate() {
        let share = share_secret(&e, j);
        partials.push(prove_partial(&share, &e.ac, &e.handle.public.a_seed(), evil.u(), &subset, COMMITTEE_SIZE, THRESHOLD, &[0x70 + k as u8; 32]).unwrap());
        w_js.push(share_commitment_publics(&e, j));
    }
    let su = combine_partials(&partials, &e.ac, &e.handle.public.a_seed(), evil.u(), &subset, &w_js, COMMITTEE_SIZE, THRESHOLD).unwrap();
    match decode_chunk(1, &su, evil.v()) {
        Err(RevealError::DigitSumAboveEnvelope { slot: 0, value, envelope: 255 }) => {
            assert!(value >= 401, "injected value {value}");
        }
        other => panic!("expected envelope rejection, got {other:?}"),
    }

}
