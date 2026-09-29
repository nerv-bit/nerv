
//! The wire protocol (erratum 144): a 1-RTT mutual PQ handshake and
//! framed AEAD sessions, generic over AsyncRead + AsyncWrite + Unpin.


use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};


use nerv_core::constants::{NET_HELLO, NET_SESSION};
use nerv_core::hash::Hash256;
use nerv_crypto::aead::{open, seal, AeadKey, Nonce, TAG_LEN};
use nerv_crypto::kdf::blake3_kdf;
use nerv_crypto::mlkem::{
    keypair_from_seed, CipherText, DecapsulationKey, EncapsulationKey, CT_LEN, EK_LEN, SEED_LEN,
};
use nerv_crypto::mldsa::{Signature, SigningKey, VerifyingKey, PK_LEN, SIG_LEN};


pub const VERSION: u8 = 1;
pub const MAX_FRAME: usize = 4 * 1024 * 1024;
pub const MAX_WIRE: usize = MAX_FRAME + TAG_LEN;
pub const HELLO_LEN: usize = 1 + PK_LEN + EK_LEN + CT_LEN;
pub const ACCEPT_LEN: usize = PK_LEN + CT_LEN + SIG_LEN;
pub const FINISH_LEN: usize = SIG_LEN;
const ENCAPS_LEN: usize = 32;


#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WireError {
    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),
    #[error("peer closed the stream")]
    Closed,
    #[error("message of {len} bytes exceeds the {max} cap")]
    TooLarge { len: usize, max: usize },
    #[error("protocol version {found}, expected {expected}")]
    Version { found: u8, expected: u8 },
    #[error("handshake signature failed")]
    BadSignature,
    #[error("frame authentication failed")]
    BadFrame,
    #[error("frame counter desynchronized")]
    Desync,
    #[error("peer does not match the pinned identity")]
    WrongPeer,
    #[error("malformed handshake message")]
    Malformed,
    #[error("crypto: {0}")]
    Crypto(#[from] nerv_crypto::CryptoError),
}


async fn read_exact_or_closed<S: AsyncRead + Unpin>(s: &mut S, buf: &mut [u8]) -> Result<(), WireError> {
    match s.read_exact(buf).await {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Err(WireError::Closed),
        Err(e) => Err(WireError::Io(e)),
    }
}


pub async fn write_msg<S: AsyncWrite + Unpin>(
    s: &mut S,
    msg: &[u8],
    cap: usize,
) -> Result<(), WireError> {
    if msg.len() > cap {
        return Err(WireError::TooLarge { len: msg.len(), max: cap });
    }
    s.write_all(&(msg.len() as u32).to_le_bytes()).await?;
    s.write_all(msg).await?;
    s.flush().await?;
    Ok(())
}


pub async fn read_msg<S: AsyncRead + Unpin>(s: &mut S, cap: usize) -> Result<Vec<u8>, WireError> {
    let mut lb = [0u8; 4];
    read_exact_or_closed(s, &mut lb).await?;
    let len = u32::from_le_bytes(lb) as usize;
    if len > cap {
        return Err(WireError::TooLarge { len, max: cap });
    }
    let mut buf = vec![0u8; len];
    read_exact_or_closed(s, &mut buf).await?;
    Ok(buf)
}


// ---------------------------------------------------------------------------
// Handshake messages (fixed-size, exact-capped)
// ---------------------------------------------------------------------------


fn encode_hello(ver: u8, vk: &VerifyingKey, ek: &EncapsulationKey, ct: &CipherText) -> Vec<u8> {
    let mut m = Vec::with_capacity(HELLO_LEN);
    m.push(ver);
    m.extend_from_slice(vk.as_bytes());
    m.extend_from_slice(ek.as_bytes());
    m.extend_from_slice(ct.as_bytes());
    m
}


fn decode_hello(b: &[u8]) -> Result<(u8, VerifyingKey, EncapsulationKey, CipherText), WireError> {
    if b.len() != HELLO_LEN {
        return Err(WireError::Malformed);
    }
    let ver = b[0];
    let vk = VerifyingKey::from_slice(&b[1..1 + PK_LEN])?;
    let ek = EncapsulationKey::from_slice(&b[1 + PK_LEN..1 + PK_LEN + EK_LEN])?;
    let ct: &[u8; CT_LEN] = b[1 + PK_LEN + EK_LEN..].try_into().map_err(|_| WireError::Malformed)?;
    Ok((ver, vk, ek, CipherText::from_bytes(ct)))
}


fn encode_accept(vk: &VerifyingKey, ct: &CipherText, sig: &Signature) -> Vec<u8> {
    let mut m = Vec::with_capacity(ACCEPT_LEN);
    m.extend_from_slice(vk.as_bytes());
    m.extend_from_slice(ct.as_bytes());
    m.extend_from_slice(sig.as_bytes());
    m
}


fn decode_accept(b: &[u8]) -> Result<(VerifyingKey, CipherText, Signature), WireError> {
    if b.len() != ACCEPT_LEN {
        return Err(WireError::Malformed);
    }
    let vk = VerifyingKey::from_slice(&b[..PK_LEN])?;
    let ct: &[u8; CT_LEN] =
        b[PK_LEN..PK_LEN + CT_LEN].try_into().map_err(|_| WireError::Malformed)?;
    let sig: &[u8; SIG_LEN] = b[PK_LEN + CT_LEN..].try_into().map_err(|_| WireError::Malformed)?;
    Ok((vk, CipherText::from_bytes(ct), Signature::from_bytes(sig)))
}


fn transcript(
    ver: u8,
    a_vk: &VerifyingKey,
    a_ek: &EncapsulationKey,
    a_ct: &CipherText,
    b_vk: &VerifyingKey,
    b_ct: &CipherText,
) -> Hash256 {
    let mut m = Vec::with_capacity(1 + 2 * PK_LEN + EK_LEN + 2 * CT_LEN);
    m.push(ver);
    m.extend_from_slice(a_vk.as_bytes());
    m.extend_from_slice(a_ek.as_bytes());
    m.extend_from_slice(a_ct.as_bytes());
    m.extend_from_slice(b_vk.as_bytes());
    m.extend_from_slice(b_ct.as_bytes());
    Hash256::concat(&NET_HELLO, &m)
}


fn signed_by(role_a: bool, t: &Hash256) -> Vec<u8> {
    let mut m = Vec::with_capacity(16 + 32);
    m.extend_from_slice(if role_a { b"nerv.net.hello.a" } else { b"nerv.net.hello.b" });
    m.extend_from_slice(t.as_bytes());
    m
}


// ---------------------------------------------------------------------------
// The session
// ---------------------------------------------------------------------------


#[derive(Clone, Copy)]
enum Role {
    Initiator,
    Responder,
}


fn nonce_of(counter: u64) -> Nonce {
    let mut n = [0u8; 12];
    n[4..].copy_from_slice(&counter.to_le_bytes());
    Nonce::from_bytes(n)
}


/// An authenticated, confidential, replay-protected framed session with
/// one peer. Frames are strictly in-order (the counters are the replay
/// defense; TCP ordering carries them).
#[derive(Debug)]
pub struct Session {
    send: AeadKey,
    recv: AeadKey,
    send_ctr: u64,
    recv_ctr: u64,
    peer: VerifyingKey,
}


impl Session {
    fn new(
        ss_a: &nerv_crypto::mlkem::SharedSecret,
        ss_b: &nerv_crypto::mlkem::SharedSecret,
        t: &Hash256,
        role: Role,
        peer: VerifyingKey,
    ) -> Session {
        let mut ikm = Vec::with_capacity(96);
        ikm.extend_from_slice(ss_a.as_bytes());
        ikm.extend_from_slice(ss_b.as_bytes());
        ikm.extend_from_slice(t.as_bytes());
        let c2s = blake3_kdf(&NET_SESSION, &ikm, b"c2s");
        let s2c = blake3_kdf(&NET_SESSION, &ikm, b"s2c");
        let (send, recv) = match role {
            Role::Initiator => (c2s, s2c),
            Role::Responder => (s2c, c2s),
        };
        Session {
            send: AeadKey::from_bytes(send),
            recv: AeadKey::from_bytes(recv),
            send_ctr: 0,
            recv_ctr: 0,
            peer,
        }
    }


    pub fn peer(&self) -> &VerifyingKey {
        &self.peer
    }


    pub async fn send_frame<S: AsyncWrite + Unpin>(
        &mut self,
        s: &mut S,
        payload: &[u8],
    ) -> Result<(), WireError> {
        if payload.len() > MAX_FRAME {
            return Err(WireError::TooLarge { len: payload.len(), max: MAX_FRAME });
        }
        let nonce = nonce_of(self.send_ctr);
        let sealed = seal(&self.send, &nonce, &[], payload)?;
        self.send_ctr = self.send_ctr.checked_add(1).ok_or(WireError::Desync)?;
        write_msg(s, &sealed, MAX_WIRE).await
    }


    pub async fn recv_frame<S: AsyncRead + Unpin>(&mut self, s: &mut S) -> Result<Vec<u8>, WireError> {
        let sealed = read_msg(s, MAX_WIRE).await?;
        if sealed.len() < TAG_LEN {
            return Err(WireError::BadFrame);
        }
        let nonce = nonce_of(self.recv_ctr);
        let payload = open(&self.recv, &nonce, &[], &sealed).map_err(|_| WireError::BadFrame)?;
        self.recv_ctr = self.recv_ctr.checked_add(1).ok_or(WireError::Desync)?;
        Ok(payload)
    }
}


// ---------------------------------------------------------------------------
// The handshake
// ---------------------------------------------------------------------------


/// The initiator (dialer): sends the hello, verifies the accept, finishes.
/// `fresh_seed` and `encap_randomness` are the caller's entropy (the host
/// draws them from the OS; tests pin them).
#[allow(clippy::too_many_lines)]
pub async fn dial_handshake<S: AsyncRead + AsyncWrite + Unpin>(
    s: &mut S,
    me: &SigningKey,
    peer_static_ek: &EncapsulationKey,
    fresh_seed: &[u8; SEED_LEN],
    encap_randomness: &[u8; ENCAPS_LEN],
    expected_peer: Option<&VerifyingKey>,
) -> Result<(Session, VerifyingKey), WireError> {
    let (fresh_ek, fresh_dk) = keypair_from_seed(fresh_seed)?;
    let (ss_a, ct_a) = peer_static_ek.encapsulate(encap_randomness)?;
    write_msg(s, &encode_hello(VERSION, me.verifying_key(), &fresh_ek, &ct_a), HELLO_LEN).await?;


    let accept = read_msg(s, ACCEPT_LEN).await?;
    let (b_vk, ct_b, sig_b) = decode_accept(&accept)?;
    let t = transcript(VERSION, me.verifying_key(), &fresh_ek, &ct_a, &b_vk, &ct_b);
    if !b_vk.verify(&signed_by(false, &t), &sig_b) {
        return Err(WireError::BadSignature);
    }
    if let Some(exp) = expected_peer {
        if &b_vk != exp {
            return Err(WireError::WrongPeer);
        }
    }
    let ss_b = fresh_dk.decapsulate(&ct_b)?;
    let sig_a = me.sign(&signed_by(true, &t))?;
    write_msg(s, sig_a.as_bytes(), FINISH_LEN).await?;


    Ok((Session::new(&ss_a, &ss_b, &t, Role::Initiator, b_vk), b_vk))
}


/// The responder (listener): reads the hello, encapsulates to the
/// initiator's fresh ek, signs, verifies the finish.
pub async fn accept_handshake<S: AsyncRead + AsyncWrite + Unpin>(
    s: &mut S,
    me: &SigningKey,
    my_static_dk: &DecapsulationKey,
    encap_randomness: &[u8; ENCAPS_LEN],
) -> Result<(Session, VerifyingKey), WireError> {
    let hello = read_msg(s, HELLO_LEN).await?;
    let (ver, a_vk, a_ek, ct_a) = decode_hello(&hello)?;
    if ver != VERSION {
        return Err(WireError::Version { found: ver, expected: VERSION });
    }
    let ss_a = my_static_dk.decapsulate(&ct_a)?;
    let (ss_b, ct_b) = a_ek.encapsulate(encap_randomness)?;
    let t = transcript(ver, &a_vk, &a_ek, &ct_a, me.verifying_key(), &ct_b);
    let sig_b = me.sign(&signed_by(false, &t))?;
    write_msg(s, &encode_accept(me.verifying_key(), &ct_b, &sig_b), ACCEPT_LEN).await?;


    let finish = read_msg(s, FINISH_LEN).await?;
    let sig_a: &[u8; SIG_LEN] =
        finish.as_slice().try_into().map_err(|_| WireError::Malformed)?;
    let sig_a = Signature::from_bytes(sig_a);
    if !a_vk.verify(&signed_by(true, &t), &sig_a) {
        return Err(WireError::BadSignature);
    }


    Ok((Session::new(&ss_a, &ss_b, &t, Role::Responder, a_vk), a_vk))
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::node_keys;
    use crate::testutil::SplitMix64;


    fn seeds(seed: u64) -> ([u8; SEED_LEN], [u8; ENCAPS_LEN], [u8; ENCAPS_LEN]) {
        let mut rng = SplitMix64::new(seed);
        let mut fill = |b: &mut [u8]| {
            for c in b.chunks_mut(8) {
                c.copy_from_slice(&rng.next_u64().to_le_bytes());
            }
        };
        let mut a = [0u8; SEED_LEN];
        let mut b = [0u8; ENCAPS_LEN];
        let mut c = [0u8; ENCAPS_LEN];
        fill(&mut a);
        fill(&mut b);
        fill(&mut c);
        (a, b, c)
    }


   #[allow(clippy::type_complexity)]
   async fn handshake_pair(
       seed: u64,
       pin: bool,
   ) -> (
       Session,
       Session,
       VerifyingKey,
       VerifyingKey,
       tokio::io::DuplexStream,
       tokio::io::DuplexStream,
   ) {

        let (sa, sb) = tokio::io::duplex(64 * 1024);
        let (mut sa, mut sb) = (sa, sb);
        let (sk_a, _, _) = node_keys(1);
        let (sk_b, ek_b, dk_b) = node_keys(2);
        let (fresh, enc_a, enc_b) = seeds(seed);
        let pin = if pin { Some(*sk_b.verifying_key()) } else { None };
        let (ra, rb) = tokio::join!(
            dial_handshake(&mut sa, &sk_a, &ek_b, &fresh, &enc_a, pin.as_ref()),
            accept_handshake(&mut sb, &sk_b, &dk_b, &enc_b),
        );
        let (sess_a, peer_b) = ra.unwrap();
        let (sess_b, peer_a) = rb.unwrap();
        assert_eq!(peer_b, *sk_b.verifying_key());
        assert_eq!(peer_a, *sk_a.verifying_key());
        (sess_a, sess_b, peer_a, peer_b, sa, sb)
    }


    #[tokio::test]
    async fn handshake_and_bidirectional_frames() {
        let (mut sess_a, mut sess_b, _, _, mut sa, mut sb) = handshake_pair(11, true);
        sess_a.send_frame(&mut sa, b"one").await.unwrap();
        sess_a.send_frame(&mut sa, b"two").await.unwrap();
        assert_eq!(sess_b.recv_frame(&mut sb).await.unwrap(), b"one");
        assert_eq!(sess_b.recv_frame(&mut sb).await.unwrap(), b"two");
        sess_b.send_frame(&mut sb, b"reply").await.unwrap();
        assert_eq!(sess_a.recv_frame(&mut sa).await.unwrap(), b"reply");
        sess_a.send_frame(&mut sa, b"").await.unwrap();
        assert_eq!(sess_b.recv_frame(&mut sb).await.unwrap(), b"");
    }


    #[tokio::test]
    async fn many_frames_in_order_both_directions() {
        let (mut sess_a, mut sess_b, _, _, mut sa, mut sb) = handshake_pair(12, true);
        let payload = vec![0xA5u8; 4096];
        for i in 0..25u64 {
            sess_a.send_frame(&mut sa, &payload).await.unwrap();
            sess_b.send_frame(&mut sb, &i.to_le_bytes()).await.unwrap();
            let got = sess_b.recv_frame(&mut sb).await.unwrap();
            assert_eq!(got, payload, "iteration {i}");
            let got = sess_a.recv_frame(&mut sa).await.unwrap();
            assert_eq!(got, i.to_le_bytes());
        }
    }


    #[tokio::test]
    async fn large_frame_roundtrip() {
        let (mut sess_a, mut sess_b, _, _, mut sa, mut sb) = handshake_pair(13, true);
        let payload = vec![0x5Au8; 512 * 1024];
        sess_a.send_frame(&mut sa, &payload).await.unwrap();
        assert_eq!(sess_b.recv_frame(&mut sb).await.unwrap(), payload);
    }


    #[tokio::test]
    async fn oversized_payload_rejected_locally() {
        let (mut sess_a, _, _, _, mut sa, _) = handshake_pair(14, true);
        let payload = vec![0u8; MAX_FRAME + 1];
        assert!(matches!(
            sess_a.send_frame(&mut sa, &payload).await,
            Err(WireError::TooLarge { len, max }) if len == MAX_FRAME + 1 && max == MAX_FRAME
        ));
    }


    #[tokio::test]
    async fn version_mismatch_rejected() {
        let (mut sa, mut sb) = tokio::io::duplex(8192);
        let (sk_a, _, _) = node_keys(1);
        let (sk_b, _, dk_b) = node_keys(2);
        let ek = EncapsulationKey::from_bytes([0u8; EK_LEN]);
        let ct = CipherText::from_bytes([0u8; CT_LEN]);
        write_msg(&mut sa, &encode_hello(9, sk_a.verifying_key(), &ek, &ct), HELLO_LEN)
            .await
            .unwrap();
        let (_, _, enc_b) = seeds(15);
        let err = accept_handshake(&mut sb, &sk_b, &dk_b, &enc_b).await.unwrap_err();
        assert!(matches!(err, WireError::Version { found: 9, expected: 1 }));
    }


    #[tokio::test]
    async fn tampered_transcript_signature_rejected() {
        let (sa, sb) = tokio::io::duplex(64 * 1024);
        let mut sb = sb;
        let (sk_a, _, _) = node_keys(1);
        let (sk_b, ek_b, _) = node_keys(2);
        let (fresh, enc_a, enc_b) = seeds(16);
        let vk_b = *sk_b.verifying_key();
        let dial = tokio::spawn(dial_handshake(
            sa,
            &sk_a,
            &ek_b,
            &fresh,
            &enc_a,
            Some(&vk_b),
        ));


        let hello = read_msg(&mut sb, HELLO_LEN).await.unwrap();
        let (ver, a_vk, a_ek, ct_a) = decode_hello(&hello).unwrap();
        let (ss_b, ct_b) = a_ek.encapsulate(&enc_b).unwrap();
        let t = transcript(ver, &a_vk, &a_ek, &ct_a, &vk_b, &ct_b);
        let mut wrong = t;
        let mut bytes = *wrong.as_bytes();
        bytes[0] ^= 1;
        wrong = Hash256::from_bytes(bytes);
        let sig = sk_b.sign(&signed_by(false, &wrong)).unwrap();
        write_msg(&mut sb, &encode_accept(&vk_b, &ct_b, &sig), ACCEPT_LEN)
            .await
            .unwrap();
        let _ = ss_b;


        let err = dial.await.unwrap().unwrap_err();
        assert!(matches!(err, WireError::BadSignature));
    }


    #[tokio::test]
    async fn wrong_peer_pin_rejected() {
        let (sa, sb) = tokio::io::duplex(64 * 1024);
        let mut sb = sb;
        let (sk_a, _, _) = node_keys(1);
        let (sk_c, _, _) = node_keys(3); // not the pinned key
        let (sk_b, ek_b, _) = node_keys(2);
        let (fresh, enc_a, enc_b) = seeds(17);
        let vk_b = *sk_b.verifying_key();
        let dial = tokio::spawn(dial_handshake(
            sa,
            &sk_a,
            &ek_b,
            &fresh,
            &enc_a,
            Some(&vk_b),
        ));


        let hello = read_msg(&mut sb, HELLO_LEN).await.unwrap();
        let (ver, a_vk, a_ek, ct_a) = decode_hello(&hello).unwrap();
        let (_, ct_b) = a_ek.encapsulate(&enc_b).unwrap();
        let vk_c = *sk_c.verifying_key();
        let t = transcript(ver, &a_vk, &a_ek, &ct_a, &vk_c, &ct_b);
        let sig = sk_c.sign(&signed_by(false, &t)).unwrap();
        write_msg(&mut sb, &encode_accept(&vk_c, &ct_b, &sig), ACCEPT_LEN)
            .await
            .unwrap();


        let err = dial.await.unwrap().unwrap_err();
        assert!(matches!(err, WireError::WrongPeer));
    }


    #[tokio::test]
    async fn unpinned_connection_from_unknown_peer_accepted() {
        // No pin: any properly-signing peer establishes a session; the
        // authorization filter is the host's (erratum 145).
        let (mut sa, mut sb) = tokio::io::duplex(64 * 1024);
       let (sk_a, _, _) = node_keys(1);
       let (sk_c, ek_c, dk_c) = node_keys(3);
       let (fresh, enc_a, enc_b) = seeds(18);

       let (r_a, r_b) = tokio::join!(
           dial_handshake(&mut sa, &sk_a, &ek_c, &fresh, &enc_a, None),
           accept_handshake(&mut sb, &sk_c, &dk_c, &enc_b),
       );


        assert_eq!(r_a.unwrap().1, *sk_c.verifying_key());
        assert_eq!(r_b.unwrap().1, *sk_a.verifying_key());
    }


    #[tokio::test]
    async fn replayed_frame_rejected() {
        let (mut sess_a, mut sess_b, _, _, mut sa, mut sb) = handshake_pair(19, true);
        sess_a.send_frame(&mut sa, b"first").await.unwrap();
        // A duplicate of frame 0, crafted with session internals.
        let replay = seal(&sess_a.send, &nonce_of(0), &[], b"first").unwrap();
        write_msg(&mut sa, &replay, MAX_WIRE).await.unwrap();


        assert_eq!(sess_b.recv_frame(&mut sb).await.unwrap(), b"first");
        let err = sess_b.recv_frame(&mut sb).await.unwrap_err();
        assert!(matches!(err, WireError::BadFrame));
    }


    #[tokio::test]
   async fn truncated_and_garbage_frames_rejected() {
       let (_, mut sess_b, _, _, mut sa, mut sb) = handshake_pair(20, true);
       write_msg(&mut sa, &[0u8; 8], MAX_WIRE).await.unwrap();
       let err = sess_b.recv_frame(&mut sb).await.unwrap_err();
       assert!(matches!(err, WireError::BadFrame));
   }



    #[tokio::test]
    async fn peer_closing_mid_handshake() {
        let (mut sa, sb) = tokio::io::duplex(64);
        drop(sb);
        let (sk_b, _, dk_b) = node_keys(2);
        let (_, _, enc_b) = seeds(21);
        let err = accept_handshake(&mut sa, &sk_b, &dk_b, &enc_b).await.unwrap_err();
        assert!(matches!(err, WireError::Closed));
    }


    #[tokio::test]
    async fn caps_enforced() {
        let (mut sa, mut sb) = tokio::io::duplex(64);
        let err = write_msg(&mut sa, &vec![0u8; 100], 10).await.unwrap_err();
        assert!(matches!(err, WireError::TooLarge { len: 100, max: 10 }));
        sa.write_all(&100u32.to_le_bytes()).await.unwrap();
        let err = read_msg(&mut sb, 10).await.unwrap_err();
        assert!(matches!(err, WireError::TooLarge { len: 100, max: 10 }));
    }


    #[tokio::test]
    async fn message_sizes_pinned() {
        assert_eq!(VERSION, 1);
        assert_eq!(HELLO_LEN, 4225);
        assert_eq!(ACCEPT_LEN, 6349);
        assert_eq!(FINISH_LEN, 3309);
        assert_eq!(MAX_FRAME, 4 * 1024 * 1024);
    }
}
