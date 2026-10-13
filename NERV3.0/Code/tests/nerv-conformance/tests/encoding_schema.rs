#![allow(clippy::unwrap_used, clippy::expect_used)]

use nerv_conformance::encoding::{decode, Canonical, EncError};
use nerv_conformance::schema;

fn roundtrip(v: Canonical) {
    let enc = v.encode();
    assert_eq!(decode(&enc).unwrap(), v);
    for cut in 0..enc.len() {
        assert!(decode(&enc[..cut]).is_err(), "truncation at {cut} must fail");
    }
    let mut ext = enc.clone();
    ext.push(0);
    assert!(decode(&ext).is_err(), "trailing bytes must fail");
}

#[test]
fn canonical_roundtrips() {
    roundtrip(Canonical::int(0));
    roundtrip(Canonical::int(u64::MAX));
    roundtrip(Canonical::Bool(true));
    roundtrip(Canonical::str("nerv.cm"));
    roundtrip(Canonical::bytes(&(0u8..=255).collect::<Vec<u8>>()));
    roundtrip(Canonical::seq(vec![Canonical::int(1), Canonical::Bool(false)]));
    roundtrip(Canonical::map().with("a", Canonical::int(1)).with("b", Canonical::seq(Vec::new())));
    let mut big = Canonical::map();
    for i in 0..100u64 {
        big = big.with(&format!("k{i:03}"), Canonical::int(i));
    }
    roundtrip(big);
}

#[test]
fn malformed_inputs_rejected() {
    assert!(matches!(decode(&[0xFF]), Err(EncError::UnknownTag { .. })));
    assert!(matches!(decode(&[b'b', 0x02]), Err(EncError::InvalidBool { byte: 2 })));

    // duplicate map key: M | count 2 | "a"=1 | "a"=2 (u32-LE framed keys)
    let mut b = vec![b'M'];
    b.extend(2u32.to_le_bytes());
    b.extend(1u32.to_le_bytes());
    b.push(b'a');
    b.push(b'I');
    b.extend(1u64.to_le_bytes());
    b.extend(1u32.to_le_bytes());
    b.push(b'a');
    b.push(b'I');
    b.extend(2u64.to_le_bytes());
    assert!(matches!(decode(&b), Err(EncError::DuplicateKey)));

    // nesting beyond MAX_DEPTH (128)
    let mut enc = Canonical::int(0).encode();
    for _ in 0..130 {
        let mut w = vec![b'['];
        w.extend(1u32.to_le_bytes());
        w.extend(&enc);
        enc = w;
    }
    assert!(matches!(decode(&enc), Err(EncError::DepthLimit(_))));
}

#[test]
fn from_toml_matrix() {
    let v: toml::Value = toml::from_str("a = 1\nb = \"x\"\nc = true\nd = [1, 2]").unwrap();
    let c = Canonical::from_toml(&v).unwrap();
    assert_eq!(c.get("a").unwrap().as_u64().unwrap(), 1);
    assert_eq!(c.get("b").unwrap().as_str().unwrap(), "x");
    assert_eq!(c.get("c").unwrap().as_bool().unwrap(), true);
    assert_eq!(c.get("d").unwrap().as_seq().unwrap().len(), 2);

    let v: toml::Value = toml::from_str("a = 1.5").unwrap();
    assert!(matches!(Canonical::from_toml(&v), Err(EncError::ForbiddenTomlType)));
    let v: toml::Value = toml::from_str("a = -1").unwrap();
    assert!(matches!(Canonical::from_toml(&v), Err(EncError::NegativeInt { value: -1 })));
}

#[test]
fn canonical_order_is_byte_lexicographic() {
    let vals = vec![
        Canonical::int(1),
        Canonical::int(256),
        Canonical::str("a"),
        Canonical::str("ab"),
        Canonical::seq(vec![Canonical::int(0)]),
        Canonical::Bool(false),
    ];
    for a in &vals {
        for b in &vals {
            let ord = a.encode().cmp(&b.encode());
            assert_eq!(a.cmp(b), ord);
        }
    }
    // LE byte-lex is not numeric: 256 < 1 canonically
    assert_eq!(Canonical::int(256).cmp(&Canonical::int(1)), std::cmp::Ordering::Less);
}

#[test]
fn schema_registry_shape() {
    assert_eq!(schema::FAMILIES.len(), 8);
    let deferred: Vec<&str> = schema::FAMILIES.iter().filter(|f| f.deferred).map(|f| f.name).collect();
    assert_eq!(
        deferred,
        vec!["adam_replay", "dkg_transcripts", "seal_decode", "witness_verify", "emission_audit", "fs_transcripts"]
    );
    assert!(schema::schema("container_encoding").is_some());
    assert!(schema::schema("nope").is_none());
    // schema hashes: deterministic and distinct per family
    let h1 = schema::schema_hash(schema::schema("container_encoding").unwrap());
    assert_eq!(h1, schema::schema_hash(schema::schema("container_encoding").unwrap()));
    assert_ne!(h1, schema::schema_hash(schema::schema("emission_schedule").unwrap()));
}

#[test]
fn emission_records_schema_checked() {
    let fs = schema::schema("emission_schedule").unwrap();
    let curve = Canonical::map()
        .with("bucket", Canonical::str("founder"))
        .with("kind", Canonical::str("curve"))
        .with("day", Canonical::int(360))
        .with("cumulative_nano", Canonical::int(0));
    assert!(schema::check_record(fs, &curve).is_ok());

    let missing = Canonical::map()
        .with("bucket", Canonical::str("founder"))
        .with("kind", Canonical::str("curve"))
        .with("day", Canonical::int(1));
    let e = schema::check_record(fs, &missing).unwrap_err();
    assert!(e.contains("cumulative_nano"), "{e}");

    let bad_kind = Canonical::map()
        .with("bucket", Canonical::str("founder"))
        .with("kind", Canonical::str("bogus"))
        .with("day", Canonical::int(1))
        .with("cumulative_nano", Canonical::int(1));
    let e = schema::check_record(fs, &bad_kind).unwrap_err();
    assert!(e.contains("unknown kind"), "{e}");

    let wrong_type = Canonical::map()
        .with("bucket", Canonical::str("founder"))
        .with("kind", Canonical::str("curve"))
        .with("day", Canonical::str("one"))
        .with("cumulative_nano", Canonical::int(1));
    let e = schema::check_record(fs, &wrong_type).unwrap_err();
    assert!(e.contains("expected int"), "{e}");

    let unknown_field = Canonical::map()
        .with("bucket", Canonical::str("founder"))
        .with("kind", Canonical::str("curve"))
        .with("day", Canonical::int(1))
        .with("cumulative_nano", Canonical::int(1))
        .with("junk", Canonical::int(9));
    let e = schema::check_record(fs, &unknown_field).unwrap_err();
    assert!(e.contains("unknown field `junk`"), "{e}");

    let env = Canonical::map()
        .with("bucket", Canonical::str("community"))
        .with("kind", Canonical::str("envelope"))
        .with("opens_day", Canonical::int(0))
        .with("closes_day", Canonical::int(720))
        .with("total_nano", Canonical::int(2_200_000_000_000_000_000))
        .with("burn_unclaimed", Canonical::Bool(true));
    assert!(schema::check_record(fs, &env).is_ok());

    let env_missing = Canonical::map()
        .with("bucket", Canonical::str("community"))
        .with("kind", Canonical::str("envelope"))
        .with("opens_day", Canonical::int(0))
        .with("closes_day", Canonical::int(720))
        .with("total_nano", Canonical::int(1));
    let e = schema::check_record(fs, &env_missing).unwrap_err();
    assert!(e.contains("burn_unclaimed"), "{e}");

    let fsd = schema::schema("adam_replay").unwrap();
    assert!(schema::check_record(fsd, &Canonical::map()).is_err());
}

#[test]
fn container_records_schema_checked() {
    let fs = schema::schema("container_encoding").unwrap();
    let v = Canonical::seq(vec![Canonical::int(1)]);
    let rec = Canonical::map()
        .with("value", v.clone())
        .with("encoded", Canonical::bytes(&v.encode()));
    assert!(schema::check_record(fs, &rec).is_ok());

    let bad = Canonical::map()
        .with("value", v)
        .with("encoded", Canonical::int(3));
    assert!(schema::check_record(fs, &bad).is_err());
}

