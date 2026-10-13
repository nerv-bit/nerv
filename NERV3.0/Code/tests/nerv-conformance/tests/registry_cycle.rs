#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::path::PathBuf;

use nerv_conformance::encoding::{decode, Canonical};
use nerv_conformance::error::RegError;
use nerv_conformance::registry;
use nerv_conformance::spec::LoadedSpec;
use nerv_conformance::util::locate;

fn spec() -> LoadedSpec {
    LoadedSpec::load(&locate("specs/params.toml").unwrap()).unwrap()
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("nerv-conf-{}-{tag}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn freeze_verify_cycle() {
    let s = spec();
    let d = tmp("cycle");
    let m = registry::freeze(&s, &d, false).unwrap();
    assert_eq!(m.families.len(), 2);
    let cont = m.families.iter().find(|f| f.name == "container_encoding").unwrap();
    assert!(cont.record_count >= 25);
    let emis = m.families.iter().find(|f| f.name == "emission_schedule").unwrap();
    // 1441 + 1801 + 3601 + 1 + 1 + 1801 + 3601 daily curve points + 2 envelopes
    assert_eq!(emis.record_count, 12_247);

    let r = registry::verify(&s, &d).unwrap();
    assert_eq!(r.families.len(), 2);
}

#[test]
fn refreeze_without_force_rejected() {
    let s = spec();
    let d = tmp("already");
    registry::freeze(&s, &d, false).unwrap();
    assert!(matches!(registry::freeze(&s, &d, false), Err(RegError::AlreadyFrozen)));
    registry::freeze(&s, &d, true).unwrap();
}

#[test]
fn tampered_data_rejected() {
    let s = spec();
    let d = tmp("tamper");
    registry::freeze(&s, &d, false).unwrap();
    let dp = d.join("emission_schedule.bin");
    let mut bytes = fs::read(&dp).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    fs::write(&dp, bytes).unwrap();
    match registry::verify(&s, &d) {
        Err(RegError::Tamper { family }) => assert_eq!(family, "emission_schedule"),
        other => panic!("expected Tamper, got {other:?}"),
    }
}

#[test]
fn missing_data_rejected() {
    let s = spec();
    let d = tmp("missing");
    registry::freeze(&s, &d, false).unwrap();
    fs::remove_file(d.join("container_encoding.bin")).unwrap();
    match registry::verify(&s, &d) {
        Err(RegError::MissingData { family }) => assert_eq!(family, "container_encoding"),
        other => panic!("expected MissingData, got {other:?}"),
    }
}

#[test]
fn spec_drift_rejected() {
    let s = spec();
    let d = tmp("drift");
    registry::freeze(&s, &d, false).unwrap();
    let text = fs::read_to_string(locate("specs/params.toml").unwrap()).unwrap();
    let mutated = text.replacen(
        "spec_version = \"3.0-corrigenda\"",
        "spec_version = \"3.0-x\"",
        1,
    );
    let other = LoadedSpec::from_str(&mutated).unwrap();
    match registry::verify(&other, &d) {
        Err(RegError::SpecDrift { .. }) => {}
        other => panic!("expected SpecDrift, got {other:?}"),
    }
}

#[test]
fn unknown_family_in_manifest_rejected() {
    let s = spec();
    let d = tmp("unknown-family");
    registry::freeze(&s, &d, false).unwrap();
    let mp = d.join("manifest.bin");
    let mut c = decode(&fs::read(&mp).unwrap()).unwrap();
    if let Canonical::Map(ref mut m) = c {
        if let Some(Canonical::Seq(v)) = m.get("families").cloned() {
            let mut v = v;
            if let Canonical::Map(ref mut fm) = v[0] {
                fm.insert("name".into(), Canonical::str("bogus"));
            }
            m.insert("families".into(), Canonical::seq(v));
        }
    }
    fs::write(&mp, c.encode()).unwrap();
    match registry::verify(&s, &d) {
        Err(RegError::UnknownFamily { family }) => assert_eq!(family, "bogus"),
        other => panic!("expected UnknownFamily, got {other:?}"),
    }
}

#[test]
fn deferred_family_with_data_rejected() {
    let s = spec();
    let d = tmp("unexpected");
    registry::freeze(&s, &d, false).unwrap();
    fs::write(d.join("adam_replay.bin"), b"not frozen through the registry").unwrap();
    match registry::verify(&s, &d) {
        Err(RegError::UnexpectedData { family }) => assert_eq!(family, "adam_replay"),
        other => panic!("expected UnexpectedData, got {other:?}"),
    }
}

#[test]
fn no_manifest_rejected() {
    let s = spec();
    let d = tmp("empty");
    assert!(matches!(registry::verify(&s, &d), Err(RegError::NoManifest)));
}

