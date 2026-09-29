#![allow(clippy::unwrap_used, clippy::expect_used)]

use nerv_conformance::encoding::Canonical;
use nerv_conformance::util::locate;
use nerv_core::params::{ParamValue, PARAMS_LEAF_COUNT, PARAMS_MAP, PARAMS_TOML_DIGEST};

fn real_text() -> String {
    let p = locate("specs/params.toml").expect("spec file");
    std::fs::read_to_string(p).expect("read spec")
}

fn walk(v: &toml::Value, path: &str, out: &mut Vec<(String, Canonical)>) {
    match v {
        toml::Value::Table(t) => {
            for (k, x) in t {
                let child = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                walk(x, &child, out);
            }
        }
        toml::Value::Array(a) => {
            for (i, x) in a.iter().enumerate() {
                walk(x, &format!("{path}[{i}]"), out);
            }
        }
        toml::Value::String(s) => out.push((path.to_string(), Canonical::Str(s.clone()))),
        toml::Value::Integer(n) => {
            assert!(*n >= 0, "negative leaf at {path}");
            out.push((path.to_string(), Canonical::Int(*n as u64)));
        }
        toml::Value::Boolean(b) => out.push((path.to_string(), Canonical::Bool(*b))),
        toml::Value::Float(_) | toml::Value::Datetime(_) => panic!("P5 violation at {path}"),
    }
}

#[test]
fn params_map_matches_live_toml_exactly() {
    let raw: toml::Value = toml::from_str(&real_text()).unwrap();
    let mut walked = Vec::new();
    walk(&raw, "", &mut walked);
    walked.sort_by(|a, b| a.0.cmp(&b.0));

    assert_eq!(walked.len(), PARAMS_LEAF_COUNT, "leaf count drift");
    for ((gp, gv), (mp, mv)) in walked.iter().zip(PARAMS_MAP.iter()) {
        assert_eq!(gp, mp, "path drift at {gp}");
        match (gv, mv) {
            (Canonical::Int(n), ParamValue::U64(m)) => assert_eq!(n, m, "value drift at {mp}"),
            (Canonical::Str(s), ParamValue::Str(m)) => assert_eq!(s, m, "value drift at {mp}"),
            (Canonical::Bool(b), ParamValue::Bool(m)) => assert_eq!(b, m, "value drift at {mp}"),
            other => panic!("type drift at {mp}: {other:?}"),
        }
    }
}

#[test]
fn params_toml_digest_matches_live_file() {
    let text = real_text();
    assert_eq!(&PARAMS_TOML_DIGEST[..], blake3::hash(text.as_bytes()).as_bytes());
}
