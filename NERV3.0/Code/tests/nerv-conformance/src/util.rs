//! Small shared helpers: hex rendering, leaf counting, workspace-root
//! location. No dependencies, no surprises.

use std::env;
use std::path::{Path, PathBuf};

/// Lowercase hex — used for every hash render in the tooling.
pub fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

/// Number of leaf values in a parsed TOML tree (diagnostics: "N leaves").
pub fn count_leaves(v: &toml::Value) -> usize {
    match v {
        toml::Value::Array(a) => a.iter().map(count_leaves).sum(),
        toml::Value::Table(t) => t.iter().map(|(_, x)| count_leaves(x)).sum(),
        _ => 1,
    }
}

/// Walk up from the CWD until `rel` exists; enables running the tools and
/// tests from any directory inside the workspace.
pub fn locate(rel: &str) -> Option<PathBuf> {
    let mut dir = env::current_dir().ok()?;
    loop {
        let cand = dir.join(rel);
        if cand.exists() {
            return Some(cand);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Atomic file write: temp file in the same directory, then rename.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}
