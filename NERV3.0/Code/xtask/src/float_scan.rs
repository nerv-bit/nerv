//! P5 float-ban scan: no `f32`/`f64` types and no float literals anywhere in
//! workspace source, except the documented dev-period oracle in
//! nerv-knowledge tests (DSR-11; deleted from CI at M1).
//!
//! Comment- and string-literal-aware lexer (Rust syntax subset sufficient
//! for type/literal detection). The clippy `float_arithmetic` deny covers
//! operations; this scan covers TYPES and LITERALS — two layers, one policy.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use walkdir::WalkDir;

const SCAN_ROOTS: &[&str] = &["crates", "bin", "tests", "xtask"];
const MAX_SCAN_BYTES: u64 = 8 * 1024 * 1024;

pub fn run() -> Result<()> {
    let mut violations = Vec::new();
    let mut files = 0u64;

    for root in SCAN_ROOTS {
        let root = Path::new(root);
        if !root.exists() {
            continue; // roots grow with the chunks
        }
        for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            if is_allowlisted(path) {
                continue;
            }
            let meta = fs::metadata(path)?;
            if meta.len() > MAX_SCAN_BYTES {
                bail!("{} exceeds {} bytes — refusing to scan a suspicious source file", path.display(), MAX_SCAN_BYTES);
            }
            let src = fs::read_to_string(path)?;
            files += 1;
            violations.extend(scan_file(&path.display().to_string(), &src));
        }
    }

    println!("float-scan: {files} file(s) checked");
    if violations.is_empty() {
        println!("float-scan: PASS");
        Ok(())
    } else {
        eprintln!("float-scan: {} violation(s) [P5 — integer-exact only]:", violations.len());
        for v in &violations {
            eprintln!("  - {v}");
        }
        bail!("P5 float-ban scan failed");
    }
}

/// Dev-period float oracle exception (DSR-11). Removed at M1 with the oracle.
fn is_allowlisted(path: &Path) -> bool {
    let comps: Vec<&str> = path
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();
    let in_knowledge_tests = comps.contains(&"nerv-knowledge") && comps.contains(&"tests");
    let is_oracle = path
        .file_stem()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.contains("oracle"));
    in_knowledge_tests && is_oracle
}

#[derive(Clone, Copy, PartialEq)]
enum State {
    Code,
    LineComment,
    BlockComment(usize),
    Str,
    RawStr(usize),
}

struct Scanner<'a> {
    src: &'a [u8],
    i: usize,
    line: u64,
    state: State,
}

pub fn scan_file(path: &str, src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut s = Scanner { src: src.as_bytes(), i: 0, line: 1, state: State::Code };

    macro_rules! violation {
        ($what:expr) => {
            out.push(format!("{path}:{}: P5 float {} — integer-exact only (WP §7.2)", s.line, $what))
        };
    }

    while s.i < s.src.len() {
        let c = s.src[s.i];
        if c == b'\n' {
            s.line += 1;
            if s.state == State::LineComment {
                s.state = State::Code;
            }
            s.i += 1;
            continue;
        }
        match s.state {
            State::LineComment => {
                s.i += 1;
            }
            State::BlockComment(depth) => {
                if starts_with(s.src, s.i, b"/*") {
                    s.state = State::BlockComment(depth + 1);
                    s.i += 2;
                } else if starts_with(s.src, s.i, b"*/") {
                    s.state = if depth <= 1 { State::Code } else { State::BlockComment(depth - 1) };
                    s.i += 2;
                } else {
                    s.i += 1;
                }
            }
            State::Str => {
                if c == b'\\' {
                    s.i += 2; // skip escape (incl. \n, \", \\)
                } else if c == b'"' {
                    s.state = State::Code;
                    s.i += 1;
                } else {
                    s.i += 1;
                }
            }
            State::RawStr(hashes) => {
                if c == b'"' && s.src[s.i + 1..].iter().take_while(|&&b| b == b'#').count() >= hashes {
                    // count exactly `hashes` trailing '#' then leave
                    s.i += 1 + hashes;
                    s.state = State::Code;
                } else {
                    s.i += 1;
                }
            }
            State::Code => {
                match c {
                    b'/' if next_is(s.src, s.i, b'/') => {
                        s.state = State::LineComment;
                        s.i += 2;
                    }
                    b'/' if next_is(s.src, s.i, b'*') => {
                        s.state = State::BlockComment(1);
                        s.i += 2;
                    }
                    b'"' if preceded_by_byte(s.src, s.i, b'b') => {
                        s.i += 1; // byte string: treat as Str
                        s.state = State::Str;
                    }
                    b'"' => {
                        s.state = State::Str;
                        s.i += 1;
                    }
                    b'r' if raw_string_ahead(s.src, s.i) => {
                        let hashes = s.src[s.i + 1..].iter().take_while(|&&b| b == b'#').count();
                        let quote = s.i + 1 + hashes;
                        s.i = quote + 1; // step into content
                        s.state = State::RawStr(hashes);
                    }
                    b'\'' => {
                        // char literal or lifetime — distinguish by looking ahead.
                        if let Some(skip) = char_literal_len(s.src, s.i) {
                            s.i += skip;
                        } else {
                            s.i += 1; // lifetime tick; following ident handled below
                        }
                    }
                    _ if c == b'_' || c.is_ascii_alphabetic() => {
                        let start = s.i;
                        while s.i < s.src.len()
                            && (s.src[s.i] == b'_' || s.src[s.i].is_ascii_alphanumeric())
                        {
                            s.i += 1;
                        }
                        let word = &s.src[start..s.i];
                        if word == b"f32" || word == b"f64" {
                            violation!("type/identifier `f32/f64`");
                        }
                    }
                    _ if c.is_ascii_digit() => {
                        let start = s.i;
                        let mut is_float = false;
                        // hex literal: ints only in Rust
                        if starts_with(s.src, s.i, b"0x") || starts_with(s.src, s.i, b"0X") {
                            s.i += 2;
                            while s.i < s.src.len() && (s.src[s.i].is_ascii_hexdigit() || s.src[s.i] == b'_') {
                                s.i += 1;
                            }
                        } else {
                            while s.i < s.src.len() && (s.src[s.i].is_ascii_digit() || s.src[s.i] == b'_') {
                                s.i += 1;
                            }
                            // fractional part: '.' followed by a digit (not a method call / range)
                            if s.i < s.src.len() && s.src[s.i] == b'.' && next_is_digit(s.src, s.i) {
                                is_float = true;
                                s.i += 1;
                                while s.i < s.src.len() && (s.src[s.i].is_ascii_digit() || s.src[s.i] == b'_') {
                                    s.i += 1;
                                }
                            }
                            // exponent: 'e'/'E' followed by optional sign and digit
                            if s.i < s.src.len() && (s.src[s.i] == b'e' || s.src[s.i] == b'E') {
                                let mut j = s.i + 1;
                                if j < s.src.len() && (s.src[j] == b'+' || s.src[j] == b'-') {
                                    j += 1;
                                }
                                if j < s.src.len() && s.src[j].is_ascii_digit() {
                                    is_float = true;
                                    s.i = j;
                                    while s.i < s.src.len() && (s.src[s.i].is_ascii_digit() || s.src[s.i] == b'_') {
                                        s.i += 1;
                                    }
                                }
                            }
                        }
                        // suffix
                        let suffix_start = s.i;
                        while s.i < s.src.len() && (s.src[s.i].is_ascii_alphanumeric() || s.src[s.i] == b'_') {
                            s.i += 1;
                        }
                        let suffix = &s.src[suffix_start..s.i];
                        if suffix == b"f32" || suffix == b"f64" {
                            is_float = true;
                        }
                        if is_float {
                            let lit = String::from_utf8_lossy(&s.src[start..s.i]).into_owned();
                            violation!(format!("literal `{lit}`"));
                        }
                    }
                    _ => {
                        s.i += 1;
                    }
                }
            }
        }
    }
    out
}

fn starts_with(b: &[u8], i: usize, pat: &[u8]) -> bool {
    b.len() >= i + pat.len() && &b[i..i + pat.len()] == pat
}
fn next_is(b: &[u8], i: usize, c: u8) -> bool {
    i + 1 < b.len() && b[i + 1] == c
}
fn next_is_digit(b: &[u8], i: usize) -> bool {
    i + 1 < b.len() && b[i + 1].is_ascii_digit()
}
fn preceded_by_byte(b: &[u8], i: usize, c: u8) -> bool {
    i > 0 && b[i - 1] == c
}
fn raw_string_ahead(b: &[u8], i: usize) -> bool {
    // 'r' followed by zero or more '#', then '"' (and not the ident 'r' alone)
    let mut j = i + 1;
    while j < b.len() && b[j] == b'#' {
        j += 1;
    }
    j < b.len() && b[j] == b'"'
}
/// Length of a char literal starting at the opening quote, if it is one.
fn char_literal_len(b: &[u8], i: usize) -> Option<usize> {
    // 'x' | '\n' | '\'' | '\\'
    let j = i + 1;
    if j >= b.len() {
        return None;
    }
    if b[j] == b'\\' {
        // escape: \ + one char + closing '
        if j + 2 < b.len() && b[j + 2] == b'\'' {
            return Some(3);
        }
        // \u{...}
        if j + 2 < b.len() && b[j + 1] == b'u' && b[j + 2] == b'{' {
            let mut k = j + 3;
            while k < b.len() && b[k] != b'}' {
                k += 1;
            }
            if k + 1 < b.len() && b[k + 1] == b'\'' {
                return Some(k + 2 - i);
            }
        }
        return None;
    }
    if b[j] == b'\'' {
        return None; // two consecutive ticks — not a char literal in Rust
    }
    if j + 1 < b.len() && b[j + 1] == b'\'' {
        Some(3)
    } else {
        None // lifetime: 'ident
    }
}
