//! Freeze/verify of the conformance vector registry.
//!
//! freeze: regenerate families deterministically from the spec, validate
//! every record against its schema, write data files + manifest atomically.
//! verify: manifest format, spec-hash, schema hashes, data hashes, record
//! counts, record schema validity, and full re-derivation — catching
//! tampering, spec drift, schema drift, and generator drift respectively.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::encoding::{decode, Canonical};
use crate::error::RegError;
use crate::schema::{self, FamilySchema};
use crate::spec::LoadedSpec;
use crate::util::{hex, write_atomic};
use crate::vectors;

pub const MANIFEST_FORMAT_VERSION: u64 = 1;
const MANIFEST_FILE: &str = "manifest.bin";

pub struct FamilyManifest {
    pub name: String,
    pub schema_hash: [u8; 32],
    pub data_hash: [u8; 32],
    pub record_count: u64,
}

pub struct Manifest {
    pub format_version: u64,
    pub spec_hash: [u8; 32],
    pub families: Vec<FamilyManifest>,
}

pub struct FamilyReport {
    pub name: String,
    pub record_count: u64,
    pub schema_hash: [u8; 32],
    pub data_hash: [u8; 32],
}

pub struct Report {
    pub spec_hash: [u8; 32],
    pub families: Vec<FamilyReport>,
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "conformance report (spec-hash {}):", hex(&self.spec_hash))?;
        for fam in &self.families {
            writeln!(
                f,
                "  {:<20} {:>7} records  schema {}…  data {}…",
                fam.name,
                fam.record_count,
                hex(&fam.schema_hash[..8]),
                hex(&fam.data_hash[..8])
            )?;
        }
        Ok(())
    }
}

fn manifest_path(dir: &Path) -> PathBuf {
    dir.join(MANIFEST_FILE)
}

fn data_path(dir: &Path, family: &str) -> PathBuf {
    dir.join(format!("{family}.bin"))
}

/// The spec's declared family list and the code's schema registry must agree
/// in both directions — this is the spec↔code coupling that catches typos.
fn family_alignment(spec: &LoadedSpec) -> Result<(), RegError> {
    let declared: BTreeSet<&str> = spec.spec.conformance.vector_families.iter().map(|s| s.as_str()).collect();
    for fam in &declared {
        if schema::schema(fam).is_none() {
            return Err(RegError::UnknownFamily { family: fam.to_string() });
        }
    }
    for fs in schema::FAMILIES {
        if !declared.contains(fs.name) {
            return Err(RegError::UnknownFamily { family: fs.name.to_string() });
        }
    }
    Ok(())
}

fn family_to_canonical(f: &FamilyManifest) -> Canonical {
    Canonical::map()
        .with("name", Canonical::str(&f.name))
        .with("schema_hash", Canonical::bytes(&f.schema_hash))
        .with("data_hash", Canonical::bytes(&f.data_hash))
        .with("record_count", Canonical::int(f.record_count))
}

fn manifest_to_canonical(m: &Manifest) -> Canonical {
    Canonical::map()
        .with("format_version", Canonical::int(m.format_version))
        .with("spec_hash", Canonical::bytes(&m.spec_hash))
        .with(
            "families",
            Canonical::seq(m.families.iter().map(family_to_canonical).collect()),
        )
}

fn hash32(b: &[u8]) -> Result<[u8; 32], RegError> {
    b.try_into().map_err(|_| RegError::Generator("hash field is not 32 bytes".to_string()))
}

fn manifest_from_bytes(bytes: &[u8]) -> Result<Manifest, RegError> {
    let c = decode(bytes)?;
    let m = c.as_map()?;
    let format_version = m.get("format_version")?.as_u64()?;
    let spec_hash = hash32(m.get("spec_hash")?.as_bytes()?)?;
    let fams = m.get("families")?.as_seq()?;
    let mut families = Vec::with_capacity(fams.len());
    for f in fams {
        let fm = f.as_map()?;
        families.push(FamilyManifest {
            name: fm.get("name")?.as_str()?.to_string(),
            schema_hash: hash32(fm.get("schema_hash")?.as_bytes()?)?,
            data_hash: hash32(fm.get("data_hash")?.as_bytes()?)?,
            record_count: fm.get("record_count")?.as_u64()?,
        });
    }
    Ok(Manifest { format_version, spec_hash, families })
}

pub fn freeze(spec: &LoadedSpec, dir: &Path, force: bool) -> Result<Manifest, RegError> {
    family_alignment(spec)?;
    let mpath = manifest_path(dir);
    if mpath.exists() && !force {
        return Err(RegError::AlreadyFrozen);
    }
    fs::create_dir_all(dir).map_err(|source| RegError::Io { path: dir.into(), source })?;
    let mut families = Vec::new();
    for fam in &spec.spec.conformance.vector_families {
        let fs: &FamilySchema = schema::schema(fam)
            .ok_or_else(|| RegError::UnknownFamily { family: fam.clone() })?;
        if fs.deferred {
            continue;
        }
        let records = vectors::generate(fam, spec)?;
        for (i, rec) in records.iter().enumerate() {
            schema::check_record(fs, rec)
                .map_err(|e| RegError::Generator(format!("family `{fam}` record {i}: {e}")))?;
        }
        let record_count = records.len() as u64;
        let data = Canonical::seq(records).encode();
        let data_hash = schema::hash_domain("nerv.data.v1", &data);
        let dpath = data_path(dir, fam);
        write_atomic(&dpath, &data).map_err(|source| RegError::Io { path: dpath, source })?;
        families.push(FamilyManifest {
            name: fam.clone(),
            schema_hash: schema::schema_hash(fs),
            data_hash,
            record_count,
        });
    }
    let manifest = Manifest { format_version: MANIFEST_FORMAT_VERSION, spec_hash: spec.hash, families };
    let bytes = manifest_to_canonical(&manifest).encode();
    write_atomic(&mpath, &bytes).map_err(|source| RegError::Io { path: mpath.clone(), source })?;
    Ok(manifest)
}

pub fn verify(spec: &LoadedSpec, dir: &Path) -> Result<Report, RegError> {
    family_alignment(spec)?;
    let mpath = manifest_path(dir);
    if !mpath.exists() {
        return Err(RegError::NoManifest);
    }
    let bytes = fs::read(&mpath).map_err(|source| RegError::Io { path: mpath.clone(), source })?;
    let manifest = manifest_from_bytes(&bytes)?;
    if manifest.format_version != MANIFEST_FORMAT_VERSION {
        return Err(RegError::FormatVersion { found: manifest.format_version, expected: MANIFEST_FORMAT_VERSION });
    }
    if manifest.spec_hash != spec.hash {
        return Err(RegError::SpecDrift { frozen: hex(&manifest.spec_hash), current: hex(&spec.hash) });
    }

    let mut reports = Vec::new();
    let mut seen = BTreeSet::new();
    for entry in &manifest.families {
        seen.insert(entry.name.clone());
        let fs = schema::schema(&entry.name)
            .ok_or_else(|| RegError::UnknownFamily { family: entry.name.clone() })?;
        if fs.deferred || !vectors::has_generator(&entry.name) {
            return Err(RegError::MissingGenerator { family: entry.name.clone() });
        }
        if schema::schema_hash(fs) != entry.schema_hash {
            return Err(RegError::SchemaDrift { family: entry.name.clone() });
        }
        let dpath = data_path(dir, &entry.name);
        if !dpath.exists() {
            return Err(RegError::MissingData { family: entry.name.clone() });
        }
        let data = fs::read(&dpath).map_err(|source| RegError::Io { path: dpath.clone(), source })?;
        if schema::hash_domain("nerv.data.v1", &data) != entry.data_hash {
            return Err(RegError::Tamper { family: entry.name.clone() });
        }
        let decoded = decode(&data)?;
        let records = decoded.as_seq()?;
        if records.len() as u64 != entry.record_count {
            return Err(RegError::RecordCount {
                family: entry.name.clone(),
                found: records.len() as u64,
                expected: entry.record_count,
            });
        }
        for rec in records {
            schema::check_record(fs, rec).map_err(|_| RegError::Tamper { family: entry.name.clone() })?;
        }
        let regenerated = vectors::generate(&entry.name, spec)?;
        let drift = if records.len() != regenerated.len() {
            records.len().min(regenerated.len()) as u64
        } else {
            match records.iter().zip(&regenerated).position(|(x, y)| x != y) {
                Some(i) => i as u64,
                None => u64::MAX,
            }
        };
        if drift != u64::MAX {
            return Err(RegError::GeneratorDrift { family: entry.name.clone(), first_diff: drift });
        }
        reports.push(FamilyReport {
            name: entry.name.clone(),
            record_count: entry.record_count,
            schema_hash: entry.schema_hash,
            data_hash: entry.data_hash,
        });
    }

    for fam in &spec.spec.conformance.vector_families {
        let fs = schema::schema(fam)
            .ok_or_else(|| RegError::UnknownFamily { family: fam.clone() })?;
        if fs.deferred {
            if data_path(dir, fam).exists() {
                return Err(RegError::UnexpectedData { family: fam.clone() });
            }
            continue;
        }
        if !seen.contains(fam) {
            return Err(RegError::MissingFamily { family: fam.clone() });
        }
    }

    Ok(Report { spec_hash: spec.hash, families: reports })
}

