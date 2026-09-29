//! Family schemas: the declarative record contracts of the vector registry.
//! Fully declarative by design — the schema descriptor IS the validation
//! logic, so its BLAKE3 hash detects code-side schema drift on verify.

use blake3::Hasher;

use crate::encoding::Canonical;
use crate::error::EncError;

pub fn hash_domain(domain: &str, msg: &[u8]) -> [u8; 32] {
    let mut h = Hasher::new_derive_key(domain);
    h.update(msg);
    *h.finalize().as_bytes()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldType {
    Any,
    Int,
    Str,
    Bytes,
    Bool,
}

impl FieldType {
    pub fn name(&self) -> &'static str {
        match self {
            FieldType::Any => "any",
            FieldType::Int => "int",
            FieldType::Str => "str",
            FieldType::Bytes => "bytes",
            FieldType::Bool => "bool",
        }
    }

    pub fn check(&self, v: &Canonical) -> bool {
        matches!(
            (self, v),
            (FieldType::Any, _)
                | (FieldType::Int, Canonical::Int(_))
                | (FieldType::Str, Canonical::Str(_))
                | (FieldType::Bytes, Canonical::Bytes(_))
                | (FieldType::Bool, Canonical::Bool(_))
        )
    }
}

pub struct FieldSpec {
    pub name: &'static str,
    pub ty: FieldType,
}

pub struct VariantSpec {
    /// Value of the `kind` field selecting this variant ("" for the sole
    /// variant of single-variant families, which carry no `kind` field).
    pub tag: &'static str,
    pub required: &'static [&'static str],
}

pub struct FamilySchema {
    pub name: &'static str,
    pub fields: &'static [FieldSpec],
    pub variants: &'static [VariantSpec],
    /// Deferred: record schema pinned by the owning chunk when it lands;
    /// never frozen before then.
    pub deferred: bool,
}

macro_rules! deferred {
    ($name:literal) => {
        FamilySchema {
            name: $name,
            fields: &[],
            variants: &[VariantSpec { tag: "", required: &[] }],
            deferred: true,
        }
    };
}

pub const FAMILIES: &[FamilySchema] = &[
    FamilySchema {
        name: "container_encoding",
        fields: &[
            FieldSpec { name: "value", ty: FieldType::Any },
            FieldSpec { name: "encoded", ty: FieldType::Bytes },
        ],
        variants: &[VariantSpec { tag: "", required: &["value", "encoded"] }],
        deferred: false,
    },
    FamilySchema {
        name: "emission_schedule",
        fields: &[
            FieldSpec { name: "bucket", ty: FieldType::Str },
            FieldSpec { name: "kind", ty: FieldType::Str },
            FieldSpec { name: "day", ty: FieldType::Int },
            FieldSpec { name: "cumulative_nano", ty: FieldType::Int },
            FieldSpec { name: "opens_day", ty: FieldType::Int },
            FieldSpec { name: "closes_day", ty: FieldType::Int },
            FieldSpec { name: "total_nano", ty: FieldType::Int },
            FieldSpec { name: "burn_unclaimed", ty: FieldType::Bool },
        ],
        variants: &[
            VariantSpec { tag: "curve", required: &["bucket", "day", "cumulative_nano"] },
            VariantSpec {
                tag: "envelope",
                required: &["bucket", "opens_day", "closes_day", "total_nano", "burn_unclaimed"],
            },
        ],
        deferred: false,
    },
    deferred!("adam_replay"),
    deferred!("dkg_transcripts"),
    deferred!("seal_decode"),
    deferred!("witness_verify"),
    deferred!("emission_audit"),
    deferred!("fs_transcripts"),
];

pub fn schema(name: &str) -> Option<&'static FamilySchema> {
    FAMILIES.iter().find(|f| f.name == name)
}

fn descriptor(fs: &FamilySchema) -> Canonical {
    Canonical::map()
        .with("name", Canonical::str(fs.name))
        .with(
            "fields",
            Canonical::seq(
                fs.fields
                    .iter()
                    .map(|f| Canonical::seq(vec![Canonical::str(f.name), Canonical::str(f.ty.name())]))
                    .collect(),
            ),
        )
        .with(
            "variants",
            Canonical::seq(
                fs.variants
                    .iter()
                    .map(|v| {
                        Canonical::map()
                            .with("tag", Canonical::str(v.tag))
                            .with(
                                "required",
                                Canonical::seq(v.required.iter().map(|r| Canonical::str(r)).collect()),
                            )
                    })
                    .collect(),
            ),
        )
        .with("deferred", Canonical::Bool(fs.deferred))
}

pub fn schema_hash(fs: &FamilySchema) -> [u8; 32] {
    hash_domain("nerv.schema.v1", &descriptor(fs).encode())
}

pub fn check_record(fs: &FamilySchema, rec: &Canonical) -> Result<(), String> {
    if fs.deferred {
        return Err(format!("family `{}` is deferred: schema pinned by its owning chunk", fs.name));
    }
    let m: &std::collections::BTreeMap<String, Canonical> =
        rec.as_map().map_err(|e: EncError| e.to_string())?;
    let variant = if fs.variants.len() > 1 {
        let tag = m.get("kind").ok_or_else(|| "missing `kind`".to_string())?;
        let tag = tag.as_str().map_err(|e| e.to_string())?;
        fs.variants.iter().find(|v| v.tag == tag).ok_or_else(|| format!("unknown kind `{tag}`"))?
    } else {
        &fs.variants[0]
    };
    for (k, v) in m {
        match fs.fields.iter().find(|f| f.name == *k) {
            Some(f) => {
                if !f.ty.check(v) {
                    return Err(format!("field `{k}`: expected {}", f.ty.name()));
                }
            }
            None => return Err(format!("unknown field `{k}`")),
        }
    }
    for req in variant.required {
        if !m.contains_key(*req) {
            return Err(format!("missing required field `{req}`"));
        }
    }
    Ok(())
}


