//! Vector generators. Families generated here are frozen by the registry;
//! deferred families error until their owning chunk registers a generator.

use crate::encoding::Canonical;
use crate::error::RegError;
use crate::schedule::{Curve, EmissionSchedule};
use crate::schema;
use crate::spec::LoadedSpec;

pub fn has_generator(family: &str) -> bool {
    matches!(family, "container_encoding" | "emission_schedule")
}

pub fn generate(family: &str, spec: &LoadedSpec) -> Result<Vec<Canonical>, RegError> {
    match family {
        "container_encoding" => Ok(container_encoding_records()),
        "emission_schedule" => emission_schedule_records(spec),
        other => {
            if schema::schema(other).is_some() {
                Err(RegError::MissingGenerator { family: other.to_string() })
            } else {
                Err(RegError::UnknownFamily { family: other.to_string() })
            }
        }
    }
}

fn container_encoding_records() -> Vec<Canonical> {
    let mut vals: Vec<Canonical> = Vec::new();
    for n in [0u64, 1, 2, 255, 256, 65535, 65536, u32::MAX as u64, 1 << 32, 1 << 63, u64::MAX, 123456789] {
        vals.push(Canonical::int(n));
    }
    vals.push(Canonical::Bool(true));
    vals.push(Canonical::Bool(false));
    let long = "a".repeat(300);
    for s in ["", "nerv.cm", "nerv.nf", "τ-delta-Ω", long.as_str()] {
        vals.push(Canonical::Str(s.to_string()));
    }
    vals.push(Canonical::Bytes(Vec::new()));
    vals.push(Canonical::bytes(&[0u8; 1]));
    vals.push(Canonical::bytes(&(0u8..=255).collect::<Vec<u8>>()));
    vals.push(Canonical::bytes(&(0u16..1040).map(|x| x as u8).collect::<Vec<u8>>()));
    vals.push(Canonical::seq(Vec::new()));
    vals.push(Canonical::seq(vec![Canonical::int(1), Canonical::int(2), Canonical::int(3)]));
    vals.push(Canonical::seq(vec![
        Canonical::seq(vec![Canonical::int(1)]),
        Canonical::seq(vec![Canonical::int(2), Canonical::int(3)]),
    ]));
    let mut deep = Canonical::int(7);
    for _ in 0..40 {
        deep = Canonical::seq(vec![deep]);
    }
    vals.push(deep);
    vals.push(
        Canonical::map()
            .with("alpha", Canonical::int(1))
            .with("beta", Canonical::seq(vec![Canonical::int(2)]))
            .with("gamma", Canonical::str("x")),
    );
    vals.push(Canonical::map());
    let mut big = Canonical::map();
    for i in 0..100u64 {
        big = big.with(&format!("key{i:03}"), Canonical::int(i));
    }
    vals.push(big);

    vals.into_iter()
        .map(|v| {
            let enc = v.encode();
            Canonical::map()
                .with("value", v)
                .with("encoded", Canonical::bytes(&enc))
        })
        .collect()
}

fn as_u64(v: u128) -> Result<u64, RegError> {
    u64::try_from(v).map_err(|_| RegError::Generator("value exceeds u64".to_string()))
}

fn emission_schedule_records(spec: &LoadedSpec) -> Result<Vec<Canonical>, RegError> {
    let p = &spec.spec.protocol;
    let sched = EmissionSchedule::from_economy(&spec.spec.economy, p.supply_nerv, p.nano_per_nerv)?;
    let mut out = Vec::new();
    for b in sched.buckets() {
        match &b.curve {
            Curve::Envelope { total_nano, opens_day, closes_day, burn_unclaimed } => {
                out.push(
                    Canonical::map()
                        .with("bucket", Canonical::str(&b.name))
                        .with("kind", Canonical::str("envelope"))
                        .with("opens_day", Canonical::int(*opens_day))
                        .with("closes_day", Canonical::int(*closes_day))
                        .with("total_nano", Canonical::int(as_u64(*total_nano)?))
                        .with("burn_unclaimed", Canonical::Bool(*burn_unclaimed)),
                );
            }
            curve => {
                let term = curve.term_day();
                for day in 0..=term {
                    let cum = match curve.cumulative_nano(day)? {
                        Some(c) => c,
                        None => {
                            return Err(RegError::Generator(
                                "envelope curve in the deterministic branch".to_string(),
                            ))
                        }
                    };
                    out.push(
                        Canonical::map()
                            .with("bucket", Canonical::str(&b.name))
                            .with("kind", Canonical::str("curve"))
                            .with("day", Canonical::int(day))
                            .with("cumulative_nano", Canonical::int(as_u64(cum)?)),
                    );
                }
            }
        }
    }
    Ok(out)
}
