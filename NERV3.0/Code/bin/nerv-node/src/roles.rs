//! Role definitions and dispatch (design doc; erratum 195).


use anyhow::{bail, Result};
use nerv_core::types::ShardId;


#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Role {
    /// A validator for a specific shard: runs the executor and signs blocks.
    Validator { shard: ShardId },
    /// A block producer for a specific shard.
    Producer { shard: ShardId },
    /// A beacon committee member.
    Committee,
    /// An archival node: full history, serves DA and witness regeneration.
    Archival,
    /// A light node: tracks the beacon only.
    Light,
}


pub fn parse_role(s: &str) -> Result<Role> {
    match s {
        "validator" => Ok(Role::Validator {
            shard: ShardId::new(6, 0).context("default shard")?,
        }),
        "producer" => Ok(Role::Producer {
            shard: ShardId::new(6, 0).context("default shard")?,
        }),
        "committee" => Ok(Role::Committee),
        "archival" => Ok(Role::Archival),
        "light" => Ok(Role::Light),
        other => bail!(
            "unknown role `{other}` (expected: validator, producer, committee, archival, light)"
        ),
    }
}


impl Role {
    /// Does this role need the full gossip engine?
    pub fn needs_gossip(&self) -> bool {
        true
    }


    /// Does this role need the DA layer?
    pub fn needs_da(&self) -> bool {
        !matches!(self, Role::Light)
    }


    /// Does this role need the executor (shard state)?
    pub fn needs_executor(&self) -> bool {
        matches!(self, Role::Validator { .. } | Role::Producer { .. } | Role::Archival)
    }


    /// Does this role need the registry (bundle verification)?
    pub fn needs_registry(&self) -> bool {
        !matches!(self, Role::Light)
    }
}
