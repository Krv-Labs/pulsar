//! The `io.phirewall/release` v1 block: declares every group a cluster
//! payload speaks about as twin row ranges and labels every group-derived
//! count, share, statistic, category level and record list by RFC 6901 JSON
//! Pointer so the Phirewall gateway can verify disclosure centrally.
//!
//! Emitted only when the run's dataset record has `DatasetSource::Path`;
//! inline datasets have no host path for Phirewall to resolve against its
//! twin registry.
//!
//! - `capability`: whether the client asked for the block, and the dataset
//!   path a run's block names.
//! - `builder`: [`ReleaseBuilder`], its groups (twin row sets) and labels.
//! - `attach`: which payload paths of each tool get which labels.

mod attach;
mod builder;
mod capability;
#[cfg(test)]
mod tests;

pub use attach::{
    attach_compare, attach_detail, attach_dossier, attach_matrix, attach_merge, attach_profile,
    attach_signal,
};
pub use builder::{group_id, groups_of, ReleaseBuilder};
pub use capability::declared_by;

/// The structured-content key.
pub const KEY: &str = "io.phirewall/release";
