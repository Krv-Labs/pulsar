//! Typed handles. Each is `<prefix>_<12 hex>`; the prefixes match the
//! Python v0.5 registry so ids stay recognisable across the port.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::StoreError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HandleKind {
    Dataset,
    Run,
    ClusterAssignment,
    Longitudinal,
    Upload,
}

impl HandleKind {
    pub fn prefix(self) -> &'static str {
        match self {
            Self::Dataset => "ds",
            Self::Run => "run",
            Self::ClusterAssignment => "ca",
            Self::Longitudinal => "lg",
            Self::Upload => "upload",
        }
    }

    /// Parameter name an agent sees, e.g. `run_id`.
    pub fn field(self) -> &'static str {
        match self {
            Self::Dataset => "dataset_id",
            Self::Run => "run_id",
            Self::ClusterAssignment => "cluster_assignment_id",
            Self::Longitudinal => "longitudinal_id",
            Self::Upload => "upload_id",
        }
    }

    /// Error code for a lookup miss.
    pub fn unknown_code(self) -> &'static str {
        match self {
            Self::Dataset => "DATASET_ID_UNKNOWN",
            Self::Run => "RUN_ID_UNKNOWN",
            Self::ClusterAssignment => "CLUSTER_ASSIGNMENT_ID_UNKNOWN",
            Self::Longitudinal => "LONGITUDINAL_ID_UNKNOWN",
            Self::Upload => "UPLOAD_ID_UNKNOWN",
        }
    }
}

impl fmt::Display for HandleKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.field())
    }
}

pub(crate) const HEX_LEN: usize = 12;

pub(crate) fn random_hex() -> String {
    let bits = rand::random::<u64>() & 0xffff_ffff_ffff;
    format!("{bits:012x}")
}

fn validate(kind: HandleKind, value: &str) -> Result<(), StoreError> {
    let invalid = || StoreError::InvalidHandle {
        kind,
        value: value.to_owned(),
    };
    let rest = value
        .strip_prefix(kind.prefix())
        .and_then(|r| r.strip_prefix('_'))
        .ok_or_else(invalid)?;
    if rest.len() != HEX_LEN || !rest.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return Err(invalid());
    }
    Ok(())
}

macro_rules! handle {
    ($(#[$meta:meta])* $name:ident => $kind:expr) => {
        $(#[$meta])*
        #[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(String);

        impl $name {
            pub const KIND: HandleKind = $kind;

            /// A fresh random handle.
            pub fn random() -> Self {
                Self(format!("{}_{}", Self::KIND.prefix(), random_hex()))
            }

            /// Validate a string an agent passed back.
            pub fn parse(value: &str) -> Result<Self, StoreError> {
                validate(Self::KIND, value)?;
                Ok(Self(value.to_owned()))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0)
            }
        }

        impl FromStr for $name {
            type Err = StoreError;
            fn from_str(s: &str) -> Result<Self, StoreError> {
                Self::parse(s)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str(&self.0)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let raw = String::deserialize(d)?;
                Self::parse(&raw).map_err(serde::de::Error::custom)
            }
        }
    };
}

handle!(
    /// A registered dataset: content-addressed by path, size and mtime.
    DatasetId => HandleKind::Dataset
);
handle!(
    /// A run. The same id names the job that produces it.
    RunId => HandleKind::Run
);
handle!(
    /// A persisted per-row cluster label vector.
    ClusterAssignmentId => HandleKind::ClusterAssignment
);
handle!(
    /// A built longitudinal panel.
    LongitudinalId => HandleKind::Longitudinal
);
handle!(
    /// An in-progress chunked upload.
    UploadId => HandleKind::Upload
);

impl DatasetId {
    /// A content-addressed id from a 12-hex digest.
    pub(crate) fn from_hex(hex: &str) -> Self {
        debug_assert_eq!(hex.len(), HEX_LEN);
        Self(format!("{}_{hex}", Self::KIND.prefix()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_validate_prefix_and_hex() {
        let id = RunId::random();
        assert_eq!(RunId::parse(id.as_str()).unwrap(), id);
        assert_eq!(id.as_str().len(), "run_".len() + HEX_LEN);

        for bad in [
            "",
            "run_",
            "run_123",
            "ds_0123456789ab",
            "run_0123456789AB",
            "run-0123456789ab",
        ] {
            let err = RunId::parse(bad).unwrap_err();
            assert_eq!(err.code(), "INVALID_HANDLE", "{bad}");
        }
        assert!(DatasetId::parse("ds_0123456789ab").is_ok());
    }

    #[test]
    fn handles_round_trip_through_serde_as_strings() {
        let id = ClusterAssignmentId::random();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, format!("\"{id}\""));
        let back: ClusterAssignmentId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, id);
        assert!(serde_json::from_str::<ClusterAssignmentId>("\"run_0123456789ab\"").is_err());
    }
}
