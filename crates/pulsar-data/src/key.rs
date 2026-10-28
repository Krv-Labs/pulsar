//! [`Key`]: one value of an identifier column (a longitudinal panel's entity
//! or time key), ordered and printed the way Python orders and prints it.

use std::cmp::Ordering;
use std::fmt;

use serde::{Serialize, Serializer};

use crate::preprocess::py_float_repr;

/// A scalar cell used as an identifier. Within one column every key has the
/// same variant; across variants the order is Bool < Int < Float < Text.
#[derive(Debug, Clone)]
pub enum Key {
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
}

impl Key {
    fn rank(&self) -> u8 {
        match self {
            Self::Bool(_) => 0,
            Self::Int(_) => 1,
            Self::Float(_) => 2,
            Self::Text(_) => 3,
        }
    }
}

impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Bool(a), Self::Bool(b)) => a.cmp(b),
            (Self::Int(a), Self::Int(b)) => a.cmp(b),
            (Self::Float(a), Self::Float(b)) => a.total_cmp(b),
            (Self::Text(a), Self::Text(b)) => a.cmp(b),
            _ => self.rank().cmp(&other.rank()),
        }
    }
}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Key {}

impl std::hash::Hash for Key {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.rank().hash(state);
        match self {
            Self::Bool(b) => b.hash(state),
            Self::Int(i) => i.hash(state),
            Self::Float(f) => f.to_bits().hash(state),
            Self::Text(s) => s.hash(state),
        }
    }
}

/// Python `str()` of the value: `3`, `3.0`, `True`, `abc`.
impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(true) => f.write_str("True"),
            Self::Bool(false) => f.write_str("False"),
            Self::Int(i) => write!(f, "{i}"),
            Self::Float(x) => f.write_str(&py_float_repr(*x)),
            Self::Text(s) => f.write_str(s),
        }
    }
}

/// The natural JSON scalar (a non-finite float becomes `null`).
impl Serialize for Key {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Bool(b) => s.serialize_bool(*b),
            Self::Int(i) => s.serialize_i64(*i),
            Self::Float(x) if x.is_finite() => s.serialize_f64(*x),
            Self::Float(_) => s.serialize_none(),
            Self::Text(t) => s.serialize_str(t),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_print_order_and_serialize_like_python_scalars() {
        assert_eq!(Key::Int(3).to_string(), "3");
        assert_eq!(Key::Float(3.0).to_string(), "3.0");
        assert_eq!(Key::Bool(true).to_string(), "True");
        assert_eq!(Key::Text("p1".into()).to_string(), "p1");
        let mut keys = vec![Key::Int(10), Key::Int(2), Key::Int(-1)];
        keys.sort();
        assert_eq!(keys, vec![Key::Int(-1), Key::Int(2), Key::Int(10)]);
        assert!(Key::Text("a".into()) < Key::Text("b".into()));
        assert_eq!(Key::Float(0.5), Key::Float(0.5));
        assert_eq!(serde_json::to_value(Key::Int(4)).unwrap(), 4);
        assert_eq!(serde_json::to_value(Key::Text("x".into())).unwrap(), "x");
        assert!(serde_json::to_value(Key::Float(f64::NAN))
            .unwrap()
            .is_null());
    }
}
