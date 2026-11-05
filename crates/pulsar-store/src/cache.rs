//! A bounded, process-wide LRU of live models keyed by run id.
//!
//! The store persists what a run *produced*; a fitted model (ball maps,
//! embeddings, the sparse accumulator) stays in memory only while it is
//! recently used. Capacity comes from `PULSAR_MCP_MODEL_CACHE_SIZE` (default
//! 3, minimum 1), as in v0.5. Unlike the Python cache this one is locked, so
//! worker threads can read it directly.

use std::sync::{Arc, Mutex};

use crate::handle::RunId;

pub const ENV_CAPACITY: &str = "PULSAR_MCP_MODEL_CACHE_SIZE";
const DEFAULT_CAPACITY: usize = 3;

#[derive(Debug)]
pub struct ModelCache<T> {
    capacity: usize,
    /// Least recently used first.
    entries: Mutex<Vec<(RunId, Arc<T>)>>,
}

impl<T> ModelCache<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: Mutex::new(Vec::new()),
        }
    }

    /// Capacity from the environment.
    pub fn from_env() -> Self {
        let capacity = std::env::var(ENV_CAPACITY)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_CAPACITY);
        Self::new(capacity)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<(RunId, Arc<T>)>> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Insert or refresh; evicts the least recently used entries beyond
    /// capacity and returns them.
    pub fn put(&self, run_id: RunId, model: Arc<T>) -> Vec<(RunId, Arc<T>)> {
        let mut entries = self.lock();
        entries.retain(|(id, _)| *id != run_id);
        entries.push((run_id, model));
        let overflow = entries.len().saturating_sub(self.capacity);
        entries.drain(..overflow).collect()
    }

    /// Most recently used entry matching `pred`; does not change the order.
    pub fn find(&self, pred: impl Fn(&T) -> bool) -> Option<(RunId, Arc<T>)> {
        self.lock()
            .iter()
            .rev()
            .find(|(_, model)| pred(model))
            .cloned()
    }

    /// Fetch and mark as most recently used.
    pub fn get(&self, run_id: &RunId) -> Option<Arc<T>> {
        let mut entries = self.lock();
        let pos = entries.iter().position(|(id, _)| id == run_id)?;
        let entry = entries.remove(pos);
        let model = entry.1.clone();
        entries.push(entry);
        Some(model)
    }

    pub fn contains(&self, run_id: &RunId) -> bool {
        self.lock().iter().any(|(id, _)| id == run_id)
    }

    pub fn evict(&self, run_id: &RunId) -> Option<Arc<T>> {
        let mut entries = self.lock();
        let pos = entries.iter().position(|(id, _)| id == run_id)?;
        Some(entries.remove(pos).1)
    }

    pub fn clear(&self) {
        self.lock().clear();
    }

    pub fn len(&self) -> usize {
        self.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// Run ids, least to most recently used.
    pub fn run_ids(&self) -> Vec<RunId> {
        self.lock().iter().map(|(id, _)| id.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_evicts_least_recently_used() {
        let cache = ModelCache::<String>::new(2);
        let (a, b, c) = (RunId::random(), RunId::random(), RunId::random());
        assert!(cache.put(a.clone(), Arc::new("a".into())).is_empty());
        assert!(cache.put(b.clone(), Arc::new("b".into())).is_empty());
        assert_eq!(cache.get(&a).as_deref(), Some(&"a".to_string()));
        let evicted = cache.put(c.clone(), Arc::new("c".into()));
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].0, b);
        assert_eq!(cache.run_ids(), vec![a.clone(), c.clone()]);
        assert!(cache.get(&b).is_none());
        assert!(cache.evict(&a).is_some());
        assert_eq!(cache.len(), 1);
        assert_eq!(ModelCache::<()>::new(0).capacity(), 1);
    }

    #[test]
    fn find_does_not_reorder() {
        let cache = ModelCache::<String>::new(3);
        let (a, b) = (RunId::random(), RunId::random());
        cache.put(a.clone(), Arc::new("a".into()));
        cache.put(b.clone(), Arc::new("b".into()));
        assert!(cache.find(|m| m == "x").is_none());
        assert_eq!(cache.find(|m| m == "a").map(|(id, _)| id), Some(a.clone()));
        assert_eq!(cache.run_ids(), vec![a, b]);
    }
}
