//! Rayon thread control (replaces the `RAYON_NUM_THREADS` override in
//! `pulsar.runtime.utils`).

/// Size the global rayon pool. Must run before any parallel kernel; a later
/// call fails because the global pool is already built. `None` keeps the
/// default (`RAYON_NUM_THREADS` or the number of logical cores).
pub fn configure_threads(workers: Option<usize>) -> Result<(), rayon::ThreadPoolBuildError> {
    match workers {
        Some(n) => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global(),
        None => Ok(()),
    }
}

/// Threads the global pool will use.
pub fn current_threads() -> usize {
    rayon::current_num_threads()
}
