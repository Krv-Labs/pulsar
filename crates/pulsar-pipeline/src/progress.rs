//! Stage-weighted progress: maps per-stage local progress onto one monotonic
//! `[0, 1]` span, as `pulsar.runtime.utils.ProgressTracker` did.

/// One progress tick emitted by [`ProgressTracker`].
///
/// Carries the tracker's registered stage `key` and optional 1-based `(done,
/// total)` sub-step count alongside the display `label`, so downstream sinks
/// (such as the MCP server's `io.phirewall/progress` contract builder) can map
/// stages from structure rather than parsing free-text labels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProgressEvent<'a> {
    pub key: &'a str,
    pub label: &'a str,
    pub fraction: f64,
    pub count: Option<(u64, u64)>,
}

/// A progress sink receiving structured [`ProgressEvent`] ticks.
pub type ProgressFn<'a> = dyn FnMut(ProgressEvent<'_>) + 'a;

/// Maps stage-local updates onto a single monotonic `0..1` span.
///
/// Stages are given in order with a weight each; a stage's span is its
/// share of the total weight. Fractions are rounded to six decimals and a
/// callback is only invoked when the fraction advances (or reaches 1.0).
pub struct ProgressTracker<'a, 'b> {
    spans: Vec<(String, f64, f64)>,
    last: f64,
    callback: Option<&'a mut ProgressFn<'b>>,
}

impl<'a, 'b> ProgressTracker<'a, 'b> {
    pub fn new(stages: &[(String, f64)], callback: Option<&'a mut ProgressFn<'b>>) -> Self {
        let total: f64 = stages.iter().map(|(_, w)| w.max(0.0)).sum();
        let total = if total > 0.0 { total } else { 1.0 };
        let mut cursor = 0.0;
        let spans = stages
            .iter()
            .map(|(label, weight)| {
                let start = cursor;
                cursor += weight.max(0.0) / total;
                (label.clone(), start, cursor)
            })
            .collect();
        Self {
            spans,
            last: 0.0,
            callback,
        }
    }

    /// Report `local_fraction` of `stage` done, under `label` (defaults to
    /// the stage name). Unknown stages are ignored.
    pub fn update(&mut self, stage: &str, local_fraction: f64, label: Option<&str>) {
        self.emit(stage, local_fraction, label, None);
    }

    /// Report 1-based step `done` of `total` within `stage`, under `label`.
    pub fn step(&mut self, stage: &str, done: usize, total: usize, label: &str) {
        if total == 0 {
            return;
        }
        self.emit(
            stage,
            done as f64 / total as f64,
            Some(label),
            Some((done as u64, total as u64)),
        );
    }

    pub fn complete(&mut self, stage: &str, label: Option<&str>) {
        self.update(stage, 1.0, label);
    }

    fn emit(
        &mut self,
        stage: &str,
        local_fraction: f64,
        label: Option<&str>,
        count: Option<(u64, u64)>,
    ) {
        let Some(cb) = self.callback.as_mut() else {
            return;
        };
        let Some(&(_, start, end)) = self.spans.iter().find(|(s, _, _)| s == stage) else {
            return;
        };
        let local = local_fraction.clamp(0.0, 1.0);
        let fraction = ((start + (end - start) * local) * 1e6).round() / 1e6;
        if fraction <= self.last && fraction < 1.0 {
            return;
        }
        self.last = fraction;
        cb(ProgressEvent {
            key: stage,
            label: label.unwrap_or(stage),
            fraction,
            count,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractions_are_monotonic_and_end_at_one() {
        let mut seen = Vec::new();
        let mut cb = |ev: ProgressEvent<'_>| {
            seen.push((
                ev.key.to_string(),
                ev.label.to_string(),
                ev.fraction,
                ev.count,
            ))
        };
        let stages = vec![
            ("a".to_string(), 1.0),
            ("b".to_string(), 3.0),
            ("zero".to_string(), 0.0),
        ];
        let mut t = ProgressTracker::new(&stages, Some(&mut cb));
        t.complete("a", None);
        t.step("b", 1, 2, "b half");
        t.update("b", 0.25, None); // goes backwards: dropped
        t.complete("b", None);
        t.complete("zero", None); // reaches 1.0 again: still reported
        drop(t);
        assert_eq!(
            seen,
            vec![
                ("a".to_string(), "a".to_string(), 0.25, None),
                ("b".to_string(), "b half".to_string(), 0.625, Some((1, 2))),
                ("b".to_string(), "b".to_string(), 1.0, None),
                ("zero".to_string(), "zero".to_string(), 1.0, None),
            ]
        );
    }
}
