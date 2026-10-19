//! Stage-weighted progress: maps per-stage local progress onto one monotonic
//! `[0, 1]` span, as `pulsar.runtime.utils.ProgressTracker` did.

/// A progress sink: `(stage label, cumulative fraction in [0, 1])`.
pub type ProgressFn<'a> = dyn FnMut(&str, f64) + 'a;

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
        cb(label.unwrap_or(stage), fraction);
    }

    pub fn complete(&mut self, stage: &str, label: Option<&str>) {
        self.update(stage, 1.0, label);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractions_are_monotonic_and_end_at_one() {
        let mut seen: Vec<(String, f64)> = Vec::new();
        let mut cb = |s: &str, f: f64| seen.push((s.to_string(), f));
        let stages = vec![
            ("a".to_string(), 1.0),
            ("b".to_string(), 3.0),
            ("zero".to_string(), 0.0),
        ];
        let mut t = ProgressTracker::new(&stages, Some(&mut cb));
        t.complete("a", None);
        t.update("b", 0.5, Some("b half"));
        t.update("b", 0.25, None); // goes backwards: dropped
        t.complete("b", None);
        t.complete("zero", None); // reaches 1.0 again: still reported
        drop(t);
        assert_eq!(
            seen,
            vec![
                ("a".to_string(), 0.25),
                ("b half".to_string(), 0.625),
                ("b".to_string(), 1.0),
                ("zero".to_string(), 1.0),
            ]
        );
    }
}
