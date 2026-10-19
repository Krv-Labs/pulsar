//! Bounded collection payloads (`pulsar/mcp/payloads.py`): summaries never
//! inline an O(n) list.

use serde_json::{json, Value};

/// `{total, preview, omitted}`.
pub fn bounded_list<T: serde::Serialize>(items: &[T], preview_limit: usize) -> Value {
    let preview: Vec<&T> = items.iter().take(preview_limit).collect();
    json!({
        "total": items.len(),
        "preview": preview,
        "omitted": items.len().saturating_sub(preview.len()),
    })
}

/// Bounded size telemetry: the [`bounded_list`] core (sorted descending)
/// plus the largest five, the singleton count and size buckets.
pub fn size_summary(sizes: &[usize], preview_limit: usize) -> Value {
    let mut values: Vec<usize> = sizes.iter().copied().filter(|&s| s > 0).collect();
    values.sort_unstable_by(|a, b| b.cmp(a));
    let singleton_count = values.iter().filter(|&&s| s == 1).count();
    let mut summary = bounded_list(&values, preview_limit);
    let obj = summary.as_object_mut().expect("object");
    obj.insert(
        "largest".into(),
        json!(values.iter().take(5).collect::<Vec<_>>()),
    );
    obj.insert("singleton_count".into(), json!(singleton_count));
    obj.insert(
        "size_buckets".into(),
        json!({
            "1": singleton_count,
            "2-5": values.iter().filter(|&&s| (2..=5).contains(&s)).count(),
            "6-20": values.iter().filter(|&&s| (6..=20).contains(&s)).count(),
            ">20": values.iter().filter(|&&s| s > 20).count(),
        }),
    );
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_summary_is_bounded_and_bucketed() {
        let sizes = [1, 30, 7, 2, 1, 0];
        let s = size_summary(&sizes, 3);
        assert_eq!(s["total"], 5);
        assert_eq!(s["preview"], json!([30, 7, 2]));
        assert_eq!(s["omitted"], 2);
        assert_eq!(s["largest"], json!([30, 7, 2, 1, 1]));
        assert_eq!(s["singleton_count"], 2);
        assert_eq!(
            s["size_buckets"],
            json!({"1": 2, "2-5": 1, "6-20": 1, ">20": 1})
        );
    }
}
