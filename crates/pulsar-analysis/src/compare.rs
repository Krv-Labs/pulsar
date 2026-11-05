//! Pairwise comparison of two clusters (`compare_clusters` and
//! `comparison_to_markdown` in `pulsar/mcp/interpreter.py`): for every numeric
//! column, the two cluster means, Cohen's d, a Welch t-test and a two-sample KS
//! test, ranked by effect size. Measurements only.

use pulsar_data::{Column, Table};
use serde::Serialize;

use crate::stats::{ks_two_sample_pvalue, np_mean, safe_std, welch_t_test};

/// Fewer values than this in either cluster (per column) skips the column.
const MIN_VALUES: usize = 2;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ComparisonRow {
    pub column: String,
    pub mean_a: f64,
    pub mean_b: f64,
    pub p_val_t: f64,
    pub p_val_ks: f64,
    pub cohens_d: f64,
}

fn present(values: &[f64], clusters: &[i64], id: i64) -> Vec<f64> {
    values
        .iter()
        .zip(clusters)
        .filter(|(v, &c)| c == id && !v.is_nan())
        .map(|(v, _)| *v)
        .collect()
}

/// Compare clusters `id_a` and `id_b` on every numeric column, strongest effect
/// first. Empty when either cluster has fewer than two rows.
pub fn compare_clusters(
    table: &Table,
    clusters: &[i64],
    id_a: i64,
    id_b: i64,
) -> Vec<ComparisonRow> {
    let size = |id: i64| clusters.iter().filter(|&&c| c == id).count();
    if size(id_a) < MIN_VALUES || size(id_b) < MIN_VALUES {
        return vec![];
    }
    let mut rows = vec![];
    for (name, column) in table.names.iter().zip(&table.columns) {
        let values: Vec<f64> = match column {
            Column::Float(v) => v.clone(),
            Column::Int(v) => v.iter().map(|&x| x as f64).collect(),
            _ => continue,
        };
        let (a, b) = (
            present(&values, clusters, id_a),
            present(&values, clusters, id_b),
        );
        if a.len() < MIN_VALUES || b.len() < MIN_VALUES {
            continue;
        }
        let (mean_a, mean_b) = (np_mean(&a), np_mean(&b));
        let (std_a, std_b) = (safe_std(&a), safe_std(&b));
        let (n_a, n_b) = (a.len() as f64, b.len() as f64);
        let pooled = (((n_a - 1.0) * (std_a * std_a) + (n_b - 1.0) * (std_b * std_b))
            / (n_a + n_b - 2.0))
            .sqrt();
        let (_, p_val_t) = welch_t_test(&a, &b);
        // Both samples hold at least two values here, so the test cannot fail.
        let p_val_ks = ks_two_sample_pvalue(&a, &b).unwrap_or(f64::NAN);
        rows.push(ComparisonRow {
            column: name.clone(),
            mean_a,
            mean_b,
            p_val_t,
            p_val_ks,
            cohens_d: (mean_a - mean_b) / pooled.max(1e-10),
        });
    }
    rows.sort_by(|x, y| y.cohens_d.abs().total_cmp(&x.cohens_d.abs()));
    rows
}

/// `f"{x:.{digits}f}"`: Python spells the non-finite values in lower case.
fn fixed(x: f64, digits: usize) -> String {
    if x.is_nan() {
        "nan".into()
    } else if x.is_infinite() {
        if x > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        }
    } else {
        format!("{x:.digits$}")
    }
}

/// The comparison as a Markdown table.
pub fn comparison_to_markdown(id_a: i64, id_b: i64, rows: &[ComparisonRow]) -> String {
    let mut md = vec![
        format!("# Cluster Comparison: {id_a} vs {id_b}"),
        String::new(),
        "| Feature | Mean A | Mean B | Diff | Cohen's d | T-test (p) | KS-test (p) |".into(),
        "|---|---|---|---|---|---|---|".into(),
    ];
    for r in rows {
        md.push(format!(
            "| {} | {} | {} | {} | {} | {} | {} |",
            r.column,
            fixed(r.mean_a, 3),
            fixed(r.mean_b, 3),
            fixed(r.mean_a - r.mean_b, 3),
            fixed(r.cohens_d, 2),
            fixed(r.p_val_t, 4),
            fixed(r.p_val_ks, 4),
        ));
    }
    md.join("\n")
}
