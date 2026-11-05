//! Merge the clusters under `k` patients into their nearest group, by the
//! full edge weights of the graph (not centroid distance).
//!
//! Pure and deterministic. A cluster's *nearest* neighbour is the one with
//! the most total edge weight between them; the weight is summed over every
//! cross-cluster edge, so it is the strength of the actual connection.
//!
//! Rules, in order:
//! 1. The cluster with the fewest patients under `k` goes first (ties: lowest
//!    label).
//! 2. It merges into the neighbour with the largest total edge weight. Ties
//!    go to the larger weight per patient pair (`W / (|a| * |b|)` in distinct
//!    patients, by `total_cmp`), then to the lowest label.
//! 3. A cluster with no edge to any other cluster merges into the largest one
//!    (most patients, then most rows, then lowest label), reported with
//!    [`Linkage::Isolated`].
//! 4. A merged cluster can still be under `k`; it is merged again (chains).
//! 5. When anything merges, the result is renumbered `0..` by each group's
//!    lowest member row. When nothing merges, the stored labels are returned
//!    unchanged, so any sizes reported under them match the assignment.
//!
//! Patients of the union are not the sum of the parts when a patient sits in
//! both, so [`merge_small_by`] asks the caller for the distinct patients of
//! any set of original clusters; [`merge_small`] takes per-cluster support
//! and adds it up, which is exact when no patient spans clusters.

use std::collections::BTreeMap;

use serde::Serialize;

/// How a cluster found the group it merged into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Linkage {
    /// The neighbour with the most edge weight to the cluster.
    EdgeWeight,
    /// The cluster had no edge to any other; it went into the largest one.
    Isolated,
}

/// Row count per label, ascending by label.
pub fn label_sizes(labels: &[i64]) -> BTreeMap<i64, usize> {
    let mut sizes: BTreeMap<i64, usize> = BTreeMap::new();
    for &l in labels {
        *sizes.entry(l).or_default() += 1;
    }
    sizes
}

/// One merge, in the original labels.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Merge {
    /// The original label of the cluster that was absorbed.
    pub from: i64,
    /// The original label of the cluster it was merged into, as that cluster
    /// stood then (it may itself have absorbed others, or be absorbed later).
    pub into: i64,
    /// The new label of the group the merged cluster ended up in.
    pub group: i64,
    pub linkage: Linkage,
    /// The share of `from`'s total edge weight to other clusters that went
    /// to `into`; 0 when isolated.
    pub weight_share: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Merged {
    /// One new label per row, numbered by lowest member row.
    pub labels: Vec<i64>,
    pub merges: Vec<Merge>,
    /// Each new label with the original labels it holds, ascending.
    pub groups: BTreeMap<i64, Vec<i64>>,
    /// Clusters still under `k` after every merge: only when everything has
    /// been merged into one group that is itself under `k`.
    pub unresolved: bool,
}

/// [`merge_small_by`] with the support of a union taken as the sum.
pub fn merge_small(
    labels: &[i64],
    support: &BTreeMap<i64, usize>,
    edges: &[(usize, usize, f64)],
    k: usize,
) -> Merged {
    merge_small_by(
        labels,
        &|members: &[i64]| {
            members
                .iter()
                .map(|l| support.get(l).copied().unwrap_or(0))
                .sum()
        },
        edges,
        k,
    )
}

struct Step {
    from: i64,
    into: i64,
    linkage: Linkage,
    weight_share: f64,
}

/// Select the cluster `a` merges into, its linkage and its weight share.
fn pick(
    a: i64,
    neighbours: &BTreeMap<i64, f64>,
    support: &BTreeMap<i64, usize>,
    rows: &BTreeMap<i64, usize>,
) -> (i64, Linkage, f64) {
    let total: f64 = neighbours.values().sum();
    let best = neighbours
        .iter()
        .max_by(|(&b1, &w1), (&b2, &w2)| {
            // A cluster whose patients are all blank counts zero; its density
            // is undefined, so it never wins a density tie-break.
            let per_patient = |b: i64, x: f64| {
                let denominator = support[&a] as f64 * support[&b] as f64;
                if denominator > 0.0 {
                    x / denominator
                } else {
                    0.0
                }
            };
            w1.total_cmp(&w2)
                .then_with(|| per_patient(b1, w1).total_cmp(&per_patient(b2, w2)))
                .then_with(|| b2.cmp(&b1))
        })
        .map(|(&b, &x)| (b, x));
    match best {
        Some((b, x)) => (b, Linkage::EdgeWeight, x / total),
        None => {
            let largest = support
                .keys()
                .copied()
                .filter(|&l| l != a)
                .max_by_key(|&l| (support[&l], rows[&l], std::cmp::Reverse(l)))
                .expect("another cluster exists");
            (largest, Linkage::Isolated, 0.0)
        }
    }
}

/// Fold cluster `a` (its edges, members, rows and support) into `into`.
#[allow(clippy::too_many_arguments)]
fn merge(
    a: i64,
    into: i64,
    neighbours: BTreeMap<i64, f64>,
    w: &mut BTreeMap<i64, BTreeMap<i64, f64>>,
    members: &mut BTreeMap<i64, Vec<i64>>,
    rows: &mut BTreeMap<i64, usize>,
    support: &mut BTreeMap<i64, usize>,
    support_of: &dyn Fn(&[i64]) -> usize,
) {
    for (c, x) in neighbours {
        if c == into {
            continue;
        }
        *w.entry(into).or_default().entry(c).or_default() += x;
        let back = w.entry(c).or_default();
        *back.entry(into).or_default() += x;
        back.remove(&a);
    }
    if let Some(edges) = w.get_mut(&into) {
        edges.remove(&a);
    }
    let absorbed = members.remove(&a).expect("a live cluster");
    let target = members.get_mut(&into).expect("a live cluster");
    target.extend(absorbed);
    target.sort_unstable();
    *rows.get_mut(&into).expect("a live cluster") += rows[&a];
    rows.remove(&a);
    support.remove(&a);
    support.insert(into, support_of(&members[&into]));
}

/// Renumber live groups `0..` by lowest member row, or keep `labels` unchanged
/// when nothing merged.
fn renumber(
    labels: &[i64],
    members: &BTreeMap<i64, Vec<i64>>,
    steps: Vec<Step>,
    unresolved: bool,
) -> Merged {
    // Nothing merged: the stored labels are the answer. Renumbering them
    // would report sizes under ids the returned assignment does not use.
    if steps.is_empty() {
        let groups = members.keys().map(|&l| (l, vec![l])).collect();
        return Merged {
            labels: labels.to_vec(),
            merges: Vec::new(),
            groups,
            unresolved,
        };
    }

    let mut first_row: BTreeMap<i64, usize> = BTreeMap::new();
    let mut owner: BTreeMap<i64, i64> = BTreeMap::new();
    for (&survivor, originals) in members {
        for &o in originals {
            owner.insert(o, survivor);
        }
    }
    for (i, l) in labels.iter().enumerate() {
        first_row.entry(owner[l]).or_insert(i);
    }
    let mut order: Vec<(usize, i64)> = first_row.iter().map(|(&s, &r)| (r, s)).collect();
    order.sort_unstable();
    let new_label: BTreeMap<i64, i64> = order
        .iter()
        .enumerate()
        .map(|(n, &(_, s))| (s, n as i64))
        .collect();
    let groups = members
        .iter()
        .map(|(s, originals)| (new_label[s], originals.clone()))
        .collect();
    // A merge's `into` may have been absorbed since: find its final group.
    let merges = steps
        .into_iter()
        .map(|s| Merge {
            from: s.from,
            into: s.into,
            group: new_label[&owner[&s.into]],
            linkage: s.linkage,
            weight_share: s.weight_share,
        })
        .collect();
    Merged {
        labels: labels.iter().map(|l| new_label[&owner[l]]).collect(),
        merges,
        groups,
        unresolved,
    }
}

/// Merge every cluster with fewer than `k` patients. `support_of(originals)`
/// is the distinct patients across the clusters with those original labels.
/// Edges naming a row outside `labels`, and non-positive or non-finite
/// weights, are ignored.
pub fn merge_small_by(
    labels: &[i64],
    support_of: &dyn Fn(&[i64]) -> usize,
    edges: &[(usize, usize, f64)],
    k: usize,
) -> Merged {
    // Live clusters, keyed by the original label that survives a merge.
    let mut rows = label_sizes(labels);
    let mut members: BTreeMap<i64, Vec<i64>> = rows.keys().map(|&l| (l, vec![l])).collect();
    let mut support: BTreeMap<i64, usize> =
        members.iter().map(|(&l, m)| (l, support_of(m))).collect();
    if support.values().all(|&s| s >= k) {
        return renumber(labels, &members, Vec::new(), false);
    }
    if members.len() < 2 {
        return renumber(labels, &members, Vec::new(), true);
    }
    // W[a][b]: total edge weight between two live clusters. Sum in a
    // canonical order (row pair, then weight) so the float total does not
    // depend on the order the caller listed the edges in.
    let mut canonical: Vec<(usize, usize, f64)> = edges
        .iter()
        .filter_map(|&(i, j, x)| {
            let (Some(&a), Some(&b)) = (labels.get(i), labels.get(j)) else {
                return None;
            };
            if a == b || !x.is_finite() || x <= 0.0 {
                return None;
            }
            Some((i.min(j), i.max(j), x))
        })
        .collect();
    let edge_cmp = |(i1, j1, x1): &(usize, usize, f64), (i2, j2, x2): &(usize, usize, f64)| {
        (i1, j1).cmp(&(i2, j2)).then_with(|| x1.total_cmp(x2))
    };
    if !canonical.is_sorted_by(|a, b| edge_cmp(a, b).is_le()) {
        canonical.sort_by(edge_cmp);
    }
    let mut w: BTreeMap<i64, BTreeMap<i64, f64>> = BTreeMap::new();
    for (i, j, x) in canonical {
        let (a, b) = (labels[i], labels[j]);
        *w.entry(a).or_default().entry(b).or_default() += x;
        *w.entry(b).or_default().entry(a).or_default() += x;
    }

    let mut steps: Vec<Step> = Vec::new();
    let mut unresolved = false;
    while let Some((&a, _)) = support
        .iter()
        .filter(|(_, &s)| s < k)
        .min_by_key(|(&l, &s)| (s, l))
    {
        if members.len() < 2 {
            unresolved = true;
            break;
        }
        let neighbours = w.remove(&a).unwrap_or_default();
        let (into, linkage, weight_share) = pick(a, &neighbours, &support, &rows);
        merge(
            a,
            into,
            neighbours,
            &mut w,
            &mut members,
            &mut rows,
            &mut support,
            support_of,
        );
        steps.push(Step {
            from: a,
            into,
            linkage,
            weight_share,
        });
    }

    renumber(labels, &members, steps, unresolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sup(pairs: &[(i64, usize)]) -> BTreeMap<i64, usize> {
        pairs.iter().copied().collect()
    }

    /// `n` rows of each label in turn.
    fn labels(counts: &[(i64, usize)]) -> Vec<i64> {
        counts
            .iter()
            .flat_map(|&(l, n)| std::iter::repeat_n(l, n))
            .collect()
    }

    #[test]
    fn all_large_is_returned_unchanged() {
        let l = vec![5, 5, 2, 2, 5, 2];
        let m = merge_small(&l, &sup(&[(2, 8), (5, 9)]), &[(0, 2, 1.0)], 8);
        assert!(m.merges.is_empty() && !m.unresolved);
        assert_eq!(m.labels, l);
        assert_eq!(m.groups[&5], vec![5]);
        assert_eq!(m.groups[&2], vec![2]);
    }

    #[test]
    fn a_no_op_keeps_the_stored_labels_and_their_ids() {
        // Spectral/threshold-stability order need not start with label 0.
        // Renumbering here would report sizes under ids the returned
        // assignment does not use.
        let l = labels(&[(1, 8), (0, 8)]);
        let m = merge_small(&l, &sup(&[(0, 8), (1, 8)]), &[(0, 8, 1.0)], 8);
        assert!(m.merges.is_empty() && !m.unresolved);
        assert_eq!(m.labels, l);
        assert_eq!(m.groups[&0], vec![0]);
        assert_eq!(m.groups[&1], vec![1]);
    }

    #[test]
    fn a_small_cluster_goes_to_the_neighbour_with_the_most_weight() {
        // 0: rows 0-7, 1: rows 8-15, 2: rows 16-17 (small).
        let l = labels(&[(0, 8), (1, 8), (2, 2)]);
        let edges = [(16, 0, 0.5), (17, 8, 0.9), (16, 9, 0.6)];
        let m = merge_small(&l, &sup(&[(0, 8), (1, 8), (2, 2)]), &edges, 8);
        assert_eq!(m.merges.len(), 1);
        let x = &m.merges[0];
        assert_eq!((x.from, x.into, x.linkage), (2, 1, Linkage::EdgeWeight));
        assert!((x.weight_share - 1.5 / 2.0).abs() < 1e-12);
        assert_eq!(m.labels[16], m.labels[8]);
        assert_eq!(x.group, m.labels[8]);
    }

    #[test]
    fn full_weights_beat_a_single_strong_edge() {
        let l = labels(&[(0, 8), (1, 8), (2, 2)]);
        // One 0.9 edge to cluster 0, but three 0.4 edges to cluster 1.
        let edges = [(16, 0, 0.9), (16, 8, 0.4), (17, 9, 0.4), (17, 10, 0.4)];
        let m = merge_small(&l, &sup(&[(0, 8), (1, 8), (2, 2)]), &edges, 8);
        assert_eq!(m.merges[0].into, 1);
    }

    #[test]
    fn a_weight_tie_goes_to_the_denser_pair_then_the_lowest_label() {
        // Equal total weight; cluster 1 is smaller, so W/(|a||b|) is larger.
        let l = labels(&[(0, 12), (1, 8), (2, 2)]);
        let edges = [(20, 0, 0.5), (21, 12, 0.5)];
        let m = merge_small(&l, &sup(&[(0, 12), (1, 8), (2, 2)]), &edges, 8);
        assert_eq!(m.merges[0].into, 1);
        // Fully tied: the lowest label.
        let l = labels(&[(0, 8), (1, 8), (2, 2)]);
        let edges = [(16, 0, 0.5), (17, 8, 0.5)];
        let m = merge_small(&l, &sup(&[(0, 8), (1, 8), (2, 2)]), &edges, 8);
        assert_eq!(m.merges[0].into, 0);
    }

    #[test]
    fn a_weight_tie_is_broken_by_patient_density_not_rows() {
        // Cluster 2 has 2 patients in 10 rows; both neighbours have 10
        // patients, cluster 1 in 10 rows and cluster 0 in 30. Per row cluster
        // 1 looks denser; per patient the two tie, so the lowest label wins.
        let l = labels(&[(0, 30), (1, 10), (2, 10)]);
        let s = sup(&[(0, 10), (1, 10), (2, 2)]);
        let edges = [(40, 30, 0.5), (41, 0, 0.5)];
        let m = merge_small(&l, &s, &edges, 8);
        assert_eq!(m.merges[0].from, 2);
        assert_eq!(m.merges[0].into, 0);
    }

    #[test]
    fn the_smallest_goes_first_and_ties_go_to_the_lowest_label() {
        let l = labels(&[(0, 8), (1, 3), (2, 3)]);
        let edges = [(8, 11, 0.5), (8, 0, 0.1), (11, 0, 0.2)];
        let m = merge_small(&l, &sup(&[(0, 8), (1, 3), (2, 3)]), &edges, 8);
        // 1 and 2 tie on support: 1 first, into 2 (0.5 > 0.1); the union (6)
        // is still small, so it goes on into 0.
        let steps: Vec<_> = m.merges.iter().map(|x| (x.from, x.into)).collect();
        assert_eq!(steps, vec![(1, 2), (2, 0)]);
        assert!(m.labels.iter().all(|&l| l == 0) && !m.unresolved);
        assert_eq!(m.groups[&0], vec![0, 1, 2]);
        assert!(m.merges.iter().all(|x| x.group == 0));
    }

    #[test]
    fn chains_use_the_folded_weights() {
        // 2 (small) -> 1 (small, nearest), then the union's edges decide.
        let l = labels(&[(0, 8), (1, 4), (2, 3), (3, 8)]);
        let edges = [
            (12, 8, 0.9),  // 2-1
            (12, 0, 0.1),  // 2-0
            (8, 20, 0.3),  // 1-3
            (12, 21, 0.3), // 2-3
            (8, 0, 0.2),   // 1-0
        ];
        let s = sup(&[(0, 8), (1, 4), (2, 3), (3, 8)]);
        let m = merge_small(&l, &s, &edges, 8);
        // 2 -> 1; the union 7 is small: 0 gets 0.1+0.2, 3 gets 0.3+0.3.
        let steps: Vec<_> = m.merges.iter().map(|x| (x.from, x.into)).collect();
        assert_eq!(steps, vec![(2, 1), (1, 3)]);
        assert_eq!(m.groups.len(), 2);
    }

    #[test]
    fn an_isolated_cluster_goes_into_the_largest() {
        let l = labels(&[(0, 8), (1, 20), (2, 2)]);
        let m = merge_small(&l, &sup(&[(0, 8), (1, 20), (2, 2)]), &[(0, 8, 1.0)], 8);
        let x = &m.merges[0];
        assert_eq!((x.from, x.into, x.linkage), (2, 1, Linkage::Isolated));
        assert_eq!(x.weight_share, 0.0);
    }

    #[test]
    fn bad_edges_are_ignored() {
        let l = labels(&[(0, 8), (1, 2)]);
        let edges = [(8, 0, f64::NAN), (8, 1, -1.0), (8, 999, 1.0), (8, 0, 0.0)];
        let m = merge_small(&l, &sup(&[(0, 8), (1, 2)]), &edges, 8);
        assert_eq!(m.merges[0].linkage, Linkage::Isolated);
    }

    #[test]
    fn a_dataset_under_k_is_reported_unresolved() {
        let l = labels(&[(0, 3), (1, 2)]);
        let m = merge_small(&l, &sup(&[(0, 3), (1, 2)]), &[(0, 3, 1.0)], 8);
        assert!(m.unresolved);
        assert!(m.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn a_patient_in_two_clusters_counts_once_through_patients() {
        use crate::disclosure::{Patients, Unit};
        use pulsar_data::{Column, Table};

        // Rows 8..12 are all the same patient, split over clusters 1 and 2;
        // rows 0..8 name eight other patients.
        let ids: Vec<Option<String>> = (0..12)
            .map(|i| {
                Some(if i < 8 {
                    format!("p{i}")
                } else {
                    "shared".into()
                })
            })
            .collect();
        let table = Table::new(
            vec!["pid".into(), "x".into()],
            vec![
                Column::Text(ids),
                Column::Float((0..12).map(|i| i as f64).collect()),
            ],
        )
        .unwrap();
        let patients = Patients::of(&table, &["pid".to_owned()]);
        assert_eq!(patients.unit(), Unit::Patients);
        assert_eq!(patients.total(), 9);

        let l = labels(&[(0, 8), (1, 2), (2, 2)]);
        let support_of = |members: &[i64]| {
            patients.count(
                l.iter()
                    .enumerate()
                    .filter(|&(_, x)| members.contains(x))
                    .map(|(i, _)| i),
            )
        };
        // Summed, 1 + 1 reads as 2 patients and stops; counted, it is 1.
        assert_eq!(support_of(&[1, 2]), 1);
        let edges = [(8, 10, 1.0), (10, 0, 0.1)];
        let m = merge_small_by(&l, &support_of, &edges, 2);
        let steps: Vec<_> = m.merges.iter().map(|x| (x.from, x.into)).collect();
        assert_eq!(steps, vec![(1, 2), (2, 0)]);
    }

    #[test]
    fn it_is_deterministic() {
        let l = labels(&[(0, 9), (1, 3), (2, 3), (3, 2), (4, 10)]);
        let edges: Vec<_> = (0..l.len())
            .flat_map(|i| (0..i).map(move |j| (i, j, ((i * 7 + j * 3) % 11) as f64 / 10.0)))
            .collect();
        let s = sup(&[(0, 9), (1, 3), (2, 3), (3, 2), (4, 10)]);
        let a = merge_small(&l, &s, &edges, 8);
        let mut rev = edges.clone();
        rev.reverse();
        let b = merge_small(&l, &s, &rev, 8);
        assert_eq!(a, b);
    }

    #[test]
    fn edge_order_does_not_decide_between_float_sums() {
        // 0.1 + 0.2 + 0.3 is not 0.6 in binary, and summing in list order
        // gives a different total (and weight share) from the reversed list.
        // The canonical order must make both runs identical.
        let l = labels(&[(0, 8), (1, 8), (2, 2)]);
        let s = sup(&[(0, 8), (1, 8), (2, 2)]);
        let edges = [(16, 0, 0.1), (16, 1, 0.2), (16, 2, 0.3), (17, 8, 0.6)];
        let a = merge_small(&l, &s, &edges, 8);
        let mut rev = edges;
        rev.reverse();
        let b = merge_small(&l, &s, &rev, 8);
        assert_eq!(a, b);
    }
}
