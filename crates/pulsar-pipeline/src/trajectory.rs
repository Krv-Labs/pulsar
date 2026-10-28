//! `CosmicTrajectory` (`pulsar/representations/trajectory.py`): the
//! observation-centric cosmic representation of a longitudinal panel.
//!
//! [`pulsar_core::TemporalCosmicGraph`] has one node per *entity* and every
//! edge inside one time slice. Here a node is an *observation* `(entity, t)`
//! and every observation is pooled into one geometry, so Ball Mapper covers,
//! and the CO_SIMILAR edges derived from them, span time by construction.
//!
//! Storage is sparse: the observation list, the ball list (scale, dimension,
//! size, members) and the upper-triangle CO_SIMILAR edges. Nothing is `n x n`.
//!
//! Known divergences from Python:
//! - `within_time`, `trajectory_edges`, `observation_index` and `to_networkx`
//!   are not ported: no tool reads them.
//! - Python's "pass entity_ids or snapshot_entity_ids, not both" and
//!   "snapshot must be 2-D" checks are unrepresentable here ([`EntityIds`],
//!   `Array2`).
//! - Every ball's scope is `"global"`, so it is not stored per ball.

use ndarray::{concatenate, Array2, ArrayView2, Axis};
use pulsar_config::PulsarConfig;
use pulsar_core::{ball_mapper_grid, StandardScalerInner};
use pulsar_data::Key;
use serde::Serialize;
use serde_json::{json, Value};

use crate::error::PipelineError;
use crate::fit::{check_epsilons, projection_grid, CosmicBuilder};
use crate::graph::Graph;

/// One node: entity `entity_id` observed at time step `t`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Observation {
    pub entity_id: Key,
    pub t: usize,
    /// The time step's label when the panel has one.
    pub timestamp: Option<Key>,
}

/// One cover element: a ball at radius `eps` in a `dim`-dimensional projection.
#[derive(Debug, Clone, PartialEq)]
pub struct Ball {
    pub eps: f64,
    pub dim: usize,
    /// Observation ids, ascending.
    pub members: Vec<usize>,
}

/// How snapshot rows map to entities.
#[derive(Debug, Clone)]
pub enum EntityIds {
    /// Row `i` of every snapshot is entity `i`.
    Positional,
    /// One id sequence shared by every snapshot.
    Shared(Vec<Key>),
    /// One id sequence per snapshot, for ragged panels.
    PerSnapshot(Vec<Vec<Key>>),
}

/// Observation-centric cosmic graph of a longitudinal panel.
#[derive(Debug, Clone)]
pub struct CosmicTrajectory {
    /// Indexed by `obs_id` (the stacked row position).
    pub obs: Vec<Observation>,
    /// Indexed by `ball_id`, in construction order.
    pub balls: Vec<Ball>,
    /// CO_SIMILAR: upper-triangle `(i, j, w)` edges, `w` in `(0, 1]`.
    pub similarity: Graph,
    /// Construction provenance (Python's `meta` dict).
    pub meta: Value,
}

fn input_error(msg: String) -> PipelineError {
    PipelineError::TrajectoryInput(msg)
}

/// Python's `str(list)` of integers.
fn py_int_list(items: &[usize]) -> String {
    let inner: Vec<String> = items.iter().map(usize::to_string).collect();
    format!("[{}]", inner.join(", "))
}

/// `_stack_panel`: the pooled `(N, F)` matrix and its observations.
fn stack_panel(
    snapshots: &[Array2<f64>],
    entity_ids: &EntityIds,
    timestamps: Option<&[Key]>,
) -> Result<(Array2<f64>, Vec<Observation>), PipelineError> {
    if snapshots.is_empty() {
        return Err(input_error("snapshots must be non-empty".into()));
    }
    let mut widths: Vec<usize> = snapshots.iter().map(Array2::ncols).collect();
    widths.sort_unstable();
    widths.dedup();
    if widths.len() != 1 {
        return Err(input_error(format!(
            "All snapshots must share a feature count; got {}",
            py_int_list(&widths)
        )));
    }
    let counts: Vec<usize> = snapshots.iter().map(Array2::nrows).collect();
    if counts.iter().sum::<usize>() == 0 {
        return Err(input_error("snapshots contain no rows".into()));
    }
    let per_snapshot: Vec<Vec<Key>> = match entity_ids {
        EntityIds::PerSnapshot(rows) => {
            if rows.len() != snapshots.len() {
                return Err(input_error(format!(
                    "snapshot_entity_ids has {} time slices but there are {} snapshots",
                    rows.len(),
                    snapshots.len()
                )));
            }
            rows.clone()
        }
        EntityIds::Shared(ids) => vec![ids.clone(); snapshots.len()],
        EntityIds::Positional => counts
            .iter()
            .map(|&c| (0..c as i64).map(Key::Int).collect())
            .collect(),
    };
    let bad: Vec<usize> = per_snapshot
        .iter()
        .zip(&counts)
        .enumerate()
        .filter(|(_, (row, &count))| row.len() != count)
        .map(|(t, _)| t)
        .collect();
    if !bad.is_empty() {
        return Err(input_error(format!(
            "Entity IDs must match snapshot row counts; mismatch at snapshots {}",
            py_int_list(&bad)
        )));
    }
    if let Some(stamps) = timestamps {
        if stamps.len() != snapshots.len() {
            return Err(input_error(format!(
                "timestamps has {} entries but there are {} snapshots",
                stamps.len(),
                snapshots.len()
            )));
        }
    }
    let obs = per_snapshot
        .into_iter()
        .enumerate()
        .flat_map(|(t, row)| {
            row.into_iter().map(move |entity_id| Observation {
                entity_id,
                t,
                timestamp: timestamps.map(|s| s[t].clone()),
            })
        })
        .collect();
    let views: Vec<ArrayView2<f64>> = snapshots.iter().map(Array2::view).collect();
    let pooled = concatenate(Axis(0), &views).map_err(|e| input_error(e.to_string()))?;
    Ok((pooled, obs))
}

impl CosmicTrajectory {
    /// Build the trajectory graph from per-time `(n_t, F)` snapshots sharing a
    /// feature schema (row counts may differ across `t`).
    ///
    /// `config` supplies the projection grid, the Ball Mapper epsilons and the
    /// cosmic construction (MinHash or exact). CO_SIMILAR edges at or below
    /// `similarity_threshold` are dropped. `scale` fits one standard scaler
    /// on the *pooled* matrix: scaling per time step would re-centre each
    /// slice and erase the drift cross-time edges exist to detect.
    pub fn from_snapshots(
        snapshots: &[Array2<f64>],
        config: &PulsarConfig,
        entity_ids: &EntityIds,
        timestamps: Option<&[Key]>,
        similarity_threshold: f64,
        scale: bool,
    ) -> Result<Self, PipelineError> {
        check_epsilons(&config.ball_mapper.epsilons)?;
        if similarity_threshold < 0.0 {
            return Err(input_error(format!(
                "similarity_threshold must be >= 0, got {}",
                pulsar_data::preprocess::py_float_repr(similarity_threshold)
            )));
        }
        let (pooled, obs) = stack_panel(snapshots, entity_ids, timestamps)?;
        let pooled = if scale {
            StandardScalerInner::fit_transform(&pooled).0
        } else {
            pooled
        };
        let embeddings = projection_grid(&pooled, &config.projection)?;
        Self::build(&embeddings, obs, config, similarity_threshold, scale)
    }

    fn build(
        embeddings: &[Array2<f64>],
        obs: Vec<Observation>,
        config: &PulsarConfig,
        similarity_threshold: f64,
        scaled: bool,
    ) -> Result<Self, PipelineError> {
        let n = obs.len();
        let epsilons = &config.ball_mapper.epsilons;
        let mut builder = CosmicBuilder::new(n, &config.cosmic_graph);
        let mut balls = Vec::new();
        // One grid call per embedding keeps each ball's provenance known by
        // construction rather than inferred from the grid's output order.
        for embedding in embeddings {
            let maps = ball_mapper_grid(std::slice::from_ref(embedding), epsilons);
            builder.accumulate(&maps)?;
            for map in maps {
                let eps = map.eps;
                balls.extend(map.nodes.into_iter().map(|members| Ball {
                    eps,
                    dim: embedding.ncols(),
                    members,
                }));
            }
        }
        let cosmic = builder.build();
        let edges = cosmic
            .weighted_edges()
            .into_iter()
            .filter(|&(_, _, w)| w > similarity_threshold)
            .collect();
        let construction =
            serde_json::to_value(config.cosmic_graph.construction).unwrap_or(Value::Null);
        let meta = json!({
            "n_observations": n,
            "n_balls": balls.len(),
            "n_embeddings": embeddings.len(),
            "construction": construction,
            "similarity_threshold": similarity_threshold,
            "epsilons": epsilons,
            "scaled": scaled,
            "scope": "global",
        });
        Ok(Self {
            obs,
            balls,
            similarity: Graph { n, edges },
            meta,
        })
    }

    pub fn n_observations(&self) -> usize {
        self.obs.len()
    }

    /// Distinct entities.
    pub fn n_entities(&self) -> usize {
        let mut ids: Vec<&Key> = self.obs.iter().map(|o| &o.entity_id).collect();
        ids.sort();
        ids.dedup();
        ids.len()
    }

    /// Distinct time steps.
    pub fn n_times(&self) -> usize {
        let mut ts: Vec<usize> = self.obs.iter().map(|o| o.t).collect();
        ts.sort_unstable();
        ts.dedup();
        ts.len()
    }

    /// Balls each observation belongs to (row sums of the incidence matrix).
    pub fn memberships(&self) -> Vec<usize> {
        let mut out = vec![0; self.obs.len()];
        for ball in &self.balls {
            for &m in &ball.members {
                out[m] += 1;
            }
        }
        out
    }

    /// CO_SIMILAR restricted to weights strictly above `threshold`.
    pub fn thresholded(&self, threshold: f64) -> Graph {
        Graph {
            n: self.similarity.n,
            edges: self
                .similarity
                .edges
                .iter()
                .copied()
                .filter(|&(_, _, w)| w > threshold)
                .collect(),
        }
    }

    /// CO_SIMILAR edges above `threshold` joining *different* time steps.
    pub fn cross_time(&self, threshold: f64) -> Graph {
        let mut g = self.thresholded(threshold);
        g.edges.retain(|&(i, j, _)| self.obs[i].t != self.obs[j].t);
        g
    }

    /// Connected-component label per observation on CO_SIMILAR above
    /// `threshold`, numbered in order of first appearance (scipy's order).
    pub fn cluster_labels(&self, threshold: f64) -> Vec<usize> {
        self.thresholded(threshold).component_labels()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snaps() -> Vec<Array2<f64>> {
        vec![Array2::zeros((2, 3)), Array2::zeros((3, 3))]
    }

    fn err(r: Result<(Array2<f64>, Vec<Observation>), PipelineError>) -> String {
        r.unwrap_err().to_string()
    }

    /// Entities a, b at t = 0, 1: obs 0 = (a, 0), 1 = (b, 0), 2 = (a, 1), 3 = (b, 1).
    fn hand_made() -> CosmicTrajectory {
        let ob = |e: &str, t| Observation {
            entity_id: Key::Text(e.into()),
            t,
            timestamp: None,
        };
        CosmicTrajectory {
            obs: vec![ob("a", 0), ob("b", 0), ob("a", 1), ob("b", 1)],
            balls: vec![
                Ball {
                    eps: 0.5,
                    dim: 2,
                    members: vec![0, 2],
                },
                Ball {
                    eps: 0.5,
                    dim: 2,
                    members: vec![1, 2, 3],
                },
            ],
            similarity: Graph {
                n: 4,
                edges: vec![(0, 2, 0.9), (1, 2, 0.3), (1, 3, 0.6), (2, 3, 0.2)],
            },
            meta: json!({}),
        }
    }

    #[test]
    fn slices_and_labels_have_known_answers() {
        let ct = hand_made();
        assert_eq!(
            (ct.n_observations(), ct.n_entities(), ct.n_times()),
            (4, 2, 2)
        );
        assert_eq!(ct.memberships(), vec![1, 1, 2, 1]);
        let ends = |g: Graph| g.edges.iter().map(|&(i, j, _)| (i, j)).collect::<Vec<_>>();
        assert_eq!(ends(ct.thresholded(0.25)), vec![(0, 2), (1, 2), (1, 3)]);
        // (2, 3) joins two observations at t = 1, so it is not cross-time.
        assert_eq!(ends(ct.cross_time(0.0)), vec![(0, 2), (1, 2), (1, 3)]);
        assert_eq!(ends(ct.cross_time(0.5)), vec![(0, 2), (1, 3)]);
        assert_eq!(ct.cluster_labels(0.0), vec![0, 0, 0, 0]);
        assert_eq!(ct.cluster_labels(0.5), vec![0, 1, 0, 1]);
        assert_eq!(ct.cluster_labels(0.95), vec![0, 1, 2, 3]);
    }

    #[test]
    fn stacking_numbers_observations_by_row_and_checks_ids() {
        let (pooled, obs) = stack_panel(
            &snaps(),
            &EntityIds::PerSnapshot(vec![
                vec![Key::Text("a".into()), Key::Text("b".into())],
                vec![
                    Key::Text("a".into()),
                    Key::Text("b".into()),
                    Key::Text("c".into()),
                ],
            ]),
            Some(&[Key::Int(2020), Key::Int(2021)]),
        )
        .unwrap();
        assert_eq!(pooled.dim(), (5, 3));
        assert_eq!(obs[2].entity_id, Key::Text("a".into()));
        assert_eq!(obs[2].t, 1);
        assert_eq!(obs[4].timestamp, Some(Key::Int(2021)));
        let (_, positional) = stack_panel(&snaps(), &EntityIds::Positional, None).unwrap();
        assert_eq!(positional[4].entity_id, Key::Int(2));

        assert_eq!(
            err(stack_panel(&[], &EntityIds::Positional, None)),
            "snapshots must be non-empty"
        );
        assert_eq!(
            err(stack_panel(
                &[Array2::zeros((2, 3)), Array2::zeros((2, 4))],
                &EntityIds::Positional,
                None
            )),
            "All snapshots must share a feature count; got [3, 4]"
        );
        assert_eq!(
            err(stack_panel(
                &[Array2::zeros((0, 3))],
                &EntityIds::Positional,
                None
            )),
            "snapshots contain no rows"
        );
        assert_eq!(
            err(stack_panel(
                &snaps(),
                &EntityIds::Shared(vec![Key::Int(1), Key::Int(2)]),
                None
            )),
            "Entity IDs must match snapshot row counts; mismatch at snapshots [1]"
        );
        assert_eq!(
            err(stack_panel(
                &snaps(),
                &EntityIds::PerSnapshot(vec![vec![]]),
                None
            )),
            "snapshot_entity_ids has 1 time slices but there are 2 snapshots"
        );
        assert_eq!(
            err(stack_panel(
                &snaps(),
                &EntityIds::Positional,
                Some(&[Key::Int(1)])
            )),
            "timestamps has 1 entries but there are 2 snapshots"
        );
    }
}
