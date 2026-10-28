//! Longitudinal panel support (`pulsar/mcp/longitudinal.py`): the long ->
//! panel pivot, its alignment policies and cost guards, and the payload
//! builders the longitudinal tools stay thin over.
//!
//! Two representations, two questions:
//! - [`pulsar_core::TemporalCosmicGraph`]: nodes are *entities*, edges live
//!   inside one time slice ("which entities are stably similar over time").
//! - [`pulsar_pipeline::CosmicTrajectory`]: nodes are *observations*
//!   `(entity, t)`, edges may span time ("which observations resemble each
//!   other across time", "how do entities move between cohorts").

pub mod panel;
pub mod payloads;

pub use panel::{
    adapt_config_to_panel, estimate_costs, guard_representation, max_tensor_bytes, pivot_panel,
    prepare_panel_frame, Panel, PanelError, PIVOT_POLICIES, REPRESENTATIONS,
};
pub use payloads::{
    archetype_payload, classify_labels, classify_trajectories_payload, cross_time_payload,
    select_longitudinal_entities, temporal_diagnosis, threshold_sweep, trajectory_diagnosis,
};
