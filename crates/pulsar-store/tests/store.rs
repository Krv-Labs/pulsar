//! The #62 acceptance criteria: runs survive a restart, concurrent readers
//! are safe, and stale handles carry the right error code.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use pulsar_config::{Construction, ProjectionMethod};
use pulsar_pipeline::Graph;
use pulsar_store::{
    summarize_history, ClusterAssignment, ConfigSummary, DatasetId, DatasetSource, JobKind,
    JobState, Representation, RunArtifacts, RunId, RunMetrics, RunRecord, Store, StoreOptions,
    ThresholdPolicy, ABANDONED, INTERRUPTED, SCHEMA_VERSION,
};

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(tag: &str) -> Self {
        let nonce: u64 = rand::random();
        let path = std::env::temp_dir().join(format!(
            "pulsar-store-it-{tag}-{}-{nonce:x}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn triangle_plus_singleton() -> RunArtifacts {
    let graph = Graph {
        n: 4,
        edges: vec![(0, 1, 0.9), (0, 2, 0.8), (1, 2, 0.7)],
    };
    let mut affinity = graph.clone();
    affinity.edges.push((2, 3, 0.05));
    RunArtifacts { graph, affinity }
}

fn record_for(run_id: RunId, dataset_id: Option<DatasetId>, artifacts: &RunArtifacts) -> RunRecord {
    RunRecord {
        schema: SCHEMA_VERSION,
        run_id,
        dataset_id,
        data_path: None,
        config_yaml: "run:\n  name: t\n".into(),
        config: ConfigSummary {
            run_name: "t".into(),
            projection_method: ProjectionMethod::Jl,
            projection_dimensions: vec![2, 3],
            n_seeds: 2,
            n_epsilons: 3,
            construction: Construction::Minhash,
            threshold_policy: ThresholdPolicy::Fixed,
            requested_threshold: Some(0.1),
            sparsify: false,
        },
        n_rows: artifacts.graph.n,
        n_features: 2,
        n_embeddings: 4,
        n_ball_maps: 12,
        resolved_construction_threshold: 0.1,
        weight_scale: 10.0,
        metrics: RunMetrics::from_graph(&artifacts.graph),
        threshold_stability_summary: None,
        created_at: 0.0,
    }
}

#[test]
fn runs_and_jobs_survive_a_restart() {
    let root = TempRoot::new("restart");
    let artifacts = triangle_plus_singleton();
    let run_id;
    {
        let store = Store::open_at(root.path()).unwrap();
        let job = store.submit(JobKind::Sweep, None).unwrap();
        run_id = job.id();
        assert_eq!(store.get_job(&run_id).unwrap().state, JobState::Queued);
        job.start().unwrap();
        job.progress("projection", 0.4).unwrap();
        job.flush().unwrap();
        let seen = store.get_job(&run_id).unwrap();
        assert_eq!(
            (seen.state, seen.stage.as_str(), seen.progress),
            (JobState::Running, "projection", 0.4)
        );
        job.complete(&record_for(run_id.clone(), None, &artifacts), &artifacts)
            .unwrap();
    }
    // A new process: nothing but the directory.
    let store = Store::open_at(root.path()).unwrap();
    let job = store.get_job(&run_id).unwrap();
    assert_eq!(job.state, JobState::Complete);
    assert_eq!(job.progress, 1.0);
    assert!(job.finished_at.is_some());
    let run = store.get_run(&run_id).unwrap();
    assert_eq!(run.run_id, run_id);
    assert_eq!(run.metrics.component_sizes, vec![3, 1]);
    assert_eq!(store.run_graph(&run_id).unwrap(), artifacts.graph);
    assert_eq!(store.run_affinity(&run_id).unwrap(), artifacts.affinity);
    assert!(store.run_dir(&run_id).join("record.json").exists());
    assert!(store.recover_interrupted().unwrap().is_empty());
}

#[test]
fn stale_and_unfinished_handles_carry_the_right_codes() {
    let root = TempRoot::new("codes");
    let store = Store::open_at(root.path()).unwrap();

    assert_eq!(
        RunId::parse("run-oops").unwrap_err().code(),
        "INVALID_HANDLE"
    );
    let missing = RunId::parse("run_0123456789ab").unwrap();
    let err = store.get_run(&missing).unwrap_err();
    assert_eq!(err.code(), "RUN_ID_UNKNOWN");
    assert!(err.agent_action().contains("Re-run the sweep"));
    assert_eq!(
        store.get_job(&missing).unwrap_err().code(),
        "RUN_ID_UNKNOWN"
    );
    assert_eq!(
        store.run_graph(&missing).unwrap_err().code(),
        "RUN_ID_UNKNOWN"
    );
    assert_eq!(
        store
            .get_dataset(&DatasetId::parse("ds_0123456789ab").unwrap())
            .unwrap_err()
            .code(),
        "DATASET_ID_UNKNOWN"
    );

    let queued = store.submit(JobKind::Sweep, None).unwrap();
    let err = store.get_run(&queued.id()).unwrap_err();
    assert_eq!(err.code(), "RUN_NOT_COMPLETE");
    assert!(err.to_string().contains("is queued"), "{err}");
    assert_eq!(queued.progress("x", 0.1).unwrap_err().code(), "JOB_STATE");

    queued.start().unwrap();
    queued.fail("DATA_ERROR", "no rows").unwrap();
    let err = store.get_run(&queued.id()).unwrap_err();
    assert_eq!(err.code(), "RUN_NOT_COMPLETE");
    assert!(err.to_string().contains("no rows (DATA_ERROR)"), "{err}");
    let artifacts = triangle_plus_singleton();
    let rec = record_for(queued.id(), None, &artifacts);
    assert_eq!(
        queued.complete(&rec, &artifacts).unwrap_err().code(),
        "JOB_STATE"
    );

    // A record from a job with a different id is refused before anything is written.
    let other = store.submit(JobKind::Sweep, None).unwrap();
    assert_eq!(
        other.complete(&rec, &artifacts).unwrap_err().code(),
        "INVALID_ARGUMENT"
    );
    assert!(!store.run_dir(&queued.id()).exists());

    // Dropping the last handle of an unfinished job fails it rather than
    // leaving it running forever.
    let abandoned_id = other.id();
    drop(other);
    let job = store.get_job(&abandoned_id).unwrap();
    assert_eq!(job.state, JobState::Failed);
    assert_eq!(job.error.unwrap().code, ABANDONED);

    // A job left running by a dead process is failed on recovery.
    let running = store.submit(JobKind::Dossier, None).unwrap();
    running.start().unwrap();
    let running_id = running.id();
    std::mem::forget(running);
    let recovered = store.recover_interrupted().unwrap();
    assert_eq!(recovered, vec![running_id.clone()]);
    assert_eq!(
        store.get_job(&running_id).unwrap().error.unwrap().code,
        INTERRUPTED
    );

    // A listing skips the incomplete runs and counts nothing (no run dirs).
    let listing = store.list_runs(None).unwrap();
    assert!(listing.runs.is_empty());
    assert_eq!(listing.skipped, 0);
    assert_eq!(store.list_jobs().unwrap().len(), 3);
}

#[test]
fn concurrent_readers_never_see_a_partial_record() {
    let root = TempRoot::new("readers");
    let store = Store::open_with(
        root.path(),
        StoreOptions {
            progress_flush_interval: Duration::ZERO,
            ..StoreOptions::default()
        },
    )
    .unwrap();
    let job = store.submit(JobKind::Sweep, None).unwrap();
    job.start().unwrap();
    let id = job.id();
    let done = Arc::new(AtomicBool::new(false));

    let readers: Vec<_> = (0..4)
        .map(|_| {
            let (store, id, done) = (store.clone(), id.clone(), done.clone());
            std::thread::spawn(move || {
                let mut reads = 0_usize;
                let mut last = -1.0_f64;
                while !done.load(Ordering::SeqCst) {
                    let rec = store
                        .get_job(&id)
                        .expect("every read parses a whole record");
                    assert!(rec.progress >= last, "progress went backwards");
                    last = rec.progress;
                    reads += 1;
                }
                reads
            })
        })
        .collect();

    const STEPS: usize = 400;
    for k in 0..=STEPS {
        job.progress(&format!("step {k}"), k as f64 / STEPS as f64)
            .unwrap();
    }
    let artifacts = triangle_plus_singleton();
    job.complete(&record_for(id.clone(), None, &artifacts), &artifacts)
        .unwrap();
    done.store(true, Ordering::SeqCst);
    let total: usize = readers.into_iter().map(|r| r.join().unwrap()).sum();
    assert!(total > 0);
    assert_eq!(store.get_job(&id).unwrap().state, JobState::Complete);
    assert_eq!(store.get_run(&id).unwrap().n_rows, 4);
}

#[test]
fn datasets_are_content_addressed_and_drift_is_detected() {
    let root = TempRoot::new("datasets");
    let store = Store::open_at(root.path()).unwrap();
    let csv = root.path().join("input.csv");
    std::fs::write(&csv, "a,b\n1,2\n").unwrap();

    let first = store.register_dataset(&csv).unwrap();
    let again = store.register_dataset(&csv).unwrap();
    assert_eq!(first.dataset_id, again.dataset_id);
    assert_eq!(first.source, DatasetSource::Path);
    assert_eq!(first.name, "input.csv");
    assert_eq!(
        store.get_dataset(&first.dataset_id).unwrap().path,
        first.path
    );
    assert!(Store::dataset_matches_disk(&first));
    assert_eq!(
        store.check_dataset(&first.dataset_id).unwrap(),
        store.get_dataset(&first.dataset_id).unwrap()
    );

    std::fs::write(&csv, "a,b\n1,2\n3,4\n").unwrap();
    let edited = store.register_dataset(&csv).unwrap();
    assert_ne!(edited.dataset_id, first.dataset_id);
    assert!(!Store::dataset_matches_disk(&first));
    let err = store.check_dataset(&first.dataset_id).unwrap_err();
    assert_eq!(err.code(), "DATASET_CHANGED");
    assert_eq!(store.list_datasets().unwrap().len(), 2);

    let err = store
        .register_dataset(root.path().join("missing.csv"))
        .unwrap_err();
    assert_eq!(err.code(), "DATASET_NOT_FOUND");
}

#[test]
fn uploads_become_store_owned_datasets_without_a_bom() {
    let root = TempRoot::new("uploads");
    let store = Store::open_at(root.path()).unwrap();
    let upload = store
        .begin_upload("../evil/../data.csv", "text/csv")
        .unwrap();
    assert_eq!(upload.filename, "data.csv");
    let upload = store
        .append_upload_chunk(&upload.upload_id, b"\xef\xbb\xbfa,b\n")
        .unwrap();
    let upload = store
        .append_upload_chunk(&upload.upload_id, b"1,2\n")
        .unwrap();
    assert_eq!(upload.bytes_received, 11);

    let dataset = store.finalize_upload(&upload.upload_id).unwrap();
    assert_eq!(dataset.source, DatasetSource::Upload);
    assert!(dataset.path.starts_with(store.root()));
    assert_eq!(
        std::fs::read_to_string(&dataset.path).unwrap(),
        "a,b\n1,2\n"
    );
    assert_eq!(dataset.size_bytes, 8);
    assert_eq!(
        store.get_upload(&upload.upload_id).unwrap_err().code(),
        "UPLOAD_ID_UNKNOWN"
    );

    // Same bytes, same source → same id.
    let second = store.begin_upload("data.csv", "text/csv").unwrap();
    store
        .append_upload_chunk(&second.upload_id, b"a,b\n1,2\n")
        .unwrap();
    assert_eq!(
        store.finalize_upload(&second.upload_id).unwrap().dataset_id,
        dataset.dataset_id
    );
}

#[test]
fn cluster_assignments_and_longitudinal_records_round_trip() {
    let root = TempRoot::new("assignments");
    let store = Store::open_at(root.path()).unwrap();
    let artifacts = triangle_plus_singleton();
    let job = store.submit(JobKind::Sweep, None).unwrap();
    job.complete(&record_for(job.id(), None, &artifacts), &artifacts)
        .unwrap();

    let labels = [0_i64, 0, 0, -1];
    let input = ClusterAssignment {
        run_id: Some(job.id()),
        dataset_id: None,
        method: "components".into(),
        interpretation_edge_weight_threshold: 0.1,
        threshold_source: "construction".into(),
        construction_threshold: 0.1,
        labels: &labels,
    };
    let saved = store.save_cluster_assignment(input.clone()).unwrap();
    assert_eq!(saved.label_count, 4);
    assert_eq!(saved.cluster_ids, vec![-1, 0]);
    assert_eq!(
        store
            .get_cluster_assignment(&saved.cluster_assignment_id)
            .unwrap(),
        saved
    );
    assert_eq!(
        store.cluster_labels(&saved.cluster_assignment_id).unwrap(),
        labels
    );

    let short = ClusterAssignment {
        labels: &labels[..2],
        ..input
    };
    assert_eq!(
        store.save_cluster_assignment(short).unwrap_err().code(),
        "INVALID_ARGUMENT"
    );

    let lg = store
        .save_longitudinal(
            None,
            Representation::Both,
            "x: 1\n",
            serde_json::json!({"n_entities": 3}),
        )
        .unwrap();
    let back = store.get_longitudinal(&lg.longitudinal_id).unwrap();
    assert_eq!(back, lg);
    assert_eq!(back.panel["n_entities"], 3);
}

fn assignment_dir_entries(root: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(root.join("cluster_assignments"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[test]
fn staged_assignments_are_visible_only_after_commit() {
    let root = TempRoot::new("staged");
    let store = Store::open_at(root.path()).unwrap();
    let labels = [0_i64, 1, 1, 2];
    let input = ClusterAssignment {
        run_id: None,
        dataset_id: None,
        method: "components".into(),
        interpretation_edge_weight_threshold: 0.0,
        threshold_source: "explicit".into(),
        construction_threshold: 0.0,
        labels: &labels,
    };

    let staged = store.stage_cluster_assignment(input.clone()).unwrap();
    let id = staged.record().cluster_assignment_id.clone();
    assert_eq!(
        store.get_cluster_assignment(&id).unwrap_err().code(),
        "CLUSTER_ASSIGNMENT_ID_UNKNOWN"
    );
    let pending = assignment_dir_entries(root.path());
    assert_eq!(pending.len(), 2);
    assert!(pending.iter().all(|n| n.starts_with('.')), "{pending:?}");

    let committed = staged.commit().unwrap();
    assert_eq!(committed.cluster_ids, vec![0, 1, 2]);
    assert_eq!(store.get_cluster_assignment(&id).unwrap(), committed);
    assert_eq!(store.cluster_labels(&id).unwrap(), labels);
    assert_eq!(
        assignment_dir_entries(root.path()),
        vec![format!("{id}.json"), format!("{id}.labels.arrow")]
    );

    // Dropping uncommitted is the discard: nothing is left behind.
    let discarded = store.stage_cluster_assignment(input.clone()).unwrap();
    let discarded_id = discarded.record().cluster_assignment_id.clone();
    drop(discarded);
    assert!(store.get_cluster_assignment(&discarded_id).is_err());
    assert_eq!(assignment_dir_entries(root.path()).len(), 2);

    // A crash between stage and commit leaves temp files that the next
    // open sweeps once they are old enough.
    std::mem::forget(store.stage_cluster_assignment(input).unwrap());
    assert_eq!(assignment_dir_entries(root.path()).len(), 4);
    Store::open_with(
        root.path(),
        StoreOptions {
            orphan_max_age: Duration::ZERO,
            ..StoreOptions::default()
        },
    )
    .unwrap();
    assert_eq!(assignment_dir_entries(root.path()).len(), 2);
}

#[test]
fn history_reads_from_the_store_in_creation_order() {
    let root = TempRoot::new("history");
    let store = Store::open_at(root.path()).unwrap();
    let csv = root.path().join("d.csv");
    std::fs::write(&csv, "a\n1\n").unwrap();
    let ds = store.register_dataset(&csv).unwrap().dataset_id;

    let artifacts = triangle_plus_singleton();
    let mut ids = Vec::new();
    for k in 0..3 {
        let job = store.submit(JobKind::Sweep, Some(ds.clone())).unwrap();
        let mut rec = record_for(
            job.id(),
            if k == 1 { None } else { Some(ds.clone()) },
            &artifacts,
        );
        rec.created_at = 100.0 + k as f64;
        job.complete(&rec, &artifacts).unwrap();
        ids.push(job.id());
    }
    // One job that never finished: no run dir at all.
    let pending = store.submit(JobKind::Sweep, Some(ds.clone())).unwrap();
    pending.start().unwrap();

    let all = store.list_runs(None).unwrap();
    assert_eq!(
        all.runs
            .iter()
            .map(|r| r.run_id.clone())
            .collect::<Vec<_>>(),
        ids
    );
    assert_eq!(all.skipped, 0);
    let mine = store.list_runs(Some(&ds)).unwrap();
    assert_eq!(mine.runs.len(), 2);

    let summary = summarize_history(&all.runs);
    assert_eq!(summary.n_runs, 3);
    assert_eq!(summary.fragmentation_trend.status, "stable");
    assert!(summary
        .observations
        .iter()
        .any(|o| o.starts_with("Graph health across 3 runs")));

    // Then the pending job finishes and shows up at the end.
    let rec = record_for(pending.id(), Some(ds.clone()), &artifacts);
    pending.complete(&rec, &artifacts).unwrap();
    assert_eq!(store.list_runs(Some(&ds)).unwrap().runs.len(), 3);
}

#[test]
fn fitted_models_round_trip_with_the_golden_penguins_case() {
    let case = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/golden/penguins");
    let cfg = pulsar_config::from_path(case.join("config.yaml")).unwrap();
    let table = pulsar_data::read_csv_path(case.join("input.csv")).unwrap();
    let fitted =
        pulsar_pipeline::fit(&cfg, &table, pulsar_pipeline::FitOptions::default(), None).unwrap();

    let root = TempRoot::new("penguins");
    let store = Store::open_at(root.path()).unwrap();
    let dataset = store.register_dataset(case.join("input.csv")).unwrap();
    let job = store
        .submit(JobKind::Sweep, Some(dataset.dataset_id.clone()))
        .unwrap();
    job.start().unwrap();
    let record = RunRecord::from_fitted(
        job.id(),
        &fitted,
        Some(dataset.dataset_id.clone()),
        Some("input.csv"),
    );
    let artifacts = RunArtifacts::from_fitted(&fitted);
    job.complete(&record, &artifacts).unwrap();

    let store = Store::open_at(root.path()).unwrap();
    let back = store.get_run(&job.id()).unwrap();
    assert_eq!(back, record);
    assert_eq!(back.n_rows, fitted.n);
    assert_eq!(back.n_ball_maps, fitted.ball_maps.len());
    assert_eq!(
        back.resolved_construction_threshold,
        fitted.construction_threshold
    );
    assert_eq!(back.metrics.n_edges, fitted.graph.n_edges());
    assert_eq!(back.config.threshold_policy, ThresholdPolicy::Auto);
    let stability = back
        .threshold_stability_summary
        .expect("auto threshold ran");
    assert_eq!(
        stability.selected_threshold,
        fitted.stability.as_ref().unwrap().optimal_threshold
    );
    assert!(!stability.top_plateaus.is_empty());
    assert_eq!(store.run_graph(&job.id()).unwrap(), fitted.graph);
    let affinity = store.run_affinity(&job.id()).unwrap();
    assert_eq!(affinity.edges, fitted.weighted_edges(Some(0.0)));
    assert!(affinity.n_edges() >= fitted.graph.n_edges());
    assert!(pulsar_config::from_yaml_str(&back.config_yaml).is_ok());
}
