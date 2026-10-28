//! Plan persistence: plans survive a restart, a revision swap is atomic and
//! conflicts name the current revision, and bad handles carry the right code.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use pulsar_store::{PlanId, Store, StoreError};
use serde_json::json;

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(tag: &str) -> Self {
        let nonce: u64 = rand::random();
        let path = std::env::temp_dir().join(format!(
            "pulsar-plans-it-{tag}-{}-{nonce:x}",
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

#[test]
fn a_plan_survives_a_restart_with_floats_intact() {
    let root = TempRoot::new("restart");
    let id = PlanId::random();
    let body =
        json!({"revision": 1, "missing_fraction": 0.1 + 0.2, "nested": {"a": [1, 2.5, null]}});
    {
        let store = Store::open_at(root.path()).unwrap();
        let saved = store
            .save_plan(&id, "ds_aaaaaaaaaaaa", 1, body.clone())
            .unwrap();
        assert_eq!(
            (saved.revision, saved.created_at == saved.updated_at),
            (1, true)
        );
    }
    let reopened = Store::open_at(root.path()).unwrap();
    let got = reopened.get_plan(&id).unwrap();
    assert_eq!(got.plan, body, "0.1 + 0.2 must round-trip bit for bit");
    assert_eq!(
        (got.dataset_id.as_str(), got.revision),
        ("ds_aaaaaaaaaaaa", 1)
    );
}

#[test]
fn updating_swaps_the_revision_and_keeps_the_creation_time() {
    let root = TempRoot::new("update");
    let store = Store::open_at(root.path()).unwrap();
    let id = PlanId::random();
    let first = store.save_plan(&id, "ds_x", 1, json!({"v": 1})).unwrap();
    let second = store.update_plan(&id, 1, 2, json!({"v": 2})).unwrap();
    assert_eq!(
        (second.revision, second.plan, second.created_at),
        (2, json!({"v": 2}), first.created_at)
    );
    assert!(second.updated_at >= first.updated_at);
    assert_eq!(store.get_plan(&id).unwrap().revision, 2);
}

#[test]
fn a_stale_revision_is_a_conflict_naming_the_current_one() {
    let root = TempRoot::new("conflict");
    let store = Store::open_at(root.path()).unwrap();
    let id = PlanId::random();
    store.save_plan(&id, "ds_x", 1, json!({})).unwrap();
    store.update_plan(&id, 1, 2, json!({"v": 2})).unwrap();

    let err = store
        .update_plan(&id, 1, 2, json!({"v": "late"}))
        .unwrap_err();
    assert_eq!(err.code(), "SESSION_REVISION_CONFLICT");
    assert!(matches!(
        err,
        StoreError::RevisionConflict {
            current: 2,
            expected: 1,
            ..
        }
    ));
    assert!(
        err.agent_action().contains("base_revision=2"),
        "{}",
        err.agent_action()
    );
    assert_eq!(
        store.get_plan(&id).unwrap().plan,
        json!({"v": 2}),
        "a rejected update writes nothing"
    );
}

#[test]
fn concurrent_updates_from_one_revision_let_exactly_one_win() {
    let root = TempRoot::new("race");
    let store = Arc::new(Store::open_at(root.path()).unwrap());
    let id = PlanId::random();
    store.save_plan(&id, "ds_x", 1, json!({})).unwrap();

    let results: Vec<bool> = (0..8)
        .map(|i| {
            let (store, id) = (store.clone(), id.clone());
            std::thread::spawn(move || store.update_plan(&id, 1, 2, json!({"writer": i})).is_ok())
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    assert_eq!(results.iter().filter(|&&won| won).count(), 1, "{results:?}");
    assert_eq!(store.get_plan(&id).unwrap().revision, 2);
}

#[test]
fn stores_sharing_a_root_let_exactly_one_writer_win_each_revision() {
    // Separate `Store`s have separate mutexes, like separate agent
    // processes; only the file lock orders them.
    let root = TempRoot::new("shared");
    let stores: Vec<Store> = (0..4)
        .map(|_| Store::open_at(root.path()).unwrap())
        .collect();
    let id = PlanId::random();
    stores[0].save_plan(&id, "ds_x", 1, json!({})).unwrap();

    for rev in 1..=20 {
        let barrier = Arc::new(std::sync::Barrier::new(stores.len() * 2));
        let wins = stores
            .iter()
            .flat_map(|s| [s.clone(), s.clone()])
            .enumerate()
            .map(|(i, store)| {
                let (barrier, id) = (barrier.clone(), id.clone());
                std::thread::spawn(move || {
                    barrier.wait();
                    store
                        .update_plan(&id, rev, rev + 1, json!({"writer": i}))
                        .is_ok()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .filter(|&won| won)
            .count();
        assert_eq!(wins, 1, "revision {rev}");
    }
    assert_eq!(stores[3].get_plan(&id).unwrap().revision, 21);
}

#[test]
fn saving_over_an_existing_plan_is_refused() {
    let root = TempRoot::new("exists");
    let store = Store::open_at(root.path()).unwrap();
    let id = PlanId::random();
    store.save_plan(&id, "ds_x", 1, json!({"v": 1})).unwrap();

    let err = store
        .save_plan(&id, "ds_y", 1, json!({"v": "clobber"}))
        .unwrap_err();
    assert!(matches!(err, StoreError::PlanExists { .. }), "{err:?}");
    assert_eq!(err.code(), "PLAN_ID_EXISTS");
    let kept = store.get_plan(&id).unwrap();
    assert_eq!(
        (kept.dataset_id.as_str(), kept.plan),
        ("ds_x", json!({"v": 1}))
    );
}

#[test]
fn unknown_and_malformed_ids_carry_the_right_codes() {
    let root = TempRoot::new("ids");
    let store = Store::open_at(root.path()).unwrap();
    let missing = PlanId::random();
    let err = store.get_plan(&missing).unwrap_err();
    assert_eq!(err.code(), "PLAN_ID_UNKNOWN");
    assert!(
        err.agent_action().contains("draft_preprocessing_plan"),
        "{}",
        err.agent_action()
    );
    assert_eq!(
        store
            .update_plan(&missing, 1, 2, json!({}))
            .unwrap_err()
            .code(),
        "PLAN_ID_UNKNOWN"
    );

    assert_eq!(
        PlanId::parse("pp_0123456789ab").unwrap().as_str(),
        "pp_0123456789ab"
    );
    for bad in [
        "pp_123",
        "run_0123456789ab",
        "pp_0123456789AB",
        "",
        "pp_0123456789abc",
    ] {
        let err = PlanId::parse(bad).unwrap_err();
        assert_eq!(err.code(), "INVALID_HANDLE", "{bad:?}");
    }
}

#[test]
fn listing_filters_by_dataset_and_puts_the_latest_last() {
    let root = TempRoot::new("list");
    let store = Store::open_at(root.path()).unwrap();
    let (a, b, c) = (PlanId::random(), PlanId::random(), PlanId::random());
    store.save_plan(&a, "ds_one", 1, json!({})).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(5));
    store.save_plan(&b, "ds_two", 1, json!({})).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(5));
    store.save_plan(&c, "ds_one", 1, json!({})).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(5));
    // Touching the oldest plan makes it the most recently worked on.
    store.update_plan(&a, 1, 2, json!({})).unwrap();

    let ids = |plans: Vec<pulsar_store::PlanRecord>| {
        plans.into_iter().map(|p| p.plan_id).collect::<Vec<_>>()
    };
    assert_eq!(
        ids(store.list_plans(Some("ds_one")).unwrap()),
        [c.clone(), a.clone()]
    );
    assert_eq!(
        ids(store.list_plans(Some("ds_two")).unwrap()),
        std::slice::from_ref(&b)
    );
    assert_eq!(ids(store.list_plans(None).unwrap()), [b, c, a]);
    assert!(store.list_plans(Some("ds_none")).unwrap().is_empty());
}

#[test]
fn an_unreadable_record_is_skipped_by_listing_and_reported_by_get() {
    let root = TempRoot::new("corrupt");
    let store = Store::open_at(root.path()).unwrap();
    let good = PlanId::random();
    store.save_plan(&good, "ds_x", 1, json!({})).unwrap();
    let bad = PlanId::random();
    std::fs::write(
        root.path().join("plans").join(format!("{bad}.json")),
        "{ not json",
    )
    .unwrap();

    assert_eq!(store.list_plans(None).unwrap().len(), 1);
    assert_eq!(store.get_plan(&bad).unwrap_err().code(), "RECORD_CORRUPT");
}
