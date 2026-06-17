"""Compact completed-sweep index for agent discovery.

The heavy artifact JSON is the data plane. Listing completed sweeps is control-plane
metadata and must never load the full artifact body.
"""
from __future__ import annotations

import json


def sweep_manifest_key(user_id: str, dataset_id: str, config_hash: str) -> str:
    return f"{user_id}/{dataset_id}/sweeps/{config_hash}.json"


def sweep_manifest_from_artifact(
    artifact: dict,
    *,
    user_id: str,
    dataset_id: str,
    config_hash: str,
    object_key: str,
) -> dict:
    artifact_ref = {"userId": user_id, "datasetId": dataset_id, "configHash": config_hash}
    return {
        "artifactRef": artifact_ref,
        # Backward-compatible alias for existing Python-side tests/clients.
        "artifact_ref": artifact_ref,
        "datasetId": dataset_id,
        "configHash": config_hash,
        "objectKey": object_key,
        "n": artifact.get("n"),
        "metrics": artifact.get("metrics"),
        "structureStatus": artifact.get("structureStatus"),
        "pulsarVersion": artifact.get("pulsarVersion"),
        "createdAt": artifact.get("createdAt"),
    }


def write_sweep_manifest(
    store,
    artifact: dict,
    *,
    user_id: str,
    dataset_id: str,
    config_hash: str,
    object_key: str,
) -> dict:
    manifest = sweep_manifest_from_artifact(
        artifact,
        user_id=user_id,
        dataset_id=dataset_id,
        config_hash=config_hash,
        object_key=object_key,
    )
    store.put(
        sweep_manifest_key(user_id, dataset_id, config_hash),
        json.dumps(manifest, sort_keys=True).encode("utf-8"),
    )
    return manifest


def load_sweep_manifest(store, *, user_id: str, dataset_id: str, config_hash: str) -> dict:
    return json.loads(store.get(sweep_manifest_key(user_id, dataset_id, config_hash)).decode("utf-8"))

