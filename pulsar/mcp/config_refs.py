"""Tenant-scoped references for generated sweep configs.

Agents should not shuttle large YAML blobs between tools. A config ref is the
stable hash of the parsed config, persisted under the tenant/dataset namespace.
"""

from __future__ import annotations

from typing import Any

import yaml

from pulsar.mcp.jobs import config_hash


def parse_config_yaml(config_yaml: str) -> dict[str, Any]:
    parsed = yaml.safe_load(config_yaml)
    if not isinstance(parsed, dict):
        raise ValueError("config_yaml must be a valid YAML mapping")
    return parsed


def config_ref_key(user_id: str, dataset_id: str, config_ref: str) -> str:
    if not config_ref.startswith("cfg_"):
        raise ValueError(
            "config_ref must be a config hash returned by create_config/refine_config"
        )
    return f"{user_id}/{dataset_id}/configs/{config_ref}.yaml"


def save_config_ref(
    store, *, user_id: str, dataset_id: str, config_yaml: str
) -> tuple[str, dict[str, Any]]:
    config = parse_config_yaml(config_yaml)
    ref = config_hash(config)
    store.put(config_ref_key(user_id, dataset_id, ref), config_yaml.encode("utf-8"))
    return ref, config


def load_config_ref(store, *, user_id: str, dataset_id: str, config_ref: str) -> str:
    key = config_ref_key(user_id, dataset_id, config_ref)
    if not store.exists(key):
        raise FileNotFoundError(
            f"config_ref {config_ref} not found for dataset {dataset_id}"
        )
    return store.get(key).decode("utf-8")
