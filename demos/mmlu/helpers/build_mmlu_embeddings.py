#!/usr/bin/env python3
"""
Build MMLU embedding artifacts outside the notebook.

This utility mirrors the notebook embedding pipeline and supports:
1) Listing pending model jobs
2) Building all variants for a single model (intended for parallel launch)
3) Finalizing the metadata index consumed by the notebook
"""

import argparse
import gc
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

try:
    import torch
except ImportError as exc:
    raise ImportError("Missing dependency: torch. Install with `uv sync --group demos`.") from exc

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:
    raise ImportError(
        "Missing dependency: transformers. Install with `uv sync --group demos`."
    ) from exc

try:
    from huggingface_hub import HfApi
except ImportError as exc:
    raise ImportError(
        "Missing dependency: huggingface-hub. Install with `uv sync --group demos`."
    ) from exc

try:
    from datasets import load_dataset
except ImportError as exc:
    raise ImportError("Missing dependency: datasets. Install with `uv sync --group demos`.") from exc


MMLU_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = MMLU_DIR / "data"
MMLU_CACHE_PATH = DATA_DIR / "mmlu_questions.csv"
MODEL_SNAPSHOT_PATH = DATA_DIR / "hf_top10_embedding_models_snapshot.json"
EMBED_METADATA_PATH = DATA_DIR / "mmlu_embedding_artifacts_metadata.json"


VARIANT_SPECS = [
    {"variant_id": "v01", "prefix": "", "max_length": 256, "normalize": True, "pooling": "mean"},
    {"variant_id": "v02", "prefix": "", "max_length": 384, "normalize": True, "pooling": "mean"},
    {"variant_id": "v03", "prefix": "", "max_length": 512, "normalize": True, "pooling": "mean"},
    {
        "variant_id": "v04",
        "prefix": "Represent this question for retrieval: ",
        "max_length": 256,
        "normalize": True,
        "pooling": "mean",
    },
    {
        "variant_id": "v05",
        "prefix": "Represent this question for retrieval: ",
        "max_length": 384,
        "normalize": True,
        "pooling": "mean",
    },
    {
        "variant_id": "v06",
        "prefix": "Represent this question for retrieval: ",
        "max_length": 512,
        "normalize": True,
        "pooling": "mean",
    },
    {"variant_id": "v07", "prefix": "Query: ", "max_length": 384, "normalize": True, "pooling": "mean"},
    {"variant_id": "v08", "prefix": "Query: ", "max_length": 384, "normalize": False, "pooling": "mean"},
    {"variant_id": "v09", "prefix": "Query: ", "max_length": 384, "normalize": True, "pooling": "cls"},
    {"variant_id": "v10", "prefix": "", "max_length": 384, "normalize": False, "pooling": "cls"},
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def normalize_limit(value: int) -> int | None:
    return value if value and value > 0 else None


def slugify_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", model_id)


def detect_device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but MPS is not available.")
        return requested

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def build_cache_path(model_id: str, variant_id: str, sample_tag: str, n_rows: int, seed: int) -> Path:
    model_slug = slugify_model_id(model_id)
    filename = f"emb_{model_slug}_{variant_id}_{sample_tag}_n{n_rows}_s{seed}.npy"
    return DATA_DIR / filename


def metadata_sidecar_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, arr)
    tmp_path.replace(path)


def save_json_atomic(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def fetch_top_models(snapshot_path: Path, top_k: int, refresh: bool) -> list[dict]:
    fallback_ids = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        "intfloat/e5-small-v2",
        "intfloat/e5-base-v2",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "thenlper/gte-small",
    ]

    if snapshot_path.exists() and not refresh:
        with snapshot_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload["models"]

    try:
        api = HfApi()
        models = api.list_models(filter="sentence-transformers", sort="downloads", limit=200)

        selected = []
        for entry in models:
            model_id = entry.id
            if "cross-encoder" in model_id.lower():
                continue
            selected.append(
                {
                    "model_id": model_id,
                    "downloads": int(getattr(entry, "downloads", 0) or 0),
                }
            )
            if len(selected) == top_k:
                break

        if len(selected) < top_k:
            raise RuntimeError("Insufficient model results from Hugging Face API")

        payload = {
            "source": "huggingface_hub",
            "metric": "downloads",
            "top_k": top_k,
            "retrieved_at": utc_now_iso(),
            "models": selected,
        }
        save_json_atomic(snapshot_path, payload)
        return payload["models"]

    except Exception as exc:  # noqa: BLE001
        print(f"Warning: model ranking fetch failed ({exc}). Using fallback list.")
        fallback = [{"model_id": model_id, "downloads": -1} for model_id in fallback_ids[:top_k]]
        payload = {
            "source": "fallback",
            "metric": "manual",
            "top_k": top_k,
            "retrieved_at": utc_now_iso(),
            "models": fallback,
        }
        save_json_atomic(snapshot_path, payload)
        return fallback


def load_mmlu_dataframe(allow_download: bool) -> pd.DataFrame:
    if MMLU_CACHE_PATH.exists():
        return pd.read_csv(MMLU_CACHE_PATH)

    if not allow_download:
        raise FileNotFoundError(
            f"Missing {MMLU_CACHE_PATH}. Run with --download-mmlu, "
            "or run notebook Step 2 once to cache the dataset."
        )

    print("Downloading MMLU from Hugging Face (cais/mmlu, test split)...")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    df_mmlu = cast(pd.DataFrame, dataset.to_pandas())
    df_mmlu["choices"] = df_mmlu["choices"].apply(lambda choices: "|||".join(choices))
    df_mmlu.to_csv(MMLU_CACHE_PATH, index=False)
    print(f"Saved {len(df_mmlu)} questions to {MMLU_CACHE_PATH}")
    return df_mmlu


def build_subsample(df_mmlu: pd.DataFrame, run_mode: str, n_subsample: int, seed: int) -> tuple[pd.DataFrame, list[str], str]:
    rng = np.random.default_rng(seed)

    if run_mode == "full":
        indices = list(range(len(df_mmlu)))
    else:
        indices = []
        for subject in df_mmlu["subject"].unique():
            subject_indices = df_mmlu.index[df_mmlu["subject"] == subject].tolist()
            n_take = max(10, int(n_subsample * len(subject_indices) / len(df_mmlu)))
            n_take = min(n_take, len(subject_indices))
            chosen = rng.choice(subject_indices, size=n_take, replace=False)
            indices.extend(chosen.tolist())

        if len(indices) > n_subsample:
            indices = sorted(rng.choice(indices, size=n_subsample, replace=False))
        else:
            indices = sorted(indices)

    df_sub = df_mmlu.iloc[indices].reset_index(drop=True)
    texts_sub = df_sub["question"].tolist()
    sample_tag = "full" if run_mode == "full" else "subsample"
    return df_sub, texts_sub, sample_tag


def select_models(top_models: list[dict], include_models: list[str], max_models_run: int | None) -> list[dict]:
    model_lookup = {entry["model_id"]: entry for entry in top_models}

    if include_models:
        selected = []
        for model_id in include_models:
            selected.append(model_lookup.get(model_id, {"model_id": model_id, "downloads": -1}))
    else:
        selected = list(top_models)

    if max_models_run:
        selected = selected[:max_models_run]

    if not selected:
        raise RuntimeError("No models selected.")

    return selected


def select_variants(variant_ids: list[str], max_variants_run: int | None) -> list[dict]:
    variant_lookup = {variant["variant_id"]: variant for variant in VARIANT_SPECS}

    if variant_ids:
        selected = []
        for variant_id in variant_ids:
            if variant_id not in variant_lookup:
                raise ValueError(f"Unknown variant_id: {variant_id}")
            selected.append(variant_lookup[variant_id])
    else:
        selected = list(VARIANT_SPECS)

    if max_variants_run:
        selected = selected[:max_variants_run]

    if not selected:
        raise RuntimeError("No variants selected.")

    return selected


def encode_variant(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    batch_size: int,
    max_length: int,
    pooling: str,
    normalize: bool,
) -> np.ndarray:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state

            if pooling == "cls":
                batch_emb = hidden[:, 0]
            else:
                batch_emb = mean_pool(hidden, encoded["attention_mask"])

            if normalize:
                batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=1)

            # Some models emit bfloat16 tensors; convert to float32 for numpy compatibility.
            outputs.append(batch_emb.detach().to(torch.float32).cpu().numpy())

    return np.vstack(outputs).astype(np.float64)


def build_metadata_row(
    artifact_key: str,
    model_id: str,
    model_downloads: int,
    variant: dict,
    emb_shape: tuple[int, int],
    sample_tag: str,
    seed: int,
    device: str,
    batch_size: int,
    cache_path: Path,
    cache_hit: bool,
    elapsed_sec: float,
) -> dict:
    return {
        "artifact_key": artifact_key,
        "model_id": model_id,
        "model_downloads": model_downloads,
        "variant_id": variant["variant_id"],
        "prefix": variant["prefix"],
        "max_length": int(variant["max_length"]),
        "normalize": bool(variant["normalize"]),
        "pooling": variant["pooling"],
        "rows": int(emb_shape[0]),
        "dims": int(emb_shape[1]),
        "sample_tag": sample_tag,
        "seed": int(seed),
        "device": device,
        "batch_size": int(batch_size),
        "cache_path": str(cache_path),
        "cache_hit": bool(cache_hit),
        "elapsed_sec": float(round(elapsed_sec, 3)),
        "updated_at": utc_now_iso(),
    }


def resolve_runtime_context(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[str], str, str]:
    df_mmlu = load_mmlu_dataframe(allow_download=args.download_mmlu)
    _, texts_sub, sample_tag = build_subsample(
        df_mmlu=df_mmlu,
        run_mode=args.run_mode,
        n_subsample=args.n_subsample,
        seed=args.seed,
    )

    top_models = fetch_top_models(
        snapshot_path=MODEL_SNAPSHOT_PATH,
        top_k=args.top_k_models,
        refresh=args.refresh_model_snapshot,
    )

    selected_models = select_models(
        top_models=top_models,
        include_models=parse_csv_list(args.models),
        max_models_run=normalize_limit(args.max_models_run),
    )

    selected_variants = select_variants(
        variant_ids=parse_csv_list(args.variant_ids),
        max_variants_run=normalize_limit(args.max_variants_run),
    )

    return selected_models, selected_variants, texts_sub, sample_tag, detect_device(args.device)


def cmd_list_models(args: argparse.Namespace) -> int:
    selected_models, selected_variants, texts_sub, sample_tag, _ = resolve_runtime_context(args)

    for model_info in selected_models:
        model_id = model_info["model_id"]
        pending = False
        for variant in selected_variants:
            cache_path = build_cache_path(
                model_id=model_id,
                variant_id=variant["variant_id"],
                sample_tag=sample_tag,
                n_rows=len(texts_sub),
                seed=args.seed,
            )
            if args.force or not cache_path.exists():
                pending = True
                break

        if args.include_existing or pending:
            print(model_id)

    return 0


def cmd_build_model(args: argparse.Namespace) -> int:
    DATA_DIR.mkdir(exist_ok=True)

    selected_models, selected_variants, texts_sub, sample_tag, device = resolve_runtime_context(args)
    model_lookup = {entry["model_id"]: entry for entry in selected_models}

    model_info = model_lookup.get(args.model_id)
    if model_info is None:
        raise RuntimeError(
            f"Model '{args.model_id}' is not in the selected model set. "
            "Use --models/--top-k-models/--max-models-run to include it."
        )

    model_downloads = int(model_info.get("downloads", -1))
    print(
        f"[build-model] model={args.model_id} variants={len(selected_variants)} "
        f"rows={len(texts_sub)} device={device}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True)
    model.to(device)
    model.eval()

    failures = []
    completed = 0

    try:
        for variant in selected_variants:
            variant_id = variant["variant_id"]
            artifact_key = f"{args.model_id}::{variant_id}"
            cache_path = build_cache_path(
                model_id=args.model_id,
                variant_id=variant_id,
                sample_tag=sample_tag,
                n_rows=len(texts_sub),
                seed=args.seed,
            )
            sidecar_path = metadata_sidecar_path(cache_path)

            started = time.time()

            try:
                if cache_path.exists() and not args.force:
                    emb = np.load(cache_path)
                    cache_hit = True
                else:
                    prefixed_texts = [variant["prefix"] + text for text in texts_sub]
                    emb = encode_variant(
                        model=model,
                        tokenizer=tokenizer,
                        texts=prefixed_texts,
                        device=device,
                        batch_size=args.batch_size,
                        max_length=variant["max_length"],
                        pooling=variant["pooling"],
                        normalize=variant["normalize"],
                    )
                    save_npy_atomic(cache_path, emb)
                    cache_hit = False

                elapsed = time.time() - started
                metadata = build_metadata_row(
                    artifact_key=artifact_key,
                    model_id=args.model_id,
                    model_downloads=model_downloads,
                    variant=variant,
                    emb_shape=emb.shape,
                    sample_tag=sample_tag,
                    seed=args.seed,
                    device=device,
                    batch_size=args.batch_size,
                    cache_path=cache_path,
                    cache_hit=cache_hit,
                    elapsed_sec=elapsed,
                )
                save_json_atomic(sidecar_path, metadata)
                completed += 1

                status = "cache" if cache_hit else "built"
                print(
                    f"  - {variant_id}: {status} | shape={emb.shape} | "
                    f"pool={variant['pooling']} | norm={variant['normalize']} | "
                    f"max_len={variant['max_length']} | {elapsed:.1f}s"
                )

            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "artifact_key": artifact_key,
                        "model_id": args.model_id,
                        "variant_id": variant_id,
                        "error": str(exc),
                    }
                )
                print(f"  - {variant_id}: FAILED ({exc})")

    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print(
        f"[build-model] completed={completed}/{len(selected_variants)} "
        f"failures={len(failures)} model={args.model_id}"
    )

    if failures and args.strict:
        raise RuntimeError(
            f"Model {args.model_id} had {len(failures)} failed variants in strict mode. "
            f"Example: {failures[0]}"
        )

    return 0


def cmd_finalize_index(args: argparse.Namespace) -> int:
    DATA_DIR.mkdir(exist_ok=True)

    selected_models, selected_variants, texts_sub, sample_tag, device = resolve_runtime_context(args)

    artifact_rows = []
    missing_artifacts = []

    for model_info in selected_models:
        model_id = model_info["model_id"]
        model_downloads = int(model_info.get("downloads", -1))

        for variant in selected_variants:
            artifact_key = f"{model_id}::{variant['variant_id']}"
            cache_path = build_cache_path(
                model_id=model_id,
                variant_id=variant["variant_id"],
                sample_tag=sample_tag,
                n_rows=len(texts_sub),
                seed=args.seed,
            )
            sidecar_path = metadata_sidecar_path(cache_path)

            if sidecar_path.exists():
                with sidecar_path.open("r", encoding="utf-8") as handle:
                    artifact_rows.append(json.load(handle))
                continue

            if cache_path.exists():
                emb = np.load(cache_path, mmap_mode="r")
                metadata = build_metadata_row(
                    artifact_key=artifact_key,
                    model_id=model_id,
                    model_downloads=model_downloads,
                    variant=variant,
                    emb_shape=(int(emb.shape[0]), int(emb.shape[1])),
                    sample_tag=sample_tag,
                    seed=args.seed,
                    device=device,
                    batch_size=args.batch_size,
                    cache_path=cache_path,
                    cache_hit=True,
                    elapsed_sec=0.0,
                )
                artifact_rows.append(metadata)
                save_json_atomic(sidecar_path, metadata)
                continue

            missing_artifacts.append(
                {
                    "artifact_key": artifact_key,
                    "model_id": model_id,
                    "variant_id": variant["variant_id"],
                    "error": "missing cache and metadata sidecar",
                }
            )

    payload = {
        "created_at": utc_now_iso(),
        "run_mode": args.run_mode,
        "sample_tag": sample_tag,
        "rows": len(texts_sub),
        "seed": args.seed,
        "top_k_models": len(selected_models),
        "variants_per_model": len(selected_variants),
        "target_artifacts": len(selected_models) * len(selected_variants),
        "completed_artifacts": len(artifact_rows),
        "failed_artifacts": missing_artifacts,
        "artifacts": artifact_rows,
    }
    save_json_atomic(EMBED_METADATA_PATH, payload)

    print(
        f"[finalize-index] completed={len(artifact_rows)} "
        f"target={len(selected_models) * len(selected_variants)} "
        f"missing={len(missing_artifacts)}"
    )
    print(f"[finalize-index] wrote {EMBED_METADATA_PATH}")

    if missing_artifacts and args.strict:
        raise RuntimeError(
            f"Missing {len(missing_artifacts)} artifacts in strict mode. "
            f"Example: {missing_artifacts[0]}"
        )

    return 0


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-mode", choices=["subsample", "full"], default="subsample")
    parser.add_argument("--n-subsample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-models", type=int, default=10)
    parser.add_argument("--refresh-model-snapshot", action="store_true")
    parser.add_argument("--max-models-run", type=int, default=0)
    parser.add_argument("--max-variants-run", type=int, default=0)
    parser.add_argument("--models", default="", help="Comma-separated model ids to include")
    parser.add_argument("--variant-ids", default="", help="Comma-separated variant ids to include")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--download-mmlu", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rebuild even if cache files already exist")
    parser.add_argument("--strict", action="store_true", help="Fail if any expected artifacts are missing")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MMLU embedding build utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_models = subparsers.add_parser("list-models", help="List selected model ids")
    add_shared_args(list_models)
    list_models.add_argument(
        "--include-existing",
        action="store_true",
        help="Include models even if all selected variants are already cached",
    )

    build_model = subparsers.add_parser(
        "build-model",
        help="Build all selected variants for one model id",
    )
    add_shared_args(build_model)
    build_model.add_argument("--model-id", required=True)

    finalize_index = subparsers.add_parser(
        "finalize-index",
        help="Write mmlu_embedding_artifacts_metadata.json from artifact sidecars",
    )
    add_shared_args(finalize_index)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list-models":
        return cmd_list_models(args)
    if args.command == "build-model":
        return cmd_build_model(args)
    if args.command == "finalize-index":
        return cmd_finalize_index(args)

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
