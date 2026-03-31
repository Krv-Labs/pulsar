#!/usr/bin/env python3
"""
Offline MMLU evaluation script.

Runs the 5k-question subsample through multiple models via API and saves
per-question results as CSV. The notebook loads this CSV — users never need
API keys.

Usage:
    export OPENAI_API_KEY=...
    export ANTHROPIC_API_KEY=...
    export GEMINI_API_KEY=...
    export XAI_API_KEY=...
    cd demos/mmlu
    uv run python helpers/run_model_evals.py

Produces: data/model_eval_results.csv
"""

import asyncio
import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# ---------------------------------------------------------------------------
# Models — edit this to add/remove models
# ---------------------------------------------------------------------------
# Each entry: display name -> (provider, api_model_id, env_var, rate_limit)
MODEL_REGISTRY = {
    "gpt-4o-mini": {
        "provider": "openai",
        "api_model": "gpt-4o-mini",
        "env_var": "OPENAI_API_KEY",
        "rate_limit": 20,
    },
    "claude-3-haiku": {
        "provider": "anthropic",
        "api_model": "claude-3-haiku-20240307",
        "env_var": "ANTHROPIC_API_KEY",
        "rate_limit": 15,
    },
    "claude-haiku-4-5": {
        "provider": "anthropic",
        "api_model": "claude-haiku-4-5-20251001",
        "env_var": "ANTHROPIC_API_KEY",
        "rate_limit": 15,
    },
    "gemini-2.5-flash": {
        "provider": "gemini",
        "api_model": "gemini-2.5-flash",
        "env_var": "GEMINI_API_KEY",
        "rate_limit": 15,
    },
    "grok-fast": {
        "provider": "xai",
        "api_model": "grok-4.20-beta-latest-non-reasoning",
        "env_var": "XAI_API_KEY",
        "rate_limit": 15,
    },
}

# ---------------------------------------------------------------------------
# General config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_CSV = DATA_DIR / "model_eval_results.csv"
PARTIAL_DIR = DATA_DIR / "eval_partial"
FINAL_DIR = DATA_DIR / "eval_results"

SYSTEM_PROMPT = (
    "You are an expert test-taker. Read the question and the choices. "
    "Output exactly one letter: A, B, C, or D. "
    "Do not output any other text or explanation."
)

ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
LETTER_SET = {"A", "B", "C", "D"}

BATCH_SIZE = 50
DEFAULT_REQUEST_TIMEOUT = 15.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_FAIL_FAST_ERRORS = 20

console = Console()


@dataclass
class ModelRunResult:
    model_name: str
    status: str
    results: dict[int, int]
    errors: int = 0
    message: str = ""
    first_error: str = ""


# ---------------------------------------------------------------------------
# Build the subsample (must match notebook exactly)
# ---------------------------------------------------------------------------
def load_subsample() -> pd.DataFrame:
    """Load MMLU and produce the same 5k subsample the notebook uses."""
    mmlu_cache = DATA_DIR / "mmlu_questions.csv"
    if not mmlu_cache.exists():
        raise FileNotFoundError(
            f"{mmlu_cache} not found. Run the notebook through Cell 4 first "
            "to download and cache the MMLU dataset."
        )

    df = pd.read_csv(mmlu_cache)
    n_sub = 5000
    rng = np.random.default_rng(42)

    subjects = df["subject"].unique()
    n_total = len(df)
    indices = []

    for subj in subjects:
        subj_idx = df.index[df["subject"] == subj].tolist()
        n_want = max(10, int(round(len(subj_idx) / n_total * n_sub)))
        chosen = rng.choice(subj_idx, size=min(n_want, len(subj_idx)), replace=False)
        indices.extend(chosen.tolist())

    rng.shuffle(indices)
    if len(indices) > n_sub:
        indices = indices[:n_sub]
    elif len(indices) < n_sub:
        remaining = list(set(range(n_total)) - set(indices))
        extra = rng.choice(remaining, size=n_sub - len(indices), replace=False)
        indices.extend(extra.tolist())

    indices.sort()
    df_sub = df.iloc[indices].copy()
    df_sub = df_sub.reset_index(drop=True)
    return df_sub


def format_prompt(row: pd.Series) -> str:
    """Build the user prompt for a single MMLU question."""
    choices_raw = row["choices"]
    if isinstance(choices_raw, str):
        parts = choices_raw.split("|||")
    else:
        parts = list(choices_raw)

    lines = [f"Question: {row['question']}"]
    for i, label in enumerate(["A", "B", "C", "D"]):
        if i < len(parts):
            lines.append(f"{label}) {parts[i].strip()}")
    return "\n".join(lines)


def parse_answer(response: str) -> str | None:
    """Extract a single letter (A/B/C/D) from model response."""
    text = response.strip().upper()
    if text and text[0] in LETTER_SET:
        return text[0]
    for ch in text:
        if ch in LETTER_SET:
            return ch
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MMLU model evaluations with per-model resume and caching."
    )
    parser.add_argument(
        "--overwrite",
        choices=["none", "all"],
        default="none",
        help="Overwrite policy for per-model finalized results.",
    )
    parser.add_argument(
        "--overwrite-models",
        default="",
        help="Comma-separated model names to overwrite. Applies even when --overwrite=none.",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model names to run (allowlist). Defaults to all.",
    )
    parser.add_argument(
        "--exclude-models",
        default="",
        help="Comma-separated model names to skip after --models is applied.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available model names and exit.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum attempts per request, including the first try.",
    )
    parser.add_argument(
        "--fail-fast-errors",
        type=int,
        default=DEFAULT_FAIL_FAST_ERRORS,
        help="Abort a model after this many consecutive request errors. Use 0 to disable.",
    )
    return parser.parse_args()


def parse_name_list(raw: str) -> set[str]:
    return {name.strip() for name in raw.split(",") if name.strip()}


def parse_overwrite_models(raw: str) -> set[str]:
    return parse_name_list(raw)


def select_models(
    model_registry: dict[str, dict],
    include_raw: str,
    exclude_raw: str,
) -> dict[str, dict]:
    include = parse_name_list(include_raw)
    exclude = parse_name_list(exclude_raw)
    known = set(model_registry)

    unknown_include = include - known
    if unknown_include:
        raise ValueError(
            f"Unknown --models entries: {', '.join(sorted(unknown_include))}"
        )
    unknown_exclude = exclude - known
    if unknown_exclude:
        raise ValueError(
            f"Unknown --exclude-models entries: {', '.join(sorted(unknown_exclude))}"
        )

    selected_names = list(model_registry.keys())
    if include:
        selected_names = [name for name in selected_names if name in include]
    if exclude:
        selected_names = [name for name in selected_names if name not in exclude]
    if not selected_names:
        raise ValueError("No models selected after applying --models/--exclude-models.")

    return {name: model_registry[name] for name in selected_names}


def compute_sample_hash(df_sub: pd.DataFrame) -> str:
    """Build a stable hash of the sampled question set."""
    hasher = hashlib.sha256()
    cols = ["question", "choices", "answer", "subject"]
    for row in df_sub[cols].itertuples(index=False, name=None):
        for value in row:
            hasher.update(str(value).encode("utf-8"))
            hasher.update(b"\x1f")
        hasher.update(b"\x1e")
    return hasher.hexdigest()


def format_exception(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= 200 else text[:197] + "..."


def compact_error_message(message: str, max_len: int = 90) -> str:
    single_line = " ".join(message.split())
    if len(single_line) <= max_len:
        return single_line
    return single_line[: max_len - 3] + "..."


def _extract_gemini_text(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        prompt_feedback = payload.get("promptFeedback")
        raise RuntimeError(
            f"Gemini returned no candidates. promptFeedback={prompt_feedback!r}"
        )

    candidate = candidates[0]
    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    for part in parts:
        text = part.get("text")
        if text:
            return text

    finish_reason = candidate.get("finishReason")
    safety_ratings = candidate.get("safetyRatings")
    prompt_feedback = payload.get("promptFeedback")
    raise RuntimeError(
        "Gemini candidate had no text parts. "
        f"finishReason={finish_reason!r}, "
        f"promptFeedback={prompt_feedback!r}, "
        f"safetyRatings={safety_ratings!r}, "
        f"content_keys={sorted(content.keys())!r}"
    )


# ---------------------------------------------------------------------------
# Generic API caller — dispatches by provider
# ---------------------------------------------------------------------------
async def call_model(
    prompt: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    provider: str,
    api_model: str,
    api_key: str,
    max_retries: int,
) -> str:
    """Route to the correct API based on provider."""
    for attempt in range(max_retries):
        try:
            async with semaphore:
                if provider == "openai":
                    return await _call_openai(client, prompt, api_model, api_key)
                elif provider == "anthropic":
                    return await _call_anthropic(client, prompt, api_model, api_key)
                elif provider == "gemini":
                    return await _call_gemini(client, prompt, api_model, api_key)
                elif provider == "xai":
                    return await _call_xai(client, prompt, api_model, api_key)
                else:
                    raise ValueError(f"Unknown provider: {provider}")
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)
            else:
                raise e


async def _call_openai(
    client: httpx.AsyncClient, prompt: str, model: str, api_key: str
) -> str:
    resp = await client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1,
            "temperature": 0,
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _call_anthropic(
    client: httpx.AsyncClient, prompt: str, model: str, api_key: str
) -> str:
    resp = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 1,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


async def _call_gemini(
    client: httpx.AsyncClient, prompt: str, model: str, api_key: str
) -> str:
    resp = await client.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={
            "content-type": "application/json",
            "x-goog-api-key": api_key,
        },
        json={
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 8,
                "temperature": 0,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        },
    )
    resp.raise_for_status()
    return _extract_gemini_text(resp.json())


async def _call_xai(
    client: httpx.AsyncClient, prompt: str, model: str, api_key: str
) -> str:
    resp = await client.post(
        "https://api.x.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            # Stable conversation id improves xAI prompt-cache reuse for
            # repeated shared prefixes like our system prompt.
            "x-grok-conv-id": f"mmlu-eval-{model}",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1,
            "temperature": 0,
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Partial result caching
# ---------------------------------------------------------------------------
def load_partial(model_name: str) -> dict[int, int]:
    """Load partial results for a model (question_index -> is_correct)."""
    path = PARTIAL_DIR / f"{model_name}.csv"
    if not path.exists():
        return {}
    results = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[int(row["question_index"])] = int(row["is_correct"])
    return results


def load_final(model_name: str) -> dict[int, int]:
    """Load finalized results for a model (question_index -> is_correct)."""
    path = FINAL_DIR / f"{model_name}.csv"
    if not path.exists():
        return {}
    results = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[int(row["question_index"])] = int(row["is_correct"])
    return results


def _write_results_csv_atomic(path: Path, results: dict[int, int]):
    path.parent.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, newline="", dir=path.parent, prefix=f".{path.name}."
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=["question_index", "is_correct"])
        writer.writeheader()
        for qi in sorted(results):
            writer.writerow({"question_index": qi, "is_correct": results[qi]})
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def save_partial(model_name: str, results: dict[int, int]):
    """Save partial results for a model."""
    _write_results_csv_atomic(PARTIAL_DIR / f"{model_name}.csv", results)


def save_final(model_name: str, results: dict[int, int]):
    """Save finalized results for a model."""
    _write_results_csv_atomic(FINAL_DIR / f"{model_name}.csv", results)


def save_final_meta(
    model_name: str,
    config: dict,
    sample_hash: str,
    n_questions: int,
):
    FINAL_DIR.mkdir(exist_ok=True)
    path = FINAL_DIR / f"{model_name}.meta.json"
    payload = {
        "version": 1,
        "model_name": model_name,
        "provider": config["provider"],
        "api_model": config["api_model"],
        "sample_hash": sample_hash,
        "n_questions": n_questions,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, prefix=f".{path.name}."
    ) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def is_reusable_final(
    model_name: str,
    config: dict,
    sample_hash: str,
    n_questions: int,
) -> bool:
    result_path = FINAL_DIR / f"{model_name}.csv"
    meta_path = FINAL_DIR / f"{model_name}.meta.json"
    if not (result_path.exists() and meta_path.exists()):
        return False

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    if meta.get("provider") != config["provider"]:
        return False
    if meta.get("api_model") != config["api_model"]:
        return False
    if meta.get("sample_hash") != sample_hash:
        return False
    if meta.get("n_questions") != n_questions:
        return False

    results = load_final(model_name)
    return len(results) == n_questions


def list_reusable_final_models(
    sample_hash: str,
    n_questions: int,
) -> dict[str, dict]:
    """Discover all finalized model runs on disk that are reusable for this sample."""
    discovered: dict[str, dict] = {}
    FINAL_DIR.mkdir(exist_ok=True)

    for meta_path in sorted(FINAL_DIR.glob("*.meta.json")):
        model_name = meta_path.name[: -len(".meta.json")]
        csv_path = FINAL_DIR / f"{model_name}.csv"
        if not csv_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("sample_hash") != sample_hash:
            continue
        if meta.get("n_questions") != n_questions:
            continue
        results = load_final(model_name)
        if len(results) != n_questions:
            continue
        discovered[model_name] = {
            "provider": meta.get("provider", "unknown"),
            "api_model": meta.get("api_model", model_name),
        }

    # Backward-compatible fallback: include legacy finalized CSVs without meta
    # if they match the expected question count and are not already covered.
    for csv_path in sorted(FINAL_DIR.glob("*.csv")):
        model_name = csv_path.stem
        if model_name in discovered:
            continue
        results = load_final(model_name)
        if len(results) != n_questions:
            continue
        discovered[model_name] = {
            "provider": "legacy",
            "api_model": model_name,
        }

    return discovered


def write_available_aggregate_csv(
    sample_hash: str,
    n_questions: int,
) -> pd.DataFrame:
    """Write aggregate CSV from all reusable finalized model outputs currently on disk."""
    rows = []
    for model_name in list_reusable_final_models(sample_hash, n_questions):
        results = load_final(model_name)
        for qi in sorted(results):
            rows.append(
                {
                    "question_index": qi,
                    "model_name": model_name,
                    "is_correct": int(results[qi]),
                }
            )

    out_df = pd.DataFrame(rows, columns=["question_index", "model_name", "is_correct"])
    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    return out_df


# ---------------------------------------------------------------------------
# Eval loop for a single model
# ---------------------------------------------------------------------------
async def eval_model(
    model_name: str,
    config: dict,
    df_sub: pd.DataFrame,
    progress: Progress,
    task_id: int,
    request_timeout: float,
    max_retries: int,
    fail_fast_errors: int,
    use_partial: bool = True,
) -> tuple[dict[int, int], int, str]:
    """Evaluate a single model on all questions, with caching + progress bar."""
    results = load_partial(model_name) if use_partial else {}
    remaining = [i for i in range(len(df_sub)) if i not in results]
    errors = 0
    consecutive_errors = 0
    first_error = ""

    def update_status() -> None:
        completed = len(results)
        attempted = progress.tasks[task_id].completed
        if completed:
            acc = sum(results.values()) / completed
            detail = f"{int(attempted)}/{len(df_sub)} done, {acc:.1%} acc, {errors} err"
        else:
            detail = f"{int(attempted)}/{len(df_sub)} done, {errors} err"
        progress.update(task_id, description=f"[cyan]{model_name}[/] ({detail})")

    # Advance progress bar for cached results
    if results:
        progress.update(task_id, advance=len(results))
    update_status()

    if not remaining:
        return results, 0, ""

    provider = config["provider"]
    api_model = config["api_model"]
    api_key = os.environ[config["env_var"]]
    semaphore = asyncio.Semaphore(config["rate_limit"])
    timeout = httpx.Timeout(request_timeout)
    limits = httpx.Limits(max_connections=config["rate_limit"] * 2)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for batch_start in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[batch_start : batch_start + BATCH_SIZE]

            async def process_one(qi: int) -> tuple[int, int, str]:
                row = df_sub.iloc[qi]
                prompt = format_prompt(row)
                try:
                    response = await call_model(
                        prompt,
                        client,
                        semaphore,
                        provider,
                        api_model,
                        api_key,
                        max_retries,
                    )
                    answer = parse_answer(response)
                    correct_letter = ANSWER_MAP[int(row["answer"])]
                    return qi, 1 if answer == correct_letter else 0, ""
                except Exception as exc:
                    return qi, -1, format_exception(exc)

            tasks = [asyncio.create_task(process_one(qi)) for qi in batch]

            try:
                for future in asyncio.as_completed(tasks):
                    qi, ic, err_msg = await future
                    progress.update(task_id, advance=1)
                    if ic >= 0:
                        results[qi] = ic
                        consecutive_errors = 0
                    else:
                        errors += 1
                        consecutive_errors += 1
                        if not first_error:
                            first_error = err_msg
                        if (
                            fail_fast_errors > 0
                            and consecutive_errors >= fail_fast_errors
                        ):
                            raise RuntimeError(
                                f"aborting after {consecutive_errors} consecutive errors; "
                                f"first error: {first_error or 'unknown'}"
                            )
                    update_status()
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            save_partial(model_name, results)

    return results, errors, first_error


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run_one_model(
    model_name: str,
    config: dict,
    df_sub: pd.DataFrame,
    sample_hash: str,
    aggregate_lock: asyncio.Lock,
    progress: Progress,
    task_id: int,
    request_timeout: float,
    max_retries: int,
    fail_fast_errors: int,
    overwrite_all: bool,
    overwrite_models: set[str],
) -> ModelRunResult:
    """Run one model with resume/overwrite behavior and per-model persistence."""

    def log_full_error(prefix: str, message: str) -> None:
        if message:
            progress.console.print(f"[red]{model_name} {prefix}:[/] {message}")

    if config["env_var"] not in os.environ:
        progress.update(
            task_id,
            description=f"[yellow]{model_name}[/] (missing {config['env_var']})",
        )
        return ModelRunResult(
            model_name=model_name,
            status="missing-key",
            results={},
            message=f"missing {config['env_var']}",
        )

    should_overwrite = overwrite_all or model_name in overwrite_models
    if not should_overwrite and is_reusable_final(
        model_name, config, sample_hash, len(df_sub)
    ):
        results = load_final(model_name)
        progress.update(
            task_id,
            completed=len(df_sub),
            description=f"[green]{model_name}[/] (reused cached final)",
        )
        return ModelRunResult(
            model_name=model_name,
            status="skipped-existing",
            results=results,
        )

    try:
        results, errors, first_error = await eval_model(
            model_name,
            config,
            df_sub,
            progress,
            task_id,
            request_timeout=request_timeout,
            max_retries=max_retries,
            fail_fast_errors=fail_fast_errors,
            use_partial=not should_overwrite,
        )
    except Exception as e:
        message = format_exception(e)
        progress.update(
            task_id,
            description=(
                f"[red]{model_name}[/] " f"(failed: {compact_error_message(message)})"
            ),
        )
        log_full_error("failed", message)
        return ModelRunResult(
            model_name=model_name,
            status="failed",
            results={},
            message=message,
            first_error=message,
        )

    if len(results) == len(df_sub):
        save_final(model_name, results)
        save_final_meta(model_name, config, sample_hash, len(df_sub))
        async with aggregate_lock:
            write_available_aggregate_csv(sample_hash, len(df_sub))
        progress.update(task_id, description=f"[green]{model_name}[/] (completed)")
        return ModelRunResult(
            model_name=model_name,
            status="completed",
            results=results,
            errors=errors,
            first_error=first_error,
        )

    incomplete_detail = f"{len(results)}/{len(df_sub)} complete"
    if first_error:
        incomplete_detail += f"; first error: {compact_error_message(first_error)}"
    progress.update(
        task_id,
        description=f"[yellow]{model_name}[/] (incomplete: {incomplete_detail})",
    )
    log_full_error("first error", first_error)
    return ModelRunResult(
        model_name=model_name,
        status="incomplete",
        results=results,
        errors=errors,
        message=f"{len(results)}/{len(df_sub)} questions complete",
        first_error=first_error,
    )


async def main(args: argparse.Namespace):
    if args.list_models:
        console.print("Available models:")
        for model_name in MODEL_REGISTRY:
            console.print(f"  - {model_name}")
        return

    try:
        selected_registry = select_models(
            MODEL_REGISTRY, args.models, args.exclude_models
        )
    except ValueError as e:
        console.print(f"[red bold]Invalid model selection:[/] {e}")
        return
    if args.request_timeout <= 0:
        console.print(
            "[red bold]Invalid runtime settings:[/] --request-timeout must be > 0"
        )
        return
    if args.max_retries < 1:
        console.print(
            "[red bold]Invalid runtime settings:[/] --max-retries must be >= 1"
        )
        return
    if args.fail_fast_errors < 0:
        console.print(
            "[red bold]Invalid runtime settings:[/] --fail-fast-errors must be >= 0"
        )
        return

    console.rule("[bold]MMLU Model Evaluation[/bold]")
    console.print("Selected models: " + ", ".join(selected_registry.keys()))
    console.print(
        "Runtime settings: "
        f"timeout={args.request_timeout:.1f}s, "
        f"retries={args.max_retries}, "
        f"fail_fast_errors={args.fail_fast_errors}"
    )

    console.print("\nLoading 5k subsample (must match notebook exactly)...")
    df_sub = load_subsample()
    console.print(
        f"  [green]{len(df_sub)}[/] questions, "
        f"[green]{df_sub['subject'].nunique()}[/] subjects\n"
    )
    sample_hash = compute_sample_hash(df_sub)
    existing_out_df = write_available_aggregate_csv(sample_hash, len(df_sub))
    if not existing_out_df.empty:
        console.print(
            f"Seeded aggregate CSV with [green]{len(existing_out_df)}[/] rows "
            f"from available finalized model runs."
        )

    overwrite_models = parse_overwrite_models(args.overwrite_models)
    unknown = overwrite_models - set(selected_registry)
    if unknown:
        console.print(
            f"[yellow]Ignoring unknown --overwrite-models entries:[/] "
            f"{', '.join(sorted(unknown))}"
        )
        overwrite_models -= unknown

    t0 = time.time()
    model_status: dict[str, ModelRunResult] = {}
    aggregate_lock = asyncio.Lock()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        # Create a progress task for each model
        model_tasks = {}
        for name in selected_registry:
            tid = progress.add_task(
                f"[cyan]{name}[/]",
                total=len(df_sub),
            )
            model_tasks[name] = tid

        task_map: dict[str, asyncio.Task[ModelRunResult]] = {}
        if hasattr(asyncio, "TaskGroup"):
            async with asyncio.TaskGroup() as tg:
                for name, cfg in selected_registry.items():
                    task_map[name] = tg.create_task(
                        run_one_model(
                            name,
                            cfg,
                            df_sub,
                            sample_hash,
                            aggregate_lock,
                            progress,
                            model_tasks[name],
                            request_timeout=args.request_timeout,
                            max_retries=args.max_retries,
                            fail_fast_errors=args.fail_fast_errors,
                            overwrite_all=(args.overwrite == "all"),
                            overwrite_models=overwrite_models,
                        )
                    )
            model_status = {name: task.result() for name, task in task_map.items()}
        else:
            coros = [
                run_one_model(
                    name,
                    cfg,
                    df_sub,
                    sample_hash,
                    aggregate_lock,
                    progress,
                    model_tasks[name],
                    request_timeout=args.request_timeout,
                    max_retries=args.max_retries,
                    fail_fast_errors=args.fail_fast_errors,
                    overwrite_all=(args.overwrite == "all"),
                    overwrite_models=overwrite_models,
                )
                for name, cfg in selected_registry.items()
            ]
            results = await asyncio.gather(*coros)
            model_status = {result.model_name: result for result in results}

    elapsed = time.time() - t0

    out_df = write_available_aggregate_csv(sample_hash, len(df_sub))
    models_with_results = out_df["model_name"].nunique() if not out_df.empty else 0

    # Summary table
    console.print()
    console.rule("[bold]Results[/bold]")
    console.print(f"\nCompleted in [green]{elapsed:.0f}s[/]")
    console.print(f"Saved: [blue]{OUTPUT_CSV}[/]")
    console.print(
        f"  {len(out_df)} rows "
        f"({models_with_results} models with finalized results)\n"
    )

    table = Table(title="Overall Accuracy")
    table.add_column("Model", style="cyan")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")

    for model_name in selected_registry:
        model_df = out_df[out_df["model_name"] == model_name]
        if model_df.empty:
            table.add_row(model_name, "-", "-", "0")
            continue
        acc = model_df["is_correct"].mean()
        correct = model_df["is_correct"].sum()
        total = len(model_df)
        table.add_row(model_name, f"{acc:.1%}", str(int(correct)), str(total))

    console.print(table)

    status_table = Table(title="Run Status")
    status_table.add_column("Model", style="cyan")
    status_table.add_column("Status")
    status_table.add_column("Details")
    for model_name in selected_registry:
        status = model_status.get(model_name)
        if status is None:
            status_table.add_row(model_name, "unknown", "")
            continue
        details = status.message or status.first_error
        status_table.add_row(model_name, status.status, details)
    console.print(status_table)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
