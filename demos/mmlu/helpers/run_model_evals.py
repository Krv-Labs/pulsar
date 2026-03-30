#!/usr/bin/env python3
"""
Offline MMLU evaluation script.

Runs the 5k-question subsample through multiple models via API and saves
per-question results as CSV. The notebook loads this CSV — users never need
API keys.

Usage:
    export OPENAI_API_KEY=...
    export ANTHROPIC_API_KEY=...
    export GROQ_API_KEY=...
    export GEMINI_API_KEY=...
    cd demos/mmlu
    uv run python helpers/run_model_evals.py

Produces: data/model_eval_results.csv
"""

import asyncio
import csv
import os
import time
from pathlib import Path

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
    "gemini-2.0-flash": {
        "provider": "gemini",
        "api_model": "gemini-2.0-flash",
        "env_var": "GEMINI_API_KEY",
        "rate_limit": 15,
    },
    "llama-3-8b": {
        "provider": "groq",
        "api_model": "llama3-8b-8192",
        "env_var": "GROQ_API_KEY",
        "rate_limit": 25,
    },
}

# ---------------------------------------------------------------------------
# General config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_CSV = DATA_DIR / "model_eval_results.csv"
PARTIAL_DIR = DATA_DIR / "eval_partial"

SYSTEM_PROMPT = (
    "You are an expert test-taker. Read the question and the choices. "
    "Output exactly one letter: A, B, C, or D. "
    "Do not output any other text or explanation."
)

ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
LETTER_SET = {"A", "B", "C", "D"}

MAX_RETRIES = 3
BATCH_SIZE = 50

console = Console()


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


# ---------------------------------------------------------------------------
# Generic API caller — dispatches by provider
# ---------------------------------------------------------------------------
async def call_model(
    prompt: str,
    semaphore: asyncio.Semaphore,
    provider: str,
    api_model: str,
    api_key: str,
) -> str:
    """Route to the correct API based on provider."""
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                if provider == "openai":
                    return await _call_openai(prompt, api_model, api_key)
                elif provider == "anthropic":
                    return await _call_anthropic(prompt, api_model, api_key)
                elif provider == "gemini":
                    return await _call_gemini(prompt, api_model, api_key)
                elif provider == "groq":
                    return await _call_groq(prompt, api_model, api_key)
                else:
                    raise ValueError(f"Unknown provider: {provider}")
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise e


async def _call_openai(prompt: str, model: str, api_key: str) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
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


async def _call_anthropic(prompt: str, model: str, api_key: str) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
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


async def _call_gemini(prompt: str, model: str, api_key: str) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
            headers={"content-type": "application/json"},
            json={
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 1, "temperature": 0},
            },
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


async def _call_groq(prompt: str, model: str, api_key: str) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
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


def save_partial(model_name: str, results: dict[int, int]):
    """Save partial results for a model."""
    PARTIAL_DIR.mkdir(exist_ok=True)
    path = PARTIAL_DIR / f"{model_name}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question_index", "is_correct"])
        writer.writeheader()
        for qi in sorted(results):
            writer.writerow({"question_index": qi, "is_correct": results[qi]})


# ---------------------------------------------------------------------------
# Eval loop for a single model
# ---------------------------------------------------------------------------
async def eval_model(
    model_name: str,
    config: dict,
    df_sub: pd.DataFrame,
    progress: Progress,
    task_id: int,
) -> dict[int, int]:
    """Evaluate a single model on all questions, with caching + progress bar."""
    results = load_partial(model_name)
    remaining = [i for i in range(len(df_sub)) if i not in results]

    # Advance progress bar for cached results
    if results:
        progress.update(task_id, advance=len(results))

    if not remaining:
        return results

    provider = config["provider"]
    api_model = config["api_model"]
    api_key = os.environ[config["env_var"]]
    semaphore = asyncio.Semaphore(config["rate_limit"])
    errors = 0

    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start:batch_start + BATCH_SIZE]

        async def process_one(qi: int) -> tuple[int, int]:
            row = df_sub.iloc[qi]
            prompt = format_prompt(row)
            try:
                response = await call_model(
                    prompt, semaphore, provider, api_model, api_key
                )
                answer = parse_answer(response)
                correct_letter = ANSWER_MAP[int(row["answer"])]
                return qi, 1 if answer == correct_letter else 0
            except Exception:
                return qi, -1

        tasks = [process_one(qi) for qi in batch]
        batch_results = await asyncio.gather(*tasks)

        batch_ok = 0
        for qi, ic in batch_results:
            if ic >= 0:
                results[qi] = ic
                batch_ok += 1
            else:
                errors += 1

        save_partial(model_name, results)
        progress.update(task_id, advance=batch_ok)

        # Update description with running accuracy
        if results:
            acc = sum(results.values()) / len(results)
            progress.update(
                task_id,
                description=f"[cyan]{model_name}[/] ({acc:.1%} acc, {errors} err)",
            )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    console.rule("[bold]MMLU Model Evaluation[/bold]")

    # Check API keys
    missing = []
    for name, cfg in MODEL_REGISTRY.items():
        if cfg["env_var"] not in os.environ:
            missing.append(f"  {cfg['env_var']:25s} (for {name})")
    if missing:
        console.print("\n[red bold]Missing environment variables:[/]")
        for m in missing:
            console.print(m)
        console.print("\nSet them and re-run.")
        return

    console.print("\nLoading 5k subsample (must match notebook exactly)...")
    df_sub = load_subsample()
    console.print(
        f"  [green]{len(df_sub)}[/] questions, "
        f"[green]{df_sub['subject'].nunique()}[/] subjects\n"
    )

    t0 = time.time()
    all_results = {}

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
        for name in MODEL_REGISTRY:
            tid = progress.add_task(
                f"[cyan]{name}[/]",
                total=len(df_sub),
            )
            model_tasks[name] = tid

        # Evaluate models sequentially (each model is internally parallel)
        for name, cfg in MODEL_REGISTRY.items():
            all_results[name] = await eval_model(
                name, cfg, df_sub, progress, model_tasks[name]
            )

    elapsed = time.time() - t0

    # Merge into final CSV
    rows = []
    for model_name, results in all_results.items():
        for qi in sorted(results):
            rows.append({
                "question_index": qi,
                "model_name": model_name,
                "is_correct": results[qi],
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    # Summary table
    console.print()
    console.rule("[bold]Results[/bold]")
    console.print(f"\nCompleted in [green]{elapsed:.0f}s[/]")
    console.print(f"Saved: [blue]{OUTPUT_CSV}[/]")
    console.print(
        f"  {len(out_df)} rows "
        f"({len(out_df) // len(MODEL_REGISTRY)} questions "
        f"x {len(MODEL_REGISTRY)} models)\n"
    )

    table = Table(title="Overall Accuracy")
    table.add_column("Model", style="cyan")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")

    for model_name in MODEL_REGISTRY:
        model_df = out_df[out_df["model_name"] == model_name]
        acc = model_df["is_correct"].mean()
        correct = model_df["is_correct"].sum()
        total = len(model_df)
        table.add_row(model_name, f"{acc:.1%}", str(int(correct)), str(total))

    console.print(table)


if __name__ == "__main__":
    asyncio.run(main())
