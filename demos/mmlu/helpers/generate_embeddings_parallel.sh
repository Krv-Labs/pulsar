#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Generate MMLU embeddings in parallel across models.

Usage:
  helpers/generate_embeddings_parallel.sh [options]

Options:
  --jobs N                    Parallel model workers (default: 2)
  --run-mode MODE             subsample|full (default: subsample)
  --n-subsample N             Subsample size when run-mode=subsample (default: 5000)
  --seed N                    Sampling/cache seed (default: 42)
  --top-k-models N            Number of top HF models to target (default: 10)
  --batch-size N              Embedding batch size per worker (default: 64)
  --models CSV                Comma-separated model ids to include
  --variant-ids CSV           Comma-separated variant ids to include (e.g. v01,v02)
  --max-models-run N          Keep only first N selected models
  --max-variants-run N        Keep only first N selected variants
  --device MODE               auto|cpu|cuda|mps (default: auto)
  --refresh-model-snapshot    Re-fetch top models list from Hugging Face
  --download-mmlu             Download MMLU if data/mmlu_questions.csv is missing
  --include-existing          Include models that are already fully cached
  --force                     Rebuild embeddings even if caches already exist
  --strict                    Fail if any expected artifacts are missing (default)
  --no-strict                 Allow partial completion
  -h, --help                  Show help
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/build_mmlu_embeddings.py"

JOBS=2
RUN_MODE="subsample"
N_SUBSAMPLE=5000
SEED=42
TOP_K_MODELS=10
BATCH_SIZE=64
MAX_MODELS_RUN=0
MAX_VARIANTS_RUN=0
MODELS=""
VARIANT_IDS=""
DEVICE="auto"
REFRESH_MODEL_SNAPSHOT=0
DOWNLOAD_MMLU=0
INCLUDE_EXISTING=0
FORCE=0
STRICT=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --run-mode)
      RUN_MODE="$2"
      shift 2
      ;;
    --n-subsample)
      N_SUBSAMPLE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --top-k-models)
      TOP_K_MODELS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --models)
      MODELS="$2"
      shift 2
      ;;
    --variant-ids)
      VARIANT_IDS="$2"
      shift 2
      ;;
    --max-models-run)
      MAX_MODELS_RUN="$2"
      shift 2
      ;;
    --max-variants-run)
      MAX_VARIANTS_RUN="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --refresh-model-snapshot)
      REFRESH_MODEL_SNAPSHOT=1
      shift
      ;;
    --download-mmlu)
      DOWNLOAD_MMLU=1
      shift
      ;;
    --include-existing)
      INCLUDE_EXISTING=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --strict)
      STRICT=1
      shift
      ;;
    --no-strict)
      STRICT=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$JOBS" -lt 1 ]]; then
  echo "--jobs must be >= 1" >&2
  exit 1
fi

COMMON_ARGS=(
  --run-mode "$RUN_MODE"
  --n-subsample "$N_SUBSAMPLE"
  --seed "$SEED"
  --top-k-models "$TOP_K_MODELS"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
)

if [[ "$MAX_MODELS_RUN" -gt 0 ]]; then
  COMMON_ARGS+=(--max-models-run "$MAX_MODELS_RUN")
fi
if [[ "$MAX_VARIANTS_RUN" -gt 0 ]]; then
  COMMON_ARGS+=(--max-variants-run "$MAX_VARIANTS_RUN")
fi
if [[ -n "$MODELS" ]]; then
  COMMON_ARGS+=(--models "$MODELS")
fi
if [[ -n "$VARIANT_IDS" ]]; then
  COMMON_ARGS+=(--variant-ids "$VARIANT_IDS")
fi
if [[ "$REFRESH_MODEL_SNAPSHOT" -eq 1 ]]; then
  COMMON_ARGS+=(--refresh-model-snapshot)
fi
if [[ "$DOWNLOAD_MMLU" -eq 1 ]]; then
  COMMON_ARGS+=(--download-mmlu)
fi

LIST_ARGS=()
if [[ "$INCLUDE_EXISTING" -eq 1 ]]; then
  LIST_ARGS+=(--include-existing)
fi
if [[ "$FORCE" -eq 1 ]]; then
  LIST_ARGS+=(--force)
fi

BUILD_ARGS=()
if [[ "$FORCE" -eq 1 ]]; then
  BUILD_ARGS+=(--force)
fi
if [[ "$STRICT" -eq 1 ]]; then
  BUILD_ARGS+=(--strict)
fi

FINALIZE_ARGS=()
if [[ "$STRICT" -eq 1 ]]; then
  FINALIZE_ARGS+=(--strict)
fi

TASK_FILE="$(mktemp)"
trap 'rm -f "$TASK_FILE"' EXIT

cd "$DEMO_DIR"

echo "Listing target models..."
uv run python "$PY_SCRIPT" list-models "${COMMON_ARGS[@]}" "${LIST_ARGS[@]}" > "$TASK_FILE"

MODEL_COUNT="$(grep -cve '^\s*$' "$TASK_FILE" || true)"
echo "Queued models: $MODEL_COUNT"

FAIL=0
ACTIVE=0

if [[ "$MODEL_COUNT" -gt 0 ]]; then
  while IFS= read -r MODEL_ID; do
    [[ -z "$MODEL_ID" ]] && continue

    (
      echo "[start] $MODEL_ID"
      uv run python "$PY_SCRIPT" build-model \
        --model-id "$MODEL_ID" \
        "${COMMON_ARGS[@]}" \
        "${BUILD_ARGS[@]}"
      echo "[done] $MODEL_ID"
    ) &

    ACTIVE=$((ACTIVE + 1))
    if [[ "$ACTIVE" -ge "$JOBS" ]]; then
      if ! wait -n; then
        FAIL=1
      fi
      ACTIVE=$((ACTIVE - 1))
    fi
  done < "$TASK_FILE"

  while [[ "$ACTIVE" -gt 0 ]]; do
    if ! wait -n; then
      FAIL=1
    fi
    ACTIVE=$((ACTIVE - 1))
  done
fi

echo "Finalizing embedding metadata index..."
if ! uv run python "$PY_SCRIPT" finalize-index "${COMMON_ARGS[@]}" "${FINALIZE_ARGS[@]}"; then
  FAIL=1
fi

if [[ "$FAIL" -ne 0 ]]; then
  echo "Embedding generation completed with failures." >&2
  exit 1
fi

echo "Embedding generation completed successfully."
