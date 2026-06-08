#!/usr/bin/env bash
set -euo pipefail

ATLAS_PATH="/lustre/groups/ml01/workspace/hpca/hpca_downstream/2026_final_object_healthy_hvg.h5ad"
QUERY_DIR="/lustre/groups/ml01/datasets/projects/20230301_Sander_SpatialPancreas_sara.jimenez/spatial"
OUT_DIR="/lustre/groups/ml01/workspace/hpca/hpca_downstream/label_transfer_spatial/TACCO/spatialpancreas_tacco_healthy"
LABEL_KEYS="Level_3 Level_4"
METHOD="OT"
COUNTS_LAYER="counts"
MAX_PARALLEL="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --atlas_path) ATLAS_PATH="$2"; shift 2 ;;
    --query_dir) QUERY_DIR="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --label_keys) LABEL_KEYS="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --counts_layer) COUNTS_LAYER="$2"; shift 2 ;;
    --max_parallel) MAX_PARALLEL="$2"; shift 2 ;;
    *)
      echo "ERROR: unknown argument: $1"
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/tacco_label_transfer.py"
PYTHON_BIN="${SCRIPT_DIR}/.conda-env/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: TACCO environment python not found at $PYTHON_BIN"
  echo "Run: bash ${SCRIPT_DIR}/create_tacco_env.sh"
  exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "ERROR: script not found: $PYTHON_SCRIPT"
  exit 1
fi

if [[ ! -f "$ATLAS_PATH" ]]; then
  echo "ERROR: atlas file not found: $ATLAS_PATH"
  exit 1
fi

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/status" "$OUT_DIR/csv" "$OUT_DIR/h5ad"

QUERIES=()
if [[ -f "$QUERY_DIR" && "$QUERY_DIR" == *.h5ad ]]; then
  QUERIES+=("$QUERY_DIR")
elif [[ -d "$QUERY_DIR" ]]; then
  for file in "$QUERY_DIR"/*.h5ad; do
    [[ -f "$file" ]] && QUERIES+=("$file")
  done
else
  echo "ERROR: query_dir must be a .h5ad file or a directory containing .h5ad files."
  exit 1
fi

if [[ ${#QUERIES[@]} -eq 0 ]]; then
  echo "ERROR: no .h5ad files found in $QUERY_DIR"
  exit 1
fi

echo "Total query files: ${#QUERIES[@]}"
echo "Parallel jobs: $MAX_PARALLEL"

active_pids=()
active_samples=()
failed=()

launch_job() {
  local query_path="$1"
  local sample
  sample="$(basename "$query_path" .h5ad)"
  local done_marker="${OUT_DIR}/status/${sample}.done"
  local log_file="${OUT_DIR}/logs/${sample}.log"

  if [[ -f "$done_marker" ]]; then
    echo "Skipping completed sample: $sample"
    return 0
  fi

  echo "Starting $sample"
  "$PYTHON_BIN" "$PYTHON_SCRIPT" \
    --atlas_path "$ATLAS_PATH" \
    --query_path "$query_path" \
    --out_dir "$OUT_DIR" \
    --sample_name "$sample" \
    --label_keys $LABEL_KEYS \
    --method "$METHOD" \
    --counts_layer "$COUNTS_LAYER" \
    --assume_valid_counts \
    >"$log_file" 2>&1 &

  active_pids+=("$!")
  active_samples+=("$sample")
}

reap_finished() {
  local new_pids=()
  local new_samples=()
  for i in "${!active_pids[@]}"; do
    local pid="${active_pids[$i]}"
    local sample="${active_samples[$i]}"
    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=("$pid")
      new_samples+=("$sample")
      continue
    fi
    if wait "$pid"; then
      echo "Finished $sample"
    else
      echo "FAILED $sample (see ${OUT_DIR}/logs/${sample}.log)"
      failed+=("$sample")
    fi
  done
  active_pids=("${new_pids[@]}")
  active_samples=("${new_samples[@]}")
}

for query in "${QUERIES[@]}"; do
  while [[ ${#active_pids[@]} -ge $MAX_PARALLEL ]]; do
    reap_finished
    sleep 2
  done
  launch_job "$query"
done

while [[ ${#active_pids[@]} -gt 0 ]]; do
  reap_finished
  sleep 2
done

echo "TACCO run complete."
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed samples (${#failed[@]}): ${failed[*]}"
  exit 1
fi
