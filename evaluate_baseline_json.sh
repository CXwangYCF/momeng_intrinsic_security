#!/usr/bin/env bash

set -Eeuo pipefail

WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$WORKSPACE/.venv/bin/python"
ENV_FILE="$WORKSPACE/.env"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

DATA="${MUSLR_DATA:-$WORKSPACE/data}"
OUTPUT_DIR="${MUSLR_OUTPUT_DIR:-$WORKSPACE/outputs}"
PREDICTIONS="${MUSLR_BASELINE_JSON_OUTPUT:-$OUTPUT_DIR/muslr_baseline_json_qwen3vl8b.jsonl}"

export PYTHONPATH="$WORKSPACE/src:$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m evaluation.evaluate \
  --data "$DATA" \
  --predictions "$PREDICTIONS" \
  --model "Qwen3-VL-8B + Baseline-JSON" \
  --output "$OUTPUT_DIR/muslr_baseline_json_qwen3vl8b.summary.json" \
  --report "$OUTPUT_DIR/muslr_baseline_json_qwen3vl8b.report.md" \
  --table-output "$OUTPUT_DIR/muslr_baseline_json_qwen3vl8b.table.md" \
  --compact-table-output "$OUTPUT_DIR/muslr_baseline_json_qwen3vl8b.compact.table.md" \
  "$@"
