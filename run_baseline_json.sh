#!/usr/bin/env bash

set -Eeuo pipefail

WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$WORKSPACE/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python environment: $PYTHON" >&2
  exit 1
fi

export PYTHONPATH="$WORKSPACE/src:$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m baseline_json.run "$@"
