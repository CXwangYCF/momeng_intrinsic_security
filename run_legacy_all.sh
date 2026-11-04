#!/usr/bin/env bash

set -Eeuo pipefail

WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$WORKSPACE/.env"
PYTHON="$WORKSPACE/.venv/bin/python"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing configuration: $ENV_FILE" >&2
  echo "Create it with: cp $WORKSPACE/.env.example $ENV_FILE" >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python environment: $PYTHON" >&2
  echo "See $WORKSPACE/README.md for environment setup." >&2
  exit 1
fi

export PYTHONPATH="$WORKSPACE/src:$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m legacy.run_all "$@"
