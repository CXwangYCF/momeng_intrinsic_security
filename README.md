# MuSLR local inference

Qwen3-VL-8B experiments on MuSLR through a RIPO OpenAI-compatible endpoint.
The current runners use JSON Schema responses. Older free-text code is kept in
`src/legacy/` for reproduction.

## What this package contains

`src/baseline_json/` and `src/logicam_json/` contain the two active runners.
`src/muslr_core/` contains shared dataset and inference code. `src/evaluation/`
produces metrics and reports. `src/result_dashboard/` browses results.
`src/scripts/` contains shell entry points; `src/tests/` contains local tests.

This source package does not include MuSLR data, official prompt files, model
weights, RIPO, predictions, or a Python environment. Obtain the dataset and
official prompts from the [MuSLR project](https://github.com/Aiden0526/MuSLR)
and install [RIPO](https://github.com/2-mo/RIPO) separately. This code was
validated against RIPO commit `ed37a8770c522cee0a2d8a4b8b65f0ba6cd3b8bc`.
RIPO later reorganized these APIs, so use that commit for reproduction. The
runner imports `pi_engine` and `ripo` from `RIPO_ROOT/src`; its Python
environment must also have RIPO's runtime dependencies installed.

## Setup

First check out the compatible RIPO version:

```bash
git clone https://github.com/2-mo/RIPO.git
git -C RIPO checkout ed37a8770c522cee0a2d8a4b8b65f0ba6cd3b8bc
```

From the extracted `myMuSLR` directory, with Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e /absolute/path/to/RIPO
cp .env.example .env
```

Use the full path to the checkout above. Install or run your model endpoint
following that RIPO revision's setup guide.

Edit `.env` to set `MUSLR_DATA`, `MUSLR_PROMPT_DIR`, `RIPO_ROOT`, endpoint,
and served model name. Use absolute paths. `MUSLR_DATA` must point to the
downloaded MuSLR directory containing `metadata.csv` and `images/`.
`MUSLR_PROMPT_DIR` must contain `baseline/baseline_CoT.txt` and
`logicam/prompt.txt` plus `logicam/prompt_fol.txt`. Keep API keys in `.env`;
the file is ignored by Git.

The endpoint must support image input and `json_schema` response format.
Set the two `*_ENDPOINTS` values to `auto` only when RIPO endpoint discovery
is running. Output defaults to `myMuSLR/outputs/` and is ignored by Git.

## Run

Check local inputs without a model request:

```bash
src/scripts/run_baseline_json.sh --dry-run --limit 1
src/scripts/run_logicam_json.sh --dry-run --limit 1
```

Run and evaluate each method:

```bash
src/scripts/run_baseline_json.sh
src/scripts/evaluate_baseline_json.sh
src/scripts/run_logicam_json.sh
src/scripts/evaluate_logicam_json.sh
```

The runners use temperature 0, one request attempt, incremental JSONL
checkpoints, and resume by default. Set `--output` for a separate experiment;
use `--no-resume` to restart one. Evaluation may download pretrained models
for optional reasoning metrics on first use.

Browse LogiCAM results after running inference:

```bash
src/scripts/run_dashboard.sh
# http://127.0.0.1:5060
```

More details: [experiment contracts](src/README.md) and
[LogiCAM audit](src/logicam_json/AUDIT.md).

## Tests

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s src/tests -v
```
