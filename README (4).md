# MuSLR structured-output experiments

This source tree contains two Qwen3-VL-8B experiments using the RIPO structured
JSON interface demonstrated in `02_pi_engine_text_inference.ipynb`, section 5.

## Layout

```text
src/
├── baseline_json/       official few-shot baseline + minimal JSON contract
├── logicam_json/        official LogiCAM few-shot prompts + minimal JSON wrapper
├── muslr_core/          shared dataset, checkpoint, parsing, and RIPO utilities
├── evaluation/          metrics, comparisons, and publication tables
├── result_dashboard/    interactive result inspection page
├── legacy/              archived free-text runners and superseded dashboard
├── scripts/             run, evaluate, and dashboard shell entry points
└── tests/               schema and prompt contract tests
```

`baseline_json` deliberately constrains only `reasoning` and `final_answer`. It
continues to use the official three-example baseline CoT prompt and introduces no
formal notation, rule catalog, candidate consistency gate, or LogiCAM behavior.

`logicam_json` preserves the complete released `prompt.txt` / `prompt_fol.txt`,
including their example images and original reasoning demonstrations. JSON only
wraps original-style `reasoning_steps` and schema-constrained `final_answer`; it does
not add a rule enum, step limit, formalization template, or consistency gate.

## Baseline-JSON

Validate configuration without contacting the model:

```bash
cd /path/to/myMuSLR
src/scripts/run_baseline_json.sh --dry-run
```

Run a separate 10-sample smoke test:

```bash
src/scripts/run_baseline_json.sh \
  --limit 10 \
  --no-resume \
  --output outputs/muslr_baseline_json_smoke.jsonl
```

Run all 1,093 samples with checkpoint resume, then evaluate:

```bash
src/scripts/run_baseline_json.sh
src/scripts/evaluate_baseline_json.sh
```

Each evaluator also writes a compact publication table with `Method`, `Overall`,
`Traffic`, `Science`, `Finance`, and `Others`. All cells use strict E2E accuracy;
`Others` pools every remaining sample before computing the percentage.

## LogiCAM-JSON

```bash
src/scripts/run_logicam_json.sh --dry-run
src/scripts/run_logicam_json.sh
src/scripts/evaluate_logicam_json.sh
```

Recover legacy results that were rejected by the former strict consistency gate:

```bash
PYTHONPATH=src:. .venv/bin/python -m logicam_json.recover_results \
  /absolute/path/to/predictions.jsonl
```

## Dashboard

```bash
src/scripts/run_dashboard.sh
# http://127.0.0.1:5060
```

Choose another result with
`src/scripts/run_dashboard.sh --predictions /absolute/path/to/result.jsonl`.

Both runners use one model request attempt, Rich progress with completed/total,
temperature 0, checkpoint resume, and path/configuration values from `.env`.
LogiCAM-JSON uses 4,096 output tokens to match the released free-text run; the
shorter baseline remains at 1,024.
