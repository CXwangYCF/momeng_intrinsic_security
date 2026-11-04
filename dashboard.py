#!/usr/bin/env python3
"""Flask dashboard for MuSLR baseline vs LogiCAM results."""

from __future__ import annotations

import argparse
import json
import mimetypes
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template_string, send_file, send_from_directory

from evaluation.paper_comparison import build_comparison, method_label
from muslr_core.runtime import Sample, load_jsonl, load_samples, sample_index


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
DATA_PATH = ROOT / "data"

BASELINE_PREDICTIONS = OUTPUTS / "baseline_ripo_3gpu.jsonl"
LOGICAM_PREDICTIONS = OUTPUTS / "logicam_ripo_3gpu.jsonl"
BASELINE_SUMMARY = OUTPUTS / "baseline_ripo_3gpu.summary.json"
LOGICAM_SUMMARY = OUTPUTS / "logicam_ripo_3gpu.summary.json"
COMPARISON_SUMMARY = OUTPUTS / "baseline_vs_logicam.summary.json"

app = Flask(__name__)

DOMAIN_LABELS = {
    "medical": "Healthcare",
    "traffic": "Traffic",
    "Sports": "Sports",
    "Entertainment": "Ent.",
    "Social/Humanity": "Social",
    "Science": "Science",
    "finance": "Finance",
    "General Knowledge/Everyday Knowledge": "General",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def predictions_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in load_jsonl(path) if row.get("id") is not None}


def fmt_percent(value: float) -> str:
    return f"{value:.2f}%"


def fmt_delta(value: float, suffix: str = " pp") -> str:
    return f"{value:+.2f}{suffix}"


def _prediction_label(record: dict[str, Any] | None) -> str:
    if not record:
        return "-"
    prediction = record.get("prediction")
    return str(prediction) if isinstance(prediction, str) and prediction else "-"


def _is_correct(record: dict[str, Any] | None, sample: Sample) -> bool:
    prediction = record.get("prediction") if record else None
    return isinstance(prediction, str) and prediction.upper() == sample.answer.upper()


def _record_status(record: dict[str, Any] | None) -> str:
    if not record:
        return "missing"
    return str(record.get("status", "unknown"))


def _usage_value(record: dict[str, Any] | None, key: str) -> int | None:
    usage = record.get("usage") if record else None
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    return int(value) if isinstance(value, int | float) else None


def _local_image_path(sample: Sample) -> Path | None:
    value = sample.image
    if isinstance(value, dict):
        value = value.get("path") or value.get("local_path") or value.get("filename")
    if not isinstance(value, str) or not value.strip():
        return None

    ref = value.strip()
    if ref.startswith(("http://", "https://", "data:")):
        return None

    candidate = Path(ref)
    candidates = [candidate] if candidate.is_absolute() else [
        DATA_PATH / candidate,
        DATA_PATH / candidate.name,
        DATA_PATH / "images" / candidate.name,
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def _extract_reasoning_types(output: str) -> list[str]:
    types: list[str] = []
    for line in output.splitlines():
        if "reasoning type" not in line.lower():
            continue
        _, _, value = line.partition(":")
        for item in value.replace("/", ",").replace(";", ",").split(","):
            cleaned = item.strip(" .:-")
            if cleaned and cleaned not in types:
                types.append(cleaned)
    return types


def _chain_type(record: dict[str, Any] | None, method: str) -> str:
    if not record:
        return "Missing"
    output = str(record.get("output", ""))
    extracted = _extract_reasoning_types(output)
    if extracted:
        return " + ".join(extracted)
    if method == "logicam":
        return "LogiCAM structured CoT"
    return "Free-form CoT"


def _gold_reasoning_tags(sample: Sample) -> list[str]:
    tags: list[str] = []
    for key in sample.reasoning:
        prefix = "".join(ch for ch in key if ch.isalpha()).upper() or key.upper()
        if prefix not in tags:
            tags.append(prefix)
    return tags


def _correct_label(record: dict[str, Any] | None, sample: Sample) -> str:
    if not record:
        return "missing"
    if record.get("status") != "completed":
        return str(record.get("status", "unknown"))
    return "correct" if _is_correct(record, sample) else "wrong"


def domain_label(domain: str) -> str:
    return DOMAIN_LABELS.get(domain, domain)


def _row_category(sample: Sample, baseline: dict[str, Any] | None, logicam: dict[str, Any] | None) -> str:
    base_correct = _is_correct(baseline, sample)
    logicam_correct = _is_correct(logicam, sample)
    if base_correct and logicam_correct:
        return "both_correct"
    if not base_correct and logicam_correct:
        return "logicam_fix"
    if base_correct and not logicam_correct:
        return "regression"
    if logicam and logicam.get("status") == "unparsed":
        return "logicam_unparsed"
    if baseline and baseline.get("status") == "unparsed":
        return "baseline_unparsed"
    return "both_incorrect"


@lru_cache(maxsize=1)
def dashboard_data() -> dict[str, Any]:
    samples = sample_index(load_samples(DATA_PATH))
    baseline_records = predictions_by_id(BASELINE_PREDICTIONS)
    logicam_records = predictions_by_id(LOGICAM_PREDICTIONS)
    comparison = load_json(COMPARISON_SUMMARY)

    sample_rows: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    for sample in samples.values():
        baseline = baseline_records.get(sample.id)
        logicam = logicam_records.get(sample.id)
        category = _row_category(sample, baseline, logicam)
        category_counts[category] = category_counts.get(category, 0) + 1
        sample_rows.append(
            {
                "id": sample.id,
                "domain": domain_label(sample.domain),
                "raw_domain": sample.domain,
                "symbol": sample.symbol.upper(),
                "depth": sample.depth,
                "answer": sample.answer.upper(),
                "baseline_prediction": _prediction_label(baseline),
                "logicam_prediction": _prediction_label(logicam),
                "baseline_status": _record_status(baseline),
                "logicam_status": _record_status(logicam),
                "category": category,
            }
        )
    sample_rows.sort(key=lambda row: (str(row["category"]), int(row["depth"] or 0), row["id"]))

    return {
        "baseline_summary": load_json(BASELINE_SUMMARY),
        "logicam_summary": load_json(LOGICAM_SUMMARY),
        "comparison": comparison,
        "samples": sample_rows,
        "category_counts": category_counts,
        "baseline_records": baseline_records,
        "logicam_records": logicam_records,
        "samples_by_id": samples,
    }


def _metric_pair(metric: dict[str, Any], key: str) -> dict[str, Any]:
    row = metric[key]
    return {
        "baseline": row["baseline"],
        "logicam": row["logicam"],
        "delta": row["delta"],
    }


def sample_metric_rows(sample: Sample, comparison: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    overall = comparison["overall"]
    rows.append(
        {
            "label": "Overall E2E",
            **_metric_pair(overall, "strict_end_to_end_accuracy_percent"),
            "suffix": "%",
        }
    )
    domain = domain_label(sample.domain)
    domain_row = comparison["by_domain"].get(domain)
    if isinstance(domain_row, dict):
        rows.append(
            {
                "label": f"{domain} E2E",
                **_metric_pair(domain_row, "accuracy_end_to_end_percent"),
                "suffix": "%",
            }
        )
    symbol = sample.symbol.upper()
    symbol_row = comparison["by_symbol"].get(symbol)
    if isinstance(symbol_row, dict):
        rows.append(
            {
                "label": f"{symbol} E2E",
                **_metric_pair(symbol_row, "accuracy_end_to_end_percent"),
                "suffix": "%",
            }
        )
    return rows


def review_samples() -> list[dict[str, Any]]:
    data = dashboard_data()
    rows = list(data["samples"])
    priority = {
        "logicam_fix": 0,
        "regression": 1,
        "logicam_unparsed": 2,
        "baseline_unparsed": 3,
        "both_incorrect": 4,
        "both_correct": 5,
    }
    rows.sort(key=lambda row: (priority.get(str(row["category"]), 9), str(row["domain"]), str(row["id"])))
    return rows


def chart_rows(groups: dict[str, Any], metric_key: str = "accuracy_end_to_end_percent") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, group in groups.items():
        metric = group[metric_key]
        rows.append(
            {
                "label": label,
                "baseline": float(metric["baseline"]),
                "logicam": float(metric["logicam"]),
                "delta": float(metric["delta"]),
                "width_baseline": min(100.0, max(0.0, float(metric["baseline"]))),
                "width_logicam": min(100.0, max(0.0, float(metric["logicam"]))),
            }
        )
    return rows


def reasoning_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, metric in comparison["reasoning_quality"].items():
        rows.append(
            {
                "label": label,
                "baseline": float(metric["baseline"]),
                "logicam": float(metric["logicam"]),
                "delta": float(metric["delta"]),
                "width_baseline": min(100.0, max(0.0, float(metric["baseline"]) * 100)),
                "width_logicam": min(100.0, max(0.0, float(metric["logicam"]) * 100)),
            }
        )
    return rows


def main_cards(comparison: dict[str, Any]) -> list[dict[str, str]]:
    overall = comparison["overall"]
    return [
        {
            "label": "Strict E2E Accuracy",
            "baseline": fmt_percent(overall["strict_end_to_end_accuracy_percent"]["baseline"]),
            "logicam": fmt_percent(overall["strict_end_to_end_accuracy_percent"]["logicam"]),
            "delta": fmt_delta(overall["strict_end_to_end_accuracy_percent"]["delta"]),
            "tone": "good",
        },
        {
            "label": "Parsed-Only Accuracy",
            "baseline": fmt_percent(overall["parsed_only_accuracy_percent"]["baseline"]),
            "logicam": fmt_percent(overall["parsed_only_accuracy_percent"]["logicam"]),
            "delta": fmt_delta(overall["parsed_only_accuracy_percent"]["delta"]),
            "tone": "good",
        },
        {
            "label": "Answer Coverage",
            "baseline": fmt_percent(overall["answer_coverage_percent"]["baseline"]),
            "logicam": fmt_percent(overall["answer_coverage_percent"]["logicam"]),
            "delta": fmt_delta(overall["answer_coverage_percent"]["delta"]),
            "tone": "bad",
        },
        {
            "label": "Correct Answers",
            "baseline": f"{overall['correct']['baseline']:.0f}",
            "logicam": f"{overall['correct']['logicam']:.0f}",
            "delta": f"{overall['correct']['delta']:+.0f}",
            "tone": "good",
        },
    ]


def sample_excerpt(record: dict[str, Any] | None, limit: int = 4000) -> dict[str, Any] | None:
    if not record:
        return None
    output = str(record.get("output", ""))
    return {
        "status": record.get("status"),
        "prediction": record.get("prediction"),
        "usage": record.get("usage"),
        "latency_seconds": record.get("latency_seconds"),
        "output": output[:limit] + ("\n...[truncated]" if len(output) > limit else ""),
    }


@app.get("/dashboard")
def dashboard_index() -> str:
    data = dashboard_data()
    comparison = data["comparison"]
    return render_template_string(
        TEMPLATE,
        cards=main_cards(comparison),
        symbol_rows=chart_rows(comparison["by_symbol"]),
        domain_rows=chart_rows(comparison["by_domain"]),
        depth_rows=chart_rows(comparison["by_depth_bucket"]),
        reasoning_rows=reasoning_rows(comparison),
        samples=data["samples"],
        category_counts=data["category_counts"],
        paired=comparison["paired_common_parsed"],
        diagnostics=comparison["diagnostics"],
        baseline_report="outputs/baseline_ripo_3gpu.report.md",
        logicam_report="outputs/logicam_ripo_3gpu.report.md",
        comparison_report="outputs/baseline_vs_logicam.report.md",
    )


@app.get("/")
def review_index() -> str:
    data = dashboard_data()
    comparison = data["comparison"]
    samples = review_samples()
    return render_template_string(
        REVIEW_TEMPLATE,
        samples=samples,
        cards=main_cards(comparison),
        category_counts=data["category_counts"],
        paired=comparison["paired_common_parsed"],
        diagnostics=comparison["diagnostics"],
        overall=comparison["overall"],
        dashboard_href="/dashboard",
    )


@app.get("/image/<sample_id>")
def sample_image(sample_id: str) -> Any:
    sample = dashboard_data()["samples_by_id"].get(sample_id)
    if sample is None:
        abort(404)
    path = _local_image_path(sample)
    if path is None:
        abort(404)
    return send_file(path, mimetype=mimetypes.guess_type(path.name)[0] or "image/jpeg")


@app.get("/paper-table")
def paper_table_page() -> str:
    return render_template_string(
        PAPER_TABLE_TEMPLATE,
        table=build_comparison(),
        method_label=method_label,
    )


@app.get("/api/sample/<sample_id>")
def sample_detail(sample_id: str) -> Any:
    data = dashboard_data()
    sample = data["samples_by_id"].get(sample_id)
    if sample is None:
        abort(404)
    baseline = data["baseline_records"].get(sample_id)
    logicam = data["logicam_records"].get(sample_id)
    image_path = _local_image_path(sample)
    return jsonify(
        {
            "id": sample.id,
            "domain": domain_label(sample.domain),
            "raw_domain": sample.domain,
            "symbol": sample.symbol.upper(),
            "depth": sample.depth,
            "answer": sample.answer.upper(),
            "question": sample.question,
            "choices": sample.choices,
            "context": sample.context,
            "gold_reasoning_tags": _gold_reasoning_tags(sample),
            "gold_reasoning": sample.reasoning,
            "image_url": f"/image/{sample.id}" if image_path else None,
            "image_path": str(image_path) if image_path else None,
            "category": _row_category(sample, baseline, logicam),
            "metrics": sample_metric_rows(sample, data["comparison"]),
            "baseline": {
                **(sample_excerpt(baseline, limit=7000) or {}),
                "chain_type": _chain_type(baseline, "baseline"),
                "correct_label": _correct_label(baseline, sample),
                "completion_tokens": _usage_value(baseline, "completion_tokens"),
                "total_tokens": _usage_value(baseline, "total_tokens"),
                "finish_reason": baseline.get("finish_reason") if baseline else None,
            },
            "logicam": {
                **(sample_excerpt(logicam, limit=7000) or {}),
                "chain_type": _chain_type(logicam, "logicam"),
                "correct_label": _correct_label(logicam, sample),
                "completion_tokens": _usage_value(logicam, "completion_tokens"),
                "total_tokens": _usage_value(logicam, "total_tokens"),
                "finish_reason": logicam.get("finish_reason") if logicam else None,
            },
        }
    )


@app.get("/outputs/<path:filename>")
def output_file(filename: str) -> Any:
    return send_from_directory(OUTPUTS, filename)


@app.get("/api/summary")
def summary_api() -> Any:
    data = dashboard_data()
    return jsonify(
        {
            "comparison": data["comparison"],
            "category_counts": data["category_counts"],
            "sample_count": len(data["samples"]),
        }
    )


PAPER_TABLE_TEMPLATE = r'''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MuSLR Paper Comparison</title>
  <style>
    :root{--bg:#fff;--line:#c9cdd2;--heavy:#222;--text:#171717;--muted:#60656d;--rank1:#d62728;--rank2:#1f77b4;--rank3:#2ca02c;--logicam:#111}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:13px/1.4 "Times New Roman",Times,serif}header{background:#fff;border-bottom:1px solid var(--heavy);padding:16px 22px;display:flex;align-items:center;justify-content:space-between;gap:16px}h1,h2,p{margin:0}h1{font-size:21px}h2{font-size:16px;margin:18px 0 8px}.muted{color:var(--muted);font-family:Inter,ui-sans-serif,system-ui,sans-serif}a,button{border:1px solid var(--line);border-radius:3px;background:#fff;color:var(--text);padding:7px 10px;text-decoration:none;font:12px/1.2 Inter,ui-sans-serif,system-ui,sans-serif}.nav{display:flex;gap:8px;align-items:center}main{padding:18px 22px 28px;max-width:1800px;margin:auto}.note{border-top:1.5px solid var(--heavy);border-bottom:1px solid var(--line);padding:10px 0 12px;margin-bottom:14px}.table-wrap{background:#fff;overflow:auto;max-height:calc(100vh - 220px);margin-bottom:16px;border-top:1.5px solid var(--heavy);border-bottom:1.5px solid var(--heavy)}table{width:100%;min-width:1260px;border-collapse:collapse;white-space:nowrap}table.domain{min-width:980px}table.diag{min-width:1120px}th,td{border:0;padding:8px 10px;text-align:right}thead th{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid var(--heavy);font-size:11px;font-family:Inter,ui-sans-serif,system-ui,sans-serif;text-transform:uppercase}tbody tr+tr td{border-top:1px solid #e2e2e2}th:nth-child(-n+2),td:nth-child(-n+2){text-align:left}th:nth-child(1),td:nth-child(1){position:sticky;left:0;z-index:3;min-width:230px;background:#fff}table:not(.domain):not(.diag) th:nth-child(2),table:not(.domain):not(.diag) td:nth-child(2){position:sticky;left:230px;z-index:3;background:#fff;min-width:62px}.domain th:nth-child(2),.domain td:nth-child(2),.diag th:nth-child(2),.diag td:nth-child(2){position:static;text-align:right;min-width:0}.logicam-row td{font-weight:800}.symbol{font-weight:750}.overall{font-weight:800}.rank-1{color:var(--rank1);font-weight:900}.rank-2{color:var(--rank2);font-weight:900}.rank-3{color:var(--rank3);font-weight:900}.rank-mark{font:800 9px/1 Inter,ui-sans-serif,system-ui,sans-serif;vertical-align:super;margin-left:2px}.pos{color:#057a55;font-weight:800}.neg{color:#b42318;font-weight:800}.legend{display:flex;gap:12px;margin-top:8px;color:var(--muted);font:12px/1.3 Inter,ui-sans-serif,system-ui,sans-serif}.swatch{width:10px;height:10px;display:inline-block;margin-right:4px;vertical-align:-1px}.swatch.rank1{background:var(--rank1)}.swatch.rank2{background:var(--rank2)}.swatch.rank3{background:var(--rank3)}ul{margin:8px 0 0;padding-left:20px}
  </style>
</head>
<body>
  <header><div><h1>{{ table.title }}</h1><div class="muted">{{ table.metric }}</div></div><nav class="nav"><a href="/">Sample review</a><a href="/dashboard">Summary</a><a href="/outputs/paper_vs_ours.table.md">Markdown</a></nav></header>
  <main>
    <section class="note"><h2>整体对比</h2><div class="muted">论文方法、本地 Qwen3-VL-8B 结果、领域汇总和 Symbol × Domain 明细放在同一页。当前指标是端到端准确率，未解析输出按错误计入。</div><div class="legend"><span><i class="swatch rank1"></i>Rank 1</span><span><i class="swatch rank2"></i>Rank 2</span><span><i class="swatch rank3"></i>Rank 3</span><span>LogiCAM rows are bold.</span></div><ul>{% for note in table.notes %}<li>{{ note }}</li>{% endfor %}</ul></section>
    <h2>Domain Overall E2E Accuracy</h2>
    <div class="table-wrap"><table class="domain"><thead><tr>{% for column in table.domain_columns %}<th>{{ column }}</th>{% endfor %}</tr></thead><tbody>{% for row in table.domain_rows %}<tr class="{{ 'logicam-row' if 'LogiCAM' in row.method else '' }}"><td>{{ method_label(row) }}</td>{% set overall_rank = row.ranks.get('overall') %}<td class="overall {% if overall_rank %}rank-{{ overall_rank }}{% endif %}">{{ "%.2f"|format(row.overall) }}{% if overall_rank %}<sup class="rank-mark">{{ overall_rank }}</sup>{% endif %}</td>{% for column in table.domain_columns[2:] %}{% set rank = row.ranks.get(column) %}<td class="{% if rank %}rank-{{ rank }}{% endif %}">{{ "%.2f"|format(row[column]) }}{% if rank %}<sup class="rank-mark">{{ rank }}</sup>{% endif %}</td>{% endfor %}</tr>{% endfor %}</tbody></table></div>
    <h2>Our Domain Diagnostics</h2>
    <div class="table-wrap"><table class="diag"><thead><tr><th>Domain</th><th>Samples</th><th>Baseline E2E</th><th>LogiCAM E2E</th><th>E2E delta</th><th>Baseline parsed-only</th><th>LogiCAM parsed-only</th><th>Parsed delta</th><th>Baseline coverage</th><th>LogiCAM coverage</th><th>Coverage delta</th></tr></thead><tbody>{% for row in table.ours_domain_diagnostics %}<tr><td>{{ row.domain }}</td><td>{{ row.samples }}</td><td>{{ "%.2f"|format(row.baseline_e2e) }}</td><td>{{ "%.2f"|format(row.logicam_e2e) }}</td><td class="{{ 'pos' if row.e2e_delta >= 0 else 'neg' }}">{{ "%+.2f"|format(row.e2e_delta) }}</td><td>{{ "%.2f"|format(row.baseline_parsed) }}</td><td>{{ "%.2f"|format(row.logicam_parsed) }}</td><td class="{{ 'pos' if row.parsed_delta >= 0 else 'neg' }}">{{ "%+.2f"|format(row.parsed_delta) }}</td><td>{{ "%.2f"|format(row.baseline_coverage) }}</td><td>{{ "%.2f"|format(row.logicam_coverage) }}</td><td class="{{ 'pos' if row.coverage_delta >= 0 else 'neg' }}">{{ "%+.2f"|format(row.coverage_delta) }}</td></tr>{% endfor %}</tbody></table></div>
    <h2>Symbol × Domain E2E Accuracy</h2>
    <div class="table-wrap"><table><thead><tr>{% for column in table.columns %}<th>{{ column }}</th>{% endfor %}</tr></thead><tbody>{% for row in table.rows %}<tr class="{{ 'logicam-row' if 'LogiCAM' in row.method else '' }}"><td>{% if loop.previtem is not defined or loop.previtem.method != row.method %}{{ method_label(row) }}{% endif %}</td><td class="symbol">{{ row.symbol }}</td><td class="overall">{% if row.overall is not none %}{{ "%.2f"|format(row.overall) }}{% endif %}</td>{% for column in table.columns[3:] %}<td>{{ "%.2f"|format(row[column]) }}</td>{% endfor %}</tr>{% endfor %}</tbody></table></div>
  </main>
</body>
</html>
'''


REVIEW_TEMPLATE = r'''
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MuSLR Sample Review</title>
  <style>
    :root{--bg:#f4f6f8;--panel:#fff;--panel2:#f8fafc;--line:#d7dde5;--text:#17202a;--muted:#66758a;--base:#4f6df5;--logicam:#0f9f86;--good:#057a55;--bad:#b42318;--warn:#a16207;--dark:#111827}
    *{box-sizing:border-box}body{margin:0;overflow:hidden;background:var(--bg);color:var(--text);font:13px/1.4 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    header{height:58px;border-bottom:1px solid var(--line);background:#fff;display:flex;align-items:center;justify-content:space-between;gap:14px;padding:10px 16px}
    h1,h2,h3,p{margin:0}h1{font-size:18px;font-weight:750}h2{font-size:14px;font-weight:750}h3{font-size:12px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.02em}.muted{color:var(--muted)}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
    .topline{display:flex;align-items:center;gap:12px;min-width:0}.topline small{display:block;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.links{display:flex;gap:8px;align-items:center}.links a,.links button,.links select,.links input{border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--text);height:34px;padding:0 10px;text-decoration:none;font:inherit}.links button{cursor:pointer}.links input{width:230px}
    .summary{height:70px;display:grid;grid-template-columns:repeat(4,minmax(0,1fr)) 1.15fr;gap:10px;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel2)}
    .metric{background:#fff;border:1px solid var(--line);border-radius:8px;padding:8px 10px;display:flex;align-items:center;justify-content:space-between;gap:10px;min-width:0}.metric strong{font-size:19px}.delta{font-weight:750;border-radius:999px;background:#eef2f7;padding:3px 7px;white-space:nowrap}.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}
    .review{height:calc(100vh - 128px);padding:12px 16px;display:grid;grid-template-columns:minmax(360px,42fr) minmax(300px,27fr) minmax(360px,31fr);gap:12px;min-height:0}
    .panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;min-height:0;overflow:hidden;display:flex;flex-direction:column}.panel-head{height:42px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:8px;padding:10px 12px;background:#fff}.panel-body{padding:12px;min-height:0;overflow:auto}.tight{display:grid;gap:10px}
    .image-wrap{flex:1;min-height:0;background:#10151f;display:flex;align-items:center;justify-content:center}.image-wrap img{display:block;max-width:100%;max-height:100%;object-fit:contain}.image-path{height:38px;padding:8px 12px;border-top:1px solid #263244;background:#151d29;color:#b9c4d3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .sample-title{display:flex;gap:8px;align-items:center;min-width:0}.sample-title strong{font-size:15px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.pill{display:inline-flex;align-items:center;border-radius:999px;padding:3px 8px;font-size:12px;font-weight:750;background:#eef2f7;white-space:nowrap}.pill.logicam_fix,.pill.correct{color:var(--good)}.pill.regression,.pill.wrong{color:var(--bad)}.pill.logicam_unparsed,.pill.unparsed{color:var(--warn)}.pill.both_correct{color:#075985}
    .facts{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.fact{border:1px solid var(--line);border-radius:7px;background:var(--panel2);padding:8px}.fact span{display:block;color:var(--muted);font-size:11px}.fact strong{font-size:15px}
    .choices{display:grid;gap:6px}.choice{border:1px solid var(--line);border-radius:6px;padding:7px 8px;background:#fff}.section{border:1px solid var(--line);border-radius:7px;background:#fff;overflow:hidden}.section h3{padding:8px 10px;background:var(--panel2);border-bottom:1px solid var(--line)}.section .content{padding:10px;max-height:168px;overflow:auto}.context .content{max-height:188px}
    .compare{display:grid;grid-template-rows:auto 1fr 1fr;gap:10px;min-height:0}.metric-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.mini-metric{border:1px solid var(--line);border-radius:7px;background:var(--panel2);padding:8px;min-width:0}.mini-metric span{display:block;color:var(--muted);font-size:11px}.mini-metric strong{font-size:14px}.mini-metric em{font-style:normal;font-weight:750}
    .method{border:1px solid var(--line);border-radius:8px;overflow:hidden;display:flex;flex-direction:column;min-height:0}.method.base{border-top:4px solid var(--base)}.method.logicam{border-top:4px solid var(--logicam)}.method-head{padding:9px 10px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:8px}.method-head strong{font-size:14px}.method-meta{display:grid;grid-template-columns:1.1fr .8fr .8fr .8fr;gap:6px;padding:8px 10px;border-bottom:1px solid var(--line);background:var(--panel2)}.method-meta div{min-width:0}.method-meta span{display:block;color:var(--muted);font-size:10px}.method-meta strong{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.output{padding:10px;margin:0;overflow:auto;white-space:pre-wrap;word-break:break-word;font:12px/1.42 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:#dce6f5;background:var(--dark);flex:1;min-height:0}
    .gold-steps{display:flex;gap:5px;flex-wrap:wrap}.empty{color:var(--muted);font-style:italic}.hidden{display:none!important}
    @media(max-width:1180px){body{overflow:auto}.summary{height:auto;grid-template-columns:repeat(2,minmax(0,1fr))}.review{height:auto;grid-template-columns:1fr}.panel{min-height:420px}.links input{width:170px}}
  </style>
</head>
<body>
  <header>
    <div class="topline">
      <div>
        <h1>MuSLR Sample Review</h1>
        <small>一屏一个样本：本地图像、题目、答案、思维链类型和指标对比</small>
      </div>
    </div>
    <div class="links">
      <button id="prev">Prev</button>
      <button id="next">Next</button>
      <select id="category">
        <option value="all">All outcomes</option>
        <option value="logicam_fix">LogiCAM fixes</option>
        <option value="regression">Regressions</option>
        <option value="logicam_unparsed">LogiCAM unparsed</option>
        <option value="both_incorrect">Both incorrect</option>
        <option value="both_correct">Both correct</option>
      </select>
      <input id="search" type="search" placeholder="id / domain / symbol">
      <a href="/paper-table">Paper table</a>
      <a href="{{ dashboard_href }}">Summary</a>
    </div>
  </header>

  <section class="summary">
    {% for card in cards %}
    <div class="metric">
      <div><h3>{{ card.label }}</h3><strong>{{ card.logicam }}</strong><span class="muted">Baseline {{ card.baseline }}</span></div>
      <span class="delta {{ card.tone }}">{{ card.delta }}</span>
    </div>
    {% endfor %}
    <div class="metric">
      <div><h3>Common Parsed</h3><strong>{{ paired.common_parsed_samples }}</strong><span class="muted">LogiCAM-only correct {{ paired.logicam_only_correct }} · Baseline-only {{ paired.baseline_only_correct }}</span></div>
      <span class="delta good">{{ "%+.2f"|format(paired.logicam_minus_baseline_points) }} pp</span>
    </div>
  </section>

  <main class="review">
    <section class="panel">
      <div class="panel-head">
        <div class="sample-title"><strong id="sample-id">-</strong><span id="sample-category" class="pill">-</span></div>
        <span id="sample-counter" class="muted mono">-</span>
      </div>
      <div class="image-wrap"><img id="sample-image" alt="MuSLR sample image"></div>
      <div id="image-path" class="image-path mono"></div>
    </section>

    <section class="panel">
      <div class="panel-head"><h2>Sample</h2><span id="sample-meta" class="muted"></span></div>
      <div class="panel-body tight">
        <div class="facts">
          <div class="fact"><span>Answer</span><strong id="answer">-</strong></div>
          <div class="fact"><span>Symbol</span><strong id="symbol">-</strong></div>
          <div class="fact"><span>Depth</span><strong id="depth">-</strong></div>
          <div class="fact"><span>Domain</span><strong id="domain">-</strong></div>
        </div>
        <div class="section"><h3>Question</h3><div id="question" class="content"></div></div>
        <div class="section"><h3>Choices</h3><div id="choices" class="content choices"></div></div>
        <div class="section context"><h3>Context</h3><div id="context" class="content"></div></div>
        <div class="section"><h3>Gold Reasoning Types</h3><div id="gold-tags" class="content gold-steps"></div></div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head"><h2>Baseline vs LogiCAM</h2><span id="compare-note" class="muted"></span></div>
      <div class="panel-body compare">
        <div id="metric-grid" class="metric-grid"></div>
        <article id="baseline-card" class="method base"></article>
        <article id="logicam-card" class="method logicam"></article>
      </div>
    </section>
  </main>

  <script>
    const samples = {{ samples|tojson }};
    let filtered = [...samples];
    let index = 0;
    const el = id => document.getElementById(id);
    const stateClass = value => String(value || '').replaceAll(' ', '_');
    const safe = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    const pct = value => `${Number(value).toFixed(2)}%`;
    const signed = value => `${Number(value) >= 0 ? '+' : ''}${Number(value).toFixed(2)} pp`;

    function applyFilters() {
      const cat = el('category').value;
      const q = el('search').value.trim().toLowerCase();
      filtered = samples.filter(row => {
        const haystack = `${row.id} ${row.domain} ${row.symbol} ${row.category}`.toLowerCase();
        return (cat === 'all' || row.category === cat) && (!q || haystack.includes(q));
      });
      index = 0;
      loadCurrent();
    }

    async function loadCurrent() {
      if (!filtered.length) {
        el('sample-id').textContent = 'No sample';
        el('sample-counter').textContent = '0 / 0';
        return;
      }
      if (index < 0) index = filtered.length - 1;
      if (index >= filtered.length) index = 0;
      const row = filtered[index];
      const response = await fetch(`/api/sample/${encodeURIComponent(row.id)}`);
      const sample = await response.json();
      renderSample(sample, row);
    }

    function renderSample(sample, row) {
      el('sample-id').textContent = sample.id;
      el('sample-counter').textContent = `${index + 1} / ${filtered.length}`;
      el('sample-category').textContent = sample.category.replaceAll('_', ' ');
      el('sample-category').className = `pill ${sample.category}`;
      el('sample-meta').textContent = `${sample.domain} · ${sample.symbol} · depth ${sample.depth}`;
      el('answer').textContent = sample.answer || '-';
      el('symbol').textContent = sample.symbol || '-';
      el('depth').textContent = sample.depth ?? '-';
      el('domain').textContent = sample.domain || '-';
      el('question').textContent = sample.question || '';
      el('context').textContent = sample.context || '';
      el('compare-note').textContent = `${sample.baseline.correct_label} / ${sample.logicam.correct_label}`;

      const img = el('sample-image');
      if (sample.image_url) {
        img.src = `${sample.image_url}?t=${Date.now()}`;
        img.classList.remove('hidden');
        el('image-path').textContent = sample.image_path || '';
      } else {
        img.removeAttribute('src');
        img.classList.add('hidden');
        el('image-path').textContent = 'local image not found';
      }

      const choices = el('choices');
      choices.innerHTML = '';
      if (sample.choices && sample.choices.length) {
        sample.choices.forEach(choice => {
          const div = document.createElement('div');
          div.className = 'choice';
          div.textContent = choice;
          choices.appendChild(div);
        });
      } else {
        choices.innerHTML = '<span class="empty">Truth evaluation sample, no multiple-choice options.</span>';
      }

      const tags = el('gold-tags');
      tags.innerHTML = '';
      if (sample.gold_reasoning_tags && sample.gold_reasoning_tags.length) {
        sample.gold_reasoning_tags.forEach(tag => {
          const span = document.createElement('span');
          span.className = 'pill';
          span.textContent = tag;
          tags.appendChild(span);
        });
      } else {
        tags.innerHTML = '<span class="empty">No gold reasoning labels.</span>';
      }

      el('metric-grid').innerHTML = sample.metrics.map(metric => {
        const tone = Number(metric.delta) >= 0 ? 'good' : 'bad';
        return `<div class="mini-metric"><span>${safe(metric.label)}</span><strong>${pct(metric.logicam)}</strong><span>Baseline ${pct(metric.baseline)}</span><em class="${tone}">${signed(metric.delta)}</em></div>`;
      }).join('');
      el('baseline-card').innerHTML = methodHtml('Baseline', sample.baseline);
      el('logicam-card').innerHTML = methodHtml('LogiCAM', sample.logicam);
    }

    function methodHtml(label, record) {
      const status = record?.correct_label || record?.status || 'missing';
      const prediction = record?.prediction || '-';
      const tokens = record?.completion_tokens ?? '-';
      const total = record?.total_tokens ?? '-';
      const latency = record?.latency_seconds == null ? '-' : `${Number(record.latency_seconds).toFixed(2)}s`;
      const finish = record?.finish_reason || '-';
      const output = record?.output || '';
      return `
        <div class="method-head"><strong>${label}</strong><span class="pill ${stateClass(status)}">${safe(status)}</span></div>
        <div class="method-meta">
          <div><span>CoT type</span><strong title="${safe(record?.chain_type)}">${safe(record?.chain_type || '-')}</strong></div>
          <div><span>Pred</span><strong>${safe(prediction)}</strong></div>
          <div><span>Tokens</span><strong>${safe(tokens)} / ${safe(total)}</strong></div>
          <div><span>Latency</span><strong title="finish: ${safe(finish)}">${safe(latency)}</strong></div>
        </div>
        <pre class="output">${safe(output)}</pre>`;
    }

    el('prev').addEventListener('click', () => { index -= 1; loadCurrent(); });
    el('next').addEventListener('click', () => { index += 1; loadCurrent(); });
    el('category').addEventListener('input', applyFilters);
    el('search').addEventListener('input', applyFilters);
    window.addEventListener('keydown', event => {
      if (event.key === 'ArrowLeft') { index -= 1; loadCurrent(); }
      if (event.key === 'ArrowRight') { index += 1; loadCurrent(); }
    });
    loadCurrent();
  </script>
</body>
</html>
'''


TEMPLATE = r'''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MuSLR Dashboard</title>
  <style>
    :root{--bg:#f6f7f9;--panel:#fff;--soft:#f0f3f6;--line:#d9dee6;--text:#17202a;--muted:#617083;--base:#5b7cfa;--logicam:#14a38b;--good:#0b7a4b;--bad:#b42318;--warn:#9a6700}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{position:sticky;top:0;z-index:5;border-bottom:1px solid var(--line);background:rgba(255,255,255,.96);backdrop-filter:blur(10px)}.header-inner{max-width:1440px;margin:0 auto;padding:14px 24px;display:flex;align-items:center;justify-content:space-between;gap:16px}h1,h2,h3{margin:0;letter-spacing:0}h1{font-size:20px;font-weight:700}h2{font-size:16px;margin-bottom:14px}h3{font-size:13px;color:var(--muted);font-weight:650}.subtle{color:var(--muted)}.links{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}.links a,button,select,input{border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--text);padding:8px 10px;text-decoration:none;font:inherit}button{cursor:pointer}main{max-width:1440px;margin:0 auto;padding:20px 24px 36px}.grid{display:grid;gap:16px}.cards{grid-template-columns:repeat(4,minmax(0,1fr))}.two{grid-template-columns:minmax(0,1.1fr) minmax(0,.9fr);margin-top:16px}.panel,.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;min-width:0}.card{padding:15px}.metric{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:end;margin-top:10px}.metric strong{font-size:24px}.delta{border-radius:999px;padding:4px 8px;font-weight:700;background:var(--soft)}.delta.good{color:var(--good)}.delta.bad{color:var(--bad)}.bar-table{display:grid;gap:10px}.bar-row{display:grid;grid-template-columns:116px minmax(160px,1fr) 78px;gap:10px;align-items:center}.bar-row.reasoning{grid-template-columns:128px minmax(160px,1fr) 80px}.track{height:28px;background:#eef1f5;border:1px solid var(--line);border-radius:6px;overflow:hidden;display:grid;grid-template-rows:1fr 1fr}.bar{min-width:2px}.bar.base{background:var(--base)}.bar.logicam{background:var(--logicam)}.legend{display:flex;align-items:center;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-bottom:12px}.swatch{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:5px}.base-swatch{background:var(--base)}.logicam-swatch{background:var(--logicam)}.delta-pos{color:var(--good);font-weight:700}.delta-neg{color:var(--bad);font-weight:700}.insights{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px}.insight{background:var(--soft);border:1px solid var(--line);border-radius:8px;padding:12px}.insight strong{display:block;font-size:20px;margin-top:3px}.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px}.controls input{min-width:260px}.table-wrap{border:1px solid var(--line);border-radius:8px;overflow:auto;max-height:520px}table{width:100%;border-collapse:collapse;background:#fff}th,td{border-bottom:1px solid var(--line);padding:9px 10px;text-align:left;white-space:nowrap}th{position:sticky;top:0;background:#f8fafc;z-index:1;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:0}.pill{display:inline-flex;align-items:center;border-radius:999px;padding:3px 8px;font-size:12px;font-weight:700;background:var(--soft)}.pill.logicam_fix{color:var(--good)}.pill.regression{color:var(--bad)}.pill.logicam_unparsed{color:var(--warn)}.pill.both_correct{color:#075985}.sample-button{border:0;background:transparent;padding:0;color:#1d4ed8;text-decoration:underline;cursor:pointer}dialog{width:min(1080px,calc(100vw - 32px));max-height:calc(100vh - 32px);border:1px solid var(--line);border-radius:8px;padding:0;overflow:hidden}dialog::backdrop{background:rgba(15,23,42,.42)}.modal-head{padding:14px 16px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;gap:12px}.modal-body{padding:16px;overflow:auto;max-height:calc(100vh - 120px)}pre{white-space:pre-wrap;word-break:break-word;background:#0f172a;color:#e5eefb;padding:12px;border-radius:6px;max-height:360px;overflow:auto}.split{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:1000px){.cards,.two,.insights,.split{grid-template-columns:1fr}.header-inner{align-items:flex-start;flex-direction:column}.bar-row,.bar-row.reasoning{grid-template-columns:92px minmax(120px,1fr) 70px}.controls input{min-width:100%}}
  </style>
</head>
<body>
  <header><div class="header-inner"><div><h1>MuSLR Baseline vs LogiCAM</h1><div class="subtle">Qwen3-VL-8B · 1093 samples · strict E2E metrics count unparsed outputs as incorrect</div></div><nav class="links"><a href="/">Sample Review</a><a href="/paper-table">Paper Table</a><a href="/api/summary">API Summary</a><a href="{{ comparison_report }}">Comparison MD</a><a href="{{ baseline_report }}">Baseline MD</a><a href="{{ logicam_report }}">LogiCAM MD</a></nav></div></header>
  <main>
    <section class="grid cards">{% for card in cards %}<div class="card"><h3>{{ card.label }}</h3><div class="metric"><div><div class="subtle">Baseline {{ card.baseline }}</div><strong>{{ card.logicam }}</strong></div><span class="delta {{ card.tone }}">{{ card.delta }}</span></div></div>{% endfor %}</section>
    <section class="insights"><div class="insight"><span class="subtle">Common parsed samples</span><strong>{{ paired.common_parsed_samples }}</strong><span>{{ "%.2f"|format(paired.baseline_accuracy_percent) }}% -> {{ "%.2f"|format(paired.logicam_accuracy_percent) }}%</span></div><div class="insight"><span class="subtle">LogiCAM-only correct</span><strong>{{ paired.logicam_only_correct }}</strong><span>Baseline-only correct: {{ paired.baseline_only_correct }}</span></div><div class="insight"><span class="subtle">Token-limit unparsed</span><strong>{{ diagnostics.logicam_unparsed_at_token_limit }}</strong><span>Baseline: {{ diagnostics.baseline_unparsed_at_token_limit }}</span></div></section>
    <section class="grid two"><div class="panel"><h2>Accuracy By Domain</h2><div class="legend"><span><i class="swatch base-swatch"></i>Baseline</span><span><i class="swatch logicam-swatch"></i>LogiCAM</span></div><div class="bar-table">{% for row in domain_rows %}<div class="bar-row"><strong>{{ row.label }}</strong><div class="track" title="Baseline {{ row.baseline }}%, LogiCAM {{ row.logicam }}%"><div class="bar base" style="width: {{ row.width_baseline }}%"></div><div class="bar logicam" style="width: {{ row.width_logicam }}%"></div></div><span class="{{ 'delta-pos' if row.delta >= 0 else 'delta-neg' }}">{{ "%+.2f"|format(row.delta) }} pp</span></div>{% endfor %}</div></div><div class="panel"><h2>Reasoning Quality</h2><div class="legend"><span><i class="swatch base-swatch"></i>Baseline</span><span><i class="swatch logicam-swatch"></i>LogiCAM</span></div><div class="bar-table">{% for row in reasoning_rows %}<div class="bar-row reasoning"><strong>{{ row.label }}</strong><div class="track"><div class="bar base" style="width: {{ row.width_baseline }}%"></div><div class="bar logicam" style="width: {{ row.width_logicam }}%"></div></div><span class="{{ 'delta-pos' if row.delta >= 0 else 'delta-neg' }}">{{ "%+.4f"|format(row.delta) }}</span></div>{% endfor %}</div></div></section>
    <section class="grid two"><div class="panel"><h2>Accuracy By Symbol</h2><div class="bar-table">{% for row in symbol_rows %}<div class="bar-row"><strong>{{ row.label }}</strong><div class="track"><div class="bar base" style="width: {{ row.width_baseline }}%"></div><div class="bar logicam" style="width: {{ row.width_logicam }}%"></div></div><span class="{{ 'delta-pos' if row.delta >= 0 else 'delta-neg' }}">{{ "%+.2f"|format(row.delta) }} pp</span></div>{% endfor %}</div></div><div class="panel"><h2>Accuracy By Depth</h2><div class="bar-table">{% for row in depth_rows %}<div class="bar-row"><strong>{{ row.label }}</strong><div class="track"><div class="bar base" style="width: {{ row.width_baseline }}%"></div><div class="bar logicam" style="width: {{ row.width_logicam }}%"></div></div><span class="{{ 'delta-pos' if row.delta >= 0 else 'delta-neg' }}">{{ "%+.2f"|format(row.delta) }} pp</span></div>{% endfor %}</div></div></section>
    <section class="panel" style="margin-top:16px"><h2>Sample Explorer</h2><div class="controls"><input id="search" type="search" placeholder="Search id, domain, symbol, prediction"><select id="category"><option value="all">All outcomes</option><option value="logicam_fix">LogiCAM fixes</option><option value="regression">Regressions</option><option value="logicam_unparsed">LogiCAM unparsed</option><option value="both_correct">Both correct</option><option value="both_incorrect">Both incorrect</option></select><select id="domain"><option value="all">All domains</option>{% for row in domain_rows %}<option value="{{ row.label }}">{{ row.label }}</option>{% endfor %}</select><button id="reset">Reset</button><span class="subtle" id="row-count"></span></div><div class="table-wrap"><table id="sample-table"><thead><tr><th>ID</th><th>Outcome</th><th>Domain</th><th>Symbol</th><th>Depth</th><th>Answer</th><th>Baseline</th><th>LogiCAM</th><th>Status</th></tr></thead><tbody>{% for row in samples %}<tr data-category="{{ row.category }}" data-domain="{{ row.domain }}" data-search="{{ (row.id ~ ' ' ~ row.domain ~ ' ' ~ row.symbol ~ ' ' ~ row.baseline_prediction ~ ' ' ~ row.logicam_prediction ~ ' ' ~ row.category)|lower }}"><td><button class="sample-button" data-id="{{ row.id }}">{{ row.id }}</button></td><td><span class="pill {{ row.category }}">{{ row.category.replace('_', ' ') }}</span></td><td>{{ row.domain }}</td><td>{{ row.symbol }}</td><td>{{ row.depth }}</td><td>{{ row.answer }}</td><td>{{ row.baseline_prediction }}</td><td>{{ row.logicam_prediction }}</td><td>{{ row.baseline_status }} / {{ row.logicam_status }}</td></tr>{% endfor %}</tbody></table></div></section>
  </main>
  <dialog id="sample-modal"><div class="modal-head"><div><h2 id="modal-title">Sample</h2><div class="subtle" id="modal-meta"></div></div><button id="modal-close">Close</button></div><div class="modal-body" id="modal-body"></div></dialog>
  <script>
    const rows=Array.from(document.querySelectorAll('#sample-table tbody tr'));const search=document.querySelector('#search');const category=document.querySelector('#category');const domain=document.querySelector('#domain');const rowCount=document.querySelector('#row-count');const reset=document.querySelector('#reset');function applyFilters(){const q=search.value.trim().toLowerCase();const cat=category.value;const dom=domain.value;let visible=0;for(const row of rows){const show=(!q||row.dataset.search.includes(q))&&(cat==='all'||row.dataset.category===cat)&&(dom==='all'||row.dataset.domain===dom);row.style.display=show?'':'none';if(show)visible+=1}rowCount.textContent=visible+' / '+rows.length+' rows'}[search,category,domain].forEach(el=>el.addEventListener('input',applyFilters));reset.addEventListener('click',()=>{search.value='';category.value='all';domain.value='all';applyFilters()});applyFilters();
    const modal=document.querySelector('#sample-modal');const modalTitle=document.querySelector('#modal-title');const modalMeta=document.querySelector('#modal-meta');const modalBody=document.querySelector('#modal-body');document.querySelector('#modal-close').addEventListener('click',()=>modal.close());function safe(value){return String(value??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]))}function resultBlock(name,record){if(!record)return '<div><h3>'+name+'</h3><p class="subtle">Missing record</p></div>';const tokens=record.usage&&record.usage.completion_tokens;return '<div><h3>'+name+': '+safe(record.status)+' · prediction '+safe(record.prediction)+'</h3><div class="subtle">tokens '+safe(tokens)+' · latency '+safe(record.latency_seconds)+'s</div><pre>'+safe(record.output)+'</pre></div>'}document.querySelectorAll('.sample-button').forEach(button=>{button.addEventListener('click',async()=>{const id=button.dataset.id;modalTitle.textContent=id;modalMeta.textContent='Loading...';modalBody.innerHTML='';modal.showModal();const response=await fetch('/api/sample/'+encodeURIComponent(id));const sample=await response.json();modalMeta.textContent=sample.domain+' · '+sample.symbol+' · depth '+sample.depth+' · answer '+sample.answer;const choices=sample.choices&&sample.choices.length?'<h3>Choices</h3><p>'+sample.choices.map(safe).join('<br>')+'</p>':'';modalBody.innerHTML='<h3>Question</h3><p>'+safe(sample.question)+'</p>'+choices+'<h3>Context</h3><p>'+safe(sample.context)+'</p><div class="split">'+resultBlock('Baseline',sample.baseline)+resultBlock('LogiCAM',sample.logicam)+'</div>'})});
  </script>
</body>
</html>
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5057)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
