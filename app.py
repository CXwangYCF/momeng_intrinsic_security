#!/usr/bin/env python3
"""Interactive dashboard for live LogiCAM-JSON MuSLR results."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, render_template, send_file

WORKSPACE = Path(__file__).resolve().parents[2]
load_dotenv(WORKSPACE / ".env", override=False)
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from muslr_core.runtime import Sample, load_jsonl, load_samples  # noqa: E402


def _existing_default_predictions() -> Path:
    configured = (
        os.getenv("MUSLR_JSON_VIS_INPUT")
        or os.getenv("MUSLR_LOGICAM_JSON_OUTPUT")
        or os.getenv("MUSLR_JSON_OUTPUT")
    )
    if configured and Path(configured).is_file():
        return Path(configured).resolve()

    output_dir = Path(os.getenv("MUSLR_OUTPUT_DIR", WORKSPACE / "outputs"))
    candidates = sorted(
        output_dir.glob("muslr_logicam_json*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No LogiCAM-JSON result found under {output_dir}")
    return candidates[0].resolve()


def _number(record: dict[str, Any], group: str, key: str) -> float | None:
    nested = record.get(group)
    if not isinstance(nested, dict):
        return None
    value = nested.get(key)
    return float(value) if isinstance(value, int | float) else None


def _safe_average(values: list[float]) -> float | None:
    return round(mean(values), 2) if values else None


def _breakdown(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(key) or "unknown")].append(record)

    rows = []
    for label, group in groups.items():
        completed = sum(record.get("status") == "completed" for record in group)
        correct = sum(record.get("correct") is True for record in group)
        rows.append(
            {
                "label": label.upper() if key == "symbol" else label,
                "total": len(group),
                "completed": completed,
                "correct": correct,
                "accuracy": round(100 * correct / len(group), 2),
                "valid_accuracy": round(100 * correct / completed, 2) if completed else None,
            }
        )
    return sorted(rows, key=lambda row: (-row["total"], row["label"]))


def _token_histogram(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = [
        ("≤512", 0, 512),
        ("513–768", 513, 768),
        ("769–1024", 769, 1024),
        ("1025–1536", 1025, 1536),
        (">1536", 1537, float("inf")),
    ]
    counts = Counter()
    for record in records:
        tokens = _number(record, "usage", "completion_tokens")
        if tokens is None:
            continue
        for label, lower, upper in buckets:
            if lower <= tokens <= upper:
                counts[label] += 1
                break
    return [{"label": label, "count": counts[label]} for label, _, _ in buckets]


def build_payload(
    predictions_path: Path,
    samples: list[Sample],
    dataset_path: Path,
) -> dict[str, Any]:
    records = load_jsonl(predictions_path)
    output_records: list[dict[str, Any]] = []

    for record in records:
        row_index = record.get("row_index")
        sample = samples[row_index] if isinstance(row_index, int) and 0 <= row_index < len(samples) else None
        enriched = dict(record)
        enriched.pop("output", None)
        enriched["sample"] = (
            {
                "id": sample.id,
                "row_index": sample.row_index,
                "context": sample.context,
                "question": sample.question,
                "choices": sample.choices,
                "answer": sample.answer,
                "domain": sample.domain,
                "symbol": sample.symbol.upper(),
                "depth": sample.depth,
                "image_url": f"/image/{sample.row_index}",
            }
            if sample
            else None
        )
        output_records.append(enriched)

    total = len(records)
    completed = sum(record.get("status") == "completed" for record in records)
    correct = sum(record.get("correct") is True for record in records)
    tokens = [
        value
        for record in records
        if (value := _number(record, "usage", "completion_tokens")) is not None
    ]
    latencies = [
        float(record["latency_seconds"])
        for record in records
        if isinstance(record.get("latency_seconds"), int | float)
    ]
    status_counts = Counter(str(record.get("status", "unknown")) for record in records)
    warning_count = sum(bool(record.get("consistency_warnings")) for record in records)

    return {
        "meta": {
            "predictions_path": str(predictions_path),
            "predictions_name": predictions_path.name,
            "updated_at": datetime.fromtimestamp(predictions_path.stat().st_mtime).isoformat(timespec="seconds"),
            "dataset_path": str(dataset_path),
            "dataset_total": len(samples),
            "model": next((record.get("model") for record in reversed(records) if record.get("model")), "-"),
        },
        "summary": {
            "total": total,
            "completed": completed,
            "correct": correct,
            "strict_accuracy": round(100 * correct / total, 2) if total else 0,
            "valid_accuracy": round(100 * correct / completed, 2) if completed else 0,
            "structured_rate": round(100 * completed / total, 2) if total else 0,
            "dataset_progress": round(100 * total / len(samples), 2) if samples else 0,
            "mean_completion_tokens": _safe_average(tokens),
            "max_completion_tokens": max(tokens, default=None),
            "mean_latency_seconds": _safe_average(latencies),
            "status_counts": dict(status_counts),
            "warning_count": warning_count,
        },
        "by_symbol": _breakdown(records, "symbol"),
        "by_domain": _breakdown(records, "domain"),
        "token_histogram": _token_histogram(records),
        "records": output_records,
    }


def _local_image_path(sample: Sample, dataset_path: Path) -> Path | None:
    value = sample.image
    if isinstance(value, dict):
        value = value.get("path") or value.get("local_path") or value.get("filename")
    if not isinstance(value, str) or not value.strip() or value.startswith(("http://", "https://", "data:")):
        return None

    candidate = Path(value)
    dataset_dir = dataset_path if dataset_path.is_dir() else dataset_path.parent
    candidates = [candidate] if candidate.is_absolute() else [
        dataset_dir / candidate,
        dataset_dir / candidate.name,
        dataset_dir / "images" / candidate.name,
    ]
    return next((path.resolve() for path in candidates if path.is_file()), None)


def create_app(predictions_path: Path, dataset_path: Path) -> Flask:
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
    samples = load_samples(dataset_path)

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/data")
    def api_data() -> Any:
        if not predictions_path.is_file():
            return jsonify({"error": f"Result file not found: {predictions_path}"}), 404
        return jsonify(build_payload(predictions_path, samples, dataset_path))

    @app.get("/image/<int:row_index>")
    def image(row_index: int) -> Any:
        if not 0 <= row_index < len(samples):
            abort(404)
        path = _local_image_path(samples[row_index], dataset_path)
        if path is None:
            abort(404)
        return send_file(path, conditional=True, max_age=3600)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=None, help="LogiCAM-JSON JSONL file")
    parser.add_argument("--data", type=Path, default=Path(os.getenv("MUSLR_DATA", WORKSPACE / "data")))
    parser.add_argument("--host", default=os.getenv("MUSLR_JSON_DASHBOARD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MUSLR_JSON_DASHBOARD_PORT", "5060")))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = (args.predictions or _existing_default_predictions()).resolve()
    print(f"LogiCAM-JSON dashboard: http://127.0.0.1:{args.port}")
    print(f"Watching result file: {predictions}")
    create_app(predictions, args.data.resolve()).run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=args.debug,
    )


if __name__ == "__main__":
    main()
