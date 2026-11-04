#!/usr/bin/env python3
"""Build the MuSLR paper Table 1 comparison with our local runs."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from muslr_core.runtime import load_samples


WORKSPACE = Path(__file__).resolve().parents[2]
OUTPUTS = WORKSPACE.parent / "outputs"

DOMAINS = ("Healthcare", "Traffic", "Sports", "Ent.", "Social", "Science", "Finance", "General")
SYMBOLS = ("PL", "FOL", "NM")
DOMAIN_KEYS = {
    "Healthcare": "medical",
    "Traffic": "traffic",
    "Sports": "Sports",
    "Ent.": "Entertainment",
    "Social": "Social/Humanity",
    "Science": "Science",
    "Finance": "finance",
    "General": "General Knowledge/Everyday Knowledge",
}

# Values transcribed from Table 1 of arXiv:2509.25851v2.
# The paper reports the method-level overall accuracies in Section 6.2.
PAPER_ROWS: tuple[tuple[str, str, float, tuple[tuple[float, ...], ...]], ...] = (
    (
        "Qwen2.5-VL-7B-Instruct",
        "Open-weight, 3-shot CoT",
        41.63,
        (
            (50.00, 33.33, 33.33, 42.67, 36.49, 48.54, 54.17, 46.51),
            (0.00, 42.86, 50.00, 40.00, 66.67, 16.67, 22.22, 25.00),
            (43.18, 25.93, 43.75, 35.42, 23.26, 43.75, 54.24, 37.50),
        ),
    ),
    (
        "Llava-1.5-7B",
        "Open-weight, 3-shot CoT",
        35.13,
        (
            (20.45, 30.95, 37.18, 32.00, 22.97, 34.95, 27.08, 43.02),
            (0.00, 57.14, 50.00, 20.00, 44.44, 66.67, 55.56, 50.00),
            (31.82, 37.04, 45.31, 43.75, 39.53, 47.06, 25.42, 45.31),
        ),
    ),
    (
        "InternVL3-8B",
        "Open-weight, 3-shot CoT",
        45.20,
        (
            (57.95, 42.86, 37.97, 44.00, 37.84, 46.60, 51.04, 50.00),
            (50.00, 42.86, 50.00, 20.00, 66.67, 50.00, 22.22, 50.00),
            (38.64, 29.63, 46.88, 35.42, 46.51, 49.02, 45.76, 43.08),
        ),
    ),
    (
        "InstructBLIP-Vicuna-13B",
        "Open-weight, 3-shot CoT",
        35.59,
        (
            (42.05, 33.33, 39.20, 26.67, 36.49, 29.13, 40.62, 25.58),
            (50.00, 28.57, 25.00, 40.00, 55.56, 16.67, 22.22, 25.00),
            (52.27, 40.74, 53.12, 31.25, 44.19, 35.29, 2.34, 30.77),
        ),
    ),
    (
        "Claude-3.7-Sonnet",
        "Closed-weight, 3-shot CoT",
        33.49,
        (
            (44.32, 26.19, 24.36, 26.67, 28.38, 35.92, 36.46, 34.88),
            (50.00, 14.29, 50.00, 20.00, 55.56, 0.00, 44.44, 75.00),
            (29.55, 37.04, 32.81, 29.17, 30.23, 43.14, 38.98, 31.25),
        ),
    ),
    (
        "GPT-4o",
        "Closed-weight, 3-shot CoT",
        38.93,
        (
            (45.45, 40.48, 33.33, 37.50, 34.72, 37.00, 28.99, 43.90),
            (0.00, 14.29, 25.00, 0.00, 37.50, 33.33, 50.00, 33.33),
            (52.27, 48.15, 35.48, 50.00, 52.38, 45.10, 41.46, 32.81),
        ),
    ),
    (
        "GPT-4.1",
        "Closed-weight, 3-shot CoT",
        46.84,
        (
            (54.55, 50.00, 44.30, 41.33, 33.78, 43.69, 45.83, 51.16),
            (0.00, 14.29, 50.00, 20.00, 44.44, 16.67, 33.33, 0.00),
            (47.73, 59.26, 46.88, 50.00, 53.49, 56.86, 61.02, 40.62),
        ),
    ),
    (
        "LogiCAM (GPT-4.1)",
        "Paper method",
        60.97,
        (
            (63.64, 61.90, 58.23, 64.00, 56.76, 57.28, 53.68, 67.44),
            (50.00, 60.42, 50.00, 60.00, 44.44, 40.00, 75.00, 75.00),
            (63.64, 66.67, 58.23, 60.42, 74.42, 64.71, 74.14, 55.38),
        ),
    ),
)


def _load_summary(name: str) -> dict[str, Any]:
    path = OUTPUTS / name
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_for_paper() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, setting, overall, values in PAPER_ROWS:
        for symbol, row_values in zip(SYMBOLS, values):
            rows.append(
                {
                    "source": "Paper",
                    "method": method,
                    "setting": setting,
                    "symbol": symbol,
                    "overall": overall if symbol == "PL" else None,
                    **dict(zip(DOMAINS, row_values)),
                }
            )
    return rows


def _rows_for_ours() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filename, method in (
        ("baseline_ripo_3gpu.summary.json", "Qwen3-VL-8B + baseline CoT (ours)"),
        ("logicam_ripo_3gpu.summary.json", "Qwen3-VL-8B + LogiCAM (ours)"),
    ):
        summary = _load_summary(filename)
        overall = summary["overall"]["accuracy_end_to_end_percent"]
        for row in summary["paper_table"]["rows"]:
            rows.append(
                {
                    "source": "Ours",
                    "method": method,
                    "setting": "local Qwen3-VL-8B, 3 vLLM endpoints",
                    "symbol": row["Symbol"],
                    "overall": overall if row["Symbol"] == "PL" else None,
                    **{domain: row[domain] for domain in DOMAINS},
                }
            )
    return rows


def _dataset_counts() -> Counter[tuple[str, str]]:
    data_path = Path(os.environ.get("MUSLR_DATA", Path(__file__).resolve().parents[2] / "data"))
    return Counter((sample.symbol.upper(), sample.domain) for sample in load_samples(data_path))


def _domain_rows_for_paper() -> list[dict[str, Any]]:
    counts = _dataset_counts()
    rows: list[dict[str, Any]] = []
    for method, setting, overall, values in PAPER_ROWS:
        by_symbol = {symbol: dict(zip(DOMAINS, row_values)) for symbol, row_values in zip(SYMBOLS, values)}
        domain_values: dict[str, float] = {}
        for domain in DOMAINS:
            raw_domain = DOMAIN_KEYS[domain]
            total = sum(counts[(symbol, raw_domain)] * by_symbol[symbol][domain] for symbol in SYMBOLS)
            denominator = sum(counts[(symbol, raw_domain)] for symbol in SYMBOLS)
            domain_values[domain] = round(total / denominator, 2) if denominator else 0.0
        rows.append(
            {
                "source": "Paper",
                "method": method,
                "setting": setting,
                "overall": overall,
                **domain_values,
            }
        )
    return rows


def _domain_rows_for_ours() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filename, method, label in (
        ("baseline_ripo_3gpu.summary.json", "Qwen3-VL-8B + baseline CoT (ours)", "baseline"),
        ("logicam_ripo_3gpu.summary.json", "Qwen3-VL-8B + LogiCAM (ours)", "logicam"),
    ):
        summary = _load_summary(filename)
        rows.append(
            {
                "source": "Ours",
                "method": method,
                "setting": "local Qwen3-VL-8B, 3 vLLM endpoints",
                "overall": summary["overall"]["accuracy_end_to_end_percent"],
                **{
                    domain: summary["by_domain"][DOMAIN_KEYS[domain]]["accuracy_end_to_end_percent"]
                    for domain in DOMAINS
                },
            }
        )
    return rows


def _ours_domain_diagnostics() -> list[dict[str, Any]]:
    baseline = _load_summary("baseline_ripo_3gpu.summary.json")
    logicam = _load_summary("logicam_ripo_3gpu.summary.json")
    rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        base = baseline["by_domain"][DOMAIN_KEYS[domain]]
        logic = logicam["by_domain"][DOMAIN_KEYS[domain]]
        rows.append(
            {
                "domain": domain,
                "samples": base["records"],
                "baseline_e2e": base["accuracy_end_to_end_percent"],
                "logicam_e2e": logic["accuracy_end_to_end_percent"],
                "e2e_delta": round(logic["accuracy_end_to_end_percent"] - base["accuracy_end_to_end_percent"], 2),
                "baseline_parsed": base["accuracy_scored_percent"],
                "logicam_parsed": logic["accuracy_scored_percent"],
                "parsed_delta": round(logic["accuracy_scored_percent"] - base["accuracy_scored_percent"], 2),
                "baseline_coverage": base["coverage_percent"],
                "logicam_coverage": logic["coverage_percent"],
                "coverage_delta": round(logic["coverage_percent"] - base["coverage_percent"], 2),
            }
        )
    return rows


def _add_domain_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach competition ranks for the domain table, with ties sharing a rank."""
    rank_columns = ["overall", *DOMAINS]
    for row in rows:
        row["ranks"] = {}
    for column in rank_columns:
        ordered = sorted(rows, key=lambda row: float(row[column]), reverse=True)
        last_value: float | None = None
        current_rank = 0
        for position, row in enumerate(ordered, start=1):
            value = float(row[column])
            if last_value is None or value != last_value:
                current_rank = position
                last_value = value
            if current_rank <= 3:
                row["ranks"][column] = current_rank
    return rows


def method_label(row: dict[str, Any]) -> str:
    """Collapse source/setting metadata into compact paper-style method markers."""
    if row["source"] == "Ours":
        marker = "§"
    elif "LogiCAM" in row["method"]:
        marker = "‡"
    elif "Open-weight" in row["setting"]:
        marker = "*"
    else:
        marker = "†"
    return f"{row['method']}{marker}"


def build_comparison() -> dict[str, Any]:
    domain_rows = _add_domain_ranks(_domain_rows_for_paper() + _domain_rows_for_ours())
    return {
        "title": "MuSLR Table 1: Paper Methods vs Our Qwen3-VL-8B Runs",
        "metric": "End-to-end exact-match accuracy (%) by Symbol x Domain; unparsed outputs count as incorrect.",
        "source_url": "https://arxiv.org/html/2509.25851",
        "columns": ["Method", "Symbol", "Overall E2E", *DOMAINS],
        "rows": _rows_for_paper() + _rows_for_ours(),
        "domain_columns": ["Method", "Overall E2E", *DOMAINS],
        "domain_rows": domain_rows,
        "ours_domain_diagnostics": _ours_domain_diagnostics(),
        "notes": [
            "* Paper open-weight VLM with three-shot CoT; † paper closed-weight VLM with three-shot CoT; ‡ paper LogiCAM on GPT-4.1; § our local anom-pi run on three vLLM endpoints.",
            "Paper domain and symbol cells are transcribed from Table 1; paper method-level overall values are reported in Section 6.2.",
            "The paper reports LogiCAM as GPT-4.1 + LogiCAM. Its 60.97 overall value is GPT-4.1's 46.84 plus the reported +14.13 percentage-point gain.",
            "The paper-only domain summary is weighted from its Symbol x Domain cells using the 1,093-sample domain distribution; the paper does not print this separate domain-total table.",
            "Our Qwen3-VL-8B deployment is served through the local anom-pi client and is not the same checkpoint as the paper's Qwen2.5-VL-7B-Instruct; these rows are a reference comparison, not a controlled model comparison.",
        ],
    }


def markdown_report(data: dict[str, Any]) -> str:
    columns = data["columns"]
    lines = [f"# {data['title']}", "", f"**Metric:** {data['metric']}", "", f"**Paper source:** [{data['source_url']}]({data['source_url']})", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join([":---"] * 2 + [":---:"] * (len(columns) - 2)) + " |")
    previous_method = None
    for row in data["rows"]:
        method = row["method"]
        method_cell = method_label(row) if method != previous_method else ""
        values = [method_cell, row["symbol"]]
        overall = row["overall"]
        bold = "LogiCAM" in row["method"]
        values.append(f"**{overall:.2f}**" if overall is not None and bold else (f"{overall:.2f}" if overall is not None else ""))
        values.extend(f"**{float(row[column]):.2f}**" if bold else f"{float(row[column]):.2f}" for column in DOMAINS)
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
        previous_method = method
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in data["notes"])
    lines.extend(["", "## Domain Overall E2E Accuracy", "", "| " + " | ".join(data["domain_columns"]) + " |", "| " + " | ".join([":---"] + [":---:"] * (len(data["domain_columns"]) - 1)) + " |"])
    for row in data["domain_rows"]:
        bold = "LogiCAM" in row["method"]
        overall = f"{row['overall']:.2f}"
        if row["ranks"].get("overall"):
            overall += f" ({row['ranks']['overall']})"
        values = [method_label(row), f"**{overall}**" if bold else overall]
        for column in DOMAINS:
            value = f"{float(row[column]):.2f}"
            if row["ranks"].get(column):
                value += f" ({row['ranks'][column]})"
            values.append(f"**{value}**" if bold else value)
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Our Domain Diagnostics", "", "| Domain | Samples | Baseline E2E | LogiCAM E2E | E2E delta | Baseline parsed-only | LogiCAM parsed-only | Parsed delta | Baseline coverage | LogiCAM coverage | Coverage delta |", "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in data["ours_domain_diagnostics"]:
        lines.append("| {domain} | {samples} | {baseline_e2e:.2f} | {logicam_e2e:.2f} | {e2e_delta:+.2f} | {baseline_parsed:.2f} | {logicam_parsed:.2f} | {parsed_delta:+.2f} | {baseline_coverage:.2f} | {logicam_coverage:.2f} | {coverage_delta:+.2f} |".format(**row))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path, default=OUTPUTS / "paper_vs_ours.table.md")
    parser.add_argument("--json", type=Path, default=OUTPUTS / "paper_vs_ours.table.json")
    args = parser.parse_args()
    data = build_comparison()
    args.markdown.write_text(markdown_report(data), encoding="utf-8")
    args.json.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.markdown)
    print(args.json)


if __name__ == "__main__":
    main()
