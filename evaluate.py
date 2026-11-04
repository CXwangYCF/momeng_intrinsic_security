#!/usr/bin/env python3
"""Evaluate MuSLR predictions and produce human- and machine-readable reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from rich import box
from rich.console import Console
from rich.table import Table

from evaluation.reasoning_metrics import compute_reasoning_metrics
from muslr_core.runtime import Sample, load_jsonl, load_samples, sample_index


PAPER_DOMAINS = (
    ("Healthcare", "medical"),
    ("Traffic", "traffic"),
    ("Sports", "Sports"),
    ("Ent.", "Entertainment"),
    ("Social", "Social/Humanity"),
    ("Science", "Science"),
    ("Finance", "finance"),
    ("General", "General Knowledge/Everyday Knowledge"),
)
PAPER_SYMBOLS = ("pl", "fol", "nm")
COMPACT_DOMAINS = (
    ("Traffic", "traffic"),
    ("Science", "Science"),
    ("Finance", "finance"),
)
DEPTH_BUCKETS = (
    ("2-3", {2, 3}),
    ("4-5", {4, 5}),
    ("6-7", {6, 7}),
    ("8-9", {8, 9}),
)


def _percent(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 2) if denominator else 0.0


def index_predictions(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(record["id"]): record
        for record in records
        if record.get("id") is not None
    }


def group_metrics(
    samples: Iterable[Sample],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    group = list(samples)
    matched = [sample for sample in group if sample.id in predictions]
    parsed = [
        sample
        for sample in matched
        if isinstance(predictions[sample.id].get("prediction"), str)
        and predictions[sample.id]["prediction"].strip()
    ]
    correct = sum(
        predictions[sample.id]["prediction"].upper() == sample.answer.upper()
        for sample in parsed
    )
    status_counts = Counter(
        str(predictions[sample.id].get("status", "unknown")) for sample in matched
    )
    total = len(group)
    return {
        "records": total,
        "prediction_records": len(matched),
        "parsed": len(parsed),
        "correct": correct,
        "missing": total - len(matched),
        "unparsed": len(matched) - len(parsed),
        "coverage_percent": _percent(len(parsed), total),
        "accuracy_scored_percent": _percent(correct, len(parsed)),
        "accuracy_end_to_end_percent": _percent(correct, total),
        "status_counts": dict(sorted(status_counts.items())),
    }


def paper_table_rows(
    records: list[dict[str, Any]],
    samples_by_id: dict[str, Sample],
    model: str,
) -> list[dict[str, Any]]:
    """Build paper Table 1-style Symbol x Domain end-to-end accuracy rows."""
    predictions = index_predictions(records)
    rows: list[dict[str, Any]] = []
    for symbol in PAPER_SYMBOLS:
        row: dict[str, Any] = {"Model": model, "Symbol": symbol.upper()}
        for column, domain in PAPER_DOMAINS:
            metrics = group_metrics(
                (
                    sample
                    for sample in samples_by_id.values()
                    if sample.symbol == symbol and sample.domain == domain
                ),
                predictions,
            )
            row[column] = metrics["accuracy_end_to_end_percent"]
        rows.append(row)
    return rows


def paper_table_markdown(rows: list[dict[str, Any]]) -> str:
    columns = ["Model", "Symbol", *(column for column, _ in PAPER_DOMAINS)]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| :--- | :--- | " + " | ".join(":---:" for _ in PAPER_DOMAINS) + " |",
    ]
    for index, row in enumerate(rows):
        values = [
            str(row["Model"]) if index == 0 else "",
            str(row["Symbol"]),
            *(f"{float(row[column]):.2f}" for column, _ in PAPER_DOMAINS),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def compact_domain_row(
    records: list[dict[str, Any]],
    samples_by_id: dict[str, Sample],
    method: str,
) -> dict[str, Any]:
    """Build a weighted E2E row for three domains and their pooled complement."""
    predictions = index_predictions(records)
    samples = list(samples_by_id.values())
    selected_domains = {domain for _, domain in COMPACT_DOMAINS}
    row: dict[str, Any] = {
        "Method": method,
        "Overall": group_metrics(samples, predictions)["accuracy_end_to_end_percent"],
    }
    for column, domain in COMPACT_DOMAINS:
        row[column] = group_metrics(
            (sample for sample in samples if sample.domain == domain),
            predictions,
        )["accuracy_end_to_end_percent"]
    other_metrics = group_metrics(
        (sample for sample in samples if sample.domain not in selected_domains),
        predictions,
    )
    row["Others"] = other_metrics["accuracy_end_to_end_percent"]
    row["Others_samples"] = other_metrics["records"]
    return row


def compact_domain_markdown(row: dict[str, Any]) -> str:
    columns = ["Method", "Overall", *(column for column, _ in COMPACT_DOMAINS), "Others"]
    values = [str(row["Method"]), *(f"{float(row[column]):.2f}" for column in columns[1:])]
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| :--- | " + " | ".join("---:" for _ in columns[1:]) + " |",
            "| " + " | ".join(values) + " |",
        ]
    )


def _load_run_metadata(predictions_path: Path) -> tuple[Path, dict[str, Any]]:
    path = predictions_path.with_suffix(predictions_path.suffix + ".meta.json")
    if not path.is_file():
        return path, {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return path, value if isinstance(value, dict) else {}


def _usage_totals(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for record in records:
        usage = record.get("usage")
        if not isinstance(usage, dict):
            continue
        for key in totals:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                totals[key] += int(value)
    return totals


def build_summary(
    *,
    samples_by_id: dict[str, Sample],
    records: list[dict[str, Any]],
    model: str,
    data_path: Path,
    predictions_path: Path,
) -> dict[str, Any]:
    predictions = index_predictions(records)
    sample_values = list(samples_by_id.values())
    record_ids = [str(record["id"]) for record in records if record.get("id") is not None]
    known_ids = set(samples_by_id)
    unknown_ids = sorted(set(predictions) - known_ids)
    duplicate_ids = len(record_ids) - len(set(record_ids))

    metadata_path, metadata = _load_run_metadata(predictions_path)
    configuration = metadata.get("configuration", {})
    if not isinstance(configuration, dict):
        configuration = {}
    max_tokens = configuration.get("max_tokens")
    max_tokens = int(max_tokens) if isinstance(max_tokens, (int, float)) else None

    unparsed_records = [
        predictions[sample.id]
        for sample in sample_values
        if sample.id in predictions
        and not isinstance(predictions[sample.id].get("prediction"), str)
    ]
    token_limit_candidates = 0
    for record in unparsed_records:
        usage = record.get("usage") or {}
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if record.get("finish_reason") == "length" or (
            max_tokens is not None
            and isinstance(completion_tokens, (int, float))
            and completion_tokens >= max_tokens
        ):
            token_limit_candidates += 1

    by_symbol = {
        symbol: group_metrics(
            (sample for sample in sample_values if sample.symbol == symbol), predictions
        )
        for symbol in PAPER_SYMBOLS
    }
    by_domain = {
        domain: group_metrics(
            (sample for sample in sample_values if sample.domain == domain), predictions
        )
        for _, domain in PAPER_DOMAINS
    }
    depths = sorted({sample.depth for sample in sample_values if sample.depth is not None})
    by_depth = {
        str(depth): group_metrics(
            (sample for sample in sample_values if sample.depth == depth), predictions
        )
        for depth in depths
    }
    by_depth_bucket = {
        label: group_metrics(
            (sample for sample in sample_values if sample.depth in depth_values), predictions
        )
        for label, depth_values in DEPTH_BUCKETS
    }
    shallow_accuracy = by_depth_bucket["2-3"]["accuracy_end_to_end_percent"]
    deep_accuracy = by_depth_bucket["8-9"]["accuracy_end_to_end_percent"]
    depth_drop_points = round(shallow_accuracy - deep_accuracy, 2)
    depth_relative_drop = (
        round(100 * depth_drop_points / shallow_accuracy, 2) if shallow_accuracy else 0.0
    )
    endpoint_counts = Counter(
        str(record["usage"]["endpoint"])
        for record in records
        if isinstance(record.get("usage"), dict) and record["usage"].get("endpoint")
    )
    table_rows = paper_table_rows(records, samples_by_id, model)
    compact_row = compact_domain_row(records, samples_by_id, model)

    return {
        "run": {
            "model": model,
            "data": str(data_path.resolve()),
            "predictions": str(predictions_path.resolve()),
            "metadata": str(metadata_path.resolve()) if metadata_path.is_file() else None,
            "method": configuration.get("method"),
            "max_tokens": max_tokens,
        },
        "overall": group_metrics(sample_values, predictions),
        "by_symbol": by_symbol,
        "by_domain": by_domain,
        "by_depth": by_depth,
        "by_depth_bucket": by_depth_bucket,
        "depth_robustness": {
            "shallow_bucket": "2-3",
            "deep_bucket": "8-9",
            "shallow_accuracy_percent": shallow_accuracy,
            "deep_accuracy_percent": deep_accuracy,
            "absolute_drop_percentage_points": depth_drop_points,
            "relative_drop_percent": depth_relative_drop,
        },
        "paper_table": {
            "metric": "end-to-end exact-match accuracy percent; unparsed or missing predictions count as incorrect",
            "rows": table_rows,
        },
        "compact_domain_table": {
            "metric": "end-to-end exact-match accuracy percent; unparsed or missing predictions count as incorrect",
            "others_definition": "all samples outside Traffic, Science, and Finance, pooled before computing accuracy",
            "row": compact_row,
        },
        "reasoning_quality": {
            "status": "not_requested",
            "reason": "Run with --reasoning-metrics to compute ROUGE-L, BERTScore-F1, and ROSCOE.",
        },
        "paper_analysis": {
            "cross_model_pearson": {
                "status": "requires_multiple_model_runs",
                "paper_metrics": ["Pearson(ROUGE-L, ROSCOE)", "Pearson(BERTScore-F1, ROSCOE)"],
                "minimum_runs": 3,
                "paper_reported": {
                    "rouge_l_vs_roscoe_r": 0.25,
                    "bertscore_f1_vs_roscoe_r": 0.65,
                },
            },
            "ablation": {
                "status": "requires_ablation_runs",
                "paper_metrics": [
                    "accuracy delta without symbolic reasoning",
                    "accuracy delta without heuristic reasoning",
                    "accuracy delta without premise selection",
                ],
                "paper_reported": {
                    "full_logicam_accuracy_percent": 61.34,
                    "without_symbolic_accuracy_percent": 56.20,
                    "without_symbolic_drop_points": 5.14,
                    "without_heuristic_accuracy_percent": 57.89,
                    "without_heuristic_drop_points": 3.45,
                    "without_premise_selection_accuracy_percent": 58.07,
                    "without_premise_selection_drop_points": 3.27,
                },
            },
            "manual_error_analysis": {
                "status": "requires_manual_annotations",
                "sample_protocol": "100 domain- and symbol-balanced examples per model",
                "categories": [
                    "incorrect application of logical rules",
                    "failure to supplement heuristic commonsense",
                    "overlooking critical visual details",
                    "visual-textual logical misalignment",
                    "improper heuristic shortcuts",
                    "object misperception",
                ],
                "paper_reported_logicam_percent": {
                    "incorrect application of logical rules": 5,
                    "failure to supplement heuristic commonsense": 5,
                    "overlooking critical visual details": 10,
                    "visual-textual logical misalignment": 67,
                    "improper heuristic shortcuts": 13,
                    "object misperception": 0,
                },
            },
        },
        "diagnostics": {
            "raw_prediction_rows": len(records),
            "unique_prediction_ids": len(predictions),
            "duplicate_prediction_rows": duplicate_ids,
            "unknown_prediction_ids": len(unknown_ids),
            "unknown_prediction_id_examples": unknown_ids[:10],
            "unparsed_at_token_limit": token_limit_candidates,
            "configured_max_tokens": max_tokens,
            "endpoint_counts": dict(sorted(endpoint_counts.items())),
            "usage": _usage_totals(records),
        },
    }


def _metric_rows(metrics: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    return [
        (
            "Strict end-to-end accuracy",
            f"{metrics['accuracy_end_to_end_percent']:.2f}%",
            f"{metrics['correct']}/{metrics['records']}",
            "Primary score; missing and unparsed predictions are incorrect",
        ),
        (
            "Parsed-only accuracy",
            f"{metrics['accuracy_scored_percent']:.2f}%",
            f"{metrics['correct']}/{metrics['parsed']}",
            "Behavior of the released evaluator; excludes unparsed outputs",
        ),
        (
            "Answer coverage",
            f"{metrics['coverage_percent']:.2f}%",
            f"{metrics['parsed']}/{metrics['records']}",
            "Share of dataset with a parseable final answer",
        ),
    ]


def _add_metric_columns(table: Table) -> None:
    table.add_column("Group", style="bold")
    table.add_column("Samples", justify="right")
    table.add_column("Parsed", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Parsed acc.", justify="right")
    table.add_column("E2E acc.", justify="right", style="bold cyan")


def _add_metric_row(table: Table, label: str, metrics: dict[str, Any]) -> None:
    table.add_row(
        label,
        str(metrics["records"]),
        str(metrics["parsed"]),
        str(metrics["correct"]),
        f"{metrics['coverage_percent']:.2f}%",
        f"{metrics['accuracy_scored_percent']:.2f}%",
        f"{metrics['accuracy_end_to_end_percent']:.2f}%",
    )


def print_rich_report(summary: dict[str, Any], console: Console | None = None) -> None:
    console = console or Console()
    run = summary["run"]
    overall = summary["overall"]
    console.print()
    console.rule(f"[bold]MuSLR Evaluation: {run['model']}[/bold]")
    console.print(f"[dim]Predictions:[/dim] {run['predictions']}")
    console.print(f"[dim]Dataset:[/dim]     {run['data']}")

    main = Table(title="Main Results", box=box.SIMPLE_HEAVY, show_lines=False)
    main.add_column("Metric", style="bold")
    main.add_column("Value", justify="right", style="bold cyan")
    main.add_column("Count", justify="right")
    main.add_column("Definition")
    for row in _metric_rows(overall):
        main.add_row(*row)
    console.print(main)

    reasoning = summary["reasoning_quality"]
    reasoning_table = Table(title="Reasoning Quality", box=box.SIMPLE_HEAVY)
    reasoning_table.add_column("Metric", style="bold")
    reasoning_table.add_column("Score", justify="right", style="bold cyan")
    reasoning_table.add_column("Definition")
    if reasoning["status"] == "computed":
        reasoning_table.add_row("ROUGE-L F1", f"{reasoning['rouge_l']['score']:.4f}", "Lexical overlap")
        reasoning_table.add_row(
            "BERTScore-F1",
            f"{reasoning['bertscore_f1']['score']:.4f}",
            "Semantic similarity",
        )
        reasoning_table.add_row(
            "ROSCOE",
            f"{reasoning['roscoe']['score']:.4f}",
            "Reference-based reasoning_alignment",
        )
        reasoning_table.add_row(
            "Overall mean",
            f"{reasoning['overall_mean']:.4f}",
            "Arithmetic mean of the three paper metrics",
        )
        reasoning_table.caption = (
            f"Scored {reasoning['samples']}/{reasoning['dataset_samples']} chains "
            f"on {reasoning['device']}"
        )
    else:
        reasoning_table.add_row("Status", "not computed", reasoning.get("reason", ""))
    console.print(reasoning_table)
    if reasoning["status"] == "computed":
        roscoe_details = Table(title="ROSCOE Semantic Components", box=box.SIMPLE)
        roscoe_details.add_column("Component", style="bold")
        roscoe_details.add_column("Score", justify="right")
        for name, score in reasoning["roscoe"]["components"].items():
            label = name.replace("_", " ").title()
            if name == reasoning["roscoe"]["paper_score_mapping"]:
                label += " (paper mapping)"
            roscoe_details.add_row(label, f"{score:.4f}")
        console.print(roscoe_details)

    symbol_table = Table(title="By Symbol", box=box.SIMPLE_HEAVY)
    _add_metric_columns(symbol_table)
    for symbol in PAPER_SYMBOLS:
        _add_metric_row(symbol_table, symbol.upper(), summary["by_symbol"][symbol])
    console.print(symbol_table)

    domain_table = Table(title="By Domain", box=box.SIMPLE_HEAVY)
    _add_metric_columns(domain_table)
    for label, domain in PAPER_DOMAINS:
        _add_metric_row(domain_table, label, summary["by_domain"][domain])
    console.print(domain_table)

    compact = Table(title="Compact Domain E2E Accuracy", box=box.SIMPLE_HEAVY)
    compact.add_column("Method", style="bold")
    for column in ("Overall", "Traffic", "Science", "Finance", "Others"):
        compact.add_column(
            column,
            justify="right",
            style="bold cyan" if column == "Overall" else None,
        )
    compact_row = summary["compact_domain_table"]["row"]
    compact.add_row(
        str(compact_row["Method"]),
        *(f"{float(compact_row[column]):.2f}" for column in ("Overall", "Traffic", "Science", "Finance", "Others")),
    )
    compact.caption = f"Others pools the remaining {compact_row['Others_samples']} samples"
    console.print(compact)

    paper = Table(title="Paper Table 1 Format: Symbol x Domain E2E Accuracy", box=box.SIMPLE)
    paper.add_column("Symbol", style="bold")
    for column, _ in PAPER_DOMAINS:
        paper.add_column(column, justify="right")
    for row in summary["paper_table"]["rows"]:
        paper.add_row(
            str(row["Symbol"]),
            *(f"{float(row[column]):.2f}" for column, _ in PAPER_DOMAINS),
        )
    console.print(paper)

    depth_table = Table(title="By Reasoning Depth", box=box.SIMPLE_HEAVY)
    _add_metric_columns(depth_table)
    for label, _ in DEPTH_BUCKETS:
        _add_metric_row(depth_table, label, summary["by_depth_bucket"][label])
    console.print(depth_table)

    robustness = summary["depth_robustness"]
    drop = robustness["absolute_drop_percentage_points"]
    trend_label = "drop" if drop >= 0 else "gain"
    console.print(
        "[dim]Depth 2-3 -> 8-9:[/dim] "
        f"{robustness['shallow_accuracy_percent']:.2f}% -> "
        f"{robustness['deep_accuracy_percent']:.2f}% "
        f"({trend_label} [bold]{abs(drop):.2f} pp[/bold], "
        f"{abs(robustness['relative_drop_percent']):.2f}% relative)"
    )

    exact_depth_table = Table(title="By Exact Depth", box=box.SIMPLE)
    _add_metric_columns(exact_depth_table)
    for depth, metrics in summary["by_depth"].items():
        _add_metric_row(exact_depth_table, depth, metrics)
    console.print(exact_depth_table)

    diagnostics = summary["diagnostics"]
    diagnostic_table = Table(title="Diagnostics", box=box.SIMPLE_HEAVY)
    diagnostic_table.add_column("Item", style="bold")
    diagnostic_table.add_column("Value", justify="right")
    diagnostic_table.add_row("Missing predictions", str(overall["missing"]))
    diagnostic_table.add_row("Unparsed predictions", str(overall["unparsed"]))
    diagnostic_table.add_row(
        "Unparsed at token limit",
        f"{diagnostics['unparsed_at_token_limit']}/{overall['unparsed']}",
    )
    diagnostic_table.add_row("Failed", str(overall["status_counts"].get("failed", 0)))
    diagnostic_table.add_row("Duplicate prediction rows", str(diagnostics["duplicate_prediction_rows"]))
    diagnostic_table.add_row("Unknown prediction IDs", str(diagnostics["unknown_prediction_ids"]))
    endpoint_text = ", ".join(
        f"{endpoint}: {count}" for endpoint, count in diagnostics["endpoint_counts"].items()
    ) or "not reported"
    diagnostic_table.add_row("Endpoint distribution", endpoint_text)
    console.print(diagnostic_table)


def _markdown_metric_table(groups: Iterable[tuple[str, dict[str, Any]]]) -> str:
    lines = [
        "| Group | Samples | Parsed | Correct | Coverage | Parsed accuracy | E2E accuracy |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, metrics in groups:
        lines.append(
            f"| {label} | {metrics['records']} | {metrics['parsed']} | {metrics['correct']} | "
            f"{metrics['coverage_percent']:.2f}% | {metrics['accuracy_scored_percent']:.2f}% | "
            f"{metrics['accuracy_end_to_end_percent']:.2f}% |"
        )
    return "\n".join(lines)


def markdown_report(summary: dict[str, Any]) -> str:
    run = summary["run"]
    overall = summary["overall"]
    diagnostics = summary["diagnostics"]
    main_lines = [
        "| Metric | Value | Count |",
        "| :--- | ---: | ---: |",
        f"| Strict end-to-end accuracy | {overall['accuracy_end_to_end_percent']:.2f}% | {overall['correct']}/{overall['records']} |",
        f"| Parsed-only accuracy | {overall['accuracy_scored_percent']:.2f}% | {overall['correct']}/{overall['parsed']} |",
        f"| Answer coverage | {overall['coverage_percent']:.2f}% | {overall['parsed']}/{overall['records']} |",
    ]
    symbols = [(symbol.upper(), summary["by_symbol"][symbol]) for symbol in PAPER_SYMBOLS]
    domains = [(label, summary["by_domain"][domain]) for label, domain in PAPER_DOMAINS]
    depths = [(label, summary["by_depth_bucket"][label]) for label, _ in DEPTH_BUCKETS]
    exact_depths = list(summary["by_depth"].items())
    robustness = summary["depth_robustness"]
    depth_drop = robustness["absolute_drop_percentage_points"]
    depth_trend = "drop" if depth_drop >= 0 else "gain"
    endpoint_text = ", ".join(
        f"`{endpoint}`: {count}" for endpoint, count in diagnostics["endpoint_counts"].items()
    ) or "not reported"
    reasoning = summary["reasoning_quality"]
    if reasoning["status"] == "computed":
        roscoe_component_lines = [
            "| ROSCOE semantic component | Score |",
            "| :--- | ---: |",
            *(
                f"| {name.replace('_', ' ').title()}"
                f"{' (paper mapping)' if name == reasoning['roscoe']['paper_score_mapping'] else ''} "
                f"| {score:.4f} |"
                for name, score in reasoning["roscoe"]["components"].items()
            ),
        ]
        reasoning_lines = [
            "| Metric | Score | Role |",
            "| :--- | ---: | :--- |",
            f"| ROUGE-L F1 | {reasoning['rouge_l']['score']:.4f} | Lexical overlap |",
            f"| BERTScore-F1 | {reasoning['bertscore_f1']['score']:.4f} | Semantic similarity |",
            f"| ROSCOE | {reasoning['roscoe']['score']:.4f} | Reference-based reasoning alignment |",
            f"| Overall mean | {reasoning['overall_mean']:.4f} | Mean of the three metrics |",
            "",
            f"Coverage: **{reasoning['samples']}/{reasoning['dataset_samples']} "
            f"({reasoning['coverage_percent']:.2f}%)**. Device: `{reasoning['device']}`.",
            "",
            "MuSLR does not publish its ROSCOE scorer configuration. This implementation uses the "
            "official default sentence encoder and maps the paper's single ROSCOE value to "
            "`reasoning_alignment`; all semantic submetrics are retained in the JSON summary.",
            "",
            "### ROSCOE Semantic Components",
            "",
            *roscoe_component_lines,
        ]
    else:
        reasoning_lines = [f"Not computed: {reasoning.get('reason', 'unknown reason')}"]

    analysis_lines = [
        "| Paper analysis | Paper-reported reference | This run | Requirement |",
        "| :--- | :--- | :--- | :--- |",
        "| Pearson(ROUGE-L, ROSCOE) | r = 0.25 | Not computed | At least 3 model summaries |",
        "| Pearson(BERTScore-F1, ROSCOE) | r = 0.65 | Not computed | At least 3 model summaries |",
        "| Without symbolic reasoning | 56.20%, drop 5.14 pp | Not computed | Ablation run |",
        "| Without heuristic reasoning | 57.89%, drop 3.45 pp | Not computed | Ablation run |",
        "| Without premise selection | 58.07%, drop 3.27 pp | Not computed | Ablation run |",
        "| Six-way error distribution | LogiCAM: 5/5/10/67/13/0% in category order below | Not computed | Manual labels on 100 balanced samples |",
        "",
        "Error category order: logical-rule application, missing commonsense, visual-detail omission, "
        "visual-textual misalignment, heuristic shortcuts, object misperception.",
    ]
    sections = [
        f"# MuSLR Evaluation: {run['model']}",
        "",
        f"- Predictions: `{run['predictions']}`",
        f"- Dataset: `{run['data']}`",
        f"- Method: `{run['method'] or 'unknown'}`",
        f"- Max output tokens: `{run['max_tokens'] if run['max_tokens'] is not None else 'unknown'}`",
        "",
        "## Main Results",
        "",
        *main_lines,
        "",
        "Strict end-to-end accuracy is the headline score. Missing or unparsed answers count as incorrect. "
        "Parsed-only accuracy matches the released evaluator's denominator behavior.",
        "",
        "## Reasoning Quality",
        "",
        *reasoning_lines,
        "",
        "## Accuracy Breakdown",
        "",
        "### By Symbol",
        "",
        _markdown_metric_table(symbols),
        "",
        "### By Domain",
        "",
        _markdown_metric_table(domains),
        "",
        "### Compact Domain E2E Accuracy",
        "",
        compact_domain_markdown(summary["compact_domain_table"]["row"]),
        "",
        f"`Others` pools the remaining **{summary['compact_domain_table']['row']['Others_samples']}** "
        "samples before computing accuracy; it is not an unweighted mean of domain percentages.",
        "",
        "### Symbol x Domain (Paper Table 1)",
        "",
        paper_table_markdown(summary["paper_table"]["rows"]),
        "",
        "Each cell is end-to-end exact-match accuracy for one `Symbol x Domain` group.",
        "",
        "## Depth Robustness",
        "",
        "### Paper Depth Buckets",
        "",
        _markdown_metric_table(depths),
        "",
        f"Depth 2-3 to 8-9: **{robustness['shallow_accuracy_percent']:.2f}% -> "
        f"{robustness['deep_accuracy_percent']:.2f}%**; {depth_trend} "
        f"**{abs(depth_drop):.2f} percentage points**, "
        f"**{abs(robustness['relative_drop_percent']):.2f}%** relative.",
        "",
        "### Exact Depth",
        "",
        _markdown_metric_table(exact_depths),
        "",
        "## Paper Analyses Requiring Additional Inputs",
        "",
        *analysis_lines,
        "",
        "## Diagnostics",
        "",
        f"- Missing predictions: **{overall['missing']}**",
        f"- Unparsed predictions: **{overall['unparsed']}**",
        f"- Unparsed at the configured token limit: **{diagnostics['unparsed_at_token_limit']}/{overall['unparsed']}**",
        f"- Failed predictions: **{overall['status_counts'].get('failed', 0)}**",
        f"- Duplicate prediction rows: **{diagnostics['duplicate_prediction_rows']}**",
        f"- Unknown prediction IDs: **{diagnostics['unknown_prediction_ids']}**",
        f"- Endpoint distribution: {endpoint_text}",
        "",
        "## Metric Definitions",
        "",
        "- **Strict end-to-end accuracy:** correct answers divided by all dataset samples.",
        "- **Parsed-only accuracy:** correct answers divided only by parseable predictions.",
        "- **Answer coverage:** parseable predictions divided by all dataset samples.",
        "- **ROUGE-L F1:** longest-common-subsequence F1 between generated and reference reasoning chains.",
        "- **BERTScore-F1:** token-level contextual semantic similarity between reasoning chains.",
        "- **ROSCOE:** official reference-based `reasoning_alignment` semantic score in this reproduction.",
        "- **Overall mean:** arithmetic mean of ROUGE-L F1, BERTScore-F1, and ROSCOE.",
    ]
    return "\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional machine-readable summary JSON path.")
    parser.add_argument("--report", type=Path, help="Optional human-readable Markdown report path.")
    parser.add_argument("--model", default="Qwen3-VL-8B", help="Model label shown in reports.")
    parser.add_argument(
        "--reasoning-metrics",
        action="store_true",
        help="Compute the paper's ROUGE-L, BERTScore-F1, ROSCOE, and overall mean.",
    )
    parser.add_argument("--reasoning-device", default="auto", help="Torch device, e.g. cuda:1, cpu, or auto.")
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    parser.add_argument("--roscoe-batch-size", type=int, default=32)
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--roscoe-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument(
        "--table-output",
        type=Path,
        help="Optional paper-style Markdown table path (kept for command compatibility).",
    )
    parser.add_argument(
        "--compact-table-output",
        type=Path,
        help="Optional compact Overall/Traffic/Science/Finance/Others Markdown table path.",
    )
    args = parser.parse_args()

    samples_by_id = sample_index(load_samples(args.data))
    records = load_jsonl(args.predictions)
    summary = build_summary(
        samples_by_id=samples_by_id,
        records=records,
        model=args.model,
        data_path=args.data,
        predictions_path=args.predictions,
    )
    if args.reasoning_metrics:
        summary["reasoning_quality"] = compute_reasoning_metrics(
            samples_by_id,
            index_predictions(records),
            bertscore_model=args.bertscore_model,
            roscoe_model=args.roscoe_model,
            device=args.reasoning_device,
            bertscore_batch_size=args.bertscore_batch_size,
            roscoe_batch_size=args.roscoe_batch_size,
        )
    print_rich_report(summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(markdown_report(summary), encoding="utf-8")
    if args.table_output:
        args.table_output.parent.mkdir(parents=True, exist_ok=True)
        args.table_output.write_text(
            paper_table_markdown(summary["paper_table"]["rows"]) + "\n",
            encoding="utf-8",
        )
    if args.compact_table_output:
        args.compact_table_output.parent.mkdir(parents=True, exist_ok=True)
        args.compact_table_output.write_text(
            compact_domain_markdown(summary["compact_domain_table"]["row"]) + "\n",
            encoding="utf-8",
        )

    written = [
        path
        for path in (args.report, args.output, args.table_output, args.compact_table_output)
        if path
    ]
    if written:
        Console().print("[dim]Written:[/dim] " + ", ".join(str(path) for path in written))


if __name__ == "__main__":
    main()
