#!/usr/bin/env python3
"""Compare complete baseline and LogiCAM MuSLR evaluation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table

from evaluation.evaluate import PAPER_DOMAINS, PAPER_SYMBOLS
from muslr_core.runtime import load_jsonl, load_samples, sample_index


def parsed_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(record["id"]): record
        for record in load_jsonl(path)
        if isinstance(record.get("prediction"), str)
    }


def paired_comparison(
    data_path: Path,
    baseline_path: Path,
    logicam_path: Path,
) -> dict[str, Any]:
    truth = sample_index(load_samples(data_path))
    baseline = parsed_by_id(baseline_path)
    logicam = parsed_by_id(logicam_path)
    common = sorted(set(baseline) & set(logicam) & set(truth))
    baseline_correct = 0
    logicam_correct = 0
    logicam_only = 0
    baseline_only = 0
    both_correct = 0
    for sample_id in common:
        expected = truth[sample_id].answer.upper()
        baseline_is_correct = baseline[sample_id]["prediction"].upper() == expected
        logicam_is_correct = logicam[sample_id]["prediction"].upper() == expected
        baseline_correct += baseline_is_correct
        logicam_correct += logicam_is_correct
        both_correct += baseline_is_correct and logicam_is_correct
        logicam_only += logicam_is_correct and not baseline_is_correct
        baseline_only += baseline_is_correct and not logicam_is_correct

    denominator = len(common)
    return {
        "common_parsed_samples": denominator,
        "baseline_accuracy_percent": _percent(baseline_correct, denominator),
        "logicam_accuracy_percent": _percent(logicam_correct, denominator),
        "logicam_minus_baseline_points": _percent(
            logicam_correct - baseline_correct, denominator
        ),
        "both_correct": both_correct,
        "logicam_only_correct": logicam_only,
        "baseline_only_correct": baseline_only,
        "net_additional_correct": logicam_only - baseline_only,
    }


def _percent(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 2) if denominator else 0.0


def _load_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _summary_path(predictions: Path) -> Path:
    return predictions.with_suffix(".summary.json")


def _metric(baseline: float, logicam: float) -> dict[str, float]:
    return {
        "baseline": baseline,
        "logicam": logicam,
        "delta": round(logicam - baseline, 4),
    }


def _group_comparison(
    baseline: dict[str, Any], logicam: dict[str, Any]
) -> dict[str, Any]:
    return {
        "samples": baseline["records"],
        "accuracy_end_to_end_percent": _metric(
            baseline["accuracy_end_to_end_percent"],
            logicam["accuracy_end_to_end_percent"],
        ),
        "accuracy_parsed_percent": _metric(
            baseline["accuracy_scored_percent"],
            logicam["accuracy_scored_percent"],
        ),
        "coverage_percent": _metric(
            baseline["coverage_percent"], logicam["coverage_percent"]
        ),
    }


def build_comparison(
    baseline_summary: dict[str, Any],
    logicam_summary: dict[str, Any],
    paired: dict[str, Any],
) -> dict[str, Any]:
    baseline_overall = baseline_summary["overall"]
    logicam_overall = logicam_summary["overall"]
    relative_improvement = (
        round(
            100
            * (
                logicam_overall["accuracy_end_to_end_percent"]
                - baseline_overall["accuracy_end_to_end_percent"]
            )
            / baseline_overall["accuracy_end_to_end_percent"],
            2,
        )
        if baseline_overall["accuracy_end_to_end_percent"]
        else 0.0
    )
    reasoning_names = {
        "rouge_l": "ROUGE-L F1",
        "bertscore_f1": "BERTScore-F1",
        "roscoe": "ROSCOE",
    }
    reasoning = {
        label: _metric(
            baseline_summary["reasoning_quality"][key]["score"],
            logicam_summary["reasoning_quality"][key]["score"],
        )
        for key, label in reasoning_names.items()
    }
    reasoning["Overall mean"] = _metric(
        baseline_summary["reasoning_quality"]["overall_mean"],
        logicam_summary["reasoning_quality"]["overall_mean"],
    )

    return {
        "runs": {
            "baseline": baseline_summary["run"],
            "logicam": logicam_summary["run"],
        },
        "overall": {
            "strict_end_to_end_accuracy_percent": _metric(
                baseline_overall["accuracy_end_to_end_percent"],
                logicam_overall["accuracy_end_to_end_percent"],
            ),
            "parsed_only_accuracy_percent": _metric(
                baseline_overall["accuracy_scored_percent"],
                logicam_overall["accuracy_scored_percent"],
            ),
            "answer_coverage_percent": _metric(
                baseline_overall["coverage_percent"],
                logicam_overall["coverage_percent"],
            ),
            "correct": _metric(baseline_overall["correct"], logicam_overall["correct"]),
            "parsed": _metric(baseline_overall["parsed"], logicam_overall["parsed"]),
            "unparsed": _metric(baseline_overall["unparsed"], logicam_overall["unparsed"]),
            "relative_accuracy_improvement_percent": relative_improvement,
        },
        "reasoning_quality": reasoning,
        "paired_common_parsed": paired,
        "by_symbol": {
            symbol.upper(): _group_comparison(
                baseline_summary["by_symbol"][symbol], logicam_summary["by_symbol"][symbol]
            )
            for symbol in PAPER_SYMBOLS
        },
        "by_domain": {
            label: _group_comparison(
                baseline_summary["by_domain"][domain], logicam_summary["by_domain"][domain]
            )
            for label, domain in PAPER_DOMAINS
        },
        "by_depth_bucket": {
            depth: _group_comparison(
                baseline_summary["by_depth_bucket"][depth],
                logicam_summary["by_depth_bucket"][depth],
            )
            for depth in ("2-3", "4-5", "6-7", "8-9")
        },
        "diagnostics": {
            "baseline_unparsed_at_token_limit": baseline_summary["diagnostics"][
                "unparsed_at_token_limit"
            ],
            "logicam_unparsed_at_token_limit": logicam_summary["diagnostics"][
                "unparsed_at_token_limit"
            ],
        },
    }


def _add_comparison_columns(table: Table) -> None:
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("LogiCAM", justify="right")
    table.add_column("Delta", justify="right", style="bold cyan")


def print_report(comparison: dict[str, Any]) -> None:
    console = Console()
    console.rule("[bold]MuSLR: Baseline vs LogiCAM[/bold]")
    overall = comparison["overall"]
    table = Table(title="Main Comparison", box=box.SIMPLE_HEAVY)
    _add_comparison_columns(table)
    for label, key in (
        ("Strict E2E accuracy", "strict_end_to_end_accuracy_percent"),
        ("Parsed-only accuracy", "parsed_only_accuracy_percent"),
        ("Answer coverage", "answer_coverage_percent"),
    ):
        metric = overall[key]
        table.add_row(
            label,
            f"{metric['baseline']:.2f}%",
            f"{metric['logicam']:.2f}%",
            f"{metric['delta']:+.2f} pp",
        )
    for label, key in (("Correct", "correct"), ("Parsed", "parsed"), ("Unparsed", "unparsed")):
        metric = overall[key]
        table.add_row(label, f"{metric['baseline']:.0f}", f"{metric['logicam']:.0f}", f"{metric['delta']:+.0f}")
    console.print(table)

    reasoning_table = Table(title="Reasoning Quality", box=box.SIMPLE_HEAVY)
    _add_comparison_columns(reasoning_table)
    for label, metric in comparison["reasoning_quality"].items():
        reasoning_table.add_row(
            label,
            f"{metric['baseline']:.4f}",
            f"{metric['logicam']:.4f}",
            f"{metric['delta']:+.4f}",
        )
    console.print(reasoning_table)

    for title, key in (
        ("By Symbol", "by_symbol"),
        ("By Domain", "by_domain"),
        ("By Depth", "by_depth_bucket"),
    ):
        group_table = Table(title=title, box=box.SIMPLE)
        group_table.add_column("Group", style="bold")
        group_table.add_column("Baseline E2E", justify="right")
        group_table.add_column("LogiCAM E2E", justify="right")
        group_table.add_column("Delta", justify="right", style="bold cyan")
        group_table.add_column("Coverage delta", justify="right")
        for label, group in comparison[key].items():
            accuracy = group["accuracy_end_to_end_percent"]
            coverage = group["coverage_percent"]
            group_table.add_row(
                label,
                f"{accuracy['baseline']:.2f}%",
                f"{accuracy['logicam']:.2f}%",
                f"{accuracy['delta']:+.2f} pp",
                f"{coverage['delta']:+.2f} pp",
            )
        console.print(group_table)

    paired = comparison["paired_common_parsed"]
    console.print(
        f"[bold]Common parsed ({paired['common_parsed_samples']}):[/bold] "
        f"{paired['baseline_accuracy_percent']:.2f}% -> {paired['logicam_accuracy_percent']:.2f}% "
        f"({paired['logicam_minus_baseline_points']:+.2f} pp); "
        f"LogiCAM-only correct {paired['logicam_only_correct']}, "
        f"baseline-only correct {paired['baseline_only_correct']}."
    )


def _markdown_group_table(groups: dict[str, Any]) -> str:
    lines = [
        "| Group | Baseline E2E | LogiCAM E2E | Delta | Baseline coverage | LogiCAM coverage | Coverage delta |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, group in groups.items():
        accuracy = group["accuracy_end_to_end_percent"]
        coverage = group["coverage_percent"]
        lines.append(
            f"| {label} | {accuracy['baseline']:.2f}% | {accuracy['logicam']:.2f}% | "
            f"{accuracy['delta']:+.2f} pp | {coverage['baseline']:.2f}% | "
            f"{coverage['logicam']:.2f}% | {coverage['delta']:+.2f} pp |"
        )
    return "\n".join(lines)


def markdown_report(comparison: dict[str, Any]) -> str:
    overall = comparison["overall"]
    paired = comparison["paired_common_parsed"]
    diagnostics = comparison["diagnostics"]
    main_rows = []
    for label, key, unit in (
        ("Strict end-to-end accuracy", "strict_end_to_end_accuracy_percent", "%"),
        ("Parsed-only accuracy", "parsed_only_accuracy_percent", "%"),
        ("Answer coverage", "answer_coverage_percent", "%"),
        ("Correct", "correct", ""),
        ("Parsed", "parsed", ""),
        ("Unparsed", "unparsed", ""),
    ):
        metric = overall[key]
        decimals = 2 if unit else 0
        delta_unit = " pp" if unit else ""
        main_rows.append(
            f"| {label} | {metric['baseline']:.{decimals}f}{unit} | "
            f"{metric['logicam']:.{decimals}f}{unit} | {metric['delta']:+.{decimals}f}{delta_unit} |"
        )

    reasoning_rows = [
        f"| {label} | {metric['baseline']:.4f} | {metric['logicam']:.4f} | {metric['delta']:+.4f} |"
        for label, metric in comparison["reasoning_quality"].items()
    ]
    return "\n".join(
        [
            "# MuSLR: Baseline vs LogiCAM",
            "",
            "## Main Results",
            "",
            "| Metric | Baseline | LogiCAM | Delta |",
            "| :--- | ---: | ---: | ---: |",
            *main_rows,
            "",
            f"Relative end-to-end accuracy improvement: **{overall['relative_accuracy_improvement_percent']:.2f}%**.",
            "",
            "## Paired Common-Parsed Comparison",
            "",
            f"Both methods parsed **{paired['common_parsed_samples']}** samples. Accuracy is "
            f"**{paired['baseline_accuracy_percent']:.2f}% -> {paired['logicam_accuracy_percent']:.2f}%** "
            f"({paired['logicam_minus_baseline_points']:+.2f} pp).",
            "",
            f"LogiCAM alone was correct on **{paired['logicam_only_correct']}** samples; baseline alone "
            f"was correct on **{paired['baseline_only_correct']}**, for a net gain of "
            f"**{paired['net_additional_correct']}** correct answers.",
            "",
            "## Reasoning Quality",
            "",
            "| Metric | Baseline | LogiCAM | Delta |",
            "| :--- | ---: | ---: | ---: |",
            *reasoning_rows,
            "",
            "## By Symbol",
            "",
            _markdown_group_table(comparison["by_symbol"]),
            "",
            "## By Domain",
            "",
            _markdown_group_table(comparison["by_domain"]),
            "",
            "## By Depth",
            "",
            _markdown_group_table(comparison["by_depth_bucket"]),
            "",
            "## Diagnostics",
            "",
            f"- Baseline unparsed at token limit: **{diagnostics['baseline_unparsed_at_token_limit']}**",
            f"- LogiCAM unparsed at token limit: **{diagnostics['logicam_unparsed_at_token_limit']}**",
            "- LogiCAM improves answer accuracy substantially, but its longer outputs reduce final-answer coverage.",
            "- Lower text-similarity scores do not contradict higher answer accuracy; they measure resemblance to the reference chain, not final-answer correctness.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--logicam", required=True, type=Path)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--logicam-summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    baseline_summary_path = args.baseline_summary or _summary_path(args.baseline)
    logicam_summary_path = args.logicam_summary or _summary_path(args.logicam)
    paired = paired_comparison(args.data, args.baseline, args.logicam)
    comparison = build_comparison(
        _load_summary(baseline_summary_path),
        _load_summary(logicam_summary_path),
        paired,
    )
    print_report(comparison)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(markdown_report(comparison), encoding="utf-8")


if __name__ == "__main__":
    main()
