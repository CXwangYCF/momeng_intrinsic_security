#!/usr/bin/env python3
"""Recover parseable final answers previously rejected by strict consistency checks."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from muslr_core.runtime import load_jsonl, load_samples  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    args = parser.parse_args()

    path = args.predictions.resolve()
    rows = load_jsonl(path)
    samples = load_samples(args.data)
    recovered = 0

    for row in rows:
        if row.get("status") != "inconsistent":
            continue
        structured = json.loads(str(row["output"]))
        row_index = int(row["row_index"])
        sample = samples[row_index]
        prediction = str(structured["final_answer"]).upper()
        original_error = str(row.pop("error", ""))
        warnings = [original_error.partition(": ")[2] or original_error] if original_error else []
        row.update(
            status="completed",
            prediction=prediction,
            correct=prediction == sample.answer.upper(),
            structured_output=structured,
            consistency_warnings=warnings,
        )
        recovered += 1

    backup = path.with_name(f"{path.stem}.strict-backup{path.suffix}")
    if backup.exists():
        parser.error(f"Backup already exists: {backup}")
    shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    print(f"Recovered {recovered} final answers")
    print(f"Backup: {backup}")


if __name__ == "__main__":
    main()
