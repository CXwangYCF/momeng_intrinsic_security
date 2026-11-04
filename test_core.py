from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.evaluate import (
    build_summary,
    compact_domain_markdown,
    compact_domain_row,
    paper_table_rows,
)
from evaluation.reasoning_metrics import generated_steps, reasoning_pairs
from muslr_core.ripo_batch import to_ripo_messages
from muslr_core.runtime import Sample, load_jsonl, load_samples, parse_answer, parse_template_content, run_samples


class MuSLRTests(unittest.TestCase):
    def test_huggingface_json_fields_are_normalized(self) -> None:
        record = [
            {
                "id": "hf-1",
                "image": {"src": "https://example.invalid/image.jpg"},
                "symbol": "PL",
                "depth": 3,
                "full_context": "premise",
                "question": "question",
                "choices": '["A. first", "B. second"]',
                "answer": "A",
                "reasoning": '{"SR1": "first step", "CR1": "second step"}',
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            sample = load_samples(path)[0]
        self.assertEqual(sample.symbol, "pl")
        self.assertEqual(sample.id, "hf-1")
        self.assertEqual(sample.source_id, "hf-1")
        self.assertEqual(sample.choices, ["A. first", "B. second"])
        self.assertEqual(sample.task_type, "multiple_choice")
        self.assertEqual(sample.reasoning, {"SR1": "first step", "CR1": "second step"})

    def test_directory_csv_and_duplicate_ids_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.csv").write_text(
                "id,file_name,domain,symbol,depth,full_context,question,choices,answer,reasoning\n"
                'dup,images/a.jpg,Domain,pl,2,ctx,q,"[""A. x"", ""B. y""]",A,{}\n'
                'dup,images/b.jpg,Domain,nm,3,ctx,q,"[""A. x"", ""B. y""]",B,{}\n',
                encoding="utf-8",
            )
            samples = load_samples(root)
        self.assertEqual([sample.id for sample in samples], ["dup", "dup#2"])
        self.assertEqual([sample.source_id for sample in samples], ["dup", "dup"])
        self.assertEqual(samples[1].image, "images/b.jpg")

    def test_answer_parser_uses_last_explicit_answer(self) -> None:
        text = "Answer: {A}\nReconsidered.\nFinal Answer: {C}"
        self.assertEqual(parse_answer(text, "multiple_choice"), "C")
        self.assertEqual(parse_answer("Answer: {Unknown}", "truth_evaluation"), "UNKNOWN")
        self.assertIsNone(parse_answer("I choose E", "multiple_choice"))

    def test_embedded_jpeg_becomes_image_message(self) -> None:
        jpeg = base64.b64encode(b"\xff\xd8\xffexample").decode("ascii")
        content = parse_template_content(f"Before\nImage: {jpeg}\nAfter")
        self.assertEqual([part["type"] for part in content], ["text", "image_url", "text"])
        self.assertNotIn(jpeg, content[0]["text"])

    def test_openai_local_content_converts_to_ripo_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.jpg"
            image.write_bytes(b"jpeg")
            messages = to_ripo_messages(
                [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": image.as_uri()}},
                ]
            )
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"][1]["type"], "image")
        self.assertTrue(messages[0]["content"][1]["path"].endswith("image.jpg"))

    def test_paper_table_uses_symbol_domain_cells_and_full_denominator(self) -> None:
        samples = {
            "correct": Sample("correct", "correct", 0, "a.jpg", "medical", "pl", 2, "", "", [], "TRUE"),
            "unparsed": Sample("unparsed", "unparsed", 1, "b.jpg", "medical", "pl", 2, "", "", [], "FALSE"),
            "finance": Sample("finance", "finance", 2, "c.jpg", "finance", "nm", 2, "", "", [], "A"),
        }
        records = [
            {"id": "correct", "prediction": "TRUE"},
            {"id": "unparsed", "prediction": None},
            {"id": "finance", "prediction": "A"},
        ]
        rows = paper_table_rows(records, samples, "test-model")
        self.assertEqual(rows[0]["Healthcare"], 50.0)
        self.assertEqual(rows[2]["Finance"], 100.0)

    def test_compact_domain_table_pools_other_samples(self) -> None:
        samples = {
            "traffic": Sample("traffic", "traffic", 0, "a.jpg", "traffic", "pl", 2, "", "", [], "TRUE"),
            "science": Sample("science", "science", 1, "b.jpg", "Science", "pl", 2, "", "", [], "TRUE"),
            "finance": Sample("finance", "finance", 2, "c.jpg", "finance", "pl", 2, "", "", [], "TRUE"),
            "other-1": Sample("other-1", "other-1", 3, "d.jpg", "medical", "pl", 2, "", "", [], "TRUE"),
            "other-2": Sample("other-2", "other-2", 4, "e.jpg", "Sports", "pl", 2, "", "", [], "TRUE"),
            "other-3": Sample("other-3", "other-3", 5, "f.jpg", "Entertainment", "pl", 2, "", "", [], "TRUE"),
        }
        records = [
            {"id": "traffic", "prediction": "TRUE"},
            {"id": "science", "prediction": "FALSE"},
            {"id": "finance", "prediction": "TRUE"},
            {"id": "other-1", "prediction": "TRUE"},
            {"id": "other-2", "prediction": "TRUE"},
            {"id": "other-3", "prediction": "FALSE"},
        ]
        row = compact_domain_row(records, samples, "test-method")
        self.assertEqual(row["Overall"], 66.67)
        self.assertEqual(row["Traffic"], 100.0)
        self.assertEqual(row["Science"], 0.0)
        self.assertEqual(row["Finance"], 100.0)
        self.assertEqual(row["Others"], 66.67)
        self.assertEqual(row["Others_samples"], 3)
        self.assertIn(
            "| test-method | 66.67 | 100.00 | 0.00 | 100.00 | 66.67 |",
            compact_domain_markdown(row),
        )

    def test_reasoning_pairs_strip_final_answer_and_keep_reference_order(self) -> None:
        sample = Sample(
            "sample",
            "sample",
            0,
            "a.jpg",
            "medical",
            "pl",
            2,
            "If premise, then conclusion",
            "question",
            [],
            "TRUE",
            {"SR1": "reference one", "CR1": "reference two"},
        )
        output = "1. generated one\n\n2. generated two\n\nAnswer: True"
        pairs = reasoning_pairs({"sample": sample}, {"sample": {"output": output}})
        self.assertEqual(generated_steps(output), ["generated one", "generated two"])
        self.assertEqual(pairs[0]["reference_steps"], ["reference one", "reference two"])

    def test_depth_robustness_reports_absolute_and_relative_drop(self) -> None:
        samples = {
            "shallow": Sample("shallow", "shallow", 0, "a.jpg", "medical", "pl", 2, "", "", [], "TRUE"),
            "deep": Sample("deep", "deep", 1, "b.jpg", "medical", "pl", 8, "", "", [], "TRUE"),
        }
        records = [
            {"id": "shallow", "prediction": "TRUE"},
            {"id": "deep", "prediction": "FALSE"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            summary = build_summary(
                samples_by_id=samples,
                records=records,
                model="test",
                data_path=root / "data.json",
                predictions_path=predictions,
            )
        self.assertEqual(summary["depth_robustness"]["absolute_drop_percentage_points"], 100.0)
        self.assertEqual(summary["depth_robustness"]["relative_drop_percent"], 100.0)

    def test_retry_statuses_replace_unparsed_without_duplicate_rows(self) -> None:
        samples = [
            Sample("done", "done", 0, "a.jpg", "medical", "pl", 2, "", "", [], "TRUE"),
            Sample("retry", "retry", 1, "b.jpg", "medical", "pl", 2, "", "", [], "TRUE"),
        ]

        class FakeBatchClient:
            uses_ripo_batch = True

            def run_batch(self, tasks, run_one, *, on_outcome, **kwargs):
                for sample in tasks:
                    on_outcome(
                        {
                            "id": sample.id,
                            "row_index": sample.row_index,
                            "status": "completed",
                            "prediction": "TRUE",
                        }
                    )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            output.write_text(
                json.dumps({"id": "done", "row_index": 0, "status": "completed", "prediction": "TRUE"})
                + "\n"
                + json.dumps({"id": "retry", "row_index": 1, "status": "unparsed", "prediction": None})
                + "\n",
                encoding="utf-8",
            )
            counters = run_samples(
                samples=samples,
                output_path=output,
                completion_client=FakeBatchClient(),
                method="logicam",
                template_dir=Path(directory),
                dataset_dir=Path(directory),
                image_root=None,
                media_mode="local",
                local_media_root=Path(directory),
                workers=1,
                resume=True,
                retry_statuses={"unparsed"},
            )
            records = load_jsonl(output)

        self.assertEqual(counters["resumed"], 1)
        self.assertEqual(len(records), 2)
        self.assertEqual({record["id"] for record in records}, {"done", "retry"})
        self.assertEqual(records[1]["prediction"], "TRUE")


if __name__ == "__main__":
    unittest.main()
