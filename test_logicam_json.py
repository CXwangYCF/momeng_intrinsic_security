from __future__ import annotations

import unittest
from pathlib import Path

from logicam_json.run import template_path
from logicam_json.schema import problem_text, response_format_for
from muslr_core.runtime import Sample


def sample(symbol: str = "pl", *, choices: bool = True) -> Sample:
    return Sample(
        id="sample-1",
        source_id="sample-1",
        row_index=0,
        image="image.jpg",
        domain="Science",
        symbol=symbol,
        depth=2,
        context="If P then Q.",
        question="Which statement follows?" if choices else "Is Q true?",
        choices=["A. P", "B. Q", "C. R", "D. S"] if choices else [],
        answer="B" if choices else "True",
    )


class LogiCAMJSONTests(unittest.TestCase):
    def test_schema_is_a_minimal_presentation_wrapper(self) -> None:
        schema = response_format_for(sample())["json_schema"]["schema"]
        self.assertEqual(set(schema["properties"]), {"reasoning_steps", "final_answer"})
        self.assertEqual(schema["properties"]["reasoning_steps"]["maxItems"], 16)
        self.assertEqual(schema["properties"]["final_answer"]["enum"], ["A", "B", "C", "D"])

    def test_truth_schema_constrains_truth_labels(self) -> None:
        schema = response_format_for(sample(choices=False))["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["final_answer"]["enum"], ["True", "False", "Unknown"])

    def test_fol_uses_the_released_fol_prompt(self) -> None:
        root = Path("/prompts")
        self.assertEqual(template_path(root, "fol"), root / "logicam" / "prompt_fol.txt")
        self.assertEqual(template_path(root, "pl"), root / "logicam" / "prompt.txt")
        self.assertEqual(template_path(root, "nm"), root / "logicam" / "prompt.txt")

    def test_query_keeps_released_field_order_without_custom_rules(self) -> None:
        rendered = problem_text(sample())
        self.assertLess(rendered.index("Image:"), rendered.index("Context:"))
        self.assertLess(rendered.index("Context:"), rendered.index("Question:"))
        self.assertLess(rendered.index("Question:"), rendered.index("Options:"))
        self.assertNotIn("12 derivation", rendered)
        self.assertNotIn("candidate_evaluations", rendered)
        self.assertNotIn("allowed rule", rendered.lower())


if __name__ == "__main__":
    unittest.main()
