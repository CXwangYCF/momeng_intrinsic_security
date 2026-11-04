from __future__ import annotations

import unittest

from baseline_json.schema import problem_text, response_format_for, system_prompt
from muslr_core.runtime import Sample


def sample(*, choices: bool = True) -> Sample:
    return Sample(
        id="sample-1",
        source_id="sample-1",
        row_index=0,
        image="image.jpg",
        domain="Science",
        symbol="pl",
        depth=2,
        context="If P then Q.",
        question="Which statement follows?" if choices else "Is Q true?",
        choices=["A. P", "B. Q", "C. R", "D. S"] if choices else [],
        answer="B" if choices else "True",
    )


class BaselineJSONTests(unittest.TestCase):
    def test_schema_is_minimal_for_multiple_choice(self) -> None:
        schema = response_format_for(sample())["json_schema"]["schema"]
        self.assertEqual(set(schema["properties"]), {"reasoning", "final_answer"})
        self.assertEqual(schema["properties"]["final_answer"]["enum"], ["A", "B", "C", "D"])

    def test_schema_constrains_truth_labels(self) -> None:
        schema = response_format_for(sample(choices=False))["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["final_answer"]["enum"], ["True", "False", "Unknown"])

    def test_prompt_preserves_natural_language_baseline(self) -> None:
        self.assertIn("natural-language baseline", system_prompt())
        self.assertIn("Do not use a formal-logic template", system_prompt())
        rendered = problem_text(sample())
        self.assertIn("A. P", rendered)
        self.assertIn("If P then Q.", rendered)


if __name__ == "__main__":
    unittest.main()
