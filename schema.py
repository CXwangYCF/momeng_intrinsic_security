"""Minimal JSON contract for the official MuSLR baseline prompt."""

from __future__ import annotations

from typing import Any

from muslr_core.runtime import Sample

SCHEMA_VERSION = "baseline-json-v1"
RETURN_EXAMPLE = {"reasoning": "", "final_answer": ""}


def answer_values(sample: Sample) -> list[str]:
    if sample.task_type == "multiple_choice":
        return ["A", "B", "C", "D"]
    return ["True", "False", "Unknown"]


def response_format_for(sample: Sample) -> dict[str, Any]:
    """Constrain only the baseline reasoning text and its final answer."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"muslr_baseline_{sample.task_type}",
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "final_answer": {
                        "type": "string",
                        "enum": answer_values(sample),
                    },
                },
                "required": list(RETURN_EXAMPLE),
                "additionalProperties": False,
            },
        },
    }


def system_prompt() -> str:
    return """You are solving MuSLR with the official natural-language baseline method.

Follow the few-shot examples and reason naturally from the image and supplied context. Do not use a formal-logic template, named inference rules, or LogiCAM-specific fields.

Your response is constrained by a JSON Schema. Put the same chain-of-thought style response that the baseline would normally produce in reasoning, and put only the selected A-D or True/False/Unknown label in final_answer. Return only the JSON object."""


def problem_text(sample: Sample) -> str:
    sections = ["\n\n---\n\nNew instance:\n\nQuestion:\n", sample.question]
    if sample.choices:
        sections.extend(["\n\nOptions:\n", "\n".join(sample.choices)])
    sections.extend(
        [
            "\n\nContext:\n",
            sample.context,
            "\n\nImage:\n",
            "Reason carefully in the same natural-language style as the examples, then fill reasoning and final_answer in the required JSON object.",
        ]
    )
    return "".join(sections)


__all__ = [
    "RETURN_EXAMPLE",
    "SCHEMA_VERSION",
    "answer_values",
    "problem_text",
    "response_format_for",
    "system_prompt",
]
