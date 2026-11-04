"""Minimal JSON contract around the official MuSLR LogiCAM prompt."""

from __future__ import annotations

from typing import Any

from muslr_core.runtime import Sample

SCHEMA_VERSION = "logicam-json-v2"
RETURN_EXAMPLE = {"reasoning_steps": [], "final_answer": ""}


def answer_values(sample: Sample) -> list[str]:
    if sample.task_type == "multiple_choice":
        return ["A", "B", "C", "D"]
    return ["True", "False", "Unknown"]


def response_format_for(sample: Sample) -> dict[str, Any]:
    """Constrain presentation without changing the LogiCAM reasoning method."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"muslr_logicam_{sample.task_type}",
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning_steps": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 16,
                        "items": {"type": "string"},
                    },
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


def problem_text(sample: Sample) -> str:
    """Render a query in the same field order as the released runner."""
    sections = ["\n\n---\n\nNew instance:\nImage:"]
    sections.extend(["\n\nContext:\n", sample.context, "\n\nQuestion:\n", sample.question])
    if sample.choices:
        sections.extend(["\n\nOptions:\n", "\n".join(sample.choices)])
    sections.extend(
        [
            "\n\nFollow the LogiCAM reasoning process demonstrated in the examples above. "
            "Put each complete original-style Step in one JSON reasoning_steps array item, in order. "
            "Stop adding steps as soon as Completion check is Yes, then put only the final "
            "A-D or True/False/Unknown label in final_answer. "
            "Return only the schema-constrained JSON object.",
        ]
    )
    return "".join(sections)


__all__ = [
    "RETURN_EXAMPLE",
    "SCHEMA_VERSION",
    "answer_values",
    "problem_text",
    "response_format_for",
]
