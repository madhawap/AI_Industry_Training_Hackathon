"""Offline evaluators for the finance hackathon dataset.

Passed directly to ``langsmith.aevaluate(..., evaluators=[...])`` in
``run_eval_offline.py``. Each function follows LangSmith's local-evaluator
interface: ``fn(run, example) -> dict``, where ``run`` carries the agent's
actual output and ``example`` carries the dataset row (inputs + reference
outputs). Uses only the standard library, same as ``evals/evaluators-offline.py``.
"""

from __future__ import annotations

from typing import Any


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _run_outputs(run: Any) -> dict[str, Any]:
    outputs = _field(run, "outputs", {}) or {}
    return outputs if isinstance(outputs, dict) else {"messages": outputs}


def _reference_outputs(example: Any) -> dict[str, Any]:
    outputs = _field(example, "outputs", {}) or {}
    return outputs if isinstance(outputs, dict) else {}


def _final_text(outputs: dict[str, Any]) -> str:
    messages = outputs.get("messages", [])
    for message in reversed(messages):
        role = str(_field(message, "type", _field(message, "role", ""))).lower()
        content = _field(message, "content")
        if role in {"ai", "assistant"} and content:
            return str(content)
    answer = outputs.get("answer")
    return answer if isinstance(answer, str) else ""


def _tool_calls(outputs: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in outputs.get("messages", []):
        message_calls = _field(message, "tool_calls", [])
        if isinstance(message_calls, list):
            calls.extend(call for call in message_calls if isinstance(call, dict))
    return calls


def tool_trajectory(run: Any, example: Any) -> dict[str, Any]:
    """Pass when expected tools form a prefix of the actual calls and
    any specified args match. Extra calls after the expected prefix are
    tolerated (mirrors ``evals/evaluators-offline.py``)."""
    reference = _reference_outputs(example)
    expected_tools = reference.get("expected_tools", [])
    expected_args = reference.get("expected_tool_args", [])
    actual_calls = _tool_calls(_run_outputs(run))
    actual_names = [call.get("name") for call in actual_calls]

    if actual_names[: len(expected_tools)] != expected_tools:
        return {
            "key": "finance_tool_trajectory",
            "score": 0,
            "comment": f"expected prefix {expected_tools!r}, got {actual_names!r}",
        }

    for index, specification in enumerate(expected_args):
        actual = actual_calls[index] if index < len(actual_calls) else {}
        expected_name = specification.get("name")
        if actual.get("name") != expected_name:
            return {
                "key": "finance_tool_trajectory",
                "score": 0,
                "comment": (
                    f"call #{index} name mismatch: expected {expected_name!r}, "
                    f"got {actual.get('name')!r}"
                ),
            }
        for key, value in specification.get("args", {}).items():
            actual_value = actual.get("args", {}).get(key)
            if actual_value != value:
                return {
                    "key": "finance_tool_trajectory",
                    "score": 0,
                    "comment": (
                        f"call #{index} arg {key!r} mismatch: expected "
                        f"{value!r}, got {actual_value!r}"
                    ),
                }

    return {
        "key": "finance_tool_trajectory",
        "score": 1,
        "comment": f"trajectory matched: {actual_names}",
    }


def grading_components_present(run: Any, example: Any) -> dict[str, Any]:
    """Fractional score = the proportion of ``grading_components`` (facts
    the reference answer states -- numbers, dates, names) that appear
    case-insensitively in the agent's final answer text.

    A fractional score (rather than pass/fail) is more informative here
    because most finance answers state several facts at once and partial
    credit distinguishes "got 2 of 3 numbers right" from "got none."
    """
    required = _reference_outputs(example).get("grading_components", [])
    text = _final_text(_run_outputs(run)).lower()
    if not required:
        return {
            "key": "finance_grading_components",
            "score": 1,
            "comment": "no grading components declared for this example",
        }
    matched = [c for c in required if c.lower() in text]
    missing = [c for c in required if c.lower() not in text]
    return {
        "key": "finance_grading_components",
        "score": round(len(matched) / len(required), 3),
        "comment": (
            f"matched {len(matched)}/{len(required)}; missing: {missing}"
            if missing
            else f"matched all {len(matched)}/{len(required)} components"
        ),
    }
