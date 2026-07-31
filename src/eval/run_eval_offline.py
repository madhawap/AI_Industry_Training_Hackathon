"""Run the finance hackathon agent evaluation in LangSmith.

Run from the project root (this file lives in ``evals-hackathon/``, which is
its own flat module namespace, not a Python package -- see the README in
this directory for why):

    .venv/bin/python evals-hackathon/run_eval_offline.py

What this does, each run:
  1. Upserts a LangSmith dataset from ``dataset.py``'s examples (deletes and
     re-uploads existing rows so the dataset always matches the current
     file). Every example's reference answer/grading components was
     computed from the real RBA/ASX/AFR data -- see ``dataset.py``.
  2. Invokes the two-stage finance agent (``finance_agent.py``: Qwen plans
     and calls the real tools in ``finance_tools.py``, then a synthesis
     stage -- Nemotron once ``DOMAIN_PREDICT_MODE=llm``, otherwise Qwen's
     own reply -- writes the final answer) against every example.
  3. Scores each run with the two local evaluators in
     ``evaluators_offline.py`` (tool trajectory + grading-component
     coverage) and prints/records the resulting LangSmith experiment.

Because ``LANGSMITH_TRACING=true`` is already set in ``.env``, every model
and tool call inside the agent is traced automatically -- open the printed
experiment URL in LangSmith to see the full per-example trace (planner
tool calls, tool results, synthesis call) alongside the scores. You can
also open ``finance-agent`` in LangGraph Studio (it's registered in
``langgraph.json``) to step through the two-node graph interactively.

This intentionally uses the simpler ``aevaluate(..., evaluators=[...])``
path (local evaluator callables passed straight to the run) rather than the
more elaborate reusable-evaluator-registration-and-dataset-binding flow in
``evals/run-eval-offline.py``. That flow is worth graduating to later (it
gives you evaluators you can see/re-run from the LangSmith UI without this
script), but it depends on more LangSmith API surface area. This script
optimizes for "get it running end-to-end today."
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client, aevaluate

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dataset import DEFAULT_DATASET_NAME, to_langsmith_examples  # noqa: E402
from evaluators_offline import (  # noqa: E402
    grading_components_present,
    tool_trajectory,
)

DATASET_DESCRIPTION = (
    "Finance hackathon examples (RBA cash rate, ASX stock prices, AFR "
    "articles), verified against the real datasets under "
    "AI_Industry_Training_Hackathon/data set."
)


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _final_text(outputs: dict[str, Any]) -> str:
    messages = outputs.get("messages", [])
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        role = getattr(message, "type", None) or (
            message.get("type", message.get("role"))
            if isinstance(message, dict)
            else None
        )
        if role in {"ai", "assistant"} and content:
            return str(content)
    return ""


def _upsert_dataset(client: Client, name: str) -> tuple[str, str]:
    datasets = list(client.list_datasets(dataset_name=name))
    if len(datasets) > 1:
        raise RuntimeError(f"Multiple datasets named {name!r} found.")

    if datasets:
        dataset = datasets[0]
        for example in client.list_examples(dataset_id=dataset.id):
            client.delete_example(example.id)
        action = "Updated"
    else:
        dataset = client.create_dataset(
            dataset_name=name,
            description=DATASET_DESCRIPTION,
        )
        action = "Created"

    rows = to_langsmith_examples()
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[row["inputs"] for row in rows],
        outputs=[row["outputs"] for row in rows],
        metadata=[row["metadata"] for row in rows],
    )
    return str(dataset.id), action


async def _target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run the two-stage finance agent for one LangSmith dataset example."""
    # Imported after _load_env() so the model config sees values from .env.
    from finance_agent import graph

    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": inputs["question"]}]}
    )
    return {**result, "answer": _final_text(result)}


async def amain(args: argparse.Namespace) -> int:
    if not os.getenv("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY is not configured.", file=sys.stderr)
        return 2

    client = Client()
    try:
        dataset_id, dataset_action = _upsert_dataset(client, args.dataset_name)
        print(f"{dataset_action} dataset {args.dataset_name!r} ({dataset_id}).")

        experiment = await aevaluate(
            _target,
            data=args.dataset_name,
            evaluators=[tool_trajectory, grading_components_present],
            experiment_prefix=args.experiment_prefix,
            max_concurrency=args.max_concurrency,
            metadata={"environment": "hackathon-offline-evaluation"},
            client=client,
        )
        print(f"Experiment complete: {experiment}")
    finally:
        client.close(timeout=0)
    return 0


def main() -> int:
    _load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name",
        default=os.getenv("FINANCE_DATASET_NAME", DEFAULT_DATASET_NAME),
    )
    parser.add_argument(
        "--experiment-prefix",
        default="finance-hackathon-agent",
        help="Prefix used for the experiment name shown in LangSmith.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=2,
        help="Maximum number of examples evaluated at the same time.",
    )
    args = parser.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
