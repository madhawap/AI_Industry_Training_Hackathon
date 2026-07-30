"""Offline evaluation harness: questions in, component score out.

Runs the complete pipeline -- planner, TFQL execution, synthesis -- over a
question file and reports what the organisers' harness will measure: component
correctness, latency against the 60-second cliff, and how many tool calls each
question needed.

Reads both formats:

  * ``public_questions.jsonl`` -- one JSON object per line with ``prompt`` and
    ``grading.components[].expected_fact`` plus per-component ``points``.
  * ``mock_questions.json`` -- a list with ``question`` and a flat
    ``grading_components`` list of strings, each worth one point.

Scoring here is a **substring approximation** of the real LLM judge, which
accepts equivalent date formats and sentiment synonyms. Treat a miss as worth
investigating, not as a definite failure.

    python -m src.eval.run_questions data/mock_questions.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.llm.client import LLMClients
from src.tfql import PlanRequest, Store
from src.tools.registry import ToolRegistry
from src.tools.tfql_tool import TOOL_NAME, make_executor, register_tfql

SLOW_SECONDS = 60.0
"""Above this the organisers deduct 20% of the points earned for a question."""

TIMEOUT_SECONDS = 300.0
"""Above this the question scores zero."""


@dataclass(slots=True)
class Case:
    qid: str
    prompt: str
    components: list[tuple[str, float]]  # (expected_fact, points)
    difficulty: str = "unknown"


@dataclass(slots=True)
class Outcome:
    case: Case
    answer: str
    seconds: float
    operations: int
    tool_calls: int
    matched: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def earned(self) -> float:
        points = dict(self.case.components)
        raw = sum(points[f] for f in self.matched)
        # Mirror the organisers' response-time rules.
        if self.seconds > TIMEOUT_SECONDS:
            return 0.0
        if self.seconds > SLOW_SECONDS:
            return raw * 0.8
        return raw

    @property
    def possible(self) -> float:
        return sum(p for _, p in self.case.components)


def load_cases(path: Path) -> list[Case]:
    """Read either question format into a common shape."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"{path} is empty")

    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        loaded = json.loads(text)
        rows = loaded if isinstance(loaded, list) else loaded.get("questions", [])

    cases: list[Case] = []
    for index, row in enumerate(rows):
        prompt = row.get("prompt") or row.get("question") or ""
        if not prompt:
            continue
        grading = row.get("grading")
        if isinstance(grading, dict):
            components = [
                (str(c.get("expected_fact", "")), float(c.get("points", 1)))
                for c in grading.get("components", [])
            ]
        else:
            components = [(str(c), 1.0) for c in row.get("grading_components", [])]
        cases.append(
            Case(
                qid=str(row.get("id", index)),
                prompt=prompt,
                components=[c for c in components if c[0]],
                difficulty=str(row.get("difficulty", "unknown")),
            )
        )
    return cases


_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers_in(text: str) -> list[float]:
    out = []
    for token in _NUMBER.findall(text):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return out


def _matches(fact: str, answer: str) -> bool:
    """Approximate the judge's per-component check.

    The brief states the judge accepts "harmless numeric formatting
    differences", so numeric facts are compared as numbers rather than
    strings: ``18439800`` matches ``18,439,800`` and ``-5.09%`` matches
    ``-5.0931%``. Tolerance is the looser of 0.02 absolute (the brief's stated
    tolerance for calculated returns) or 0.5% relative, which lets a rounded
    figure match a fuller one without letting genuinely different values pass.

    Non-numeric facts fall back to a case-insensitive substring test.
    """
    fact = fact.strip()
    if not fact:
        return False

    cleaned = fact.rstrip("%").replace(",", "").lstrip("+")
    try:
        target = float(cleaned)
    except ValueError:
        return fact.lower() in answer.lower()

    # Dates arrive as facts too; keep those exact rather than numeric.
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", fact):
        return fact in answer

    tolerance = max(0.02, abs(target) * 0.005)
    return any(abs(value - target) <= tolerance for value in _numbers_in(answer))


async def run_case(
    case: Case,
    llm: LLMClients,
    tools: list[dict[str, Any]],
    run_plan,
    brain_system: str,
    synthesis_system: str,
) -> Outcome:
    start = time.perf_counter()
    try:
        planned = await llm.brain_chat(
            [
                {"role": "system", "content": brain_system},
                {"role": "user", "content": case.prompt},
            ],
            tools=tools,
        )
        message = planned.choices[0].message
        calls = [
            c for c in (message.tool_calls or []) if c.function.name == TOOL_NAME
        ]
        if not calls:
            elapsed = time.perf_counter() - start
            answer = (message.content or "").strip()
            return _score(case, answer, elapsed, 0, 0, error="no tool call")

        plan = PlanRequest(**json.loads(calls[0].function.arguments))
        bundle = await run_plan(plan)
        answer = await llm.synthesize(
            f"Question: {case.prompt}\n\n"
            f"Verified results:\n{json.dumps(bundle.data)}",
            system=synthesis_system,
        )
        elapsed = time.perf_counter() - start
        return _score(case, answer, elapsed, len(plan.operations), len(calls))
    except Exception as exc:  # noqa: BLE001 - one bad case must not stop the run
        elapsed = time.perf_counter() - start
        return _score(case, "", elapsed, 0, 0, error=str(exc)[:200])


def _score(
    case: Case,
    answer: str,
    seconds: float,
    operations: int,
    tool_calls: int,
    error: str | None = None,
) -> Outcome:
    matched = [f for f, _ in case.components if _matches(f, answer)]
    missed = [f for f, _ in case.components if f not in matched]
    return Outcome(
        case=case,
        answer=answer,
        seconds=seconds,
        operations=operations,
        tool_calls=tool_calls,
        matched=matched,
        missed=missed,
        error=error,
    )


def report(outcomes: list[Outcome]) -> float:
    earned = sum(o.earned for o in outcomes)
    possible = sum(o.possible for o in outcomes)
    latencies = sorted(o.seconds for o in outcomes)

    for o in outcomes:
        flag = "ok " if not o.missed and not o.error else "   "
        print(
            f"{flag}[{o.case.difficulty:6s}] {o.seconds:5.1f}s ops={o.operations} "
            f"{len(o.matched)}/{len(o.case.components)}  {o.case.prompt[:64]}"
        )
        if o.error:
            print(f"       ERROR: {o.error}")
        if o.missed:
            print(f"       missed: {o.missed}")
            print(f"       answer: {o.answer[:170]}")

    pct = (earned / possible * 100) if possible else 0.0
    slow = sum(1 for o in outcomes if o.seconds > SLOW_SECONDS)
    no_tool = sum(1 for o in outcomes if o.tool_calls == 0)
    p95 = latencies[int(len(latencies) * 0.95) - 1] if latencies else 0.0

    print(f"\n{'=' * 62}")
    print(f"component score   {earned:.1f}/{possible:.0f}  ({pct:.1f}%)")
    print(f"perfect answers   {sum(1 for o in outcomes if not o.missed)}/{len(outcomes)}")
    print(f"latency           mean {sum(latencies) / len(latencies):.1f}s  p95 {p95:.1f}s")
    print(f"over 60s          {slow}  (each loses 20% of its earned points)")
    print(f"no tool call      {no_tool}")
    print(f"mean operations   {sum(o.operations for o in outcomes) / len(outcomes):.1f}")
    return pct


async def main_async(args: argparse.Namespace) -> int:
    from src.graph.nodes import BRAIN_SYSTEM, SYNTHESIS_SYSTEM

    cases = load_cases(Path(args.questions))
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} questions from {args.questions}\n")

    store = Store.build()
    registry = ToolRegistry()
    register_tfql(registry, store)
    llm = LLMClients(get_settings())
    run_plan = make_executor(store)
    tools = registry.openai_tools()

    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(case: Case) -> Outcome:
        async with semaphore:
            return await run_case(
                case, llm, tools, run_plan, BRAIN_SYSTEM, SYNTHESIS_SYSTEM
            )

    started = time.perf_counter()
    outcomes = await asyncio.gather(*(guarded(c) for c in cases))
    wall = time.perf_counter() - started

    pct = report(list(outcomes))
    print(f"wall clock        {wall:.1f}s at concurrency {args.concurrency}")
    await llm.aclose()
    store.close()
    return 0 if pct >= args.threshold else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", help="path to a .json or .jsonl question file")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="simultaneous questions (harness default is 3)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="exit non-zero below this component percentage")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
