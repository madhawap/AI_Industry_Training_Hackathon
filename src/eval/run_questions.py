"""Run public questions against a live agent ``POST /query`` and write results.

For each row in ``Participant_Package/public_questions.jsonl`` (or another
``.jsonl`` / ``.json`` question file):

1. POST ``{"question": <prompt>}`` to the agent ``/query`` endpoint
2. Record the full JSON response (``answer``, ``steps``, ``tool_trace``)
3. Pair it with ``required_facts``, ``scoring_notes``, and ``reference_answer``

Writes ``src/eval/results.json`` by default.

    python -m src.eval.run_questions
    python -m src.eval.run_questions Participant_Package/public_questions.jsonl \\
        --endpoint http://localhost:8011 --concurrency 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = REPO_ROOT / "Participant_Package" / "public_questions.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results.json"
DEFAULT_ENDPOINT = "http://localhost:8011"


@dataclass(slots=True)
class Case:
    qid: str
    question: str
    required_facts: list[str]
    scoring_notes: str
    reference_answer: str
    difficulty: str = "unknown"


def load_cases(path: Path) -> list[Case]:
    """Load question rows; keep the fields needed for the results file."""
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
        question = (row.get("prompt") or row.get("question") or "").strip()
        if not question:
            continue
        facts = row.get("required_facts")
        if not isinstance(facts, list):
            facts = []
        cases.append(
            Case(
                qid=str(row.get("id", index)),
                question=question,
                required_facts=[str(f) for f in facts],
                scoring_notes=str(row.get("scoring_notes") or ""),
                reference_answer=str(row.get("reference_answer") or ""),
                difficulty=str(row.get("difficulty") or "unknown"),
            )
        )
    return cases


async def query_one(
    client: httpx.AsyncClient,
    case: Case,
    *,
    endpoint: str,
) -> dict[str, Any]:
    """POST one question and return a results-file record."""
    url = f"{endpoint.rstrip('/')}/query"
    started = time.perf_counter()
    query_response: dict[str, Any]
    error: str | None = None

    try:
        resp = await client.post(url, json={"question": case.question})
        elapsed = time.perf_counter() - started
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001
            body = {"raw": resp.text}

        if resp.is_success and isinstance(body, dict):
            # Keep the full /query payload shape (answer, steps, tool_trace).
            query_response = {
                "answer": body.get("answer", ""),
                "steps": body.get("steps", 0),
                "tool_trace": body.get("tool_trace", []),
            }
            # Preserve any extra keys the server may add.
            for key, value in body.items():
                if key not in query_response:
                    query_response[key] = value
        else:
            query_response = body if isinstance(body, dict) else {"raw": body}
            error = f"HTTP {resp.status_code}"
    except Exception as exc:  # noqa: BLE001 - one failure must not abort the run
        elapsed = time.perf_counter() - started
        query_response = {}
        error = str(exc)[:300]

    record: dict[str, Any] = {
        "id": case.qid,
        "difficulty": case.difficulty,
        "question": case.question,
        "query_response": query_response,
        "required_facts": case.required_facts,
        "scoring_notes": case.scoring_notes,
        "reference_answer": case.reference_answer,
        "seconds": round(elapsed, 3),
    }
    if error:
        record["error"] = error
    return record


async def main_async(args: argparse.Namespace) -> int:
    questions_path = Path(args.questions)
    output_path = Path(args.output)
    cases = load_cases(questions_path)
    if args.limit:
        cases = cases[: args.limit]

    endpoint = args.endpoint.rstrip("/")
    print(f"{len(cases)} questions from {questions_path}")
    print(f"endpoint  {endpoint}/query")
    print(f"output    {output_path}\n")

    # Probe health so a downed agent fails fast with a clear message.
    async with httpx.AsyncClient(timeout=args.timeout) as probe:
        try:
            health = await probe.get(f"{endpoint}/health")
            if health.status_code != 200:
                print(
                    f"WARN  GET /health returned {health.status_code}; "
                    "continuing anyway",
                    file=sys.stderr,
                )
        except Exception as exc:  # noqa: BLE001
            print(
                f"FAIL  cannot reach {endpoint}/health: {exc}\n"
                "Start the agent (python -m src.main) before running eval.",
                file=sys.stderr,
            )
            return 2

    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[dict[str, Any] | None] = [None] * len(cases)

    async with httpx.AsyncClient(timeout=args.timeout) as client:

        async def guarded(index: int, case: Case) -> None:
            async with semaphore:
                print(f"→ [{case.qid}] {case.question[:72]}...")
                record = await query_one(client, case, endpoint=endpoint)
                results[index] = record
                status = "ERR" if record.get("error") else "ok "
                answer = ""
                qr = record.get("query_response") or {}
                if isinstance(qr, dict):
                    answer = str(qr.get("answer") or "")[:80]
                print(
                    f"{status} [{case.qid}] {record['seconds']:.1f}s  "
                    f"{answer or record.get('error', '')}"
                )

        wall0 = time.perf_counter()
        await asyncio.gather(*(guarded(i, c) for i, c in enumerate(cases)))
        wall = time.perf_counter() - wall0

    records = [r for r in results if r is not None]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    errors = sum(1 for r in records if r.get("error"))
    print(f"\n{'=' * 62}")
    print(f"wrote             {output_path}  ({len(records)} records)")
    print(f"errors            {errors}/{len(records)}")
    print(f"wall clock        {wall:.1f}s at concurrency {args.concurrency}")
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "questions",
        nargs="?",
        default=str(DEFAULT_QUESTIONS),
        help=f"path to .json / .jsonl (default: {DEFAULT_QUESTIONS})",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"agent base URL (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"results JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="simultaneous /query requests (harness default is 3)",
    )
    parser.add_argument("--limit", type=int, default=0, help="only the first N")
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="per-request HTTP timeout in seconds",
    )
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
