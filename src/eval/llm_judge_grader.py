"""Grade agent answers the same way the organizers' hidden-question harness
does: component-by-component, LLM-as-judge, YES/NO per fact, partial credit.

Mirrors ``AI_Industry_Training_Hackathon/Participant_Package/handout/03_scoring_and_examples.md``:

  * Each question has one or more **components** -- a specific fact worth
    some number of points (``expected_fact`` + ``points``).
  * A judge model receives the question, the candidate answer, and ONE
    expected fact at a time, and replies YES or NO.
  * Equivalent formatting is accepted (``"1,234"`` == ``"1234"``,
    ``"Jan 2024"`` == ``"2024-01"``, minor rephrasing that preserves
    meaning) -- but a correct number in the wrong context, hedging language,
    or a refusal is NOT accepted.
  * ``hidden_question_score = sum(earned_points) / sum(max_points) * 100%``,
    with a 20% penalty on a question's earned points if the answer took more
    than 60s, and 0 points past 300s.

Ground truth (the ``expected_fact``s) come from real facts -- either
``dataset.py``'s ``EXAMPLES`` (computed directly from the datasets under
``AI_Industry_Training_Hackathon/data set/``, see that module's docstring),
or an external question file in one of the formats the organizers use:
``public_questions.jsonl`` (rich ``grading.components[]`` schema) or
``mock_questions.json`` (flat ``grading_components`` list).

The judge model is independent of the agent under test
--------------------------------------------------------
This whole script is offline, private tooling: we're rehearsing the
organizers' own hidden-question grading on ourselves before submitting, not
part of the submission itself. The judge (``_build_judge``) is its own
``ChatOpenAI`` instance, matching the organizers' real judge choice
(Qwen3.6-35B-A3B-FP8 via the private ``agent-brain`` service) by default,
but it is never the same object, prompt, or conversation as
``finance_agent.py``'s planner ("the Qwen brain inside the submitted
agent"), and it never calls or substitutes for ``finance_agent.synthesize``
(the required fine-tuned Nemotron step). See ``_build_judge``'s docstring
for the exact env var fallback chain.

Why a hybrid deterministic + LLM judge, not pure LLM
-----------------------------------------------------
The brief's own worked example ("41 records... 20 of which are holds") shows
the real judge rejects a correct number appearing in the wrong context --
so a fact being *present as a substring* is not sufficient evidence of a
YES, and this grader never fast-paths to YES on that basis alone. The one
shortcut taken here is the reverse case: if a numeric or date fact does not
appear *anywhere* in the answer (in any equivalent format we check), it
cannot possibly be satisfied regardless of context, so that's a safe,
deterministic NO with no LLM call needed. Everything else -- present
numbers/dates (context must be checked) and every non-numeric fact --
always goes to the LLM judge. Pass ``--llm-only`` to disable the shortcut
entirely and match the organizers' pure-LLM methodology exactly.

Three ways to get the answers being graded
--------------------------------------------
1. **In-process** (default) -- calls ``finance_agent.graph`` directly.
2. **``--endpoint URL``** -- POSTs each question to a running ``/query``
   service (this repo's ``api.py``, or the real submission service in
   ``cognitivo_prep/src/main.py`` once its model backend is reachable) and
   grades whatever comes back. This is how you check a *deployed* answer --
   including one synthesized by a real fine-tuned Nemotron, not just the
   ``DOMAIN_PREDICT_MODE=mock`` fallback.
3. **``--answers-file PATH``** -- grades pre-recorded answers with no agent
   or endpoint call at all (e.g. answers you already collected, or a
   ``curl .../query`` response you pasted into a file).

See ``GRADER_README.md`` in this directory for the full writeup: rubric,
scoring formula, every CLI flag, and worked examples of all three modes.

Usage
-----
    .venv/bin/python evals-hackathon/llm_judge_grader.py
    .venv/bin/python evals-hackathon/llm_judge_grader.py --questions path/to/public_questions.jsonl
    .venv/bin/python evals-hackathon/llm_judge_grader.py --endpoint http://127.0.0.1:8001 --sample 10
    .venv/bin/python evals-hackathon/llm_judge_grader.py --answers-file recorded.json --sample 10
    .venv/bin/python evals-hackathon/llm_judge_grader.py --llm-only --limit 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dataset import EXAMPLES  # noqa: E402
from finance_agent import extract_tool_trace, final_answer_text, graph  # noqa: E402

SLOW_SECONDS = 60.0
TIMEOUT_SECONDS = 300.0


# ---------------------------------------------------------------------------
# Grading case model + loaders
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Component:
    expected_fact: str
    points: float = 1.0


@dataclass(slots=True)
class GradingCase:
    qid: str
    question: str
    components: list[Component]
    difficulty: str = "unknown"
    tolerance_note: str | None = None

    @property
    def max_score(self) -> float:
        return sum(c.points for c in self.components)


def _components_from_row(row: dict[str, Any]) -> tuple[list[Component], str | None]:
    grading = row.get("grading")
    if isinstance(grading, dict):
        components = [
            Component(str(c.get("expected_fact", "")), float(c.get("points", 1)))
            for c in grading.get("components", [])
        ]
        return [c for c in components if c.expected_fact], grading.get("tolerance_note")
    components = [
        Component(str(fact), 1.0) for fact in row.get("grading_components", []) if fact
    ]
    return components, None


def load_cases_from_file(path: Path) -> list[GradingCase]:
    """Read either the rich ``public_questions.jsonl`` schema or the flat
    ``mock_questions.json`` / ``questions_template.json`` schema."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"{path} is empty")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        loaded = json.loads(text)
        rows = loaded if isinstance(loaded, list) else [loaded]

    cases: list[GradingCase] = []
    for index, row in enumerate(rows):
        question = row.get("prompt") or row.get("question") or ""
        if not question:
            continue
        components, tolerance_note = _components_from_row(row)
        cases.append(
            GradingCase(
                qid=str(row.get("id", index)),
                question=question,
                components=components,
                difficulty=str(row.get("difficulty", "unknown")),
                tolerance_note=tolerance_note,
            )
        )
    return cases


def load_cases_from_dataset() -> list[GradingCase]:
    """Default ground truth: this repo's own dataset.py examples, already
    verified against the real RBA/ASX/AFR data (see dataset.py's docstring)."""
    return [
        GradingCase(
            qid=example.id,
            question=example.question,
            components=[Component(fact, 1.0) for fact in example.grading_components],
            difficulty=example.difficulty,
        )
        for example in EXAMPLES
    ]


# ---------------------------------------------------------------------------
# Running the agent under test
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class AgentAnswer:
    answer: str
    steps: int
    tool_trace: list[dict[str, Any]]
    seconds: float


async def run_agent_in_process(question: str) -> AgentAnswer:
    """Invoke finance_agent.graph directly in this process."""
    start = time.perf_counter()
    result = await graph.ainvoke({"messages": [{"role": "user", "content": question}]})
    elapsed = time.perf_counter() - start
    messages = result["messages"]
    return AgentAnswer(
        answer=final_answer_text(messages),
        steps=len(extract_tool_trace(messages)),
        tool_trace=extract_tool_trace(messages),
        seconds=elapsed,
    )


async def run_agent_via_endpoint(client: httpx.AsyncClient, base_url: str, question: str) -> AgentAnswer:
    """POST to a running service's ``/query`` -- the same contract
    ``api.py`` and the real submission service (``cognitivo_prep/src/main.py``)
    both implement. Lets this grader test whatever is actually deployed,
    including a live fine-tuned-Nemotron synthesis stage, not just the
    in-process graph."""
    start = time.perf_counter()
    response = await client.post(f"{base_url.rstrip('/')}/query", json={"question": question})
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    payload = response.json()
    return AgentAnswer(
        answer=str(payload.get("answer", "")),
        steps=int(payload.get("steps", 0)),
        tool_trace=list(payload.get("tool_trace", [])),
        seconds=elapsed,
    )


def load_recorded_answers(path: Path) -> dict[str, AgentAnswer]:
    """Load pre-recorded answers -- e.g. pasted from a ``curl .../query``
    response -- keyed by ``id`` (preferred) or exact ``question`` text, so
    they can be graded without calling any agent or endpoint at all. Each
    row uses the same ``answer``/``steps``/``tool_trace`` shape as
    ``answer_template.json``, plus ``id`` and/or ``question`` to match it
    to a ``GradingCase``. ``seconds`` is optional (default 0 -- no slow
    penalty applies to answers you didn't just time yourself)."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    answers: dict[str, AgentAnswer] = {}
    for row in rows:
        answer = AgentAnswer(
            answer=str(row.get("answer", "")),
            steps=int(row.get("steps", 0)),
            tool_trace=list(row.get("tool_trace", [])),
            seconds=float(row.get("seconds", 0.0)),
        )
        if row.get("id"):
            answers[str(row["id"])] = answer
        if row.get("question"):
            answers[str(row["question"])] = answer
    return answers


# ---------------------------------------------------------------------------
# Deterministic pre-check -- safe fast NO only (see module docstring)
# ---------------------------------------------------------------------------
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_TOLERANCE_NOTE = re.compile(r"([+\-]?/?-?)\s*0*\.?\d+")


def _numbers_in(text: str) -> list[float]:
    out = []
    for token in _NUMBER.findall(text):
        cleaned = token.replace(",", "")
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


def _tolerance_from_note(note: str | None, magnitude: float) -> float:
    default = max(0.02, abs(magnitude) * 0.005)
    if not note:
        return default
    match = re.search(r"[+\-]/?-?\s*(\d+(?:\.\d+)?)", note)
    if not match:
        return default
    return max(float(match.group(1)), default)


_DATE_FORMATS = ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%b %Y", "%B %Y", "%Y-%m")


def _date_variants(fact: str) -> list[str] | None:
    """If ``fact`` parses as a date, return equivalent string forms to search
    for (covering the brief's own example: "Jan 2024" == "2024-01")."""
    fact = fact.strip()
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(fact, fmt)
        except ValueError:
            continue
        if fmt in ("%b %Y", "%B %Y", "%Y-%m"):
            return [parsed.strftime("%b %Y"), parsed.strftime("%B %Y"), parsed.strftime("%Y-%m")]
        return [parsed.strftime("%Y-%m-%d"), parsed.strftime("%d %b %Y"), parsed.strftime("%d %B %Y")]
    return None


def deterministic_absent(fact: str, answer: str, tolerance_note: str | None) -> bool:
    """True only when ``fact`` provably cannot be satisfied: a numeric or
    date fact that appears nowhere in ``answer``, in any equivalent form.
    False means "can't tell for free" -- always defer to the LLM judge,
    never treat this as a deterministic YES (see module docstring)."""
    fact = fact.strip()
    if not fact:
        return True

    date_variants = _date_variants(fact)
    if date_variants is not None:
        lowered = answer.lower()
        return not any(variant.lower() in lowered for variant in date_variants)

    cleaned = fact.rstrip("%").replace(",", "").lstrip("+")
    try:
        target = float(cleaned)
    except ValueError:
        return False  # non-numeric, non-date fact -- always defer to the LLM

    tolerance = _tolerance_from_note(tolerance_note, target)
    found = _numbers_in(answer)
    return not any(abs(value - target) <= tolerance for value in found)


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------
class ComponentVerdict(BaseModel):
    verdict: Literal["YES", "NO"] = Field(
        description="YES only if the answer states this exact fact, in the right context."
    )
    reason: str = Field(description="One short sentence explaining the verdict.")


JUDGE_SYSTEM_PROMPT = """\
You are grading one component of a financial research agent's answer, using \
the exact rubric the hackathon organizers use for hidden-question scoring.

You will be given the user's question, the agent's full answer, and ONE \
expected fact. Reply YES only if the answer clearly states that fact; \
otherwise reply NO.

ACCEPT as equivalent (reply YES):
- Equivalent numeric formatting: "1,234" and "1234"; "-5.09%" and "-5.0931%" \
  are the same value within normal rounding.
- Equivalent date formatting: "Jan 2024" and "2024-01" are the same.
- Minor rephrasing that preserves the exact meaning.
- A stated numeric tolerance note, if given, overrides the default rounding \
  tolerance.

REJECT (reply NO):
- The right number or date appears, but attached to the wrong thing (e.g. \
  the fact is a count of "increases" but the answer only gives a total \
  count with no split, or attaches the number to a different metric).
- Hedging language that changes the claim ("approximately", "roughly", \
  "around", "about") when the expected fact is a precise value.
- The answer refuses to answer, redirects to a different question, or \
  never actually states the fact.
- A value that is simply wrong or missing.

Be strict and literal. When genuinely uncertain whether the context matches, \
prefer NO -- partial credit rewards precision, not generous interpretation.
"""


def _build_judge(model_name: str | None = None) -> Any:
    """Build the judge as its own, independent ``ChatOpenAI`` instance --
    never the same Python object as ``finance_agent.planner_model``.

    The organizers' real hidden-question judge is Qwen3.6-35B-A3B-FP8,
    reached through the private ``agent-brain`` service -- explicitly *not*
    the team's submitted agent, and *not* a stand-in for the required
    fine-tuned Nemotron synthesis step (see
    ``Participant_Package/handout/03_scoring_and_examples.md`` and
    ``Challenge_Brief.md``). This grader is offline, private tooling we run
    on ourselves before submitting -- effectively rehearsing the
    organizers' own grading -- so it deliberately mirrors that judge choice
    while keeping the call fully separate from the pipeline under test:

    1. ``JUDGE_MODEL_*`` -- set these to point the judge at a dedicated
       endpoint (e.g. the real Qwen agent-brain alias, reached the same way
       ``finance_agent.py``'s planner reaches it, but as a second,
       independent client/credential set).
    2. Falls back to ``AGENT_BRAIN_*`` -- reuses the *connection details*
       for the same Qwen deployment the planner uses, since that's where
       the real judge model lives, but constructs its own ``ChatOpenAI``
       object and makes its own API calls. This is judging, not planning:
       nothing here shares state, a prompt, or a conversation with
       ``finance_agent.planner_model``, and nothing here calls or
       replaces ``finance_agent.synthesize`` (Nemotron).
    3. Falls back further to this repo's generic ``MODEL_NAME`` gateway
       when neither is set, so the grader still runs before any Qwen
       endpoint is reachable.
    """
    verify_env = os.getenv(
        "JUDGE_MODEL_VERIFY_SSL",
        os.getenv("AGENT_BRAIN_VERIFY_SSL", os.getenv("OPENAI_VERIFY_SSL", "true")),
    ).strip().lower()
    verify_ssl = verify_env not in {"false", "0", "no"}
    api_version = os.getenv("OPENAI_API_VERSION", "2024-08-01-preview")
    model = ChatOpenAI(
        openai_api_key=os.getenv(
            "JUDGE_MODEL_API_KEY",
            os.getenv("AGENT_BRAIN_API_KEY", os.getenv("OPENAI_API_KEY", "sk-123456")),
        ),
        openai_api_base=os.getenv(
            "JUDGE_MODEL_BASE_URL",
            os.getenv(
                "AGENT_BRAIN_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://ai-gateway:4000")
            ),
        ),
        model_name=model_name
        or os.getenv(
            "JUDGE_MODEL_NAME",
            os.getenv("AGENT_BRAIN_MODEL_NAME", os.getenv("MODEL_NAME", "llama-distributed")),
        ),
        temperature=0,
        http_async_client=httpx.AsyncClient(verify=verify_ssl),
        default_query={"api-version": api_version} if api_version else None,
    )
    return model.with_structured_output(ComponentVerdict)


async def llm_judge_component(
    judge: Any, question: str, answer: str, fact: str, tolerance_note: str | None
) -> ComponentVerdict:
    tolerance_line = f"\nStated tolerance for this fact: {tolerance_note}" if tolerance_note else ""
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"AGENT'S ANSWER:\n{answer}\n\n"
        f"EXPECTED FACT:\n{fact}{tolerance_line}"
    )
    return await judge.ainvoke([
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human", prompt),
    ])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ComponentOutcome:
    component: Component
    matched: bool
    mode: Literal["deterministic", "llm"]
    reason: str


@dataclass(slots=True)
class CaseOutcome:
    case: GradingCase
    agent: AgentAnswer
    components: list[ComponentOutcome]

    @property
    def earned(self) -> float:
        raw = sum(c.component.points for c in self.components if c.matched)
        if self.agent.seconds > TIMEOUT_SECONDS:
            return 0.0
        if self.agent.seconds > SLOW_SECONDS:
            return raw * 0.8
        return raw

    @property
    def possible(self) -> float:
        return self.case.max_score


async def grade_case(
    case: GradingCase,
    judge: Any,
    llm_only: bool,
    concurrency: asyncio.Semaphore,
    get_answer: Callable[[GradingCase], Awaitable[AgentAnswer | None]],
) -> CaseOutcome | None:
    async with concurrency:
        agent = await get_answer(case)
    if agent is None:
        print(f"skipping {case.qid!r} -- no recorded answer found", file=sys.stderr)
        return None

    async def grade_one(component: Component) -> ComponentOutcome:
        if not llm_only:
            if deterministic_absent(component.expected_fact, agent.answer, case.tolerance_note):
                return ComponentOutcome(
                    component, matched=False, mode="deterministic",
                    reason="fact not found in the answer in any equivalent numeric/date form",
                )
        verdict = await llm_judge_component(
            judge, case.question, agent.answer, component.expected_fact, case.tolerance_note
        )
        return ComponentOutcome(
            component, matched=verdict.verdict == "YES", mode="llm", reason=verdict.reason,
        )

    outcomes = await asyncio.gather(*(grade_one(c) for c in case.components))
    return CaseOutcome(case=case, agent=agent, components=list(outcomes))


def report(outcomes: list[CaseOutcome]) -> float:
    earned_total = sum(o.earned for o in outcomes)
    possible_total = sum(o.possible for o in outcomes)

    for o in outcomes:
        missed = [c for c in o.components if not c.matched]
        flag = "ok " if not missed else "   "
        print(
            f"{flag}[{o.case.difficulty:6s}] {o.agent.seconds:5.1f}s "
            f"{len(o.components) - len(missed)}/{len(o.components)}  {o.case.question[:64]}"
        )
        for c in missed:
            print(f"       missed ({c.mode}): {c.component.expected_fact!r} -- {c.reason}")
        if missed:
            print(f"       answer: {o.agent.answer[:200]}")

    pct = (earned_total / possible_total * 100) if possible_total else 0.0
    print(f"\n{'=' * 62}")
    print(f"component score   {earned_total:.1f}/{possible_total:.0f}  ({pct:.1f}%)")
    print(f"perfect answers   {sum(1 for o in outcomes if o.earned == o.possible)}/{len(outcomes)}")
    slow = sum(1 for o in outcomes if o.agent.seconds > SLOW_SECONDS)
    print(f"over 60s          {slow} (each loses 20% of its earned points)")
    return pct


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_answer_source(
    args: argparse.Namespace,
) -> tuple[Callable[[GradingCase], Awaitable[AgentAnswer | None]], httpx.AsyncClient | None]:
    """Pick one of the three ways to obtain an answer for a question:
    a live HTTP endpoint, pre-recorded answers, or the in-process graph
    (default). Returns the async lookup function plus an httpx client to
    close afterwards (only when endpoint mode is active)."""
    if args.endpoint:
        client = httpx.AsyncClient(timeout=TIMEOUT_SECONDS)

        async def via_endpoint(case: GradingCase) -> AgentAnswer:
            return await run_agent_via_endpoint(client, args.endpoint, case.question)

        return via_endpoint, client

    if args.answers_file:
        recorded = load_recorded_answers(Path(args.answers_file))

        async def via_recorded(case: GradingCase) -> AgentAnswer | None:
            return recorded.get(case.qid) or recorded.get(case.question)

        return via_recorded, None

    async def via_graph(case: GradingCase) -> AgentAnswer:
        return await run_agent_in_process(case.question)

    return via_graph, None


async def amain(args: argparse.Namespace) -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    cases = (
        load_cases_from_file(Path(args.questions)) if args.questions else load_cases_from_dataset()
    )
    if args.sample:
        rng = random.Random(args.seed)
        cases = rng.sample(cases, min(args.sample, len(cases)))
    if args.limit:
        cases = cases[: args.limit]
    if not any(c.components for c in cases):
        print(
            "No grading components found for any question -- nothing to score. "
            "Pass --questions pointing at a file with grading_components / "
            "grading.components, or omit it to use dataset.py's examples.",
            file=sys.stderr,
        )
        return 2

    source_label = (
        f"endpoint {args.endpoint}" if args.endpoint
        else f"recorded answers {args.answers_file}" if args.answers_file
        else "in-process finance_agent.graph"
    )
    print(f"{len(cases)} question(s) -- answers from: {source_label}")
    print(f"judge model: "
          f"{args.judge_model or os.getenv('JUDGE_MODEL_NAME', os.getenv('MODEL_NAME', 'llama-distributed'))}"
          f"{' (deterministic fast-path disabled)' if args.llm_only else ''}\n")

    judge = _build_judge(args.judge_model)
    get_answer, client = _build_answer_source(args)
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    try:
        raw_outcomes = await asyncio.gather(
            *(grade_case(c, judge, args.llm_only, semaphore, get_answer) for c in cases)
        )
    finally:
        if client is not None:
            await client.aclose()
    wall = time.perf_counter() - started

    outcomes = [o for o in raw_outcomes if o is not None]
    if not outcomes:
        print("No answers were found for any question -- nothing scored.", file=sys.stderr)
        return 2

    pct = report(outcomes)
    print(f"wall clock        {wall:.1f}s at concurrency {args.concurrency}")
    return 0 if pct >= args.threshold else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        help="path to a public_questions.jsonl / mock_questions.json-shaped file "
        "(defaults to this repo's dataset.py examples)",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--endpoint",
        help="base URL of a running /health+/query service (e.g. http://127.0.0.1:8001) -- "
        "grades whatever is actually deployed there instead of the in-process graph",
    )
    source.add_argument(
        "--answers-file",
        help="path to a JSON list of pre-recorded {id|question, answer, steps, tool_trace} "
        "rows (e.g. pasted curl .../query output) -- grades them with no agent/endpoint call",
    )
    parser.add_argument("--judge-model", default=None, help="override JUDGE_MODEL_NAME/.env default")
    parser.add_argument("--llm-only", action="store_true",
                         help="disable the deterministic fast-NO path; judge every component with the LLM")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="simultaneous questions (harness default is 3)")
    parser.add_argument("--sample", type=int, default=0,
                        help="grade a random sample of N questions instead of all of them")
    parser.add_argument("--seed", type=int, default=None, help="random seed for --sample")
    parser.add_argument("--limit", type=int, default=0,
                         help="only grade the first N questions (applied after --sample)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="exit non-zero below this component percentage")
    args = parser.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
