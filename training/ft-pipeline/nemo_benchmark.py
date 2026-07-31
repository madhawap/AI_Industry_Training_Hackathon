"""NVIDIA nemo-evaluator Bring-Your-Own-Benchmark registration for
`training/data/generated_questions.jsonl`.

Not installed by default -- `nemo-evaluator` is not in requirements.txt, and
is untested in the `ft-pipeline` conda env (that env's `python` resolves to a
pyenv 3.11 interpreter built without `_sqlite3`, which nemo-evaluator's
on-disk response cache imports unconditionally; a plain venv off a
system Python with sqlite3 support installs and imports it fine). Install
into whichever interpreter you actually run this from:

    pip install nemo-evaluator

What's verified (see test_nemo_benchmark.py): the benchmark registers, loads
`generated_questions.jsonl` for real, and `seed()`/`verify()` produce correct
prompts and scores -- all offline, no model call. What's NOT verified here:
an actual `run_evaluation()` end-to-end run, which needs a live
OpenAI-compatible endpoint (`ModelClient(base_url=..., model=...)` -- your
LiteLLM proxy or a directly-served checkpoint). Not reachable from this
sandbox, so treat the `run_evaluation` snippet in the module docstring below
as the documented next step, not something already exercised.

Usage, once a live endpoint exists:

    from nemo_evaluator import ModelClient, ChatSolver, run_evaluation, get_environment
    import nemo_benchmark  # registers "cognitivo-generated" as a side effect

    env = get_environment("cognitivo-generated")()
    client = ModelClient(base_url="http://localhost:4000/v1", model="domain-ft")
    solver = ChatSolver(client)
    results = run_evaluation(env, solver, max_concurrent=3)  # brief's own concurrency bar
"""

from __future__ import annotations

import re

from nemo_evaluator import ScorerInput, benchmark, scorer

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
    """Same approximation as src/eval/run_questions.py's judge-substitute:
    numeric facts compared as numbers (tolerant of formatting), everything
    else as a case-insensitive substring. Kept identical on purpose so a
    fine-tune's nemo-evaluator score and its ftpipe component_match score
    should broadly agree -- a large gap between the two would mean one of
    them has a bug, not that the model behaves differently under each.
    """
    fact = fact.strip()
    if not fact:
        return False
    cleaned = fact.rstrip("%").replace(",", "").lstrip("+")
    try:
        target = float(cleaned)
    except ValueError:
        return fact.lower() in answer.lower()
    tolerance = max(0.02, abs(target) * 0.005)
    return any(abs(n - target) <= tolerance for n in _numbers_in(answer))


@benchmark(
    name="cognitivo-generated",
    dataset="../data/generated_questions.jsonl",
    prompt="QUESTION: {prompt}\nTOOL_TRACE: {tool_trace}\nMETHOD: {derivation_methodology}",
    target_field="answer",
    endpoint_type="chat",
)
@scorer
def component_recall(sample: ScorerInput) -> dict:
    """Fraction of `required_facts` (carried through in `metadata` -- BYOB
    copies every dataset column into `SeedResult.metadata`) found in the
    model's response. `correct` requires every component, matching
    ftpipe's own component_match grader's `all_components` metric."""
    facts = sample.metadata.get("required_facts") or []
    if not facts:
        return {"reward": 0.0, "correct": False, "n_components": 0}
    hits = [_matches(f, sample.response) for f in facts]
    recall = sum(hits) / len(hits)
    return {"reward": recall, "correct": all(hits), "n_components": len(hits)}
