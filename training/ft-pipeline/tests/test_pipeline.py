"""Tests for the properties that are easy to break and expensive to discover late.

Runs with plain python (no pytest needed):

    conda run -n ft-pipeline python tests/test_pipeline.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ftpipe import registry  # noqa: E402
from ftpipe.schema import Record, SchemaError, make_id, read_jsonl, validate, write_jsonl  # noqa: E402

PASS, FAIL = "\033[0;32m✓\033[0m", "\033[0;31m✗\033[0m"
_results: list[tuple[str, str | None]] = []


def test(fn):
    try:
        fn()
        _results.append((fn.__name__, None))
    except AssertionError as exc:
        _results.append((fn.__name__, str(exc) or "assertion failed"))
    except Exception as exc:  # noqa: BLE001
        _results.append((fn.__name__, f"{exc.__class__.__name__}: {exc}"))
    return fn


def _rec(**kw):
    base = dict(id="x1", task="t", target="an answer", inputs={"question": "q"}, eval={}, meta={})
    base.update(kw)
    return Record(**base)


# --- schema -----------------------------------------------------------------
@test
def schema_rejects_empty_target():
    try:
        validate(_rec(target="   "))
    except SchemaError:
        return
    raise AssertionError("empty target should be rejected")


@test
def schema_rejects_unknown_fields():
    try:
        Record.from_dict({"id": "a", "task": "t", "target": "x", "surprise": 1})
    except SchemaError:
        return
    raise AssertionError("unknown field should be rejected")


@test
def schema_roundtrips_opaque_payloads():
    """`inputs`/`eval` must survive untouched — they hold undecided contract detail."""
    weird = {"nested": {"a": [1, 2, {"b": None}]}, "unicode": "café"}
    rec = _rec(inputs=weird, eval={"components": ["1", "2"]})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.jsonl"
        write_jsonl([rec], path)
        back = read_jsonl(path)[0]
    assert back.inputs == weird, "opaque inputs mutated in round-trip"
    assert back.eval == {"components": ["1", "2"]}


@test
def ids_are_content_derived_and_stable():
    assert make_id("a", {"b": 1}) == make_id("a", {"b": 1})
    assert make_id("a", {"b": 1}) != make_id("a", {"b": 2})


# --- curate: the leakage property ------------------------------------------
@test
def curate_never_splits_a_group_across_splits():
    from ftpipe.artifacts import Run
    from ftpipe.stages import curate, ingest

    cfg = {
        "seed": 0,
        "adapter": {"name": "stub", "n": 120, "seed": 1},
        "curate": {"splits": [0.6, 0.2, 0.2]},
    }
    with tempfile.TemporaryDirectory() as tmp:
        import ftpipe.artifacts as artifacts

        original = artifacts.RUNS_DIR
        artifacts.RUNS_DIR = Path(tmp)
        try:
            run = Run("t", cfg)
            ingest.run(cfg, run)
            curate.run(cfg, run)
            groups = {}
            for split in ("train", "val", "test"):
                for rec in read_jsonl(Path(tmp) / "t" / "curate" / f"{split}.jsonl"):
                    key = rec.meta["group_key"]
                    groups.setdefault(key, set()).add(split)
        finally:
            artifacts.RUNS_DIR = original

    leaked = {g: s for g, s in groups.items() if len(s) > 1}
    assert not leaked, f"group(s) leaked across splits: {leaked}"


# --- renderer: train/serve parity -------------------------------------------
@test
def renderer_builds_identical_prompts_for_train_and_serve():
    """The whole anti-skew argument rests on this."""
    renderer = registry.get("renderer", "chat")({
        "system_prompt": "SYS", "user_template": "Q: {question}\nF: {context}"
    })
    rec = _rec(inputs={"question": "how many?", "context": {"a": 1}})

    train_msgs = renderer.render(rec, purpose="train")["messages"]
    serve_msgs = renderer.serving_payload(rec.inputs)["messages"]

    assert train_msgs[:-1] == serve_msgs, "training prompt differs from serving prompt"
    assert train_msgs[-1] == {"role": "assistant", "content": rec.target}


@test
def renderer_is_strict_about_missing_keys():
    renderer = registry.get("renderer", "chat")({"user_template": "{question} {absent}"})
    try:
        renderer.serving_payload({"question": "q"})
    except KeyError:
        return
    raise AssertionError("missing template key should raise, not silently render")


@test
def contract_change_drill():
    """Changing the prompt format must not require touching anything but the
    renderer config. If this ever fails, contract knowledge has leaked."""
    rec = _rec(inputs={"alpha": "A", "beta": {"b": 2}})
    for template in ("{alpha}", "X {alpha} Y {beta}", "{beta}\n---\n{alpha}"):
        renderer = registry.get("renderer", "chat")({"user_template": template})
        messages = renderer.serving_payload(rec.inputs)["messages"]
        assert messages[-1]["role"] == "user" and messages[-1]["content"]


# --- graders ----------------------------------------------------------------
@test
def component_match_handles_equivalent_expressions():
    grader = registry.get("grader", "component_match")({"numeric_tolerance": 0.0})
    rec = _rec(eval={"components": ["0.1", "2020-11-04", "16"]})
    scores = grader.score(rec, "The lowest was 0.10, effective 4 November 2020, across 16 records.")
    assert scores["component_recall"] == 1.0, scores


@test
def component_match_gives_partial_credit():
    grader = registry.get("grader", "component_match")({})
    rec = _rec(eval={"components": ["4.75", "2010-11-02", "11"]})
    scores = grader.score(rec, "The highest was 4.75 on 2010-11-03, across 11 records.")
    assert abs(scores["component_recall"] - 2 / 3) < 1e-6, scores
    assert scores["all_components"] == 0.0


@test
def format_health_flags_invented_numbers():
    grader = registry.get("grader", "format_health")({})
    rec = _rec(inputs={"context": {"a": 41}})
    clean = grader.score(rec, "There were 41 changes.")
    dirty = grader.score(rec, "There were 41 changes worth 987 points.")
    assert clean["hallucinated_number_rate"] == 0.0, clean
    assert dirty["hallucinated_number_rate"] == 1.0, dirty


@test
def format_health_treats_0_10_as_0_1():
    grader = registry.get("grader", "format_health")({})
    rec = _rec(inputs={"context": {"rate": 0.1}})
    assert grader.score(rec, "The rate was 0.10.")["hallucinated_number_rate"] == 0.0


@test
def reference_overlap_scores_an_exact_match_as_one():
    grader = registry.get("grader", "reference_overlap")({})
    rec = _rec(target="the rate was 0.1 on 4 november")
    scores = grader.score(rec, "The rate was 0.1 on 4 november")
    assert scores["token_f1"] == 1.0, scores


@test
def reference_overlap_scores_disjoint_text_as_zero():
    grader = registry.get("grader", "reference_overlap")({})
    rec = _rec(target="the rate was 0.1")
    scores = grader.score(rec, "completely unrelated words here")
    assert scores["token_f1"] == 0.0, scores


@test
def llm_judge_rejects_an_unknown_provider_before_any_network_call():
    try:
        registry.get("grader", "llm_judge")({"provider": "not-a-real-provider"})
    except ValueError as exc:
        assert "provider" in str(exc)
        return
    raise AssertionError("unknown judge_provider should raise, not silently pick one")


# --- selection policy -------------------------------------------------------
@test
def guardrails_reject_a_higher_scoring_but_hallucinating_checkpoint():
    policy = registry.get("policy", "guardrailed")({
        "primary": "component_recall",
        "guardrails": {"hallucinated_number_rate": {"max": 0.05}},
    })
    decision = policy.choose([
        {"checkpoint": "a", "step": 20, "label": "step-20",
         "metrics": {"component_recall": 0.70, "hallucinated_number_rate": 0.00}},
        {"checkpoint": "b", "step": 40, "label": "step-40",
         "metrics": {"component_recall": 0.95, "hallucinated_number_rate": 0.40}},
    ])
    assert decision["chosen"] == "a", decision


@test
def policy_prefers_the_earliest_checkpoint_within_tolerance():
    policy = registry.get("policy", "guardrailed")({
        "primary": "component_recall", "tie_tolerance": 0.01,
    })
    decision = policy.choose([
        {"checkpoint": "early", "step": 20, "label": "step-20", "metrics": {"component_recall": 0.900}},
        {"checkpoint": "late", "step": 100, "label": "step-100", "metrics": {"component_recall": 0.905}},
    ])
    assert decision["chosen"] == "early", decision


@test
def policy_reports_when_nothing_passes():
    policy = registry.get("policy", "guardrailed")({
        "primary": "component_recall", "guardrails": {"hallucinated_number_rate": {"max": 0.0}},
    })
    decision = policy.choose([
        {"checkpoint": "a", "step": 20, "label": "step-20",
         "metrics": {"component_recall": 0.9, "hallucinated_number_rate": 0.5}},
    ])
    assert decision["chosen"] is None and decision["candidates"][0]["violations"]


# --- config -----------------------------------------------------------------
@test
def undecided_config_raises_a_named_error():
    from ftpipe import config

    try:
        config.require({"train": {"seq_len": None}}, "train.seq_len", "measure it first")
    except config.UndecidedError as exc:
        assert "train.seq_len" in str(exc)
        return
    raise AssertionError("null config value should raise UndecidedError")


@test
def run_id_is_stable_across_set_overrides():
    """Regression: run_id used to hash the whole config, so `--set` between
    two `stage` invocations silently created a second run directory and the
    later stage could not see the earlier stage's output."""
    from ftpipe.cli import _run_id

    base = {"name": "demo", "run_id": "auto", "train": {"seq_len": None}}
    changed = {"name": "demo", "run_id": "auto", "train": {"seq_len": 1024}}
    assert _run_id(base) == _run_id(changed) == "demo"


if __name__ == "__main__":
    failures = [(n, e) for n, e in _results if e]
    for name, error in _results:
        print(f"  {FAIL if error else PASS} {name}" + (f"\n      {error}" if error else ""))
    print(f"\n{len(_results) - len(failures)}/{len(_results)} passed")
    sys.exit(1 if failures else 0)
