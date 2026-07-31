"""Offline validation for nemo_benchmark.py: registration, real dataset load,
and scoring -- everything short of an actual `run_evaluation()` call, which
needs a live OpenAI-compatible endpoint this sandbox doesn't have. Run with
whichever interpreter has `nemo-evaluator` installed:

    python3 test_nemo_benchmark.py
"""

from __future__ import annotations

import asyncio

import nemo_benchmark  # noqa: F401 -- registers "cognitivo-generated" as a side effect
from nemo_evaluator.environments.registry import get_environment, list_environments


def main() -> None:
    assert "cognitivo-generated" in list_environments(), list_environments()
    print("OK: benchmark registered under 'cognitivo-generated'")

    env = get_environment("cognitivo-generated", num_examples=5)
    assert len(env._dataset) == 5
    print(f"OK: loaded {len(env._dataset)} real rows from generated_questions.jsonl")

    seed = asyncio.run(env.seed(0))
    assert "QUESTION:" in seed.prompt and "TOOL_TRACE:" in seed.prompt
    assert seed.expected_answer
    assert seed.metadata.get("required_facts")
    print("OK: seed() renders the prompt template and carries required_facts through metadata")
    print(f"    sample prompt (truncated): {seed.prompt[:120]}...")

    # A model producing the exact gold answer should score perfectly.
    verdict_perfect = asyncio.run(env.verify(seed.expected_answer, seed.expected_answer, **seed.metadata))
    assert verdict_perfect.reward == 1.0, verdict_perfect
    print(f"OK: gold answer scores reward=1.0 ({verdict_perfect.scoring_details})")

    # A model producing an unrelated, factually-empty answer should score near zero.
    verdict_wrong = asyncio.run(env.verify("I don't know.", seed.expected_answer, **seed.metadata))
    assert verdict_wrong.reward < verdict_perfect.reward, verdict_wrong
    print(f"OK: an unrelated answer scores lower (reward={verdict_wrong.reward})")

    print("\nAll offline checks passed. NOT exercised: an actual run_evaluation() "
          "call against a live model endpoint -- needs ModelClient(base_url=...).")


if __name__ == "__main__":
    main()
