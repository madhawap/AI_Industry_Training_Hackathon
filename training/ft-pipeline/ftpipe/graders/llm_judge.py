"""LLM-as-judge grader — groundedness / correctness / concision, 1-5 each,
normalised to 0-1 like every other grader here.

The one metric in this pipeline that can catch hallucination independent of
phrasing overlap: it scores whether the answer only uses facts present in
`rec.inputs`, not whether it happens to share words with `rec.target`. Not in
any default `evaluate.graders` list — it costs a network call per example and
needs an API key — opt in explicitly:

    evaluate:
      graders: [component_match, format_health, llm_judge]
      llm_judge: {provider: anthropic, model: claude-sonnet-5}

`provider` is "anthropic" (needs `ANTHROPIC_API_KEY`) or "openai" (needs
`OPENAI_API_KEY`).
"""

from __future__ import annotations

import json
import os

from ftpipe.registry import register

_JUDGE_PROMPT = """You are grading whether a model's answer is grounded in the context it was given (no unsupported claims), and whether it is correct.

Question: {question}
Context / tool results provided to the model: {context}
Reference answer (may be partial or absent): {reference}
Model's answer: {answer}

Score 1-5 on each axis:
- groundedness: does it only use facts present in the context (no hallucination)?
- correctness: does it match the reference's key facts (ignore if reference is empty)?
- concision: is it appropriately brief, not padded?

Respond with ONLY a JSON object: {{"groundedness": int, "correctness": int, "concision": int, "reasoning": "<one sentence>"}}"""


class LLMJudge:
    name = "llm_judge"

    def __init__(self, cfg: dict):
        self.provider = cfg.get("provider", "anthropic")
        if self.provider not in ("anthropic", "openai"):
            raise ValueError(f"llm_judge.provider must be 'anthropic' or 'openai', got {self.provider!r}")
        self.model = cfg.get("model", "claude-sonnet-5")

    def _call(self, prompt: str) -> str:
        if self.provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            resp = client.messages.create(
                model=self.model, max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    def score(self, rec, prediction: str) -> dict[str, float]:
        question = rec.inputs.get("question", json.dumps(rec.inputs, ensure_ascii=False))
        context = json.dumps(rec.inputs, ensure_ascii=False, sort_keys=True)
        prompt = _JUDGE_PROMPT.format(question=question, context=context, reference=rec.target, answer=prediction)

        raw = self._call(prompt).strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        verdict = json.loads(raw)
        return {
            "judge_groundedness": verdict["groundedness"] / 5.0,
            "judge_correctness": verdict["correctness"] / 5.0,
            "judge_concision": verdict["concision"] / 5.0,
        }


@register("grader", "llm_judge")
def build(cfg: dict) -> LLMJudge:
    return LLMJudge(cfg)
