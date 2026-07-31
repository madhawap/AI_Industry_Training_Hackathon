"""Chat renderer: canonical record -> the exact text the model reads.

THE ONE RULE OF THIS FILE
    Training data and live requests are both built here, by `build_messages`.
    Nothing else in the codebase — and nothing in the serving app — may format
    a prompt. Train/serve prompt skew is the most common silent cause of a
    fine-tune that "trained fine but got worse in production", and keeping a
    single builder makes it structurally impossible rather than a thing you
    have to remember.

The prompt LAYOUT is config, not code: `user_template` is a format string over
whatever keys `inputs` happens to have. So when the contract changes, you edit
config (or at most this file), never the pipeline.
"""

from __future__ import annotations

import json
from typing import Any

from ftpipe.registry import register


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ChatRenderer:
    """Args (from config.renderer):
    system_prompt : str | null  — set null until the serving system prompt is decided
    user_template : str         — format string over `inputs` keys, e.g.
                                  "QUESTION: {question}\nFACTS: {context}"
    strict        : bool        — error if the template references a missing key
    """

    def __init__(self, cfg: dict):
        self.system_prompt = cfg.get("system_prompt")
        self.user_template = cfg.get("user_template") or "{question}"
        self.strict = bool(cfg.get("strict", True))

    # -- the single prompt builder -------------------------------------------
    def build_messages(self, inputs: dict) -> list[dict]:
        fields = {k: _stringify(v) for k, v in inputs.items()}
        try:
            user = self.user_template.format(**fields)
        except KeyError as exc:
            if self.strict:
                raise KeyError(
                    f"renderer.user_template references {exc} but `inputs` has {sorted(fields)}. "
                    f"Either fix the template or the adapter."
                ) from None
            user = self.user_template.format_map({**{k: "" for k in _template_keys(self.user_template)}, **fields})

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user})
        return messages

    # -- training-time --------------------------------------------------------
    def render(self, rec, purpose: str = "train") -> dict:
        out = {"id": rec.id, "task": rec.task, "messages": self.build_messages(rec.inputs)}
        if purpose == "train":
            out["messages"] = out["messages"] + [{"role": "assistant", "content": rec.target}]
        return out

    # -- serving-time: identical prompt, no assistant turn --------------------
    def serving_payload(self, inputs: dict) -> dict:
        return {"messages": self.build_messages(inputs)}

    # -- model output -> gradeable answer ------------------------------------
    def parse(self, raw_output: str) -> str:
        return raw_output.strip()


def _template_keys(template: str) -> list[str]:
    import string

    return [f for _, f, _, _ in string.Formatter().parse(template) if f]


@register("renderer", "chat")
def build(cfg: dict) -> ChatRenderer:
    return ChatRenderer(cfg)
