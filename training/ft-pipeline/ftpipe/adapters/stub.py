"""Stub source adapter — synthetic records with NO dependency on the real
data, schema, or contract.

Its whole job is to let the eight stages run end-to-end today. The shape of
`inputs` here is arbitrary; that is the point. When the real contract lands,
write a sibling adapter and change one line of config.
"""

from __future__ import annotations

import random

from ftpipe.registry import register
from ftpipe.schema import Record, make_id

# A tiny closed world so targets are deterministic and gradeable without any
# model, dataset, or tool layer being decided.
_TEMPLATES = [
    ("count", "How many {thing} are in the {place}?", "There are {a} {thing} in the {place}."),
    ("extreme", "What is the largest {thing} in the {place}?", "The largest {thing} in the {place} is {b}."),
    ("compare", "Are there more {thing} in the {place} or the {place2}?",
     "There are more {thing} in the {place} ({a}) than in the {place2} ({c})."),
    ("missing", "How many {thing} are in the {place2} before {year}?",
     "The data does not cover the {place2} before {year}, so this cannot be answered."),
]
_THINGS = ["widgets", "sprockets", "cogs", "levers", "valves"]
_PLACES = ["north depot", "south depot", "east yard", "west yard", "central store"]


@register("adapter", "stub")
def load(cfg: dict) -> list[Record]:
    n = int(cfg.get("n", 200))
    rng = random.Random(int(cfg.get("seed", 0)))
    records: list[Record] = []

    for i in range(n):
        task, q_tpl, a_tpl = _TEMPLATES[i % len(_TEMPLATES)]
        thing = rng.choice(_THINGS)
        place, place2 = rng.sample(_PLACES, 2)
        a, c = rng.randint(2, 99), rng.randint(2, 99)
        b = f"{rng.randint(100, 999)}mm"
        year = rng.choice([2015, 2016, 2017])

        fields = dict(thing=thing, place=place, place2=place2, a=a, b=b, c=c, year=year)
        question = q_tpl.format(**fields)
        target = a_tpl.format(**fields)

        # `inputs` is opaque to the pipeline: only the renderer reads it.
        inputs = {
            "question": question,
            "context": {"facts": {k: v for k, v in fields.items() if k in ("a", "b", "c", "year")}},
        }
        # `eval` is opaque too: only the grader reads it. Components are the
        # substrings a component-recall grader will look for.
        components = [str(fields[k]) for k in ("a", "b", "c") if str(fields[k]) in target]
        if task == "missing":
            components = ["does not cover", str(year)]

        records.append(
            Record(
                id=make_id("stub", i, question),
                task=task,
                target=target,
                inputs=inputs,
                eval={"components": components},
                meta={
                    "source": "stub",
                    # Group by template so curate cannot split near-duplicates
                    # across train/val — the inflated-score trap.
                    "group_key": f"{task}:{thing}",
                    "difficulty": ["easy", "medium", "hard"][i % 3],
                },
            )
        )
    return records
