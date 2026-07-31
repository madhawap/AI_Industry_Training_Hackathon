"""Reference-overlap grader — ROUGE-L, BLEU, token-F1 against `rec.target`.

`component_match` is the sharp signal when `rec.eval["components"]` exists, but
it scores nothing when it doesn't. This grader needs no grading components at
all, only the target answer every record already has, so it is the fallback
that keeps `evaluate` meaningful on data that hasn't got a components field yet.

ROUGE-L and BLEU come from optional packages (`rouge-score`, `sacrebleu`). If
they are not installed, this grader still runs — it just reports `token_f1`
and warns once, rather than failing the whole `evaluate` stage over an
optional metric.
"""

from __future__ import annotations

import re
import warnings

from ftpipe.registry import register

_missing_deps_warned = False


def _normalize(text: str) -> str:
    text = text.lower().strip().replace(",", "")
    return re.sub(r"\s+", " ", text)


def _token_f1(pred: str, ref: str) -> float:
    p, r = _normalize(pred).split(), _normalize(ref).split()
    if not p or not r:
        return 0.0
    common = set(p) & set(r)
    if not common:
        return 0.0
    overlap = sum(min(p.count(w), r.count(w)) for w in common)
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def _load_optional_scorers():
    global _missing_deps_warned
    try:
        from rouge_score import rouge_scorer
        import sacrebleu

        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return rouge, sacrebleu
    except ImportError:
        if not _missing_deps_warned:
            warnings.warn(
                "reference_overlap: 'rouge-score'/'sacrebleu' not installed — "
                "reporting token_f1 only. `pip install rouge-score sacrebleu` for rougeL/bleu."
            )
            _missing_deps_warned = True
        return None, None


class ReferenceOverlap:
    name = "reference_overlap"

    def __init__(self, cfg: dict):
        self._rouge, self._sacrebleu = _load_optional_scorers()

    def score(self, rec, prediction: str) -> dict[str, float]:
        ref = rec.target
        scores = {"token_f1": _token_f1(prediction, ref)}
        if self._rouge is not None:
            scores["rougeL"] = self._rouge.score(ref, prediction)["rougeL"].fmeasure
            scores["bleu"] = self._sacrebleu.sentence_bleu(prediction, [ref]).score / 100.0
        return scores


@register("grader", "reference_overlap")
def build(cfg: dict) -> ReferenceOverlap:
    return ReferenceOverlap(cfg)
