"""Evaluation dataset for the finance hackathon agent, grounded in the real
RBA / ASX / AFR datasets under ``AI_Industry_Training_Hackathon/data set``.

Every ``answer`` and ``grading_components`` value below was computed
directly from those files with the same deterministic logic implemented in
``finance_tools.py`` (see that module's docstring) -- nothing here is
illustrative or invented. If the underlying data files change, regenerate
these numbers rather than hand-editing them; a quick way is to call the
relevant tool function directly and copy its returned figures in.

``expected_tools`` / ``expected_tool_args`` name the real tools in
``finance_tools.py`` and their real argument shapes -- no placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_DATASET_NAME = "finance-hackathon-eval"


@dataclass(frozen=True)
class FinanceExample:
    """One evaluation row.

    Attributes:
        id: Stable identifier (used as the LangSmith example name/metadata).
        difficulty: Free-form difficulty label ("easy" / "medium" / "hard").
        question: Natural-language question posed to the agent.
        answer: Reference answer, verified against the real datasets.
        grading_components: Case-insensitive substrings that MUST appear
            somewhere in the agent's final answer -- one entry per fact the
            answer must state (a number, a date, a name, ...).
        expected_tools: Ordered real tool names the agent MUST call.
        expected_tool_args: Per-tool exact-match arg expectations, aligned
            positionally with ``expected_tools``.
    """

    id: str
    difficulty: str
    question: str
    answer: str
    grading_components: list[str]
    expected_tools: list[str] = field(default_factory=list)
    expected_tool_args: list[dict[str, Any]] = field(default_factory=list)


EXAMPLES: list[FinanceExample] = [
    # ----- RBA cash-rate history ---------------------------------------
    FinanceExample(
        id="rba-lowest-rate",
        difficulty="easy",
        question=(
            "What is the lowest cash-rate target in the RBA dataset, when "
            "did it first take effect, and how many decision records show "
            "that rate?"
        ),
        answer=(
            "The lowest cash-rate target in the dataset is 0.1%, which "
            "first took effect on 2020-11-04, across 16 decision records "
            "at that rate."
        ),
        grading_components=["0.1", "2020-11-04", "16"],
        expected_tools=["rba_extreme_rate"],
        expected_tool_args=[
            {"name": "rba_extreme_rate", "args": {"mode": "lowest"}},
        ],
    ),
    FinanceExample(
        id="rba-longest-gap",
        difficulty="medium",
        question=(
            "What was the longest stretch between two non-zero RBA rate "
            "changes in this dataset?"
        ),
        answer=(
            "The longest stretch between two non-zero rate changes was "
            "1036 days, from 2016-08-03 to 2019-06-05, during which the "
            "rate held at 1.5% before changing to 1.25%."
        ),
        grading_components=["1036", "2016-08-03", "2019-06-05", "1.5", "1.25"],
        expected_tools=["rba_longest_gap"],
        expected_tool_args=[{"name": "rba_longest_gap", "args": {}}],
    ),
    FinanceExample(
        id="rba-largest-hike-cycle",
        difficulty="hard",
        question=(
            "Which tightening cycle (consecutive hikes with no intervening "
            "cuts) had the most hikes, and what was the cumulative "
            "increase?"
        ),
        answer=(
            "The largest tightening cycle ran from 2022-05-04 to "
            "2023-11-08 and comprised 13 hikes, for a cumulative increase "
            "of 4.25 percentage points. The target rate immediately before "
            "the first hike was 0.1%, and the rate reached by 2023-11-08 "
            "was 4.35%."
        ),
        grading_components=[
            "13", "2022-05-04", "2023-11-08", "4.25", "0.1", "4.35",
        ],
        expected_tools=["rba_largest_cycle"],
        expected_tool_args=[
            {"name": "rba_largest_cycle", "args": {"direction": "hikes"}},
        ],
    ),
    # ----- ASX prices (18-company basket) -------------------------------
    FinanceExample(
        id="asx-bhp-price-extremes",
        difficulty="easy",
        question=(
            "What were BHP's highest and lowest closing prices in this "
            "dataset, and on what dates did they occur?"
        ),
        answer=(
            "BHP's highest close was $33.72 on 2021-08-04, and its lowest "
            "close was $6.44 on 2016-01-21."
        ),
        grading_components=["33.72", "2021-08-04", "6.44", "2016-01-21"],
        expected_tools=["asx_price_extremes"],
        expected_tool_args=[
            {"name": "asx_price_extremes", "args": {"ticker": "BHP"}},
        ],
    ),
    FinanceExample(
        id="asx-bhp-largest-decline",
        difficulty="medium",
        question=(
            "What was BHP's largest single-day percentage decline in "
            "closing price, and when did it happen?"
        ),
        answer=(
            "BHP's largest single-day decline was -14.41%, on 2020-03-09, "
            "when the close moved from $19.07 to $16.32."
        ),
        grading_components=["-14.41%", "2020-03-09", "19.07", "16.32"],
        expected_tools=["asx_single_day_move"],
        expected_tool_args=[
            {
                "name": "asx_single_day_move",
                "args": {"ticker": "BHP", "direction": "decline"},
            }
        ],
    ),
    FinanceExample(
        id="asx-bhp-basket-rank",
        difficulty="hard",
        question=(
            "Where does BHP rank by total close-to-close return among the "
            "18-company ASX basket over the full dataset period, and what "
            "is its return?"
        ),
        answer=(
            "BHP ranks 3 of 18 companies in the ASX basket by total "
            "close-to-close return, with a return of +139.40% over the "
            "full dataset period."
        ),
        grading_components=["3 of 18", "139.40"],
        expected_tools=["asx_rank_by_return"],
        expected_tool_args=[
            {"name": "asx_rank_by_return", "args": {"ticker": "BHP"}},
        ],
    ),
    # ----- AFR articles ---------------------------------------------------
    FinanceExample(
        id="afr-total-articles",
        difficulty="easy",
        question=(
            "How many articles are in the AFR dataset in total, and what "
            "date range do they span?"
        ),
        answer=(
            "There are 219538 articles in total, spanning publication "
            "dates from 2015-01-02 to 2021-12-29."
        ),
        grading_components=["219538", "2015-01-02", "2021-12-29"],
        expected_tools=["afr_corpus_stats"],
        expected_tool_args=[{"name": "afr_corpus_stats", "args": {}}],
    ),
    FinanceExample(
        id="afr-busiest-day",
        difficulty="medium",
        question="Which single day had the most AFR articles published, and how many?",
        answer="The busiest day was 2016-08-26, with 314 articles published.",
        grading_components=["2016-08-26", "314"],
        expected_tools=["afr_busiest_day"],
        expected_tool_args=[{"name": "afr_busiest_day", "args": {}}],
    ),
    FinanceExample(
        id="afr-bank-mentions-and-longest-headline",
        difficulty="hard",
        question=(
            "How many AFR articles mention 'bank' (as a whole word) in "
            "their headline, subhead, intro, or body text, and what is the "
            "longest headline in the entire dataset by character count?"
        ),
        answer=(
            "47347 articles mention 'bank' as a whole word across their "
            "headline, subhead, intro, or body text. The longest headline "
            "in the dataset is 489 characters, published 2015-07-17 -- it "
            "is actually a scraped 'MINING & OIL' share-price table, not an "
            "editorial headline, which is a real data-quality quirk in "
            "this corpus."
        ),
        grading_components=["47347", "489", "MINING & OIL", "2015-07-17"],
        expected_tools=["afr_pattern_count", "afr_longest_headline"],
        expected_tool_args=[
            {"name": "afr_pattern_count", "args": {"pattern": "bank"}},
            {"name": "afr_longest_headline", "args": {}},
        ],
    ),
]


def to_langsmith_examples() -> list[dict[str, Any]]:
    """Serialise EXAMPLES into the LangSmith dataset upload format."""
    rows = []
    for ex in EXAMPLES:
        rows.append({
            "inputs": {"question": ex.question},
            "outputs": {
                "answer": ex.answer,
                "grading_components": ex.grading_components,
                "expected_tools": ex.expected_tools,
                "expected_tool_args": ex.expected_tool_args,
            },
            "metadata": {"example_id": ex.id, "difficulty": ex.difficulty},
        })
    return rows
