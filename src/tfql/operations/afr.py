"""AFR news-corpus operations.

The Setup Instructions call three rules non-negotiable for reproducibility, and
they are enforced structurally here rather than left to the caller:

  * **All four fields.** Counts search HEADLINE, SUBHEAD, INTRO and TEXT
    combined, via the single ``AFR_ALL_TEXT`` expression in ``store``. No
    operation can narrow the scope by accident.
  * **Once per record.** A row matches at most once however many fields or
    repetitions contain the pattern, which ``count(*)`` over rows gives us.
  * **Word boundaries.** Whole-word searches anchor with ``\\b`` so ``NAB`` does
    not match inside unrelated words.

Counting never touches the warehouse's FTS index. That index stems -- "bank"
matches "banking" and "banks" -- which is correct for ranked retrieval and wrong
for exact counts. It also omits INTRO. Retrieval uses it; counting does not.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..dates import iso, parse_date, validate_window
from ..errors import ErrorCode, TFQLError
from ..evidence import Evidence, OpOutput
from ..invariants import check, check_non_empty
from ..registry import Args, register
from ..store import AFR_ALL_TEXT, AFR_FIELDS, AFR_TABLE, Store

DATASET = "afr"

_RE_METACHARS = set(r"\.^$*+?()[]{}|")


def _escape(pattern: str) -> str:
    """Escape regex metacharacters so a pattern is matched literally."""
    return "".join("\\" + c if c in _RE_METACHARS else c for c in pattern)


def _to_regex(pattern: str, whole_word: bool) -> str:
    """Build the RE2 pattern used for matching.

    Word-boundary anchors are applied only at edges that are themselves word
    characters -- anchoring ``\\b`` beside a symbol such as the ``&`` in "S&P"
    would never match.
    """
    body = _escape(pattern)
    if not whole_word or not pattern:
        return f"(?i){body}"
    lead = r"\b" if pattern[0].isalnum() or pattern[0] == "_" else ""
    trail = r"\b" if pattern[-1].isalnum() or pattern[-1] == "_" else ""
    return f"(?i){lead}{body}{trail}"


def _window_clause(
    store: Store, start: str | None, end: str | None
) -> tuple[str, list[object], str]:
    """SQL predicate, params and a human description for an optional window."""
    lo = parse_date(start) if start else None
    hi = parse_date(end) if end else None
    validate_window(lo, hi)
    lo, hi = store.afr_coverage.clamp(lo, hi)
    return (
        "publication_date between ? and ?",
        [lo, hi],
        f"{iso(lo)} to {iso(hi)}",
    )


# -------------------------------------------------------------- pattern_count


class PatternCountArgs(Args):
    patterns: list[str] = Field(
        min_length=1,
        max_length=40,
        description=(
            "one or more terms to count. Pass them all in a single call -- the "
            "table is scanned once for the whole list."
        ),
    )
    whole_word: bool = Field(
        default=True,
        description=(
            "match whole words only. Leave true for acronyms and tickers; set "
            "false only for deliberate substring matching."
        ),
    )
    start: str | None = Field(default=None, description="ISO window start")
    end: str | None = Field(default=None, description="ISO window end")


@register(
    "afr.pattern_count",
    PatternCountArgs,
    summary=(
        "How many AFR articles mention each of the given terms, searching "
        "headline, subhead, intro and text combined, counting each article once. "
        "Returns per-term counts plus a ranking."
    ),
    datasets=("afr",),
)
def pattern_count(args: PatternCountArgs, store: Store) -> OpOutput:
    where, params, window = _window_clause(store, args.start, args.end)

    # One scan for every pattern: conditional aggregates over a single pass,
    # rather than a query per term.
    selects, regex_params = [], []
    for idx, pattern in enumerate(args.patterns):
        selects.append(f"count(*) filter (where regexp_matches({AFR_ALL_TEXT}, ?)) as p{idx}")
        regex_params.append(_to_regex(pattern, args.whole_word))

    sql = f"select {', '.join(selects)}, count(*) as total from {AFR_TABLE} where {where}"
    row = store.query(sql, regex_params + params)[0]

    counts = [int(v) for v in row[:-1]]
    total = int(row[-1])
    for pattern, count in zip(args.patterns, counts):
        # A pattern cannot match more records than the window holds.
        check(
            0 <= count <= total,
            f"count for {pattern!r} is outside the window's record count",
            pattern=pattern,
            count=count,
            total=total,
        )

    entries = [{"pattern": p, "article_count": c} for p, c in zip(args.patterns, counts)]
    ranked = sorted(entries, key=lambda e: (-int(e["article_count"]), str(e["pattern"])))
    for rank, entry in enumerate(ranked, start=1):
        entry["rank"] = rank

    data: dict[str, object] = {
        "counts": {p: c for p, c in zip(args.patterns, counts)},
        "articles_in_window": total,
        "window_start": window.split(" to ")[0],
        "window_end": window.split(" to ")[1],
    }
    if len(args.patterns) == 1:
        data["pattern"] = args.patterns[0]
        data["article_count"] = counts[0]
    else:
        data["ranked"] = ranked
        data["most_mentioned"] = ranked[0]["pattern"]

    return OpOutput(
        data=data,
        evidence=Evidence(
            dataset=DATASET,
            method=(
                "case-insensitive regex over "
                + "+".join(AFR_FIELDS)
                + " combined, one match per article"
                + (", word-boundary anchored" if args.whole_word else "")
            ),
            records_used=total,
            coverage=window,
        )
        .note("fields_searched", list(AFR_FIELDS))
        .note("whole_word", args.whole_word),
    )


# ---------------------------------------------------------- retrieve_articles


class RetrieveArticlesArgs(Args):
    query: str = Field(description="search terms")
    mode: Literal["relevance", "exact"] = Field(
        default="relevance",
        description=(
            "relevance = BM25 ranking, stems words (bank matches banking); "
            "exact = literal whole-word matching across all four fields"
        ),
    )
    limit: int = Field(default=5, ge=1, le=25)
    start: str | None = Field(default=None, description="ISO window start")
    end: str | None = Field(default=None, description="ISO window end")
    excerpt_chars: int = Field(default=400, ge=100, le=2000)


@register(
    "afr.retrieve_articles",
    RetrieveArticlesArgs,
    summary=(
        "Fetch AFR articles matching a query, with headline, date and a text "
        "excerpt. Use this for article evidence and sentiment context, never "
        "for structured RBA or ASX calculations."
    ),
    datasets=("afr",),
)
def retrieve_articles(args: RetrieveArticlesArgs, store: Store) -> OpOutput:
    where, params, window = _window_clause(store, args.start, args.end)
    excerpt = f"substr(coalesce(text, ''), 1, {args.excerpt_chars})"

    if args.mode == "relevance":
        # BM25 through the prebuilt index. Stemming is desirable here.
        sql = (
            f"select headline, subhead, publication_date, {excerpt}, score "
            f"from (select *, fts_main_{AFR_TABLE}.match_bm25(article_id, ?) "
            f"      as score from {AFR_TABLE}) "
            f"where score is not null and {where} "
            f"order by score desc, publication_date, headline limit {args.limit}"
        )
        rows = store.query(sql, [args.query] + params)
    else:
        sql = (
            f"select headline, subhead, publication_date, {excerpt}, null "
            f"from {AFR_TABLE} "
            f"where regexp_matches({AFR_ALL_TEXT}, ?) and {where} "
            f"order by publication_date, headline limit {args.limit}"
        )
        rows = store.query(sql, [_to_regex(args.query, True)] + params)

    articles = [
        {
            "headline": r[0],
            "subhead": r[1],
            "publication_date": iso(r[2]),
            "excerpt": r[3],
            **({"relevance_score": round(float(r[4]), 4)} if r[4] is not None else {}),
        }
        for r in rows
    ]

    out = OpOutput(
        data={
            "query": args.query,
            "mode": args.mode,
            "article_count": len(articles),
            "articles": articles,
        },
        evidence=Evidence(
            dataset=DATASET,
            method=(
                "BM25 relevance ranking (stemmed) over headline/subhead/text"
                if args.mode == "relevance"
                else "literal whole-word match over " + "+".join(AFR_FIELDS) + " combined"
            ),
            records_used=len(articles),
            coverage=window,
        ).note("mode", args.mode),
    )
    if not articles:
        out.warn(f"no AFR articles matched {args.query!r} within {window}")
    if args.mode == "relevance":
        out.warn(
            "relevance mode stems words, so counts derived from it will not "
            "match afr.pattern_count; use pattern_count for exact counts"
        )
    return out


# ----------------------------------------------------------------- date_count


class DateCountArgs(Args):
    granularity: Literal["day", "month", "year", "total"] = Field(
        default="total",
        description="total gives the corpus size and span; the others bucket it",
    )
    pattern: str | None = Field(default=None, description="optional term to restrict the count to")
    whole_word: bool = Field(default=True)
    field: Literal["all", "headline"] = Field(
        default="all",
        description=(
            "all = the four-field combined scope required for pattern counts; "
            "headline = headline only, for questions that ask specifically"
        ),
    )
    start: str | None = Field(default=None, description="ISO window start")
    end: str | None = Field(default=None, description="ISO window end")
    limit: int = Field(default=20, ge=1, le=200)


@register(
    "afr.date_count",
    DateCountArgs,
    summary=(
        "How many AFR articles were published, in total or bucketed by day, "
        "month or year, with the busiest bucket and the date range covered. "
        "Optionally restricted to articles mentioning a term."
    ),
    datasets=("afr",),
)
def date_count(args: DateCountArgs, store: Store) -> OpOutput:
    where, params, window = _window_clause(store, args.start, args.end)
    scope = "headline" if args.field == "headline" else AFR_ALL_TEXT

    if args.pattern:
        where += f" and regexp_matches(coalesce({scope}, ''), ?)"
        params = params + [_to_regex(args.pattern, args.whole_word)]

    totals = store.query(
        f"select count(*), min(publication_date), max(publication_date) "
        f"from {AFR_TABLE} where {where}",
        params,
    )[0]
    total, first_day, last_day = int(totals[0]), totals[1], totals[2]
    if total == 0:
        raise TFQLError(
            ErrorCode.NO_MATCHING_RECORDS,
            f"no AFR articles matched within {window}",
            dataset=DATASET,
            pattern=args.pattern,
        )

    data: dict[str, object] = {
        "article_count": total,
        "earliest_publication_date": iso(first_day),
        "latest_publication_date": iso(last_day),
        "granularity": args.granularity,
    }
    if args.pattern:
        data["pattern"] = args.pattern

    if args.granularity != "total":
        bucket = {
            "day": "publication_date",
            "month": "date_trunc('month', publication_date)",
            "year": "date_trunc('year', publication_date)",
        }[args.granularity]
        rows = store.query(
            f"select {bucket} as bucket, count(*) as n from {AFR_TABLE} "
            f"where {where} group by bucket order by n desc, bucket "
            f"limit {args.limit}",
            params,
        )
        check_non_empty(rows, "grouping produced no buckets")
        buckets = [{"period": iso(r[0]), "article_count": int(r[1])} for r in rows]
        # Buckets are a partition of the window, so they cannot exceed it.
        check(
            sum(int(b["article_count"]) for b in buckets) <= total,
            "bucket counts exceed the window total",
        )
        data["buckets"] = buckets
        data["busiest_period"] = buckets[0]["period"]
        data["busiest_period_count"] = buckets[0]["article_count"]

    return OpOutput(
        data=data,
        evidence=Evidence(
            dataset=DATASET,
            method=(
                f"article counts bucketed by {args.granularity}"
                if args.granularity != "total"
                else "total article count and publication span"
            )
            + (
                ", restricted to articles matching a term in "
                + ("headline" if args.field == "headline" else "+".join(AFR_FIELDS))
                if args.pattern
                else ""
            ),
            records_used=total,
            coverage=window,
        ).note("scope", args.field),
    )
