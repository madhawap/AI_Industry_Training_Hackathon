"""AFR matching rules.

The Setup Instructions call these non-negotiable for reproducibility: search all
four fields combined, count each article once, and anchor whole-word searches
with word boundaries. These tests exist because the shared warehouse's own FTS
index violates two of the three, so the operations must not rely on it.
"""

from __future__ import annotations

import pytest
from src.tfql.errors import ErrorCode, TFQLError
from src.tfql.operations.afr import _to_regex
from src.tfql.store import AFR_ALL_TEXT, AFR_FIELDS


class TestFourFieldScope:
    def test_all_four_required_fields_are_searched(self):
        assert set(AFR_FIELDS) == {"headline", "subhead", "intro", "text"}

    def test_the_concatenation_covers_every_field(self):
        # intro is the field the warehouse's FTS index omits.
        for field in AFR_FIELDS:
            assert field in AFR_ALL_TEXT

    def test_null_fields_do_not_void_the_concatenation(self):
        # coalesce guards every field; a null subhead must not blank the row.
        assert AFR_ALL_TEXT.count("coalesce") == len(AFR_FIELDS)

    def test_evidence_records_the_scope_searched(self, run):
        evidence = run("afr.pattern_count", patterns=["bank"]).evidence.to_dict()
        assert evidence["fields_searched"] == list(AFR_FIELDS)


class TestWordBoundaries:
    def test_whole_word_excludes_substring_matches(self, run):
        """The exact inflation the Setup Instructions warn about.

        'RBA' appears as a substring inside 'Transurban'. Counting without
        boundaries silently inflates the result.
        """
        whole = run("afr.pattern_count", patterns=["RBA"]).data["article_count"]
        substring = run("afr.pattern_count", patterns=["RBA"], whole_word=False).data[
            "article_count"
        ]
        assert whole < substring

    def test_bank_count_is_boundary_anchored(self, run):
        whole = run("afr.pattern_count", patterns=["bank"]).data["article_count"]
        substring = run("afr.pattern_count", patterns=["bank"], whole_word=False).data[
            "article_count"
        ]
        # banking / banks / Bendigo Bank inflate the substring form.
        assert whole == 13
        assert substring > whole

    def test_regex_anchors_alphanumeric_edges(self):
        assert _to_regex("NAB", True) == r"(?i)\bNAB\b"

    def test_regex_omits_anchors_beside_symbols(self):
        # \b next to '&' would never match, so it must be left off.
        assert _to_regex("S&P", True) == r"(?i)\bS&P\b"
        assert _to_regex("$A", True) == r"(?i)\$A\b"

    def test_metacharacters_are_escaped(self):
        assert _to_regex("a.b", False) == r"(?i)a\.b"
        assert _to_regex("(x)", False) == r"(?i)\(x\)"


class TestOncePerRecord:
    def test_count_never_exceeds_the_corpus(self, run):
        data = run("afr.pattern_count", patterns=["the"], whole_word=False).data
        assert data["article_count"] <= data["articles_in_window"]

    def test_a_common_word_still_counts_articles_not_occurrences(self, run):
        # 'the' appears many times per article; the count is bounded by rows.
        data = run("afr.pattern_count", patterns=["the"], whole_word=False).data
        assert data["article_count"] <= 110


class TestMonotonicity:
    def test_widening_the_window_cannot_reduce_the_count(self, run):
        narrow = run(
            "afr.pattern_count",
            patterns=["bank"],
            start="2015-02-01",
            end="2015-02-28",
        ).data["article_count"]
        wide = run(
            "afr.pattern_count",
            patterns=["bank"],
            start="2015-01-01",
            end="2015-03-31",
        ).data["article_count"]
        assert wide >= narrow


class TestBatching:
    def test_multiple_patterns_return_a_ranking(self, run):
        data = run("afr.pattern_count", patterns=["RBA", "BHP", "ANZ", "NAB"]).data
        counts = [entry["article_count"] for entry in data["ranked"]]
        assert counts == sorted(counts, reverse=True)
        assert data["most_mentioned"] == data["ranked"][0]["pattern"]

    def test_batched_counts_match_individual_counts(self, run):
        """One scan for many patterns must agree with one scan each."""
        batched = run("afr.pattern_count", patterns=["RBA", "BHP"]).data["counts"]
        for pattern, count in batched.items():
            single = run("afr.pattern_count", patterns=[pattern]).data["article_count"]
            assert single == count


class TestDateCount:
    def test_total_reports_the_corpus_span(self, run):
        data = run("afr.date_count", granularity="total").data
        assert data["article_count"] == 110
        assert data["earliest_publication_date"] == "2015-01-05"
        assert data["latest_publication_date"] == "2015-03-31"

    def test_busiest_day_is_the_top_bucket(self, run):
        data = run("afr.date_count", granularity="day").data
        assert data["busiest_period_count"] == data["buckets"][0]["article_count"]
        counts = [bucket["article_count"] for bucket in data["buckets"]]
        assert counts == sorted(counts, reverse=True)

    def test_headline_scope_is_narrower_than_all_fields(self, run):
        headline = run("afr.date_count", pattern="RBA", field="headline").data
        combined = run("afr.date_count", pattern="RBA", field="all").data
        assert headline["article_count"] <= combined["article_count"]

    def test_no_match_raises_rather_than_returning_zero(self, run):
        with pytest.raises(TFQLError) as exc:
            run("afr.date_count", pattern="zzzznotaword")
        assert exc.value.code is ErrorCode.NO_MATCHING_RECORDS


class TestRetrieval:
    def test_relevance_mode_warns_about_stemming(self, run):
        out = run("afr.retrieve_articles", query="bank", mode="relevance")
        assert any("stem" in w for w in out.warnings)

    def test_exact_mode_uses_whole_word_matching(self, run):
        out = run("afr.retrieve_articles", query="RBA", mode="exact", limit=10)
        for article in out.data["articles"]:
            assert "transurban" not in article["headline"].lower()

    def test_articles_carry_the_fields_the_synthesiser_needs(self, run):
        article = run("afr.retrieve_articles", query="iron ore", limit=1).data["articles"][0]
        assert {"headline", "publication_date", "excerpt"} <= set(article)

    def test_empty_result_is_warned_not_raised(self, run):
        out = run("afr.retrieve_articles", query="zzzznotaword", mode="exact")
        assert out.data["article_count"] == 0
        assert out.warnings
