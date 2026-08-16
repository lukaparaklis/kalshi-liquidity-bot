"""
test_market_discovery.py
------------------------
Choosing which series to look for markets in.

This exists because of a failure that produced no error at all. Paging
/markets and filtering MVE markets out worked on demo — 663 of 800 survived —
and returned literally nothing on prod. Two MVE series hold so many open
markets that 5,000 rows across 25 pages contained not one ordinary market. The
filter was correct; the enumeration never reached anything for it to pass.

Demo actively hid this: it has few enough MVE markets that the first pages look
normal. Any check that "the filter keeps some markets" would have passed on
demo and told us nothing.

So discovery goes through /series instead, where MVE is 14 entries out of
~13,000, and these tests pin the selection rules.
"""

from __future__ import annotations

import pytest

from src.scanner import FAST_FREQUENCIES, SERIES_LIMIT, is_mve, select_series


def series(ticker, *, category="Economics", frequency="daily") -> dict:
    return {"ticker": ticker, "category": category, "frequency": frequency}


class TestMveDetection:
    @pytest.mark.parametrize("ticker", [
        "KXMVECROSSCATEGORY-SHARD1-R",
        "KXMVESPORTSMULTIGAMEEXTENDED-R",
        "KXMVECROSSCATEGORY-R",
    ])
    def test_real_mve_series_tickers_are_detected(self, ticker):
        """All three MVE series observed on prod."""
        assert is_mve(ticker) is True

    @pytest.mark.parametrize("ticker", ["KXHIGHNY", "KXBTC15M", "KXNBA", "KXCPI"])
    def test_ordinary_series_are_not_mve(self, ticker):
        assert is_mve(ticker) is False

    def test_it_is_case_insensitive(self):
        assert is_mve("kxmvecrosscategory-r") is True

    def test_a_missing_ticker_is_not_mve(self):
        """Absent data must not be silently classified either way."""
        assert is_mve(None) is False
        assert is_mve("") is False


class TestSeriesSelection:
    def test_mve_series_are_excluded(self):
        """Their markets are not orderable: POSTing one returns 404."""
        chosen = select_series([
            series("KXMVECROSSCATEGORY-R"),
            series("KXHIGHNY"),
        ])

        assert [s["ticker"] for s in chosen] == ["KXHIGHNY"]

    @pytest.mark.parametrize("frequency", sorted(FAST_FREQUENCIES))
    def test_fast_series_are_excluded_by_default(self, frequency):
        """
        CLAUDE.md rejects these on the merits, not for convenience: the
        underlying trades faster elsewhere, so every fill is adversely selected.
        """
        chosen = select_series([series("KXBTC15M", frequency=frequency)])

        assert chosen == []

    def test_fast_series_can_be_opted_back_in(self):
        chosen = select_series([series("KXBTC15M", frequency="fifteen_min")],
                               include_fast=True)

        assert [s["ticker"] for s in chosen] == ["KXBTC15M"]

    def test_mve_stays_excluded_even_when_fast_is_allowed(self):
        """include_fast relaxes the strategy filter, never the tradeability one."""
        chosen = select_series([series("KXMVECROSSCATEGORY-R", frequency="hourly")],
                               include_fast=True)

        assert chosen == []

    @pytest.mark.parametrize("frequency", ["daily", "weekly", "monthly", "one_off"])
    def test_slow_series_are_kept(self, frequency):
        chosen = select_series([series("KXHIGHNY", frequency=frequency)])

        assert len(chosen) == 1

    def test_categories_filter_when_given(self):
        chosen = select_series(
            [series("KXWEATHER", category="Climate and Weather"),
             series("KXNBA", category="Sports")],
            categories={"Climate and Weather"},
        )

        assert [s["ticker"] for s in chosen] == ["KXWEATHER"]

    def test_no_category_filter_keeps_everything_tradeable(self):
        chosen = select_series([
            series("KXWEATHER", category="Climate and Weather"),
            series("KXNBA", category="Sports"),
        ])

        assert len(chosen) == 2

    def test_the_limit_is_respected(self):
        """One request per series, so the budget has to be honoured."""
        chosen = select_series([series(f"KX{i}") for i in range(100)], limit=5)

        assert len(chosen) == 5

    def test_the_limit_counts_only_series_that_survived_filtering(self):
        """
        A budget of 3 must yield 3 usable series, not 3 candidates of which
        two were discarded — otherwise a run of MVE entries starves the scan.
        """
        entries = [series("KXMVECROSSCATEGORY-R")] * 10 + [series(f"KX{i}") for i in range(10)]

        chosen = select_series(entries, limit=3)

        assert len(chosen) == 3
        assert not any(is_mve(s["ticker"]) for s in chosen)

    def test_an_empty_series_list_selects_nothing(self):
        assert select_series([]) == []

    def test_entries_without_a_ticker_are_skipped(self):
        chosen = select_series([{"category": "Economics", "frequency": "daily"},
                                series("KXHIGHNY")])

        assert [s["ticker"] for s in chosen] == ["KXHIGHNY"]

    def test_the_default_limit_is_a_bounded_request_budget(self):
        assert SERIES_LIMIT <= 100
