"""
test_pool_allocation.py
-----------------------
Splitting a reward pool across the markets it actually covers.

The incentives page lists one dollar figure per EVENT row, and an event holds
every strike within it. Treating that figure as per-market multiplies the
implied programme by the number of strikes: across five NVIDIA compute chip
families that produced $171,045/day, which is not a plausible incentive budget.
Events here hold 1 to 33 markets, averaging 10.5, so the error is not a
constant factor that cancels out of a ranking — it reweights it.
"""

from __future__ import annotations

import json

import pytest

from src.scanner import count_markets_per_event, load_pools, match_pool, pool_per_market

POOLS = {"KXB200": 195.0, "KXH200": 165.0}


def market(ticker: str, event: str) -> dict:
    return {"ticker": ticker, "event_ticker": event}


class TestMatchPool:
    def test_an_exact_event_ticker_matches(self):
        pools = {"KXB200MS-27MAR": 50.0, "KXB200": 195.0}

        assert match_pool("KXB200MS-27MAR", pools) == 50.0

    def test_an_exact_entry_beats_a_prefix(self):
        """Pinning one event from the incentives page must override the family."""
        pools = {"KXB200MS-27MAR": 50.0, "KXB200": 195.0}

        assert match_pool("KXB200MS-27MAR", pools) == 50.0
        assert match_pool("KXB200MS-27MAY", pools) == 195.0

    def test_the_longest_prefix_wins(self):
        pools = {"KX": 1.0, "KXB200": 195.0, "KXB200MS": 20.0}

        assert match_pool("KXB200MS-27MAR", pools) == 20.0

    def test_an_unknown_event_has_no_pool(self):
        """None, not zero: an unknown pool is unknown, not an absence of reward."""
        assert match_pool("KXNOTHING-26AUG", POOLS) is None

    def test_an_empty_event_ticker_has_no_pool(self):
        assert match_pool("", POOLS) is None
        assert match_pool(None, POOLS) is None


class TestMarketsPerEvent:
    def test_it_counts_markets_within_each_event(self):
        markets = [market("A-1", "A"), market("A-2", "A"), market("B-1", "B")]

        assert count_markets_per_event(markets) == {"A": 2, "B": 1}

    def test_markets_without_an_event_are_ignored(self):
        assert count_markets_per_event([{"ticker": "X"}]) == {}


class TestPoolPerMarket:
    def test_the_pool_is_split_across_the_event(self):
        """$195 over 3 strikes is $65 each, not $195 each."""
        share, count = pool_per_market(
            market("KXB200MS-27MAR-3.00", "KXB200MS-27MAR"),
            POOLS, {"KXB200MS-27MAR": 3},
        )

        assert share == pytest.approx(65.0)
        assert count == 3

    def test_a_bigger_event_dilutes_each_market(self):
        """
        The reason this is not a constant factor: events here run from 1 to 33
        markets, so per-event splitting reorders the ranking rather than just
        rescaling it.
        """
        small, _ = pool_per_market(market("m", "KXB200-A"), POOLS, {"KXB200-A": 2})
        large, _ = pool_per_market(market("m", "KXB200-B"), POOLS, {"KXB200-B": 33})

        assert small > large
        assert small == pytest.approx(97.5)
        assert large == pytest.approx(195 / 33)

    def test_a_sole_market_takes_the_whole_pool(self):
        share, count = pool_per_market(market("m", "KXB200-A"), POOLS, {"KXB200-A": 1})

        assert share == pytest.approx(195.0)
        assert count == 1

    def test_an_unknown_event_yields_no_pool(self):
        share, count = pool_per_market(market("m", "KXOTHER-A"), POOLS, {"KXOTHER-A": 5})

        assert share is None
        assert count == 5

    def test_a_missing_event_count_does_not_divide_by_zero(self):
        share, count = pool_per_market(market("m", "KXB200-A"), POOLS, {})

        assert share == pytest.approx(195.0)
        assert count == 1

    def test_the_lookup_uses_the_event_not_the_market_ticker(self):
        """
        A market ticker carries the strike (KXB200MS-27MAR-3.00); the pool is
        listed against the event (KXB200MS-27MAR). Matching on the market
        ticker happens to work with prefixes and would break the moment an
        exact event entry is pinned.
        """
        pools = {"KXB200MS-27MAR": 60.0}

        share, _ = pool_per_market(
            market("KXB200MS-27MAR-3.00", "KXB200MS-27MAR"), pools,
            {"KXB200MS-27MAR": 2},
        )

        assert share == pytest.approx(30.0)


class TestLoadPools:
    def test_underscore_keys_are_comments(self, tmp_path, monkeypatch):
        """The file is hand-maintained from a web page and needs annotating."""
        path = tmp_path / "pools.json"
        path.write_text(json.dumps({"_README": "notes", "KXB200": 195.0}))
        monkeypatch.setattr("src.scanner.POOLS_FILE", path)

        assert load_pools() == {"KXB200": 195.0}

    def test_a_missing_file_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.scanner.POOLS_FILE", tmp_path / "absent.json")

        assert load_pools() == {}

    def test_a_malformed_file_degrades_to_no_pools(self, tmp_path, monkeypatch):
        """Ranking without pools is useful; crashing on a typo is not."""
        path = tmp_path / "pools.json"
        path.write_text("{ not json")
        monkeypatch.setattr("src.scanner.POOLS_FILE", path)

        assert load_pools() == {}
