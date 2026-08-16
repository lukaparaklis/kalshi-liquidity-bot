"""
test_fill_analysis.py
---------------------
Replaying paper logs into a fill-rate estimate.

CLAUDE.md calls fill rate the metric that decides everything, which makes the
crossing test in this module the highest-consequence arithmetic in the project.
Get it backwards and a market that runs us over looks quiet — the exact
conclusion that would justify going live on a losing strategy.

The rule under test: a resting bid of P on side S is crossed when the OTHER
side's best bid reaches 100 - P, because a no bid at Q is a yes ask at 100-Q.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.monitor import analyse, crossed, load_records

START = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)


def snapshot(ticker="MKT-A", *, seconds=0, yes_bid=60, no_bid=38,
             yes_quote=58, no_quote=36, size=100, two_sided=True,
             quotable=True) -> dict:
    quotes = []
    if quotable:
        quotes = [
            {"side": "yes", "price_cents": yes_quote, "size": size,
             "reference_price": yes_quote, "ticks_below_top": yes_bid - yes_quote},
            {"side": "no", "price_cents": no_quote, "size": size,
             "reference_price": no_quote, "ticks_below_top": no_bid - no_quote},
        ]
    return {
        "ts": (START + timedelta(seconds=seconds)).isoformat(),
        "ticker": ticker,
        "size": size,
        "quotes": quotes,
        "two_sided_ok": two_sided,
        "best": {
            "yes_bid": yes_bid,
            "no_bid": no_bid,
            "spread": 100 - yes_bid - no_bid,
            "mid": (yes_bid + (100 - no_bid)) / 2,
        },
        "book": {"yes": [[yes_bid, 500]], "no": [[no_bid, 500]]},
    }


# ---------------------------------------------------------------------------
# THE CROSSING RULE
# ---------------------------------------------------------------------------

class TestCrossingDetection:
    def test_a_yes_bid_is_crossed_when_the_no_side_bids_high_enough(self):
        """
        Resting yes at 58 means we are crossed once the implied yes ask is 58
        or lower, i.e. the no bid reaches 42.
        """
        assert crossed("yes", 58, {"no_bid": 42}) is True

    def test_a_yes_bid_is_untouched_just_below_the_threshold(self):
        assert crossed("yes", 58, {"no_bid": 41}) is False

    def test_a_no_bid_is_crossed_when_the_yes_side_bids_high_enough(self):
        """Resting no at 36 is crossed once the yes bid reaches 64."""
        assert crossed("no", 36, {"yes_bid": 64}) is True

    def test_a_no_bid_is_untouched_just_below_the_threshold(self):
        assert crossed("no", 36, {"yes_bid": 63}) is False

    def test_the_two_sides_use_opposite_books(self):
        """
        Guards the copy-paste error that would make this whole module lie:
        reading our own side's bid instead of the other side's.
        """
        best = {"yes_bid": 64, "no_bid": 30}

        assert crossed("no", 36, best) is True
        assert crossed("yes", 58, best) is False

    def test_a_missing_other_side_is_not_a_fill(self):
        """An empty book is no evidence of trading; absence is not a crossing."""
        assert crossed("yes", 58, {"no_bid": None}) is False
        assert crossed("yes", 58, {}) is False

    def test_resting_deeper_is_crossed_less_easily(self):
        """The reason to rest at the Reference Price rather than the top."""
        best = {"no_bid": 41}

        assert crossed("yes", 60, best) is True     # top of book: hit
        assert crossed("yes", 58, best) is False    # reference price: untouched


# ---------------------------------------------------------------------------
# REPLAY
# ---------------------------------------------------------------------------

class TestReplay:
    def test_a_quiet_market_records_no_fills(self):
        records = [snapshot(seconds=0), snapshot(seconds=10), snapshot(seconds=20)]

        stats = analyse(records)["MKT-A"]

        assert stats.snapshots == 3
        assert stats.total_fills == 0
        assert "quiet" in stats.verdict

    def test_a_market_trading_through_us_records_a_fill(self):
        records = [
            snapshot(seconds=0),
            snapshot(seconds=10, no_bid=45),   # implied yes ask 55 < our 58
        ]

        stats = analyse(records)["MKT-A"]

        assert stats.fills["yes"] == 1
        assert stats.fills["no"] == 0

    def test_fills_are_attributed_to_the_right_side(self):
        records = [
            snapshot(seconds=0),
            snapshot(seconds=10, yes_bid=70),  # implied no ask 30 < our 36
        ]

        stats = analyse(records)["MKT-A"]

        assert stats.fills["no"] == 1
        assert stats.fills["yes"] == 0

    def test_the_last_snapshot_has_no_successor_to_judge_it(self):
        """Nothing to compare against, so no fill is inferred from it."""
        records = [snapshot(seconds=0, no_bid=45)]

        assert analyse(records)["MKT-A"].total_fills == 0

    def test_snapshots_where_we_would_not_quote_are_counted_separately(self):
        records = [
            snapshot(seconds=0, quotable=False, two_sided=False),
            snapshot(seconds=10),
            snapshot(seconds=20),
        ]

        stats = analyse(records)["MKT-A"]

        assert stats.snapshots == 3
        assert stats.quotable_snapshots == 2
        assert stats.quotable_fraction == pytest.approx(2 / 3)

    def test_earning_fraction_tracks_qualifying_snapshots(self):
        """No qualifying snapshot, no reward — this is the rent side."""
        records = [snapshot(seconds=0, two_sided=False),
                   snapshot(seconds=10, two_sided=True)]

        assert analyse(records)["MKT-A"].earning_fraction == pytest.approx(0.5)

    def test_markets_are_analysed_independently(self):
        records = [
            snapshot("MKT-A", seconds=0), snapshot("MKT-A", seconds=10, no_bid=45),
            snapshot("MKT-B", seconds=0), snapshot("MKT-B", seconds=10),
        ]

        stats = analyse(records)

        assert stats["MKT-A"].total_fills == 1
        assert stats["MKT-B"].total_fills == 0

    def test_out_of_order_records_are_sorted_before_comparison(self):
        """Logs can interleave; ordering by timestamp is what makes pairs mean anything."""
        records = [snapshot(seconds=20), snapshot(seconds=0), snapshot(seconds=10)]

        stats = analyse(records)["MKT-A"]

        assert stats.hours == pytest.approx(20 / 3600)


class TestFillRate:
    def test_fills_per_hour_uses_the_observed_window(self):
        records = [snapshot(seconds=s, no_bid=45 if s else 38)
                   for s in (0, 1800)]   # half an hour apart, one crossing

        stats = analyse(records)["MKT-A"]

        assert stats.hours == pytest.approx(0.5)
        assert stats.fills_per_hour == pytest.approx(2.0)

    def test_the_rate_is_normalised_to_a_hundred_contracts(self):
        """So markets quoted at different sizes can be compared."""
        records = [snapshot(seconds=s, size=200, no_bid=45 if s else 38)
                   for s in (0, 3600)]

        stats = analyse(records)["MKT-A"]

        assert stats.fills_per_hour == pytest.approx(1.0)
        assert stats.fills_per_hour_per_100 == pytest.approx(0.5)

    def test_a_single_snapshot_reports_no_rate_instead_of_dividing_by_zero(self):
        stats = analyse([snapshot(seconds=0)])["MKT-A"]

        assert stats.hours == 0.0
        assert stats.fills_per_hour == 0.0

    def test_staying_crossed_is_one_fill_not_many(self):
        """
        An order fills once and is gone. A market that sits beyond our price
        for twenty polls did not fill us twenty times, and counting it that way
        would turn one bad trade into an alarm.
        """
        records = [snapshot(seconds=s * 10, no_bid=45) for s in range(20)]

        assert analyse(records)["MKT-A"].fills["yes"] == 1

    def test_crossing_back_and_forth_fills_each_time(self):
        """Each fresh move through our price is a new fill: we requoted between."""
        records = [snapshot(seconds=s * 10, no_bid=45 if s % 2 else 38)
                   for s in range(20)]

        assert analyse(records)["MKT-A"].fills["yes"] == 10

    def test_a_market_filling_constantly_is_called_out(self):
        """CLAUDE.md: quotes filling constantly means stop."""
        records = [snapshot(seconds=s * 10, no_bid=45 if s % 2 else 38)
                   for s in range(20)]

        assert analyse(records)["MKT-A"].verdict == "RUN OVER"

    def test_a_market_that_never_qualifies_is_called_out(self):
        records = [snapshot(seconds=s * 10, two_sided=False) for s in range(5)]

        assert analyse(records)["MKT-A"].verdict == "rarely qualifies"


class TestAdverseSelection:
    def test_a_mid_that_keeps_falling_after_a_yes_fill_is_adverse(self):
        """We bought yes; the market kept going down. That is being run over."""
        records = [
            snapshot(seconds=0),
            snapshot(seconds=10, no_bid=45),            # crossed: we bought yes
            snapshot(seconds=25, yes_bid=50, no_bid=45),  # mid moved down
        ]

        stats = analyse(records)["MKT-A"]

        assert stats.fills["yes"] == 1
        assert stats.mean_adverse_cents > 0

    def test_a_mid_that_recovers_after_a_yes_fill_is_favourable(self):
        records = [
            snapshot(seconds=0),
            snapshot(seconds=10, no_bid=45),
            snapshot(seconds=25, yes_bid=70, no_bid=25),  # mid moved up
        ]

        assert analyse(records)["MKT-A"].mean_adverse_cents < 0

    def test_the_sign_flips_for_a_no_fill(self):
        """Buying no is short yes, so a rising mid is what hurts."""
        records = [
            snapshot(seconds=0),
            snapshot(seconds=10, yes_bid=70),             # crossed: we bought no
            snapshot(seconds=25, yes_bid=80, no_bid=15),  # mid moved up
        ]

        stats = analyse(records)["MKT-A"]

        assert stats.fills["no"] == 1
        assert stats.mean_adverse_cents > 0

    def test_no_measurement_without_a_later_snapshot(self):
        """The horizon needs a snapshot on the far side of it."""
        records = [snapshot(seconds=0), snapshot(seconds=10, no_bid=45)]

        assert analyse(records)["MKT-A"].mean_adverse_cents is None


class TestLogLoading:
    def test_it_reads_jsonl(self, tmp_path):
        path = tmp_path / "p.jsonl"
        path.write_text("\n".join(json.dumps(snapshot(seconds=s)) for s in (0, 10)))

        assert len(load_records(path)) == 2

    def test_malformed_lines_are_skipped_not_fatal(self, tmp_path):
        """A truncated final line from a killed process must not lose the run."""
        path = tmp_path / "p.jsonl"
        path.write_text(json.dumps(snapshot()) + "\n{ broken\n")

        assert len(load_records(path)) == 1

    def test_blank_lines_are_ignored(self, tmp_path):
        path = tmp_path / "p.jsonl"
        path.write_text("\n" + json.dumps(snapshot()) + "\n\n")

        assert len(load_records(path)) == 1
