"""
test_reference_price.py
-----------------------
Tests for the core mechanic. Written before book.py exists; book.py must be
written to satisfy these, not the other way around.

The rule being encoded (CLAUDE.md, "The core mechanic"):

    Kalshi computes a Reference Price per snapshot by walking down from the
    best bid to the first level where cumulative resting size reaches 1/5 of
    Target Size. Orders at or better than the Reference Price earn full credit
    (1.0x); orders below it earn DiscountFactor ^ ticks_away.

The payoff is that resting AT the Reference Price earns the same reward as
resting at top of book while filling materially less often. Every test here
exists to stop an implementation from drifting back toward the top of book.

Book model: each side is its own book of bids in its own currency (a "no" bid
at 38 is a "yes" ask at 62), best price first, exactly as Kalshi's orderbook
endpoint returns them. So "walk down from the best bid" is descending price on
that side's own book, and the two sides are scored independently.

ASSUMPTIONS FLAGGED (per CLAUDE.md):
  - Target Size is NOT published per market. 1000 is an assumption, so
    target_size is a required argument here — no default — to keep the
    assumption visible at every call site.
  - REFERENCE_FRACTION = 1/5 and the "reaches" comparison is inclusive (>=).
    Both are inferred from the program description, not from a Kalshi-provided
    worked example. If rewards ever reconcile badly, suspect these first.
"""

from __future__ import annotations

import pytest

from src.book import REFERENCE_FRACTION, OrderBook

# Target Size is only ever an assumption; see module docstring.
TARGET = 1000
NEED = int(REFERENCE_FRACTION * TARGET)  # 200 contracts


def book(yes=(), no=()) -> OrderBook:
    """Hand-build a book. Levels are (price_cents, size)."""
    return OrderBook(yes=list(yes), no=list(no))


# ---------------------------------------------------------------------------
# NORMAL BOOK
# ---------------------------------------------------------------------------

class TestNormalBook:
    """A two-sided book with enough depth to reach the threshold on both sides."""

    def test_reference_price_is_where_cumulative_depth_reaches_the_threshold(self):
        # cumulative: 50, 130, 250 -> crosses 200 at 58c
        b = book(yes=[(60, 50), (59, 80), (58, 120), (57, 300)])

        ref = b.reference_price("yes", target_size=TARGET)

        assert ref == (58, 250)
        assert ref.price == 58
        assert ref.depth == 250

    def test_reference_price_is_not_the_top_of_book(self):
        """The entire edge. If this ever equals the best bid, the edge is gone."""
        b = book(yes=[(60, 50), (59, 80), (58, 120)])

        ref = b.reference_price("yes", target_size=TARGET)

        assert ref.price == 58
        assert ref.price < 60, "resting at top of book earns the same credit for more fills"

    def test_each_side_is_scored_independently(self):
        # yes: 60, 140, 260 -> crosses at 58.  no: 60, 150, 350 -> crosses at 36.
        b = book(
            yes=[(60, 60), (59, 80), (58, 120)],
            no=[(38, 60), (37, 90), (36, 200)],
        )

        assert b.reference_price("yes", target_size=TARGET) == (58, 260)
        assert b.reference_price("no", target_size=TARGET) == (36, 350)

    def test_threshold_met_at_the_very_first_level(self):
        """A book deep at the top: reference price IS the best bid. Legitimate."""
        b = book(yes=[(60, 500), (59, 100)])

        assert b.reference_price("yes", target_size=TARGET) == (60, 500)

    def test_levels_below_the_reference_price_do_not_move_it(self):
        deep = book(yes=[(60, 50), (59, 80), (58, 120), (57, 9999)])
        shallow = book(yes=[(60, 50), (59, 80), (58, 120)])

        assert (deep.reference_price("yes", target_size=TARGET)
                == shallow.reference_price("yes", target_size=TARGET))


# ---------------------------------------------------------------------------
# BOOK TOO THIN TO REACH 1/5 OF TARGET SIZE
# ---------------------------------------------------------------------------

class TestBookTooThin:
    """
    No level reaches the threshold. There is no Reference Price.

    This must return None rather than falling back to the deepest or the best
    level. A fallback to best bid would quote top of book, which is the exact
    failure this project is built to avoid. Such a book also cannot meet full
    Target Size, so the snapshot is excluded and the reward is zero regardless.
    """

    def test_returns_none_when_total_depth_is_below_the_threshold(self):
        b = book(yes=[(60, 40), (59, 50), (58, 30)])  # 120 total, need 200

        assert b.reference_price("yes", target_size=TARGET) is None

    def test_does_not_fall_back_to_the_worst_level(self):
        b = book(yes=[(60, 40), (59, 50), (58, 30)])

        assert b.reference_price("yes", target_size=TARGET) != (58, 120)

    def test_does_not_fall_back_to_the_best_level(self):
        b = book(yes=[(60, 40), (59, 50), (58, 30)])

        assert b.reference_price("yes", target_size=TARGET) != (60, 40)

    def test_one_contract_short_of_the_threshold_is_still_none(self):
        b = book(yes=[(60, 100), (59, 99)])  # 199, need 200

        assert b.reference_price("yes", target_size=TARGET) is None

    def test_thin_on_one_side_only_does_not_affect_the_other(self):
        b = book(yes=[(60, 40)], no=[(38, 60), (37, 90), (36, 200)])

        assert b.reference_price("yes", target_size=TARGET) is None
        assert b.reference_price("no", target_size=TARGET) == (36, 350)


# ---------------------------------------------------------------------------
# ONE-SIDED BOOK
# ---------------------------------------------------------------------------

class TestOneSidedBook:
    """One side has no resting orders at all. Must not crash or leak the other side."""

    def test_empty_no_side_returns_none(self):
        b = book(yes=[(60, 250)], no=[])

        assert b.reference_price("no", target_size=TARGET) is None

    def test_populated_side_still_computes(self):
        b = book(yes=[(60, 250)], no=[])

        assert b.reference_price("yes", target_size=TARGET) == (60, 250)

    def test_empty_yes_side_returns_none(self):
        b = book(yes=[], no=[(38, 250)])

        assert b.reference_price("yes", target_size=TARGET) is None
        assert b.reference_price("no", target_size=TARGET) == (38, 250)

    def test_sides_are_not_swapped(self):
        """Guards the classic `yes_levels if side == "yes" else no_levels` bug."""
        b = book(yes=[(60, 250)], no=[(38, 250)])

        assert b.reference_price("yes", target_size=TARGET).price == 60
        assert b.reference_price("no", target_size=TARGET).price == 38


# ---------------------------------------------------------------------------
# TIES AT A PRICE LEVEL
# ---------------------------------------------------------------------------

class TestTiesAtAPriceLevel:
    """
    Two things called "ties":

    1. Several entries at the same price (our order alongside someone else's).
       They are one level and must be aggregated before the walk. Kalshi's
       credit is level-granular — an order cannot be half at the Reference
       Price — so a walk that stops partway through a level is wrong.
    2. Cumulative depth landing exactly on the threshold. "Reaches" is
       inclusive, so the level that lands on it IS the Reference Price.
    """

    def test_duplicate_entries_at_one_price_are_aggregated(self):
        b = book(yes=[(60, 100), (60, 150), (59, 80)])

        assert b.reference_price("yes", target_size=TARGET) == (60, 250)

    def test_unsorted_input_is_normalized_before_walking(self):
        """
        The dangerous case: same level split and out of order. A naive walk
        double-counts 60c and reports a level that does not exist as stated.
        """
        b = book(yes=[(60, 100), (59, 80), (60, 150)])

        assert b.reference_price("yes", target_size=TARGET) == (60, 250)

    def test_exactly_reaching_the_threshold_qualifies_that_level(self):
        b = book(yes=[(60, 200), (59, 100)])  # cum hits exactly 200 at 60c

        assert b.reference_price("yes", target_size=TARGET) == (60, 200)

    def test_one_short_of_the_threshold_walks_to_the_next_level(self):
        b = book(yes=[(60, 199), (59, 100)])

        assert b.reference_price("yes", target_size=TARGET) == (59, 299)

    def test_zero_size_levels_are_ignored(self):
        """A 0-size level must not be reported as the Reference Price."""
        b = book(yes=[(60, 0), (59, 200)])

        assert b.reference_price("yes", target_size=TARGET) == (59, 200)

    def test_gaps_between_price_levels_are_fine(self):
        """Empty ticks between levels are normal on a thin market."""
        b = book(yes=[(60, 50), (55, 80), (41, 120)])

        assert b.reference_price("yes", target_size=TARGET) == (41, 250)


# ---------------------------------------------------------------------------
# TARGET SIZE AT BOTH BOUNDS
# ---------------------------------------------------------------------------

class TestTargetSizeBounds:
    """
    CLAUDE.md states Target Size is set per period, >100 and <20,000.

    Note the inequalities are strict, so 100 and 20,000 are themselves outside
    the stated range. These tests pin the arithmetic AT those values and treat
    them as computable rather than as errors — see the open question raised
    alongside this file about whether the implementation should validate.
    """

    def test_lower_bound_target_100(self):
        # need = 20; cumulative 5, 15, 25 -> crosses at 58c
        b = book(yes=[(60, 5), (59, 10), (58, 10)])

        assert b.reference_price("yes", target_size=100) == (58, 25)

    def test_upper_bound_target_20000(self):
        # need = 4000; cumulative 1000, 2500, 4500 -> crosses at 58c
        b = book(yes=[(60, 1000), (59, 1500), (58, 2000)])

        assert b.reference_price("yes", target_size=20_000) == (58, 4500)

    def test_a_book_deep_enough_at_100_is_far_too_thin_at_20000(self):
        """Same book, opposite verdicts. Getting Target Size wrong is expensive."""
        b = book(yes=[(60, 5), (59, 10), (58, 10)])

        assert b.reference_price("yes", target_size=100) == (58, 25)
        assert b.reference_price("yes", target_size=20_000) is None

    def test_larger_target_never_yields_a_better_reference_price(self):
        """Monotonicity: raising Target Size can only push the price deeper."""
        b = book(yes=[(60, 50), (59, 80), (58, 120), (57, 300)])

        small = b.reference_price("yes", target_size=500)
        large = b.reference_price("yes", target_size=2000)

        assert small.price >= large.price


# ---------------------------------------------------------------------------
# EMPTY BOOK
# ---------------------------------------------------------------------------

class TestEmptyBook:
    """No resting orders anywhere. Must return None, not raise, not return 0."""

    def test_both_sides_return_none(self):
        b = book(yes=[], no=[])

        assert b.reference_price("yes", target_size=TARGET) is None
        assert b.reference_price("no", target_size=TARGET) is None

    def test_a_book_of_only_zero_size_levels_is_empty(self):
        b = book(yes=[(60, 0), (59, 0)], no=[(38, 0)])

        assert b.reference_price("yes", target_size=TARGET) is None
        assert b.reference_price("no", target_size=TARGET) is None

    @pytest.mark.parametrize("target_size", [100, TARGET, 20_000])
    def test_empty_book_is_none_at_every_target_size(self, target_size):
        b = book(yes=[], no=[])

        assert b.reference_price("yes", target_size=target_size) is None


# ---------------------------------------------------------------------------
# SIDE ARGUMENT
# ---------------------------------------------------------------------------

class TestSideArgument:
    """
    A bad side argument must raise, never silently resolve to a side. Quoting
    the wrong side of a market is a directional position taken by accident.
    """

    @pytest.mark.parametrize("bad_side", ["YES", "yes ", "bid", "", None])
    def test_unknown_side_raises(self, bad_side):
        b = book(yes=[(60, 250)], no=[(38, 250)])

        with pytest.raises(ValueError):
            b.reference_price(bad_side, target_size=TARGET)
