"""
test_economics.py
-----------------
Fee math is tested against reality; reward estimation is tested for structure.

That split is deliberate. The fee formula is knowable — Kalshi publishes it and
a real fill confirms it — so these tests assert exact values. The reward and
fill-cost estimates rest on unpublished parameters, so asserting exact numbers
would only pin today's guess in place. Those are tested for the properties that
must hold whatever the parameters turn out to be: monotonicity, bounds, and
behaviour at the edges.

THE ANCHOR
    On 2026-08-15 a demo order filled 1 contract at 29c and was charged
    taker_fees_dollars "0.014500". 0.07 x 0.29 x 0.71 = 0.0144130, rounded up
    to a centicent = 0.0145. That single observation is what makes the fee
    tests real rather than a restatement of the same assumption the reference
    implementation made.
"""

from __future__ import annotations

import pytest

from src.book import OrderBook
from src.economics import (
    MAKER_FEE_RATE,
    TAKER_FEE_RATE,
    TARGET_SIZE_ASSUMPTION,
    VALID_FRACTION_MARGINAL,
    assumptions,
    capital_required_dollars,
    credit_multiplier,
    estimate_fill_cost_dollars,
    estimate_reward,
    mid_cents,
    qualifying_depth,
    spread_cents,
    trade_fee_cents,
    trade_fee_dollars,
    two_sided_ok,
)

TARGET = TARGET_SIZE_ASSUMPTION


def book(yes=(), no=()) -> OrderBook:
    return OrderBook(yes=list(yes), no=list(no))


# ---------------------------------------------------------------------------
# FEES — verified against a real fill
# ---------------------------------------------------------------------------

class TestFeeAgainstRealFill:
    """The observed charge, and the arithmetic that has to reproduce it."""

    def test_matches_the_observed_taker_fee(self):
        """1 contract at 29c was charged exactly $0.0145 on demo."""
        assert trade_fee_dollars(29, 1, maker=False) == 0.0145

    def test_the_unrounded_value_is_just_under_the_charge(self):
        """0.0144130 rounded up to a centicent is 0.0145 — not 0.0144."""
        raw = 0.07 * 0.29 * 0.71

        assert raw < 0.0145
        assert raw > 0.0144
        assert trade_fee_dollars(29, 1, maker=False) == 0.0145

    def test_it_is_not_rounded_to_a_whole_cent(self):
        """
        CLAUDE.md says "rounded up to the next cent per order". It is not:
        the real charge was $0.0145, and whole-cent rounding would have been
        $0.02 — 38% higher, on a bot whose entire margin is fee-sized.
        """
        assert trade_fee_dollars(29, 1, maker=False) != 0.02
        assert trade_fee_dollars(29, 1, maker=False) < 0.02


class TestFeeRounding:
    def test_a_fee_already_on_the_grid_is_not_rounded_up(self):
        """
        Taker at 50c is 0.07 x 0.25 = 0.0175 exactly, an exact centicent
        multiple. A float ceiling computes 0.017500000000000002 and charges
        0.0176; Decimal must not.
        """
        assert trade_fee_dollars(50, 1, maker=False) == 0.0175

    def test_rounding_is_up_not_nearest(self):
        """Taker at 1c: 0.07 x 0.01 x 0.99 = 0.000693 -> 0.0007, not 0.0006."""
        assert trade_fee_dollars(1, 1, maker=False) == 0.0007

    def test_maker_at_fifty_cents_rounds_up_off_the_grid(self):
        """0.0175 x 0.25 = 0.004375, which is 43.75 centicents -> 0.0044."""
        assert trade_fee_dollars(50, 1, maker=True) == 0.0044


class TestFeeShape:
    def test_the_maker_rate_is_a_quarter_of_the_taker_rate(self):
        assert MAKER_FEE_RATE * 4 == TAKER_FEE_RATE

    def test_maker_is_cheaper_than_taker_at_every_price(self):
        for price in range(1, 100):
            assert trade_fee_dollars(price, 100, maker=True) < \
                   trade_fee_dollars(price, 100, maker=False)

    def test_the_fee_peaks_at_fifty_cents(self):
        """P(1-P) is maximal at 0.5, which is why the bot works the wings."""
        fees = {p: trade_fee_dollars(p, 1000, maker=True) for p in range(1, 100)}

        assert max(fees, key=fees.get) == 50

    def test_the_wings_are_cheaper_but_only_the_far_wing_is_a_fifth(self):
        """
        CLAUDE.md says "working 85-95c costs ~1/5 what 50c costs". That holds
        only at the top of the range. P(1-P) relative to 0.25:

            95c -> 0.0475 / 0.25 = 0.19   about a fifth, as claimed
            90c -> 0.0900 / 0.25 = 0.36   nearer a third
            85c -> 0.1275 / 0.25 = 0.51   about a half

        So the fee saving at 85c is roughly half, not a fifth. Worth knowing
        before picking a working range on the strength of that sentence.
        """
        middle = trade_fee_dollars(50, 10_000, maker=True)

        assert trade_fee_dollars(95, 10_000, maker=True) == pytest.approx(
            0.19 * middle, rel=0.02)
        assert trade_fee_dollars(90, 10_000, maker=True) == pytest.approx(
            0.36 * middle, rel=0.02)
        assert trade_fee_dollars(85, 10_000, maker=True) == pytest.approx(
            0.51 * middle, rel=0.02)

    def test_fees_fall_monotonically_from_the_middle_to_the_wing(self):
        fees = [trade_fee_dollars(p, 10_000, maker=True) for p in range(50, 100)]

        assert fees == sorted(fees, reverse=True)

    def test_it_is_symmetric_about_fifty_cents(self):
        for price in range(1, 50):
            assert trade_fee_dollars(price, 500, maker=True) == \
                   trade_fee_dollars(100 - price, 500, maker=True)

    def test_it_scales_with_contract_count(self):
        one = trade_fee_dollars(40, 1, maker=True)
        many = trade_fee_dollars(40, 1000, maker=True)

        assert many > one
        assert many == pytest.approx(1000 * 0.0175 * 0.4 * 0.6, abs=0.0001)

    def test_the_multiplier_scales_the_fee(self):
        """Some series carry M > 1; CLAUDE.md flags crypto in particular."""
        base = trade_fee_dollars(40, 100, maker=True)
        doubled = trade_fee_dollars(40, 100, maker=True, multiplier=2.0)

        assert doubled == pytest.approx(2 * base, abs=0.0001)

    @pytest.mark.parametrize("price_cents", [0, 100])
    def test_a_certain_outcome_is_free(self, price_cents):
        """P(1-P) is zero at both ends. No expected earnings, no fee."""
        assert trade_fee_dollars(price_cents, 1000, maker=True) == 0.0

    @pytest.mark.parametrize("contracts", [0, -5])
    def test_no_contracts_no_fee(self, contracts):
        assert trade_fee_dollars(50, contracts, maker=True) == 0.0

    def test_cents_helper_agrees_with_dollars(self):
        assert trade_fee_cents(29, 1, maker=False) == pytest.approx(1.45)


# ---------------------------------------------------------------------------
# BOOK-DERIVED QUANTITIES
# ---------------------------------------------------------------------------

class TestQualifyingDepth:
    def test_it_sums_the_side(self):
        b = book(yes=[(60, 100), (59, 150)])

        assert qualifying_depth(b, "yes", TARGET) == 250

    def test_it_caps_at_target_size(self):
        """The program asks whether the market reaches Target Size, not by how much."""
        b = book(yes=[(60, 50_000)])

        assert qualifying_depth(b, "yes", TARGET) == TARGET

    def test_an_empty_side_qualifies_nothing(self):
        assert qualifying_depth(book(), "yes", TARGET) == 0


class TestTwoSidedRequirement:
    def test_both_sides_at_target_qualifies(self):
        b = book(yes=[(60, TARGET)], no=[(38, TARGET)])

        assert two_sided_ok(b, TARGET) is True

    def test_one_side_short_excludes_the_snapshot(self):
        """Market-wide: everyone earns nothing from an excluded snapshot."""
        b = book(yes=[(60, TARGET)], no=[(38, TARGET - 1)])

        assert two_sided_ok(b, TARGET) is False

    def test_a_one_sided_book_is_excluded(self):
        assert two_sided_ok(book(yes=[(60, 5_000)]), TARGET) is False


class TestSpreadAndMid:
    def test_spread_is_implied_across_the_two_sides(self):
        """yes bid 60, no bid 38 -> yes ask 62 -> spread 2."""
        b = book(yes=[(60, 100)], no=[(38, 100)])

        assert spread_cents(b) == 2

    def test_mid_sits_between_the_two(self):
        b = book(yes=[(60, 100)], no=[(38, 100)])

        assert mid_cents(b) == 61.0

    def test_a_crossed_book_does_not_report_a_negative_spread(self):
        b = book(yes=[(60, 100)], no=[(45, 100)])

        assert spread_cents(b) == 0

    def test_a_one_sided_book_has_neither(self):
        b = book(yes=[(60, 100)])

        assert spread_cents(b) is None
        assert mid_cents(b) is None


class TestCreditMultiplier:
    def test_at_the_reference_price_credit_is_full(self):
        """The entire edge: same credit as top of book, far fewer fills."""
        assert credit_multiplier(0) == 1.0

    def test_better_than_the_reference_price_is_still_only_full(self):
        assert credit_multiplier(-3) == 1.0

    def test_credit_decays_with_distance(self):
        assert credit_multiplier(1) > credit_multiplier(2) > credit_multiplier(3)

    def test_it_decays_geometrically(self):
        assert credit_multiplier(2, discount_factor=0.5) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# ESTIMATES — structure, not exact values
# ---------------------------------------------------------------------------

class TestRewardEstimate:
    def test_share_is_a_fraction(self):
        b = book(yes=[(60, 500)], no=[(38, 500)])

        estimate = estimate_reward(b, my_size=100, target_size=TARGET)

        assert 0.0 < estimate.share < 1.0

    def test_a_bigger_quote_earns_a_bigger_share(self):
        b = book(yes=[(60, 500)], no=[(38, 500)])

        small = estimate_reward(b, 50, TARGET).share
        large = estimate_reward(b, 500, TARGET).share

        assert large > small

    def test_share_shrinks_as_competitors_arrive(self):
        """Rewards are a share of a fixed pool, so crowding is the main risk."""
        quiet = book(yes=[(60, 100)], no=[(38, 100)])
        crowded = book(yes=[(60, 10_000)], no=[(38, 10_000)])

        assert estimate_reward(quiet, 100, TARGET).share > \
               estimate_reward(crowded, 100, TARGET).share

    def test_an_excluded_market_is_discounted(self):
        """One side below Target Size means most snapshots do not count."""
        two_sided = book(yes=[(60, TARGET)], no=[(38, TARGET)])
        marginal = book(yes=[(60, TARGET)], no=[(38, TARGET - 50)])

        assert estimate_reward(two_sided, 100, TARGET).valid_fraction > \
               estimate_reward(marginal, 100, TARGET).valid_fraction

    def test_an_empty_market_earns_nothing_despite_a_total_share(self):
        """
        The trap this diverges from the reference to avoid. On an empty book
        our share is 100% — we would be the only liquidity — but 100 contracts
        cannot lift a market to a Target Size of 1000, so no snapshot ever
        qualifies. Owning all of nothing must not rank above owning some of
        something.
        """
        empty = book()

        estimate = estimate_reward(empty, 100, TARGET, pool_dollars_per_day=500)

        assert estimate.share == 1.0
        assert estimate.valid_fraction == 0.0
        assert estimate.dollars_per_day == 0.0

    def test_a_market_our_size_can_lift_still_counts(self):
        """Being the marginal contributor is worth something, just less."""
        nearly = book(yes=[(60, TARGET - 50)], no=[(38, TARGET - 50)])

        estimate = estimate_reward(nearly, 100, TARGET)

        assert estimate.valid_fraction == VALID_FRACTION_MARGINAL

    def test_a_market_too_thin_for_our_size_to_lift_earns_nothing(self):
        thin = book(yes=[(60, 10)], no=[(38, 10)])

        assert estimate_reward(thin, 100, TARGET).valid_fraction == 0.0

    def test_reachability_depends_on_our_size(self):
        """A bigger quote can make an otherwise dead market qualify."""
        b = book(yes=[(60, 600)], no=[(38, 600)])

        assert estimate_reward(b, 100, TARGET).valid_fraction == 0.0
        assert estimate_reward(b, 400, TARGET).valid_fraction == VALID_FRACTION_MARGINAL

    def test_dollars_are_none_without_a_known_pool(self):
        """The pool is not in the API; inventing one would be a fake number."""
        b = book(yes=[(60, 500)], no=[(38, 500)])

        assert estimate_reward(b, 100, TARGET).dollars_per_day is None

    def test_dollars_scale_with_the_pool(self):
        b = book(yes=[(60, 500)], no=[(38, 500)])

        small = estimate_reward(b, 100, TARGET, pool_dollars_per_day=10).dollars_per_day
        large = estimate_reward(b, 100, TARGET, pool_dollars_per_day=100).dollars_per_day

        assert large == pytest.approx(10 * small)

    def test_target_size_changes_the_answer(self):
        """
        Flagged because it is the assumption everything rests on: the same book
        and the same quote produce different rewards under different guesses.
        """
        b = book(yes=[(60, 500)], no=[(38, 500)])

        assert estimate_reward(b, 100, 100).valid_fraction != \
               estimate_reward(b, 100, 20_000).valid_fraction


class TestFillCost:
    def test_a_busier_market_costs_more(self):
        b = book(yes=[(60, 500)], no=[(38, 500)])

        quiet = estimate_fill_cost_dollars(b, 100, hours=24, daily_volume=100)
        busy = estimate_fill_cost_dollars(b, 100, hours=24, daily_volume=100_000)

        assert busy > quiet

    def test_resting_longer_costs_more(self):
        b = book(yes=[(60, 500)], no=[(38, 500)])

        assert estimate_fill_cost_dollars(b, 100, 24, 5_000) > \
               estimate_fill_cost_dollars(b, 100, 1, 5_000)

    def test_cost_cannot_exceed_our_whole_quote_being_filled(self):
        """Turnover is capped: we cannot be filled for more than we rested."""
        b = book(yes=[(60, 10)], no=[(38, 10)])

        cost = estimate_fill_cost_dollars(b, 100, hours=1000, daily_volume=10_000_000)
        worst = 100 * (100 + 1) / 100.0

        assert cost < worst

    def test_an_empty_book_is_infinite_cost(self):
        """Unquotable rather than free — nothing to rest against."""
        assert estimate_fill_cost_dollars(book(), 100, 24, 1_000) == float("inf")

    def test_a_one_sided_book_is_infinite_cost(self):
        b = book(yes=[(60, 500)])

        assert estimate_fill_cost_dollars(b, 100, 24, 1_000) == float("inf")


class TestCapitalRequired:
    def test_both_sides_are_funded(self):
        """Resting 100 on each side of a 60/38 book ties up $98."""
        b = book(yes=[(60, 500)], no=[(38, 500)])

        assert capital_required_dollars(b, 100) == pytest.approx(98.0)

    def test_an_empty_book_needs_nothing(self):
        assert capital_required_dollars(book(), 100) == 0.0


class TestAssumptionsAreSurfaced:
    def test_target_size_appears_with_its_value(self):
        text = " ".join(assumptions(TARGET))

        assert str(TARGET) in text
        assert "Target Size" in text

    def test_a_custom_target_size_is_the_one_reported(self):
        """Output must never claim a different number than the one used."""
        assert "777" in " ".join(assumptions(777))

    def test_every_unverified_input_is_named(self):
        text = " ".join(assumptions(TARGET)).lower()

        for topic in ("target size", "discount factor", "snapshot", "pool", "fill"):
            assert topic in text
