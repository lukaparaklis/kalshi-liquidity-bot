"""
test_order_encoding.py
----------------------
Pins the yes/no -> bid/ask translation used by the v2 order endpoint.

v2 replaced "buy yes / buy no" with one book quoted in yes terms plus a
direction, so every no-side quote is now expressed as an ask at the inverted
price. If that inversion were backwards, a 38c no quote would reach the
exchange as 62c and nothing else in the system would notice.

VERIFIED against demo on 2026-08-15 rather than inferred. Posting
side=ask price="0.9900" and reading it back gave:

    {"book_side": "ask", "outcome_side": "no",
     "yes_price_dollars": "0.9900", "no_price_dollars": "0.0100"}

— an ask at 0.99 is buying NO at 1c, which is what encode_order("no", 1)
produces. The bid leg matched too: side=bid price="0.0100" came back as
outcome_side "yes" at yes_price_dollars "0.0100".

These tests now pin verified behaviour. If they ever fail, the API changed.
"""

from __future__ import annotations

import pytest

from src.auth import encode_order


class TestYesSide:
    """Buying yes is a bid at the yes price. Confirmed by Kalshi's quick-start."""

    def test_matches_the_documented_quick_start_example(self):
        """side=bid price="0.0100" is documented as "buy 1 YES contract for 1c"."""
        assert encode_order("yes", 1) == ("bid", "0.0100")

    def test_a_mid_book_yes_quote(self):
        assert encode_order("yes", 60) == ("bid", "0.6000")

    def test_the_top_of_the_range(self):
        assert encode_order("yes", 99) == ("bid", "0.9900")


class TestNoSide:
    """
    Buying no is an ask at the inverted price: buy no @ 38c == sell yes @ 62c.

    UNVERIFIED — see the module docstring. These tests state the assumption so
    it is visible and cheap to flip, not because it has been confirmed.
    """

    def test_a_no_quote_inverts_to_an_ask(self):
        assert encode_order("no", 38) == ("ask", "0.6200")

    def test_a_cheap_no_quote_is_an_expensive_ask(self):
        assert encode_order("no", 1) == ("ask", "0.9900")

    def test_an_expensive_no_quote_is_a_cheap_ask(self):
        assert encode_order("no", 99) == ("ask", "0.0100")

    @pytest.mark.parametrize("price_cents", [1, 5, 25, 38, 50, 62, 75, 99])
    def test_the_two_sides_always_sum_to_a_dollar(self, price_cents):
        """The invariant the inversion rests on."""
        _, yes_price = encode_order("yes", price_cents)
        _, no_price = encode_order("no", price_cents)

        assert float(yes_price) + float(no_price) == pytest.approx(1.0)


class TestFormat:
    """v2 wants fixed-point dollar strings, not floats and not cents."""

    def test_prices_are_strings(self):
        _, price = encode_order("yes", 60)

        assert isinstance(price, str)

    @pytest.mark.parametrize("price_cents", [1, 7, 50, 99])
    def test_always_four_decimal_places(self, price_cents):
        _, price = encode_order("yes", price_cents)

        assert len(price.split(".")[1]) == 4

    def test_a_fifty_cent_quote_is_not_written_as_cents(self):
        """Guards the obvious catastrophe: sending 50 where 0.5000 is meant."""
        _, price = encode_order("yes", 50)

        assert price == "0.5000"
        assert price != "50"


class TestSideValidation:
    """The project speaks yes/no. Anything else must raise, never default."""

    @pytest.mark.parametrize("bad_side", ["bid", "ask", "YES", "buy", "", None])
    def test_unknown_side_raises(self, bad_side):
        with pytest.raises(ValueError):
            encode_order(bad_side, 60)
