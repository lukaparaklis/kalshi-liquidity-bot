"""
test_book_parsing.py
--------------------
Turning GET /markets/{ticker}/orderbook into an OrderBook.

This parser exists because the shape already caused one silent failure. The
scanner read `orderbook.yes` — the v1 key — against a v2 response, got nothing
back, and concluded that every market on the exchange was dead. Nothing raised;
the answer was just quietly wrong. Reading a book is now done in exactly one
place so that failure cannot be reproduced independently in three scripts.

The v2 shape, confirmed on demo:

    {"orderbook_fp": {"yes_dollars": [["0.0100", "2.00"], ["0.0200", "1.00"]],
                      "no_dollars":  []}}

Prices are dollar strings, sizes are fixed-point strings, and levels arrive
ascending. Internally everything is integer cents, best first.
"""

from __future__ import annotations

from src.book import OrderBook

# Copied from demo, 2026-08-15: KXPPLMATCH-26AUG14TPBSDS-TPB.
LIVE_ORDERBOOK_PAYLOAD = {
    "orderbook_fp": {
        "yes_dollars": [["0.0100", "2.00"], ["0.0200", "1.00"]],
        "no_dollars": [],
    }
}


class TestV2Shape:
    def test_the_live_payload_parses(self):
        b = OrderBook.from_api(LIVE_ORDERBOOK_PAYLOAD)

        assert b.yes == [(2, 1), (1, 2)]   # best first
        assert b.no == []

    def test_dollar_strings_become_cents(self):
        b = OrderBook.from_api(
            {"orderbook_fp": {"yes_dollars": [["0.6000", "100.00"]], "no_dollars": []}}
        )

        assert b.yes == [(60, 100)]

    def test_both_sides_are_read(self):
        b = OrderBook.from_api({"orderbook_fp": {
            "yes_dollars": [["0.6000", "100.00"]],
            "no_dollars": [["0.3800", "250.00"]],
        }})

        assert b.yes == [(60, 100)]
        assert b.no == [(38, 250)]

    def test_ascending_input_is_reordered_best_first(self):
        b = OrderBook.from_api({"orderbook_fp": {
            "yes_dollars": [["0.5800", "10.00"], ["0.5900", "20.00"], ["0.6000", "30.00"]],
            "no_dollars": [],
        }})

        assert [price for price, _ in b.yes] == [60, 59, 58]

    def test_sub_cent_prices_round_to_the_nearest_cent(self):
        """Kalshi quotes sub-penny on some series; the book model is integer cents."""
        b = OrderBook.from_api(
            {"orderbook_fp": {"yes_dollars": [["0.5550", "10.00"]], "no_dollars": []}}
        )

        assert b.yes == [(56, 10)]


class TestLegacyShape:
    def test_the_v1_shape_still_parses(self):
        """Integer cents, not dollars — 60 here means 60c, not $60."""
        b = OrderBook.from_api({"orderbook": {"yes": [[60, 100]], "no": [[38, 250]]}})

        assert b.yes == [(60, 100)]
        assert b.no == [(38, 250)]

    def test_the_two_shapes_are_not_confused(self):
        """
        The dangerous ambiguity: 60 as v1 cents and "0.6000" as v2 dollars are
        the same price, so a parser that guessed by value would be wrong by
        100x in one direction or the other.
        """
        v1 = OrderBook.from_api({"orderbook": {"yes": [[60, 100]], "no": []}})
        v2 = OrderBook.from_api(
            {"orderbook_fp": {"yes_dollars": [["0.6000", "100.00"]], "no_dollars": []}}
        )

        assert v1.yes == v2.yes


class TestDegenerateInput:
    def test_an_empty_v2_book_is_empty(self):
        b = OrderBook.from_api({"orderbook_fp": {"yes_dollars": [], "no_dollars": []}})

        assert b.yes == []
        assert b.no == []

    def test_a_missing_orderbook_key_is_empty_not_an_error(self):
        b = OrderBook.from_api({})

        assert b.yes == []
        assert b.no == []

    def test_null_sides_are_empty(self):
        b = OrderBook.from_api({"orderbook_fp": {"yes_dollars": None, "no_dollars": None}})

        assert b.yes == []

    def test_zero_size_levels_are_dropped(self):
        b = OrderBook.from_api({"orderbook_fp": {
            "yes_dollars": [["0.6000", "0.00"], ["0.5900", "10.00"]], "no_dollars": [],
        }})

        assert b.yes == [(59, 10)]

    def test_normalisation_still_applies(self):
        """Duplicate levels aggregate, exactly as for a hand-built book."""
        b = OrderBook.from_api({"orderbook_fp": {
            "yes_dollars": [["0.6000", "100.00"], ["0.6000", "150.00"]], "no_dollars": [],
        }})

        assert b.yes == [(60, 250)]

    def test_malformed_levels_are_skipped_not_fatal(self):
        """A scanner sweeping hundreds of markets must not die on one bad row."""
        b = OrderBook.from_api({"orderbook_fp": {
            "yes_dollars": [["0.6000", "100.00"], ["oops"], [], None],
            "no_dollars": [],
        }})

        assert b.yes == [(60, 100)]
