"""
test_paper_quoter.py
--------------------
Paper mode reads PRODUCTION books. That is the whole reason this file leads
with the safety guarantee rather than the quoting logic.

Demo has no liquidity worth quoting against, so measuring anything means
reading prod — the environment where real money lives. "The paper quoter does
not call create_order" is not a safety property; it is an observation about
today's code. What is tested here is that trading is impossible by
construction: the client's write path raises, the quoter refuses any other
client, and there is no order-placing code to reach.
"""

from __future__ import annotations

import json

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from src.auth import (
    DEMO_BASE,
    PROD_BASE,
    Credentials,
    KalshiClient,
    ReadOnlyClient,
    ReadOnlyError,
)
from src.book import OrderBook
from src.quoter import PaperQuoter, RequoteTracker, plan_quotes

KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
TARGET = 1000
NEED = 200          # 1/5 of TARGET


def prod_creds() -> Credentials:
    return Credentials(key_id="k", private_key=KEY, env="prod", base_url=PROD_BASE)


def book(yes=(), no=()) -> OrderBook:
    return OrderBook(yes=list(yes), no=list(no))


def deep_book() -> OrderBook:
    """Both sides reach the 200-contract threshold two ticks down."""
    return book(yes=[(60, 50), (59, 80), (58, 120), (57, 300)],
                no=[(38, 60), (37, 90), (36, 200)])


class RecordingSession:
    """Fails the test if anything but a GET is attempted."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.methods: list[str] = []

    def request(self, method, url, **_kwargs):
        self.methods.append(method)
        assert method == "GET", f"a write reached the network: {method} {url}"

        class Response:
            status_code = 200
            headers: dict = {}
            text = json.dumps(self.payload)
            ok = True

            def json(inner):
                return json.loads(inner.text)

        return Response()


def readonly_client(payload: dict | None = None) -> ReadOnlyClient:
    client = ReadOnlyClient(prod_creds())
    client.session = RecordingSession(payload or {"orderbook_fp": {
        "yes_dollars": [["0.6000", "50.00"], ["0.5900", "80.00"], ["0.5800", "120.00"]],
        "no_dollars": [["0.3800", "60.00"], ["0.3700", "90.00"], ["0.3600", "200.00"]],
    }})
    return client


# ---------------------------------------------------------------------------
# THE SAFETY GUARANTEE
# ---------------------------------------------------------------------------

class TestReadOnlyClientCannotWrite:
    @pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
    def test_every_write_verb_raises(self, method):
        client = readonly_client()

        with pytest.raises(ReadOnlyError):
            client.request(method, "/portfolio/events/orders")

    def test_it_raises_before_touching_the_network(self):
        """
        The refusal has to come first. A write that is built, signed and sent
        before being judged is a write.
        """
        client = readonly_client()

        with pytest.raises(ReadOnlyError):
            client.request("POST", "/portfolio/events/orders", body={"x": 1})

        assert client.session.methods == []

    def test_create_order_raises(self):
        with pytest.raises(ReadOnlyError):
            readonly_client().create_order("MKT-A", "yes", 1, 60)

    def test_cancel_order_raises(self):
        with pytest.raises(ReadOnlyError):
            readonly_client().cancel_order("o1")

    def test_reads_still_work(self):
        """Refusing writes is only useful if reading still functions."""
        client = readonly_client()

        assert client.get("/markets/X/orderbook")["orderbook_fp"] is not None
        assert client.session.methods == ["GET"]

    def test_a_write_verb_added_later_is_refused_too(self):
        """
        request() is overridden, not just the order helpers, so a new write
        method on KalshiClient inherits the refusal without anyone remembering.
        """
        assert ReadOnlyClient.WRITE_METHODS >= {"POST", "PUT", "PATCH", "DELETE"}
        with pytest.raises(ReadOnlyError):
            readonly_client().post("/some/future/endpoint", body={})


class TestPaperQuoterRefusesATradingClient:
    def test_an_ordinary_client_is_rejected_at_construction(self):
        """Fail before polling starts, not at a first order that never comes."""
        trading = KalshiClient(prod_creds())

        with pytest.raises(TypeError, match="ReadOnlyClient"):
            PaperQuoter(trading)

    def test_even_a_demo_trading_client_is_rejected(self):
        """
        The type is the guarantee, not the environment. A demo client would be
        harmless today and a prod client tomorrow after one config edit.
        """
        demo = KalshiClient(
            Credentials(key_id="k", private_key=KEY, env="demo", base_url=DEMO_BASE)
        )

        with pytest.raises(TypeError):
            PaperQuoter(demo)

    def test_a_readonly_client_is_accepted(self):
        assert PaperQuoter(readonly_client()) is not None

    def test_polling_a_market_issues_only_gets(self, tmp_path):
        quoter = PaperQuoter(readonly_client(), log_path=tmp_path / "p.jsonl")

        quoter.poll(["MKT-A"])

        assert set(quoter.client.session.methods) == {"GET"}


# ---------------------------------------------------------------------------
# WHERE WE WOULD REST
# ---------------------------------------------------------------------------

class TestPlanQuotes:
    def test_it_rests_at_the_reference_price(self):
        plan = plan_quotes(deep_book(), TARGET, size=100)

        by_side = {q.side: q for q in plan.quotes}
        assert by_side["yes"].price_cents == 58
        assert by_side["no"].price_cents == 36

    def test_it_does_not_rest_at_top_of_book(self):
        """The edge: same reward credit, far fewer fills."""
        plan = plan_quotes(deep_book(), TARGET, size=100)

        by_side = {q.side: q for q in plan.quotes}
        assert by_side["yes"].price_cents < 60
        assert by_side["no"].price_cents < 38

    def test_it_records_how_far_below_the_top_we_sit(self):
        plan = plan_quotes(deep_book(), TARGET, size=100)

        by_side = {q.side: q for q in plan.quotes}
        assert by_side["yes"].ticks_below_top == 2
        assert by_side["no"].ticks_below_top == 2

    def test_it_quotes_both_sides(self):
        plan = plan_quotes(deep_book(), TARGET, size=100)

        assert {q.side for q in plan.quotes} == {"yes", "no"}

    def test_a_thin_side_cancels_the_whole_plan(self):
        """
        Both sides or neither: a snapshot is excluded unless both sides meet
        Target Size, so a lone quote carries fill risk and earns nothing.
        """
        one_sided = book(yes=[(60, 500)], no=[(38, 10)])

        plan = plan_quotes(one_sided, TARGET, size=100)

        assert plan.quotes == []
        assert not plan.quotable
        assert "no" in plan.skipped_reason

    def test_an_empty_book_is_not_quotable(self):
        plan = plan_quotes(book(), TARGET, size=100)

        assert not plan.quotable

    def test_target_size_moves_where_we_would_rest(self):
        """Flagged: the resting price is downstream of an unpublished number."""
        shallow = plan_quotes(deep_book(), 100, size=100).quotes
        deep = plan_quotes(deep_book(), TARGET, size=100).quotes

        assert {q.side: q.price_cents for q in shallow}["yes"] == 60
        assert {q.side: q.price_cents for q in deep}["yes"] == 58


# ---------------------------------------------------------------------------
# REQUOTE HYSTERESIS — safety rule 7
# ---------------------------------------------------------------------------

class TestRequoteHysteresis:
    def test_the_first_quote_always_goes_out(self):
        assert RequoteTracker().should_requote("MKT-A", "yes", 58) is True

    def test_an_unchanged_reference_price_does_not_requote(self):
        """
        The rule's substance. Book deltas arrive constantly while the Reference
        Price rarely moves; requoting on deltas burns the rate limit and churns
        fees for no reward change.
        """
        tracker = RequoteTracker()
        tracker.record("MKT-A", "yes", 58)

        assert tracker.should_requote("MKT-A", "yes", 58) is False

    def test_a_one_tick_move_requotes(self):
        tracker = RequoteTracker()
        tracker.record("MKT-A", "yes", 58)

        assert tracker.should_requote("MKT-A", "yes", 57) is True

    def test_a_wider_threshold_suppresses_small_moves(self):
        tracker = RequoteTracker(threshold_ticks=3)
        tracker.record("MKT-A", "yes", 58)

        assert tracker.should_requote("MKT-A", "yes", 57) is False
        assert tracker.should_requote("MKT-A", "yes", 55) is True

    def test_sides_are_tracked_separately(self):
        tracker = RequoteTracker()
        tracker.record("MKT-A", "yes", 58)

        assert tracker.should_requote("MKT-A", "no", 58) is True

    def test_markets_are_tracked_separately(self):
        tracker = RequoteTracker()
        tracker.record("MKT-A", "yes", 58)

        assert tracker.should_requote("MKT-B", "yes", 58) is True


# ---------------------------------------------------------------------------
# THE LOG
# ---------------------------------------------------------------------------

class TestSnapshotLog:
    def test_it_records_everything_the_replay_needs(self, tmp_path):
        quoter = PaperQuoter(readonly_client(), log_path=tmp_path / "p.jsonl")

        record = quoter.snapshot("MKT-A")

        for key in ("ts", "ticker", "reference_price", "quotes", "book", "best",
                    "two_sided_ok", "target_size"):
            assert key in record

    def test_it_keeps_the_full_book_not_a_summary(self, tmp_path):
        """A summarised log cannot answer a question posed after it was written."""
        quoter = PaperQuoter(readonly_client(), log_path=tmp_path / "p.jsonl")

        record = quoter.snapshot("MKT-A")

        assert record["book"]["yes"] == [(60, 50), (59, 80), (58, 120)]
        assert record["book"]["no"] == [(38, 60), (37, 90), (36, 200)]

    def test_it_records_both_reference_prices(self, tmp_path):
        quoter = PaperQuoter(readonly_client(), log_path=tmp_path / "p.jsonl")

        record = quoter.snapshot("MKT-A")

        assert record["reference_price"]["yes"]["price"] == 58
        assert record["reference_price"]["no"]["price"] == 36

    def test_it_flags_that_target_size_is_assumed(self, tmp_path):
        """The log has to carry the caveat; it outlives the console output."""
        quoter = PaperQuoter(readonly_client(), log_path=tmp_path / "p.jsonl")

        assert quoter.snapshot("MKT-A")["target_size_is_an_assumption"] is True

    def test_each_poll_appends_one_json_line_per_market(self, tmp_path):
        path = tmp_path / "p.jsonl"
        quoter = PaperQuoter(readonly_client(), log_path=path)

        quoter.poll(["MKT-A"])
        quoter.poll(["MKT-A"])

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert all(json.loads(line)["ticker"] == "MKT-A" for line in lines)

    def test_the_second_identical_poll_is_not_a_requote(self, tmp_path):
        path = tmp_path / "p.jsonl"
        quoter = PaperQuoter(readonly_client(), log_path=path)

        first = quoter.poll(["MKT-A"])[0]
        second = quoter.poll(["MKT-A"])[0]

        assert sorted(first["requoted_sides"]) == ["no", "yes"]
        assert second["requoted_sides"] == []
