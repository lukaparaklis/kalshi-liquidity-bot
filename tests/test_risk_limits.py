"""
test_risk_limits.py
-------------------
Tests for the safety layer. Written before risk.py exists; risk.py must be
written to satisfy these.

CLAUDE.md, "SAFETY RULES — non-negotiable", says these protect real money and
must not be relaxed to make a feature work. Four of them are encoded here:

  1. Cancel-all on shutdown — SIGINT/SIGTERM and atexit both cancel every
     resting order. Both handlers can fire on one shutdown, so cancel-all must
     be safe to call twice.
  2. Dead-man's switch — if the main loop has not completed an iteration in
     60s, cancel everything. A hung bot with live quotes is the worst state.
  3. Hard position limits per market and in aggregate, checked BEFORE every
     order, not after.
  4. Daily loss limit that halts new quoting for the session.

Architecture this pins down: RiskManager is a gateway, not an advisor. Orders
are placed THROUGH it, so "checked before every order" is structural rather
than a convention a caller can forget. A test that only asserted a checker
returned False would prove nothing about whether an order was actually sent;
these assert the client never saw the order at all.

Time is injected. Nothing here sleeps, and the dead-man's switch is tested at
its exact boundary rather than approximately.

Startup reconciliation is part of rule 1, not an extra. A process that crashed
left its orders resting on the exchange, and the new process has no memory of
them. So RiskManager queries GET /portfolio/orders on startup and cancels what
it finds BEFORE it is allowed to place anything.

ASSUMPTIONS FLAGGED
  - Limits are worst-case potential exposure (filled position plus resting
    orders that could fill), not just filled position. A limit that ignored
    resting orders would be trivially breached by the next fill.
  - Reconciliation reads GET /portfolio/positions and seeds the limits from it,
    so exposure inherited from a previous session counts against the caps, and
    cancels resting orders from GET /portfolio/orders.
  - A reconciliation that could not read positions, or could not cancel every
    stale order, leaves the bot blocked rather than quoting on top of exposure
    it cannot see or could not clear.
  - Kalshi's market position is signed: positive is long yes, negative is long
    no. That matches the internal convention. Field named "position" — verify
    against a real payload before trusting it with money.
"""

from __future__ import annotations

import pytest

from src.risk import RiskLimits, RiskManager, RiskViolation

START = 1_000.0
DEADMAN_TIMEOUT_S = 60.0


# ---------------------------------------------------------------------------
# FAKES
# ---------------------------------------------------------------------------

class FakeClock:
    """Injected time source. Absolute sets keep boundary tests exact."""

    def __init__(self, now: float = START):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def set(self, now: float) -> None:
        self.now = now


class FakeClient:
    """
    Records what the exchange was actually asked to do.

    `created` and `cancelled` are the ground truth for "was this order really
    sent", which is the whole point of the pre-order tests.
    """

    def __init__(self):
        self.created: list[dict] = []
        self.cancelled: list[str] = []
        self.fail_cancel_ids: set[str] = set()
        self.open_orders: list[dict] = []      # what GET /portfolio/orders returns
        self.pages: list[list[dict]] | None = None   # set to exercise the cursor
        self.get_orders_calls: list[dict] = []
        self.on_cancel = None                  # fires once, mid-cancellation
        self.positions: list[dict] = []        # what GET /portfolio/positions returns
        self.position_pages: list[list[dict]] | None = None
        self.get_positions_calls: list[dict] = []
        self.fail_positions = False
        self._next_id = 0

    def get_positions(self, cursor: str | None = None) -> dict:
        self.get_positions_calls.append({"cursor": cursor})
        if self.fail_positions:
            raise RuntimeError("simulated positions fetch failure")
        if self.position_pages is not None:
            index = int(cursor) if cursor else 0
            following = index + 1
            return {
                "market_positions": self.position_pages[index],
                "cursor": str(following) if following < len(self.position_pages) else "",
            }
        return {"market_positions": list(self.positions), "cursor": ""}

    def get_orders(self, status: str | None = None, cursor: str | None = None) -> dict:
        self.get_orders_calls.append({"status": status, "cursor": cursor})
        if self.pages is not None:
            index = int(cursor) if cursor else 0
            following = index + 1
            return {
                "orders": self.pages[index],
                "cursor": str(following) if following < len(self.pages) else "",
            }
        return {"orders": list(self.open_orders), "cursor": ""}

    def create_order(self, ticker: str, side: str, count: int, price_cents: int,
                     post_only: bool = True) -> dict:
        self._next_id += 1
        order_id = f"o{self._next_id}"
        self.created.append({
            "order_id": order_id, "ticker": ticker,
            "side": side, "count": count, "price_cents": price_cents,
        })
        # The v2 create response is flat — no {"order": ...} envelope.
        return {"order_id": order_id, "fill_count": "0",
                "remaining_count": str(count), "ts_ms": 1_700_000_000_000}

    def cancel_order(self, order_id: str) -> dict:
        # Lets a test simulate SIGINT landing while cancellation is in flight.
        if self.on_cancel is not None:
            hook, self.on_cancel = self.on_cancel, None
            hook()
        if order_id in self.fail_cancel_ids:
            raise RuntimeError(f"simulated cancel failure for {order_id}")
        self.cancelled.append(order_id)
        return {"order": {"order_id": order_id, "status": "canceled"}}


def resting(order_id: str, *, ticker="MKT-A", side="yes", count=50, price=60,
            status="resting") -> dict:
    """
    An order as GET /portfolio/orders returns it, in v2 field names:
    outcome_side/book_side, fixed-point counts, dollar-string prices.
    """
    return {
        "order_id": order_id,
        "ticker": ticker,
        "outcome_side": side,
        "book_side": "bid" if side == "yes" else "ask",
        # Copied from a real payload: the legacy `side` field reads "yes" even
        # on a no-side order. Present here so the parser is tested against the
        # trap rather than a tidied-up version of it.
        "side": "yes",
        "status": status,
        "remaining_count_fp": f"{count}.00",
        "yes_price_dollars": f"{(price if side == 'yes' else 100 - price) / 100:.4f}",
        "no_price_dollars": f"{(100 - price if side == 'yes' else price) / 100:.4f}",
    }


def resting_v1(order_id: str, *, ticker="MKT-A", side="yes", count=50,
               price=60, status="resting") -> dict:
    """The legacy shape, kept to prove the parser still tolerates it."""
    return {
        "order_id": order_id, "ticker": ticker, "side": side, "status": status,
        "remaining_count": count,
        ("yes_price" if side == "yes" else "no_price"): price,
    }


def held(ticker: str, position: int) -> dict:
    """A position as GET /portfolio/positions returns it. Signed: +yes, -no."""
    return {"ticker": ticker, "position": position}


def make_risk(*, per_market=100, aggregate=250, max_daily_loss_cents=5_000,
              open_orders=None, positions=None, reconcile=True):
    """
    A RiskManager wired to fakes. Returns (risk, client, clock).

    Reconciles by default, since quoting is blocked until it has run. Tests
    that are about reconciliation itself pass reconcile=False and drive it
    explicitly.
    """
    clock = FakeClock()
    client = FakeClient()
    if open_orders is not None:
        client.open_orders = list(open_orders)
    if positions is not None:
        client.positions = list(positions)
    limits = RiskLimits(
        max_position_per_market=per_market,
        max_position_aggregate=aggregate,
        max_daily_loss_cents=max_daily_loss_cents,
        deadman_timeout_s=DEADMAN_TIMEOUT_S,
    )
    risk = RiskManager(limits, client=client, clock=clock)
    if reconcile:
        risk.reconcile()
    return risk, client, clock


# ---------------------------------------------------------------------------
# SAFETY RULE 3 — POSITION LIMITS, CHECKED PRE-ORDER
# ---------------------------------------------------------------------------

class TestPositionLimitsArePreOrder:
    """
    "Hard position limits per market and in aggregate, checked before every
    order, not after."

    Every rejection test asserts the client never received the order. Checking
    after the fact means the breach already happened on the exchange.
    """

    def test_order_within_limits_reaches_the_exchange(self):
        risk, client, _ = make_risk(per_market=100)

        risk.place_order("MKT-A", "yes", count=40, price_cents=60)

        assert len(client.created) == 1
        assert client.created[0]["count"] == 40

    def test_breaching_order_is_never_sent_to_the_exchange(self):
        risk, client, _ = make_risk(per_market=100)

        with pytest.raises(RiskViolation):
            risk.place_order("MKT-A", "yes", count=101, price_cents=60)

        assert client.created == [], "limit was checked after the order, not before"

    def test_rejected_order_leaves_no_state_behind(self):
        """A rejection must not half-register the order."""
        risk, _, _ = make_risk(per_market=100)

        with pytest.raises(RiskViolation):
            risk.place_order("MKT-A", "yes", count=101, price_cents=60)

        assert risk.working_orders == {}
        assert risk.position("MKT-A") == 0
        assert risk.exposure("MKT-A") == 0

    def test_exactly_at_the_limit_is_allowed(self):
        risk, client, _ = make_risk(per_market=100)

        risk.place_order("MKT-A", "yes", count=100, price_cents=60)

        assert len(client.created) == 1

    def test_one_contract_over_the_limit_is_rejected(self):
        risk, client, _ = make_risk(per_market=100)
        risk.place_order("MKT-A", "yes", count=100, price_cents=60)

        with pytest.raises(RiskViolation):
            risk.place_order("MKT-A", "yes", count=1, price_cents=59)

        assert len(client.created) == 1

    def test_resting_orders_count_toward_the_limit(self):
        """
        Limits are worst-case: resting size can fill. A limit that only counted
        filled position would be breached by the very next fill.
        """
        risk, _, _ = make_risk(per_market=100)
        risk.place_order("MKT-A", "yes", count=80, price_cents=60)

        assert risk.position("MKT-A") == 0
        assert risk.exposure("MKT-A") == 80

        with pytest.raises(RiskViolation):
            risk.place_order("MKT-A", "yes", count=21, price_cents=59)

    def test_quoting_both_sides_does_not_double_count(self):
        """
        Resting 100 yes and 100 no is not 200 of exposure — the worst case is
        one side filling, not both netting out. This matters: the reward
        program excludes any snapshot where both sides are not present, so
        two-sided quoting is mandatory and must not trip the limit.
        """
        risk, client, _ = make_risk(per_market=100)

        risk.place_order("MKT-A", "yes", count=100, price_cents=60)
        risk.place_order("MKT-A", "no", count=100, price_cents=38)

        assert len(client.created) == 2
        assert risk.exposure("MKT-A") == 100

    def test_a_fill_does_not_change_worst_case_exposure(self):
        """Resting size converting to position is exposure moving, not growing."""
        risk, _, _ = make_risk(per_market=100)
        order = risk.place_order("MKT-A", "yes", count=100, price_cents=60)

        risk.record_fill(order.order_id, count=40)

        assert risk.position("MKT-A") == 40
        assert risk.exposure("MKT-A") == 100

    def test_per_market_limits_are_independent(self):
        risk, client, _ = make_risk(per_market=100, aggregate=1_000)

        risk.place_order("MKT-A", "yes", count=100, price_cents=60)
        risk.place_order("MKT-B", "yes", count=100, price_cents=60)

        assert len(client.created) == 2

    def test_aggregate_limit_rejects_what_per_market_limits_allow(self):
        """Each order is fine for its own market; together they breach the book."""
        risk, client, _ = make_risk(per_market=100, aggregate=150)
        risk.place_order("MKT-A", "yes", count=100, price_cents=60)

        with pytest.raises(RiskViolation):
            risk.place_order("MKT-B", "yes", count=100, price_cents=60)

        assert len(client.created) == 1

    def test_aggregate_limit_allows_an_order_that_exactly_fills_it(self):
        risk, client, _ = make_risk(per_market=100, aggregate=150)
        risk.place_order("MKT-A", "yes", count=100, price_cents=60)

        risk.place_order("MKT-B", "yes", count=50, price_cents=60)

        assert len(client.created) == 2
        assert risk.aggregate_exposure() == 150

    def test_violation_identifies_which_limit_was_hit(self):
        """Rejections must be diagnosable in a log at 3am."""
        risk, _, _ = make_risk(per_market=100, aggregate=150)

        with pytest.raises(RiskViolation) as excinfo:
            risk.place_order("MKT-A", "yes", count=200, price_cents=60)
        assert excinfo.value.kind == "position_per_market"

        risk.place_order("MKT-A", "yes", count=100, price_cents=60)
        with pytest.raises(RiskViolation) as excinfo:
            risk.place_order("MKT-B", "yes", count=100, price_cents=60)
        assert excinfo.value.kind == "position_aggregate"


# ---------------------------------------------------------------------------
# SAFETY RULE 4 — DAILY LOSS LIMIT
# ---------------------------------------------------------------------------

class TestDailyLossLimit:
    """
    "A daily loss limit that halts new quoting for the session."

    "For the session" is a latch: once tripped it stays tripped. A limit that
    released itself when P&L recovered would let a bad day resume quoting on a
    bounce, which is exactly the behaviour the rule exists to prevent.
    """

    def test_losses_below_the_limit_do_not_halt(self):
        risk, client, _ = make_risk(max_daily_loss_cents=5_000)

        risk.record_pnl(-4_999)

        assert not risk.halted
        risk.place_order("MKT-A", "yes", count=10, price_cents=60)
        assert len(client.created) == 1

    def test_reaching_the_limit_exactly_halts(self):
        risk, _, _ = make_risk(max_daily_loss_cents=5_000)

        risk.record_pnl(-5_000)

        assert risk.halted
        assert risk.halt_reason == "daily_loss_limit"

    def test_halt_blocks_new_orders_before_they_are_sent(self):
        risk, client, _ = make_risk(max_daily_loss_cents=5_000)
        risk.record_pnl(-5_000)

        with pytest.raises(RiskViolation) as excinfo:
            risk.place_order("MKT-A", "yes", count=10, price_cents=60)

        assert excinfo.value.kind == "halted"
        assert client.created == []

    def test_losses_accumulate_across_calls(self):
        risk, _, _ = make_risk(max_daily_loss_cents=5_000)

        risk.record_pnl(-2_000)
        risk.record_pnl(-2_000)
        assert not risk.halted

        risk.record_pnl(-1_000)
        assert risk.halted

    def test_profit_never_halts(self):
        risk, _, _ = make_risk(max_daily_loss_cents=5_000)

        risk.record_pnl(50_000)

        assert not risk.halted

    def test_halt_is_latched_and_survives_a_recovery(self):
        """The critical property. A bounce must not resume quoting."""
        risk, client, _ = make_risk(max_daily_loss_cents=5_000)
        risk.record_pnl(-5_000)
        assert risk.halted

        risk.record_pnl(20_000)

        assert risk.halted, "daily loss halt released itself on a P&L recovery"
        with pytest.raises(RiskViolation):
            risk.place_order("MKT-A", "yes", count=10, price_cents=60)
        assert client.created == []

    def test_halt_never_blocks_cancelling(self):
        """
        Halted means "place nothing", never "do nothing". Losing the ability to
        cancel while halted would strand live orders — the worst state.
        """
        risk, client, _ = make_risk(max_daily_loss_cents=5_000)
        risk.place_order("MKT-A", "yes", count=10, price_cents=60)
        risk.record_pnl(-5_000)

        result = risk.cancel_all()

        assert len(result.cancelled) == 1
        assert client.cancelled == ["o1"]


# ---------------------------------------------------------------------------
# SAFETY RULE 2 — DEAD-MAN'S SWITCH
# ---------------------------------------------------------------------------

class TestDeadMansSwitch:
    """
    "If the main loop hasn't completed an iteration in 60s, cancel everything."

    Firing must both cancel and halt. Cancelling without halting would let the
    next loop iteration re-place quotes into the same hung condition.
    """

    def test_does_not_fire_before_the_timeout(self):
        risk, client, clock = make_risk()
        risk.place_order("MKT-A", "yes", count=10, price_cents=60)

        clock.set(START + 59.9)

        assert risk.check_deadman() is False
        assert not risk.halted
        assert client.cancelled == []

    def test_fires_at_exactly_the_timeout(self):
        risk, _, clock = make_risk()

        clock.set(START + DEADMAN_TIMEOUT_S)

        assert risk.check_deadman() is True
        assert risk.halted
        assert risk.halt_reason == "deadman"

    def test_firing_cancels_every_resting_order(self):
        risk, client, clock = make_risk()
        risk.place_order("MKT-A", "yes", count=10, price_cents=60)
        risk.place_order("MKT-B", "no", count=10, price_cents=38)

        clock.set(START + DEADMAN_TIMEOUT_S)
        risk.check_deadman()

        assert sorted(client.cancelled) == ["o1", "o2"]
        assert risk.working_orders == {}

    def test_a_tick_resets_the_timer(self):
        risk, client, clock = make_risk()

        clock.set(START + 50.0)
        risk.tick()
        clock.set(START + 100.0)  # 100s since start, only 50s since the tick

        assert risk.check_deadman() is False
        assert not risk.halted
        assert client.cancelled == []

    def test_fires_when_ticks_stop_arriving(self):
        risk, _, clock = make_risk()
        clock.set(START + 50.0)
        risk.tick()

        clock.set(START + 110.1)  # 60.1s since the last tick

        assert risk.check_deadman() is True

    def test_a_late_tick_does_not_revive_a_halted_bot(self):
        """
        A loop that hangs and then recovers must not silently resume quoting.
        Restarting is a human decision.
        """
        risk, client, clock = make_risk()
        clock.set(START + DEADMAN_TIMEOUT_S)
        risk.check_deadman()

        risk.tick()

        assert risk.halted
        with pytest.raises(RiskViolation):
            risk.place_order("MKT-A", "yes", count=10, price_cents=60)
        assert client.created == []

    def test_firing_twice_does_not_cancel_twice(self):
        risk, client, clock = make_risk()
        risk.place_order("MKT-A", "yes", count=10, price_cents=60)
        clock.set(START + DEADMAN_TIMEOUT_S)

        risk.check_deadman()
        risk.check_deadman()

        assert client.cancelled == ["o1"]

    def test_fires_safely_with_no_resting_orders(self):
        risk, client, clock = make_risk()

        clock.set(START + DEADMAN_TIMEOUT_S)

        assert risk.check_deadman() is True
        assert risk.halted
        assert client.cancelled == []

    def test_placing_an_order_checks_the_switch_first(self):
        """
        Defence in depth: if the watchdog itself is what hung, the order path
        must still refuse to quote into a stale loop.
        """
        risk, client, clock = make_risk()
        clock.set(START + DEADMAN_TIMEOUT_S)

        with pytest.raises(RiskViolation) as excinfo:
            risk.place_order("MKT-A", "yes", count=10, price_cents=60)

        assert excinfo.value.kind == "halted"
        assert client.created == []
        assert risk.halted


# ---------------------------------------------------------------------------
# SAFETY RULE 1 — CANCEL-ALL
# ---------------------------------------------------------------------------

class TestCancelAll:
    """
    "If the process dies, orders stay live on the exchange. Register
    SIGINT/SIGTERM handlers AND an atexit hook."

    Both handlers can fire on a single shutdown, so a second call must be a
    no-op rather than an error or a duplicate cancel. Kalshi rejects cancelling
    an already-cancelled order, so a double-cancel turns clean shutdown output
    into a wall of errors that hides real failures.
    """

    def test_cancels_every_resting_order(self):
        risk, client, _ = make_risk()
        risk.place_order("MKT-A", "yes", count=10, price_cents=60)
        risk.place_order("MKT-A", "no", count=10, price_cents=38)
        risk.place_order("MKT-B", "yes", count=10, price_cents=55)

        result = risk.cancel_all()

        assert sorted(client.cancelled) == ["o1", "o2", "o3"]
        assert sorted(result.cancelled) == ["o1", "o2", "o3"]
        assert result.failed == []
        assert risk.working_orders == {}

    def test_second_call_is_a_no_op(self):
        risk, client, _ = make_risk()
        risk.place_order("MKT-A", "yes", count=10, price_cents=60)
        risk.cancel_all()

        result = risk.cancel_all()

        assert result.cancelled == []
        assert client.cancelled == ["o1"], "order was cancelled twice"

    def test_calling_on_an_empty_book_is_safe(self):
        risk, client, _ = make_risk()

        result = risk.cancel_all()

        assert result.cancelled == []
        assert result.failed == []
        assert client.cancelled == []

    def test_one_failure_does_not_abort_the_rest(self):
        """
        The single most important property here. An exception on the first
        cancel must not strand the other orders live on the exchange.
        """
        risk, client, _ = make_risk()
        risk.place_order("MKT-A", "yes", count=10, price_cents=60)
        risk.place_order("MKT-A", "no", count=10, price_cents=38)
        risk.place_order("MKT-B", "yes", count=10, price_cents=55)
        client.fail_cancel_ids = {"o1"}

        result = risk.cancel_all()

        assert sorted(client.cancelled) == ["o2", "o3"]
        assert sorted(result.cancelled) == ["o2", "o3"]
        assert [order_id for order_id, _ in result.failed] == ["o1"]

    def test_an_order_that_failed_to_cancel_stays_tracked(self):
        """
        A failed cancel means the order may still be live. Forgetting it would
        make the bot believe it is flat when it is not.
        """
        risk, client, _ = make_risk()
        risk.place_order("MKT-A", "yes", count=10, price_cents=60)
        risk.place_order("MKT-A", "no", count=10, price_cents=38)
        client.fail_cancel_ids = {"o1"}

        risk.cancel_all()

        assert set(risk.working_orders) == {"o1"}

    def test_retry_picks_up_only_the_order_that_failed(self):
        risk, client, _ = make_risk()
        risk.place_order("MKT-A", "yes", count=10, price_cents=60)
        risk.place_order("MKT-A", "no", count=10, price_cents=38)
        client.fail_cancel_ids = {"o1"}
        risk.cancel_all()

        client.fail_cancel_ids.clear()
        result = risk.cancel_all()

        assert result.cancelled == ["o1"]
        assert client.cancelled == ["o2", "o1"]
        assert risk.working_orders == {}

    def test_does_not_cancel_orders_that_already_filled(self):
        risk, client, _ = make_risk()
        order = risk.place_order("MKT-A", "yes", count=10, price_cents=60)
        risk.record_fill(order.order_id, count=10)

        result = risk.cancel_all()

        assert result.cancelled == []
        assert client.cancelled == []

    def test_partially_filled_orders_are_still_cancelled(self):
        risk, client, _ = make_risk()
        order = risk.place_order("MKT-A", "yes", count=10, price_cents=60)
        risk.record_fill(order.order_id, count=4)

        result = risk.cancel_all()

        assert result.cancelled == ["o1"]
        assert client.cancelled == ["o1"]

    def test_a_reentrant_call_during_cancellation_is_safe(self):
        """
        SIGINT arriving while atexit's cancel_all is mid-flight, in the same
        thread. A lock held across the cancel loop would deadlock here; a plain
        re-entrant walk of the live registry would cancel everything twice.
        """
        risk, client, _ = make_risk()
        risk.place_order("MKT-A", "yes", count=10, price_cents=60)
        risk.place_order("MKT-A", "no", count=10, price_cents=38)
        client.on_cancel = risk.cancel_all

        risk.cancel_all()

        assert sorted(client.cancelled) == ["o1", "o2"]
        assert len(client.cancelled) == 2, "reentrant call double-cancelled"
        assert risk.working_orders == {}


# ---------------------------------------------------------------------------
# SAFETY RULE 1 — STARTUP RECONCILIATION
# ---------------------------------------------------------------------------

class TestStartupReconciliation:
    """
    A process that died left its orders resting on the exchange, and a fresh
    RiskManager has no memory of them. Local cancel-all cannot see them, so
    they would rest unmanaged and unhedged until they filled.

    Startup therefore queries GET /portfolio/orders and cancels what it finds
    before quoting is permitted. As with the pre-order checks, this is
    structural: place_order refuses until reconciliation has succeeded, rather
    than trusting main.py to remember to call it.
    """

    def test_quoting_is_blocked_before_reconciliation(self):
        risk, client, _ = make_risk(reconcile=False)

        with pytest.raises(RiskViolation) as excinfo:
            risk.place_order("MKT-A", "yes", count=10, price_cents=60)

        assert excinfo.value.kind == "not_reconciled"
        assert client.created == []
        assert not risk.reconciled

    def test_quoting_is_unblocked_after_reconciliation(self):
        risk, client, _ = make_risk(reconcile=False)

        risk.reconcile()

        assert risk.reconciled
        risk.place_order("MKT-A", "yes", count=10, price_cents=60)
        assert len(client.created) == 1

    def test_pre_existing_resting_orders_are_cancelled(self):
        risk, client, _ = make_risk(
            reconcile=False,
            open_orders=[resting("stale-1"), resting("stale-2", side="no", price=38)],
        )

        result = risk.reconcile()

        assert sorted(client.cancelled) == ["stale-1", "stale-2"]
        assert sorted(result.cancelled) == ["stale-1", "stale-2"]
        assert risk.working_orders == {}

    def test_it_asks_the_exchange_for_resting_orders(self):
        risk, client, _ = make_risk(reconcile=False)

        risk.reconcile()

        assert client.get_orders_calls[0]["status"] == "resting"

    def test_non_resting_orders_are_left_alone(self):
        """
        Filtered locally as well as by the query parameter. Cancelling an
        already-executed order is an error response that buries real ones.
        """
        risk, client, _ = make_risk(
            reconcile=False,
            open_orders=[
                resting("done", status="executed"),
                resting("gone", status="canceled"),
                resting("live"),
            ],
        )

        result = risk.reconcile()

        assert client.cancelled == ["live"]
        assert result.cancelled == ["live"]

    def test_it_follows_the_cursor_to_the_last_page(self):
        """
        Paginated results are the silent version of this bug: cancel page one,
        report success, leave page two resting on the exchange.
        """
        risk, client, _ = make_risk(reconcile=False)
        client.pages = [
            [resting("p1-a"), resting("p1-b")],
            [resting("p2-a")],
            [resting("p3-a")],
        ]

        result = risk.reconcile()

        assert sorted(client.cancelled) == ["p1-a", "p1-b", "p2-a", "p3-a"]
        assert len(result.cancelled) == 4
        assert len(client.get_orders_calls) == 3

    def test_no_open_orders_is_the_normal_clean_start(self):
        risk, client, _ = make_risk(reconcile=False)

        result = risk.reconcile()

        assert result.cancelled == []
        assert result.failed == []
        assert client.cancelled == []
        assert risk.reconciled

    def test_a_failed_cancel_leaves_the_bot_blocked(self):
        """
        Refusing to quote is the conservative branch: a stale order that would
        not cancel is live exposure of unknown size, and adding fresh quotes on
        top of it compounds a position nobody chose.
        """
        risk, client, _ = make_risk(
            reconcile=False,
            open_orders=[resting("stale-1"), resting("stale-2")],
        )
        client.fail_cancel_ids = {"stale-1"}

        result = risk.reconcile()

        assert [order_id for order_id, _ in result.failed] == ["stale-1"]
        assert not risk.reconciled
        with pytest.raises(RiskViolation) as excinfo:
            risk.place_order("MKT-A", "yes", count=10, price_cents=60)
        assert excinfo.value.kind == "not_reconciled"
        assert client.created == []

    def test_one_failure_does_not_strand_the_others(self):
        risk, client, _ = make_risk(
            reconcile=False,
            open_orders=[resting("stale-1"), resting("stale-2"), resting("stale-3")],
        )
        client.fail_cancel_ids = {"stale-1"}

        risk.reconcile()

        assert sorted(client.cancelled) == ["stale-2", "stale-3"]

    def test_reconciliation_can_be_retried_after_a_failure(self):
        risk, client, _ = make_risk(
            reconcile=False, open_orders=[resting("stale-1")],
        )
        client.fail_cancel_ids = {"stale-1"}
        risk.reconcile()
        assert not risk.reconciled

        client.fail_cancel_ids.clear()
        result = risk.reconcile()

        assert result.cancelled == ["stale-1"]
        assert risk.reconciled

    def test_reconciling_twice_cancels_nothing_extra(self):
        risk, client, _ = make_risk(
            reconcile=False, open_orders=[resting("stale-1")],
        )
        risk.reconcile()
        client.open_orders = []          # the exchange now reports a clean book

        result = risk.reconcile()

        assert result.cancelled == []
        assert client.cancelled == ["stale-1"]

    def test_it_does_not_touch_orders_placed_after_it_ran(self):
        risk, client, _ = make_risk(reconcile=False)
        risk.reconcile()

        risk.place_order("MKT-A", "yes", count=10, price_cents=60)

        assert client.cancelled == []
        assert set(risk.working_orders) == {"o1"}


# ---------------------------------------------------------------------------
# SAFETY RULE 3 — INHERITED POSITIONS
# ---------------------------------------------------------------------------

class TestPositionReconciliation:
    """
    Cancelling stale orders clears what was resting; it does nothing about what
    already filled. A previous session that ended holding 80 contracts leaves
    that exposure on the account, and a fresh RiskManager starting from zero
    would happily quote another 100 on top of it — breaching a limit it thinks
    it is enforcing.

    So startup seeds positions from GET /portfolio/positions. Signed, matching
    the internal convention: positive is long yes, negative is long no.
    """

    def test_inherited_positions_are_seeded(self):
        risk, _, _ = make_risk(
            reconcile=False, positions=[held("MKT-A", 80), held("MKT-B", -30)],
        )

        risk.reconcile()

        assert risk.position("MKT-A") == 80
        assert risk.position("MKT-B") == -30

    def test_inherited_exposure_is_absolute(self):
        """Long no is exposure just as much as long yes."""
        risk, _, _ = make_risk(reconcile=False, positions=[held("MKT-A", -30)])

        risk.reconcile()

        assert risk.exposure("MKT-A") == 30

    def test_it_asks_the_exchange_for_positions(self):
        risk, client, _ = make_risk(reconcile=False)

        risk.reconcile()

        assert client.get_positions_calls != []

    def test_an_inherited_position_consumes_the_per_market_cap(self):
        """The hole this closes: 80 inherited + 21 new is 101 against a cap of 100."""
        risk, client, _ = make_risk(per_market=100, positions=[held("MKT-A", 80)])

        with pytest.raises(RiskViolation) as excinfo:
            risk.place_order("MKT-A", "yes", count=21, price_cents=60)

        assert excinfo.value.kind == "position_per_market"
        assert client.created == []

    def test_an_inherited_position_leaves_room_for_a_smaller_order(self):
        risk, client, _ = make_risk(per_market=100, positions=[held("MKT-A", 80)])

        risk.place_order("MKT-A", "yes", count=20, price_cents=60)

        assert len(client.created) == 1
        assert risk.exposure("MKT-A") == 100

    def test_an_inherited_short_position_consumes_the_cap_too(self):
        risk, client, _ = make_risk(per_market=100, positions=[held("MKT-A", -80)])

        with pytest.raises(RiskViolation):
            risk.place_order("MKT-A", "no", count=21, price_cents=38)

        assert client.created == []

    def test_an_order_that_offsets_an_inherited_position_is_allowed(self):
        """
        Quoting the other side of an inherited position reduces worst-case
        exposure. A limit that summed magnitudes would block the one order that
        makes the book safer.
        """
        risk, client, _ = make_risk(per_market=100, positions=[held("MKT-A", -80)])

        risk.place_order("MKT-A", "yes", count=21, price_cents=60)

        assert len(client.created) == 1
        assert risk.exposure("MKT-A") == 80

    def test_inherited_positions_consume_the_aggregate_cap(self):
        risk, client, _ = make_risk(
            per_market=100, aggregate=150,
            positions=[held("MKT-A", 100), held("MKT-B", 40)],
        )

        with pytest.raises(RiskViolation) as excinfo:
            risk.place_order("MKT-B", "yes", count=20, price_cents=60)

        assert excinfo.value.kind == "position_aggregate"
        assert client.created == []

    def test_it_follows_the_cursor_to_the_last_page(self):
        risk, client, _ = make_risk(reconcile=False)
        client.position_pages = [
            [held("MKT-A", 10)], [held("MKT-B", 20)], [held("MKT-C", 30)],
        ]

        risk.reconcile()

        assert risk.aggregate_exposure() == 60
        assert len(client.get_positions_calls) == 3

    def test_flat_positions_do_not_affect_exposure(self):
        risk, _, _ = make_risk(
            reconcile=False, positions=[held("MKT-A", 0), held("MKT-B", 50)],
        )

        risk.reconcile()

        assert risk.aggregate_exposure() == 50

    def test_no_positions_is_the_normal_clean_start(self):
        risk, _, _ = make_risk(reconcile=False)

        risk.reconcile()

        assert risk.aggregate_exposure() == 0
        assert risk.reconciled

    def test_a_positions_failure_blocks_quoting(self):
        """
        Exposure we cannot read is exposure we cannot limit. Blocked is the
        only honest state.
        """
        risk, client, _ = make_risk(reconcile=False)
        client.fail_positions = True

        risk.reconcile()

        assert not risk.reconciled
        with pytest.raises(RiskViolation) as excinfo:
            risk.place_order("MKT-A", "yes", count=10, price_cents=60)
        assert excinfo.value.kind == "not_reconciled"
        assert client.created == []

    def test_a_positions_failure_still_cancels_stale_orders(self):
        """
        Cancelling is always risk-reducing, so it must not be skipped just
        because the positions read failed.
        """
        risk, client, _ = make_risk(
            reconcile=False, open_orders=[resting("stale-1")],
        )
        client.fail_positions = True

        risk.reconcile()

        assert client.cancelled == ["stale-1"]
        assert not risk.reconciled

    def test_v2_order_fields_parse(self):
        """
        Observed through a failed cancel, which is the one public path that
        leaves a discovered order in the registry to inspect.
        """
        risk, client, _ = make_risk(
            reconcile=False, open_orders=[resting("stale-1", count=50, price=60)],
        )
        client.fail_cancel_ids = {"stale-1"}

        risk.reconcile()

        order = risk.working_orders["stale-1"]
        assert order.side == "yes"
        assert order.count == 50
        assert order.price_cents == 60

    def test_v2_no_side_order_parses(self):
        risk, client, _ = make_risk(
            reconcile=False,
            open_orders=[resting("stale-1", side="no", count=25, price=38)],
        )
        client.fail_cancel_ids = {"stale-1"}

        risk.reconcile()

        order = risk.working_orders["stale-1"]
        assert order.side == "no"
        assert order.count == 25
        assert order.price_cents == 38

    def test_legacy_v1_order_fields_still_parse(self):
        """Tolerated on purpose: the field rename could not be verified live."""
        risk, client, _ = make_risk(
            reconcile=False,
            open_orders=[resting_v1("stale-1", count=50, price=60)],
        )
        client.fail_cancel_ids = {"stale-1"}

        risk.reconcile()

        order = risk.working_orders["stale-1"]
        assert order.side == "yes"
        assert order.count == 50
        assert order.price_cents == 60

    def test_centicount_spelling_parses(self):
        """
        The published field list says remaining_count_fp, the changelog says
        remaining_centicount_fp. Both are read; a count that silently reads as
        0 would under-count exposure.
        """
        risk, client, _ = make_risk(reconcile=False)
        client.open_orders = [{
            "order_id": "stale-1", "ticker": "MKT-A", "outcome_side": "yes",
            "status": "resting", "remaining_centicount_fp": "5000",
            "yes_price_dollars": "0.6000",
        }]
        client.fail_cancel_ids = {"stale-1"}

        risk.reconcile()

        assert risk.working_orders["stale-1"].count == 50

    def test_outcome_side_beats_the_misleading_legacy_side_field(self):
        """
        A real v2 no-side order carries side="yes" next to outcome_side="no".
        Trusting `side` would flip the sign of every no position, so this is
        the guard: the fixture above ships that exact contradiction.
        """
        risk, client, _ = make_risk(
            reconcile=False,
            open_orders=[resting("stale-1", side="no", count=10, price=38)],
        )
        client.fail_cancel_ids = {"stale-1"}

        risk.reconcile()

        assert risk.working_orders["stale-1"].side == "no"

    def test_book_side_is_used_when_outcome_side_is_missing(self):
        risk, client, _ = make_risk(reconcile=False)
        client.open_orders = [{
            "order_id": "stale-1", "ticker": "MKT-A", "book_side": "ask",
            "status": "resting", "remaining_count_fp": "10.00",
            "no_price_dollars": "0.3800",
        }]
        client.fail_cancel_ids = {"stale-1"}

        risk.reconcile()

        order = risk.working_orders["stale-1"]
        assert order.side == "no"
        assert order.price_cents == 38

    def test_reconciliation_can_be_retried_after_a_positions_failure(self):
        risk, client, _ = make_risk(reconcile=False, positions=[held("MKT-A", 80)])
        client.fail_positions = True
        risk.reconcile()
        assert not risk.reconciled

        client.fail_positions = False
        risk.reconcile()

        assert risk.reconciled
        assert risk.position("MKT-A") == 80
