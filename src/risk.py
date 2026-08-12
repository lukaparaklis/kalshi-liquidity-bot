"""
risk.py
-------
The safety layer. Written to satisfy tests/test_risk_limits.py, which was
written first.

CLAUDE.md's SAFETY RULES are non-negotiable and must not be relaxed to make a
feature work. Four are implemented here:

  1. Cancel-all on shutdown, including reconciliation of orders left resting by
     a previous process.
  2. Dead-man's switch: no loop iteration in 60s cancels everything.
  3. Hard position limits per market and in aggregate, checked before every
     order.
  4. Daily loss limit that halts new quoting for the session.

RiskManager is a GATEWAY, not an advisor. Orders are placed through it, so
"checked before every order" is structural rather than a convention that
quoter.py has to remember. Nothing in this project should call
client.create_order directly.

Startup reconciliation covers both halves of inherited exposure: resting ORDERS
from GET /portfolio/orders (cancelled) and filled POSITIONS from
GET /portfolio/positions (seeded into the limits). Cancelling orders alone
would leave a position that already filled invisible to the caps.

This module deliberately does NOT compute P&L. monitor.py drives record_pnl(),
so the daily loss limit is only as good as its caller.

The client is duck-typed. It must provide create_order(ticker, side, count,
price_cents), cancel_order(order_id), get_orders(status, cursor) and
get_positions(cursor).
"""

from __future__ import annotations

import atexit
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .auth import log

SIDES = ("yes", "no")
RESTING = "resting"

# Guards against a server that keeps handing back a cursor forever.
MAX_RECONCILE_PAGES = 100


@dataclass(frozen=True)
class RiskLimits:
    """
    Hard limits. Position limits are worst-case potential exposure, not filled
    position: resting size can fill, so a limit that ignored it would be
    breached by the very next fill.
    """

    max_position_per_market: int
    max_position_aggregate: int
    max_daily_loss_cents: int
    deadman_timeout_s: float = 60.0


class RiskViolation(Exception):
    """
    A refusal to place an order. `kind` is one of "not_reconciled", "halted",
    "position_per_market", "position_aggregate" — so a log at 3am says which
    limit fired without anyone having to parse prose.
    """

    def __init__(self, kind: str, message: str):
        self.kind = kind
        super().__init__(message)


@dataclass
class WorkingOrder:
    order_id: str
    ticker: str
    side: str
    count: int          # remaining, unfilled
    price_cents: int


@dataclass
class CancelAllResult:
    cancelled: list[str] = field(default_factory=list)
    failed: list[tuple[str, Exception]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed


class RiskManager:
    def __init__(self, limits: RiskLimits, *, client,
                 clock: Callable[[], float] = time.monotonic):
        self.limits = limits
        self.client = client
        self._clock = clock

        self._working: dict[str, WorkingOrder] = {}
        self._positions: dict[str, int] = {}
        self._daily_pnl_cents = 0
        self._halted = False
        self._halt_reason: str | None = None
        self._reconciled = False
        self._last_tick = clock()

        # Re-entrant on purpose. A signal handler runs on the main thread, so
        # SIGINT arriving while cancel_all holds a plain Lock would deadlock the
        # one code path that must never hang. Network calls are made outside the
        # lock regardless.
        self._lock = threading.RLock()

    # --- state -----------------------------------------------------------

    @property
    def halted(self) -> bool:
        with self._lock:
            return self._halted

    @property
    def halt_reason(self) -> str | None:
        with self._lock:
            return self._halt_reason

    @property
    def reconciled(self) -> bool:
        with self._lock:
            return self._reconciled

    @property
    def working_orders(self) -> dict[str, WorkingOrder]:
        with self._lock:
            return dict(self._working)

    @property
    def daily_pnl_cents(self) -> int:
        with self._lock:
            return self._daily_pnl_cents

    def position(self, ticker: str) -> int:
        """Filled position, signed: positive is long yes, negative is long no."""
        with self._lock:
            return self._positions.get(ticker, 0)

    def exposure(self, ticker: str) -> int:
        with self._lock:
            return self._exposure(ticker)

    def aggregate_exposure(self) -> int:
        with self._lock:
            return self._aggregate_exposure()

    # --- exposure --------------------------------------------------------

    def _exposure(self, ticker: str, extra_side: str | None = None,
                  extra_count: int = 0) -> int:
        """
        Worst-case contracts at risk in one market, optionally including a
        hypothetical order.

        The worst case is one side of our quotes filling, not both: resting 100
        yes and 100 no is exposure of 100, not 200. Summing both sides would
        reject two-sided quoting, which the reward program makes mandatory —
        snapshots are excluded unless both sides are present.
        """
        position = self._positions.get(ticker, 0)
        resting_yes = extra_count if extra_side == "yes" else 0
        resting_no = extra_count if extra_side == "no" else 0
        for order in self._working.values():
            if order.ticker != ticker:
                continue
            if order.side == "yes":
                resting_yes += order.count
            else:
                resting_no += order.count
        return max(abs(position + resting_yes), abs(position - resting_no))

    def _aggregate_exposure(self, ticker: str | None = None,
                            extra_side: str | None = None,
                            extra_count: int = 0) -> int:
        tickers = set(self._positions) | {o.ticker for o in self._working.values()}
        if ticker is not None:
            tickers.add(ticker)
        return sum(
            self._exposure(t, extra_side, extra_count) if t == ticker else self._exposure(t)
            for t in tickers
        )

    # --- the order gateway ------------------------------------------------

    def place_order(self, ticker: str, side: str, count: int,
                    price_cents: int) -> WorkingOrder:
        """
        The only sanctioned path to the exchange. Raises RiskViolation without
        contacting the exchange if any limit would be breached.
        """
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        if count <= 0:
            raise ValueError(f"count must be positive, got {count!r}")

        # Defence in depth: if the watchdog is what hung, the order path still
        # refuses to quote into a stale loop. Outside the lock because firing
        # cancels orders over the network.
        self.check_deadman()

        with self._lock:
            if not self._reconciled:
                raise RiskViolation(
                    "not_reconciled",
                    "startup reconciliation has not succeeded; refusing to quote",
                )
            if self._halted:
                raise RiskViolation("halted", f"halted ({self._halt_reason})")
            self._check_position_limits(ticker, side, count)

        response = self.client.create_order(
            ticker=ticker, side=side, count=count, price_cents=price_cents,
        )
        order_id = ((response or {}).get("order") or {}).get("order_id")
        if not order_id:
            # The order may or may not exist on the exchange. Say so loudly:
            # this is the one case where our view can silently diverge.
            raise RuntimeError(f"create_order returned no order_id: {response!r}")

        order = WorkingOrder(order_id, ticker, side, count, price_cents)
        with self._lock:
            self._working[order_id] = order
        return order

    def _check_position_limits(self, ticker: str, side: str, count: int) -> None:
        projected = self._exposure(ticker, side, count)
        if projected > self.limits.max_position_per_market:
            raise RiskViolation(
                "position_per_market",
                f"{ticker}: {projected} contracts would exceed per-market limit "
                f"{self.limits.max_position_per_market}",
            )

        projected_total = self._aggregate_exposure(ticker, side, count)
        if projected_total > self.limits.max_position_aggregate:
            raise RiskViolation(
                "position_aggregate",
                f"{projected_total} contracts would exceed aggregate limit "
                f"{self.limits.max_position_aggregate}",
            )

    # --- fills and P&L ----------------------------------------------------

    def record_fill(self, order_id: str, count: int) -> None:
        """Move `count` contracts from resting to position."""
        with self._lock:
            order = self._working.get(order_id)
            if order is None:
                log(f"fill for unknown order {order_id}; ignoring")
                return
            filled = min(count, order.count)
            order.count -= filled
            signed = filled if order.side == "yes" else -filled
            self._positions[order.ticker] = self._positions.get(order.ticker, 0) + signed
            if order.count <= 0:
                del self._working[order_id]

    def record_pnl(self, cents: int) -> None:
        """Apply a realized P&L delta and halt if the daily limit is reached."""
        with self._lock:
            self._daily_pnl_cents += cents
            breached = self._daily_pnl_cents <= -self.limits.max_daily_loss_cents
        if breached:
            self.halt("daily_loss_limit")

    def halt(self, reason: str) -> None:
        """
        Stop new quoting for the session. Latched: it is never cleared, so a
        P&L recovery cannot resume quoting on a bounce, and a hung loop that
        starts ticking again cannot silently revive itself. Restarting is a
        human decision.

        Halted means "place nothing". It never blocks cancelling.
        """
        with self._lock:
            if self._halted:
                return
            self._halted = True
            self._halt_reason = reason
        log(f"HALTED: {reason}")

    # --- dead-man's switch ------------------------------------------------

    def tick(self) -> None:
        """Call once per completed main-loop iteration."""
        with self._lock:
            self._last_tick = self._clock()

    def check_deadman(self) -> bool:
        """
        True if the loop has gone stale. Firing both cancels and halts:
        cancelling alone would let the next iteration re-quote into the same
        hung condition.
        """
        with self._lock:
            elapsed = self._clock() - self._last_tick
            stale = elapsed >= self.limits.deadman_timeout_s
            already_fired = self._halt_reason == "deadman"

        if not stale:
            return False
        if already_fired:
            return True

        log(f"DEAD-MAN'S SWITCH: no loop iteration in {elapsed:.1f}s")
        self.halt("deadman")
        self.cancel_all()
        return True

    # --- cancellation -----------------------------------------------------

    def cancel_all(self) -> CancelAllResult:
        """
        Cancel every order we know is resting. Safe to call twice: SIGINT and
        atexit can both fire on one shutdown, and Kalshi rejects cancelling an
        already-cancelled order, which would bury real errors in noise.

        Orders are claimed out of the registry before any network call, so a
        re-entrant call (a signal landing mid-cancellation) sees an empty
        registry and no-ops instead of cancelling everything twice.
        """
        with self._lock:
            claimed = dict(self._working)
            self._working.clear()

        result = self._cancel_each(claimed)

        if result.failed:
            # A failed cancel may still be live. Keep tracking it so a retry,
            # or the next cancel_all, picks it up. Forgetting it would make the
            # bot believe it is flat when it is not.
            with self._lock:
                for order_id, _ in result.failed:
                    self._working[order_id] = claimed[order_id]
        return result

    def _cancel_each(self, orders: dict[str, WorkingOrder]) -> CancelAllResult:
        """One failure must never strand the remaining orders on the exchange."""
        result = CancelAllResult()
        for order_id in list(orders):
            try:
                self.client.cancel_order(order_id)
            except Exception as exc:  # noqa: BLE001 - keep going, always
                log(f"cancel failed for {order_id}: {exc}")
                result.failed.append((order_id, exc))
            else:
                result.cancelled.append(order_id)
        return result

    # --- startup reconciliation -------------------------------------------

    def reconcile(self) -> CancelAllResult:
        """
        Adopt whatever a previous process left behind, then unblock quoting.

        Two kinds of inherited exposure, and both matter:
          - Positions that already filled. Seeded into the limits, so a session
            that ended holding 80 contracts cannot be topped up with another
            100 by a manager that started from zero.
          - Orders still resting. Cancelled. A fresh RiskManager has no memory
            of them, so local cancel-all cannot see them.

        Until this succeeds, place_order refuses — structural, so main.py
        cannot forget to call it. It stays blocked if positions could not be
        read (exposure we cannot see is exposure we cannot limit) or if any
        cancel failed (a stale order that would not cancel is live exposure of
        unknown size, and quoting on top of it compounds a position nobody
        chose).

        Positions are read first, but a failure there does not skip the
        cancels: cancelling is always risk-reducing. A failure to LIST orders
        propagates instead — not knowing what is resting is not a state to
        continue from quietly.
        """
        positions_ok = self._seed_positions()

        discovered = self._fetch_resting_orders()
        if discovered:
            log(f"reconcile: {len(discovered)} order(s) resting from a previous session")

        result = self._cancel_each(discovered)

        with self._lock:
            for order_id in result.cancelled:
                self._working.pop(order_id, None)
            for order_id, _ in result.failed:
                self._working[order_id] = discovered[order_id]
            self._reconciled = positions_ok and not result.failed

        if result.failed:
            log(f"reconcile INCOMPLETE: {len(result.failed)} order(s) still resting; "
                "quoting stays blocked")
        return result

    def _seed_positions(self) -> bool:
        """
        Replace the position map with what the exchange reports. False if it
        could not be read, which blocks quoting.

        A wholesale replace, not a merge: the exchange is the truth at startup,
        and a stale local entry would silently under-count exposure.
        """
        try:
            found = self._fetch_positions()
        except Exception as exc:  # noqa: BLE001 - any failure here blocks quoting
            log(f"reconcile: could not read positions ({exc}); quoting stays blocked")
            return False

        with self._lock:
            self._positions = found
        if found:
            summary = ", ".join(f"{t}={p:+d}" for t, p in sorted(found.items()))
            log(f"reconcile: inherited {len(found)} position(s): {summary}")
        return True

    def _fetch_positions(self) -> dict[str, int]:
        """
        Every page of positions, signed: positive long yes, negative long no.

        Flat positions are dropped — the API reports settled and closed markets
        too, and they are not exposure.
        """
        found: dict[str, int] = {}
        cursor: str | None = None
        seen_cursors: set[str] = set()

        for _ in range(MAX_RECONCILE_PAGES):
            page = self.client.get_positions(cursor=cursor) or {}
            for raw in page.get("market_positions") or []:
                ticker = raw.get("ticker")
                position = int(raw.get("position") or 0)
                if ticker and position:
                    found[str(ticker)] = position

            cursor = page.get("cursor") or ""
            if not cursor or cursor in seen_cursors:
                break
            seen_cursors.add(cursor)
        else:
            log(f"reconcile: stopped after {MAX_RECONCILE_PAGES} position pages")

        return found

    def _fetch_resting_orders(self) -> dict[str, WorkingOrder]:
        """
        Every page of resting orders. Pagination is the silent version of this
        bug: cancel page one, report success, leave page two on the exchange.
        """
        discovered: dict[str, WorkingOrder] = {}
        cursor: str | None = None
        seen_cursors: set[str] = set()

        for _ in range(MAX_RECONCILE_PAGES):
            page = self.client.get_orders(status=RESTING, cursor=cursor) or {}
            for raw in page.get("orders") or []:
                # Filtered locally as well as by the query parameter: the
                # server-side filter is not something to bet the account on.
                if str(raw.get("status", "")).lower() != RESTING:
                    continue
                order = self._parse_order(raw)
                if order is not None:
                    discovered[order.order_id] = order

            cursor = page.get("cursor") or ""
            if not cursor or cursor in seen_cursors:
                break
            seen_cursors.add(cursor)
        else:
            log(f"reconcile: stopped after {MAX_RECONCILE_PAGES} pages")

        return discovered

    @staticmethod
    def _parse_order(raw: dict) -> WorkingOrder | None:
        order_id = raw.get("order_id")
        if not order_id:
            return None
        side = str(raw.get("side") or "yes").lower()
        count = raw.get("remaining_count")
        if count is None:
            count = raw.get("count") or 0
        price = raw.get("yes_price" if side == "yes" else "no_price") or 0
        return WorkingOrder(
            order_id=str(order_id),
            ticker=str(raw.get("ticker") or ""),
            side=side if side in SIDES else "yes",
            count=int(count),
            price_cents=int(price),
        )

    # --- shutdown ---------------------------------------------------------

    def install_shutdown_handlers(self) -> None:
        """
        Rule 1: register SIGINT/SIGTERM handlers AND an atexit hook, because if
        the process dies the orders stay live on the exchange.

        Both can fire on a single shutdown; cancel_all is idempotent, which is
        precisely why that is tested. Not covered by the test suite — driving
        real signals under pytest is unreliable — so treat changes here with
        suspicion and verify by hand.
        """
        atexit.register(self._on_exit)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._on_signal)

    def _on_signal(self, signum, _frame) -> None:
        log(f"signal {signum}: cancelling all resting orders")
        self.halt(f"signal_{signum}")
        self.cancel_all()
        raise SystemExit(128 + signum)

    def _on_exit(self) -> None:
        if self._working:
            log("atexit: cancelling all resting orders")
            self.cancel_all()
