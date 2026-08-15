#!/usr/bin/env python3
"""
verify_safety.py
----------------
Manual verification of the two safety properties the unit tests cannot reach:
shutdown cancellation against a real exchange, and the market_positions parse
against a real filled position.

    python scripts/verify_safety.py status
    python scripts/verify_safety.py rest [--ticker T] [--price 2] [--deadman]
    python scripts/verify_safety.py fill [--ticker T] [--count 1]

DEMO ONLY. Refuses to run against prod, deliberately: `fill` crosses the spread
on purpose, which CLAUDE.md rejects as strategy and which costs real money on a
live account. There is no flag to override this. If you ever need it on prod,
make that a considered change to this file, not a command-line decision at 2am.

What each mode is for
---------------------
status  Read-only. Prints balance, resting orders and positions as the API
        returns them. Never cancels anything, so it is safe to run while a
        `rest` process is live — which is the point: it is how you confirm an
        order really is gone after shutdown.

rest    Places one contract far below the market so it will not fill, prints
        the order id and PID, then sleeps. THIS IS THE TEST: send SIGINT
        (Ctrl-C) or SIGTERM (`kill <pid>`) and the shutdown handler must cancel
        the order. Then run `status` from another shell and confirm the book is
        empty. install_shutdown_handlers() is the one part of the safety layer
        with no automated coverage, because driving real signals under pytest
        is unreliable. This is how it gets verified.

        --deadman stops ticking the loop so the dead-man's switch fires on its
        own after the timeout and cancels without any signal at all.

fill    Crosses the spread to get a genuine position, then reconciles a fresh
        RiskManager against it. This closes the last unverified assumption in
        risk.py: that GET /portfolio/positions returns a signed integer under
        `position` inside `market_positions`. If that field is named
        differently or is unsigned, inherited exposure silently reads as zero
        and the position limits are enforcing nothing.

Known obstacles
---------------
1. THE CURRENT API KEY CANNOT TRADE. Both order endpoints return
   403 insufficient_scope: write::trade required. `status` works; `rest` and
   `fill` cannot run until a demo key with trade permissions exists. This is
   also why the v2 payload shapes in auth.py are documented-but-unverified.
2. Demo books are frequently empty. `fill` needs a resting ask to cross, and
   says so plainly rather than resting a 99c order and hoping something hits
   it. If nothing is crossable, place a sell by hand in the demo UI and pass
   --ticker.
3. Balance is checked first and printed by every mode, so an underfunded
   account fails fast instead of surfacing as an opaque order rejection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.auth import KalshiAPIError, build_client, log  # noqa: E402
from src.risk import RiskLimits, RiskManager  # noqa: E402

# Far below any real bid: a buy-yes here fills only if someone dumps at 2c.
RESTING_PRICE_CENTS = 2

# Deliberately tiny. This script exists to exercise safety paths, not to trade.
LIMITS = RiskLimits(
    max_position_per_market=5,
    max_position_aggregate=5,
    max_daily_loss_cents=100,
    deadman_timeout_s=60.0,
)

MARKET_SCAN_LIMIT = 200     # per page
MARKET_SCAN_PAGES = 4       # /markets is dominated by MVE markets; one page is not enough
MARKET_PROBE_LIMIT = 30     # orderbook fetches, one request each


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------

def preflight(client) -> None:
    """Refuse prod, and surface the balance before anything is attempted."""
    if client.creds.is_prod:
        sys.exit(
            "REFUSING TO RUN: KALSHI_ENV=prod.\n"
            "This script places orders and crosses the spread on purpose. It is "
            "a demo verification tool and has no prod mode by design."
        )

    balance = client.balance().get("balance")
    log(f"env={client.creds.env}  balance={balance} cents")
    if not balance:
        log("WARNING: balance is 0 — every order below will be rejected for "
            "insufficient funds. Top up the demo account first.")


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _levels(raw, *, dollars: bool) -> list[tuple[int, int]]:
    """
    Normalise one side of a book to [(price_cents, size), ...], best first.

    v2 returns dollar strings and fixed-point sizes: [["0.0100", "2.00"], ...].
    Legacy returned integer cents. The two are indistinguishable by value alone
    (60 cents vs "0.6000" dollars), so the caller says which it read.
    """
    out = []
    for entry in raw or []:
        cents = round(_number(entry[0]) * 100) if dollars else int(_number(entry[0]))
        count = int(_number(entry[1]))
        if count > 0 and cents > 0:
            out.append((cents, count))
    return sorted(out, key=lambda level: -level[0])


def orderbook(client, ticker: str) -> tuple[list, list]:
    """
    (yes_levels, no_levels) in cents, best first.

    v2 nests the book under "orderbook_fp" as yes_dollars/no_dollars. The old
    "orderbook" -> yes/no shape is read as a fallback. Reading the wrong key
    yields an empty book, which looks like "no liquidity anywhere" rather than
    like an error — which is exactly how the old picker went wrong.
    """
    response = client.get(f"/markets/{ticker}/orderbook", params={"depth": 10})

    book = response.get("orderbook_fp")
    if book is not None:
        return (_levels(book.get("yes_dollars"), dollars=True),
                _levels(book.get("no_dollars"), dollars=True))

    book = response.get("orderbook") or {}
    return (_levels(book.get("yes"), dollars=False),
            _levels(book.get("no"), dollars=False))


def open_markets(client) -> list[dict]:
    """Open markets across several pages; one page is not the whole exchange."""
    markets: list[dict] = []
    cursor = None
    for _ in range(MARKET_SCAN_PAGES):
        params = {"status": "open", "limit": MARKET_SCAN_LIMIT}
        if cursor:
            params["cursor"] = cursor
        data = client.get("/markets", params=params)
        markets += data.get("markets") or []
        cursor = data.get("cursor") or ""
        if not cursor:
            break
    return markets


@dataclass
class Candidate:
    ticker: str
    volume: float
    yes_levels: list
    no_levels: list

    @property
    def yes_bid(self) -> int | None:
        return self.yes_levels[0][0] if self.yes_levels else None

    @property
    def yes_ask(self) -> int | None:
        """A no bid at P is a yes ask at 100-P."""
        return 100 - self.no_levels[0][0] if self.no_levels else None

    @property
    def ask_size(self) -> int:
        return self.no_levels[0][1] if self.no_levels else 0

    @property
    def has_book(self) -> bool:
        return bool(self.yes_levels or self.no_levels)


def survey(client) -> list[Candidate]:
    """
    Probe the busiest real markets and report what is on their books.

    Two filters carry the weight. MVE markets (mve_collection_ticker set) are
    multi-leg combination markets — tickers like KXMVECROSSCATEGORY-SHARD1-...
    — and POSTing an order against one returns 404 market_not_found. They are
    also the majority of what /markets returns first, which is how the old
    picker kept choosing them. Sorting by volume then puts markets that
    actually trade ahead of the long tail of dead ones.
    """
    markets = open_markets(client)
    if not markets:
        sys.exit("no open markets; nothing to verify against")

    plain = [m for m in markets if m.get("ticker") and not m.get("mve_collection_ticker")]
    plain.sort(key=lambda m: _number(m.get("volume_fp")), reverse=True)
    log(f"scanned {len(markets)} open markets: {len(markets) - len(plain)} MVE/combination "
        f"skipped, {len(plain)} tradeable; probing top {MARKET_PROBE_LIMIT} by volume")

    candidates = []
    for market in plain[:MARKET_PROBE_LIMIT]:
        ticker = market["ticker"]
        try:
            yes_levels, no_levels = orderbook(client, ticker)
        except KalshiAPIError as exc:
            log(f"  {ticker}: orderbook unavailable ({exc.status})")
            continue
        candidates.append(
            Candidate(ticker, _number(market.get("volume_fp")), yes_levels, no_levels)
        )
    return candidates


def report(candidates: list[Candidate], chosen: str | None = None) -> None:
    """Print what was considered, so a bad pick is visible rather than inferred."""
    print()
    print(f"  {'':2} {'ticker':<44} {'volume':>10}  {'yes_bid':>7} {'yes_ask':>7}  book")
    for candidate in candidates:
        mark = "->" if candidate.ticker == chosen else "  "
        print(f"  {mark} {candidate.ticker[:44]:<44} {candidate.volume:>10.2f}  "
              f"{str(candidate.yes_bid or '-'):>7} {str(candidate.yes_ask or '-'):>7}  "
              f"{len(candidate.yes_levels)}y/{len(candidate.no_levels)}n")
    print()


def pick_market(client, *, need_ask: bool) -> tuple[str, int | None]:
    """
    Find a usable market. Returns (ticker, best_yes_ask_cents).

    `rest` only needs a real market to rest against, so an empty book is
    acceptable — though one with a book is preferred as evidence it trades.
    `fill` needs a crossable ask, which exists only where the no side has depth.
    """
    candidates = survey(client)
    if not candidates:
        sys.exit("no tradeable markets found")

    if need_ask:
        usable = [c for c in candidates if c.yes_ask and 1 <= c.yes_ask <= 99]
        if not usable:
            report(candidates)
            sys.exit(
                "no probed market has a resting ask to cross.\n"
                "Demo books are thin. Place a resting sell by hand in the demo "
                "UI and pass --ticker, or retry when the book has depth."
            )
        best = max(usable, key=lambda c: c.ask_size)
        report(candidates, best.ticker)
        log(f"chose {best.ticker}: yes ask {best.yes_ask}c x{best.ask_size}")
        return best.ticker, best.yes_ask

    best = max(candidates, key=lambda c: (c.has_book, c.volume))
    report(candidates, best.ticker)
    log(f"chose {best.ticker} (volume {best.volume:.2f}, book "
        f"{len(best.yes_levels)}y/{len(best.no_levels)}n)")
    return best.ticker, None


def build_risk(client) -> RiskManager:
    risk = RiskManager(LIMITS, client=client)
    log("reconciling (cancels anything left resting from a previous run)...")
    result = risk.reconcile()
    if result.cancelled:
        log(f"  cancelled {len(result.cancelled)}: {result.cancelled}")
    if not risk.reconciled:
        sys.exit(f"reconciliation incomplete: {result.failed}; refusing to continue")
    log(f"  reconciled. aggregate exposure {risk.aggregate_exposure()}")
    return risk


# --------------------------------------------------------------------------
# MODES
# --------------------------------------------------------------------------

def cmd_status(client, _args) -> int:
    """Read-only. Deliberately does NOT reconcile — that would cancel."""
    orders = client.get_orders(status="resting")
    resting = orders.get("orders") or []
    log(f"resting orders: {len(resting)}")
    print(json.dumps(resting, indent=2))

    positions = client.get_positions()
    market_positions = positions.get("market_positions") or []
    log(f"market_positions: {len(market_positions)}")
    print(json.dumps(market_positions, indent=2))

    if market_positions:
        log("check that each entry has a signed integer under 'position' — "
            "risk.py depends on exactly that")
    return 0


def cmd_markets(client, _args) -> int:
    """Read-only: run the picker's survey and print it, placing nothing."""
    candidates = survey(client)
    report(candidates)
    crossable = [c for c in candidates if c.yes_ask and 1 <= c.yes_ask <= 99]
    log(f"{sum(1 for c in candidates if c.has_book)}/{len(candidates)} probed markets "
        f"have a book; {len(crossable)} have an ask `fill` could cross")
    return 0


def cmd_rest(client, args) -> int:
    ticker = args.ticker
    if not ticker:
        ticker, _ = pick_market(client, need_ask=False)

    risk = build_risk(client)
    risk.install_shutdown_handlers()

    try:
        order = risk.place_order(ticker, "yes", count=args.count,
                                 price_cents=args.price)
    except KalshiAPIError as exc:
        log(f"order rejected: {exc}")
        if exc.status == 403 and "scope" in exc.body:
            log("This API key is read-only. Create a demo key with write::trade "
                "scope; nothing that places orders can be verified without one.")
        return 1

    log(f"RESTING: order_id={order.order_id} {ticker} yes "
        f"{order.count}@{order.price_cents}c")
    log(f"PID {os.getpid()}")
    print()
    log("Now verify shutdown cancellation:")
    log("  Ctrl-C here (SIGINT), or `kill <pid>` from another shell (SIGTERM)")
    log("  then: python scripts/verify_safety.py status")
    log(f"  PASS = zero resting orders, and {order.order_id} is gone")
    if args.deadman:
        log(f"  --deadman: not ticking, so the switch should fire on its own "
            f"in ~{LIMITS.deadman_timeout_s:.0f}s")
    print()

    while True:
        if not args.deadman:
            risk.tick()
        if risk.check_deadman():
            log("dead-man's switch fired — orders cancelled, loop exiting")
            log("run `status` to confirm the book is empty")
            return 0
        time.sleep(1)


def cmd_fill(client, args) -> int:
    ticker, ask = args.ticker, None
    if ticker:
        _, no_levels = orderbook(client, ticker)
        if not no_levels:
            sys.exit(f"{ticker} has no resting ask to cross")
        ask = 100 - no_levels[0][0]
    else:
        ticker, ask = pick_market(client, need_ask=True)

    risk = build_risk(client)

    log(f"crossing: buy yes {args.count}@{ask}c on {ticker}")
    log("this is a deliberate spread-cross on DEMO to produce a real position; "
        "it is not a strategy change")
    try:
        # post_only=False is the whole point here: everywhere else the exchange
        # is asked to reject an order rather than let it cross.
        order = risk.place_order(ticker, "yes", count=args.count, price_cents=ask,
                                 post_only=False)
    except KalshiAPIError as exc:
        log(f"order rejected: {exc}")
        if exc.status == 403 and "scope" in exc.body:
            log("This API key is read-only. Create a demo key with write::trade "
                "scope; nothing that places orders can be verified without one.")
        return 1

    log(f"placed order_id={order.order_id}; polling for fill...")
    for _ in range(10):
        time.sleep(1)
        live = client.get(f"/portfolio/orders/{order.order_id}").get("order") or {}
        status = live.get("status")
        log(f"  status={status} remaining={live.get('remaining_count')}")
        if status in ("executed", "canceled"):
            break

    print()
    log("RAW GET /portfolio/positions:")
    print(json.dumps(client.get_positions(), indent=2))

    print()
    log("what risk.py parses from it:")
    verifier = RiskManager(LIMITS, client=client)
    verifier.reconcile()
    log(f"  position({ticker}) = {verifier.position(ticker)}")
    log(f"  exposure({ticker}) = {verifier.exposure(ticker)}")
    log(f"  aggregate_exposure = {verifier.aggregate_exposure()}")

    if verifier.position(ticker) == 0:
        log("FAIL: the position did not parse. Either the fill did not happen, "
            "or 'position' is not the field name risk.py expects — compare "
            "against the raw payload above.")
        return 1

    log("PASS: inherited position parsed and counted toward the limits.")
    log("NOTE: this position is still open on demo. It will keep consuming the "
        "caps until it settles or you close it by hand.")
    return 0


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    sub.add_parser("status", help="read-only: balance, resting orders, positions")
    sub.add_parser("markets", help="read-only: show what the picker considers")

    rest = sub.add_parser("rest", help="rest an unfillable order, then sleep")
    rest.add_argument("--ticker")
    rest.add_argument("--price", type=int, default=RESTING_PRICE_CENTS)
    rest.add_argument("--count", type=int, default=1)
    rest.add_argument("--deadman", action="store_true",
                      help="stop ticking so the dead-man's switch fires by itself")

    fill = sub.add_parser("fill", help="cross the spread to create a position")
    fill.add_argument("--ticker")
    fill.add_argument("--count", type=int, default=1)

    args = parser.parse_args()

    client = build_client()
    preflight(client)

    handlers = {"status": cmd_status, "markets": cmd_markets,
                "rest": cmd_rest, "fill": cmd_fill}
    return handlers[args.mode](client, args)


if __name__ == "__main__":
    raise SystemExit(main())
