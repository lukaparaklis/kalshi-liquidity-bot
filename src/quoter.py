"""
quoter.py
---------
Quote planning and paper mode.

Paper mode reads PRODUCTION order books and writes down what it would have
quoted. It places nothing. Demo has no liquidity worth quoting against — the
deepest demo book reaches 45 contracts against a 200-contract threshold — so
paper against demo would measure nothing at all.

That means paper mode runs against the environment where real money lives,
which is why "it doesn't call create_order" is not good enough. Three
independent things have to hold:

  1. The client is a ReadOnlyClient, whose write path raises before a request
     is even built.
  2. PaperQuoter refuses to construct with anything else.
  3. There is no order-placing code here at all — no RiskManager, no client
     write call, nothing to reach even by mistake.

Any one of those would probably do. All three cost nothing.

Where we quote
--------------
At the Reference Price on both sides, never at top of book. Orders at or better
than the Reference Price earn full credit, so the top of book earns exactly the
same reward while filling far more often. That gap is the entire edge, and this
module exists to measure whether it survives contact with a real book.

The log
-------
JSONL, one record per market per poll, carrying the full book snapshot. Verbose
on purpose: monitor.py replays these to estimate fill rate, and a summarised
log cannot answer a question you did not know to ask when you wrote it.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .auth import KalshiAPIError, ReadOnlyClient, log
from .book import OrderBook, ReferencePrice
from .economics import (
    TARGET_SIZE_ASSUMPTION,
    mid_cents,
    qualifying_depth,
    spread_cents,
    two_sided_ok,
)

DEFAULT_LOG_PATH = Path("paper_quotes.jsonl")
DEFAULT_SIZE = 100
DEFAULT_INTERVAL_S = 10.0
REQUOTE_THRESHOLD_TICKS = 1


@dataclass(frozen=True)
class IntendedQuote:
    """A quote we would have rested. Never sent anywhere."""

    side: str
    price_cents: int
    size: int
    reference_price: int
    ticks_below_top: int    # how far below best bid we would rest


@dataclass
class QuotePlan:
    ticker: str
    quotes: list[IntendedQuote] = field(default_factory=list)
    skipped_reason: str | None = None

    @property
    def quotable(self) -> bool:
        return self.skipped_reason is None


def plan_quotes(book: OrderBook, target_size: int, size: int,
                ticker: str = "") -> QuotePlan:
    """
    Where we would rest, given this book.

    Both sides or neither: a snapshot is excluded unless both sides meet Target
    Size, so a one-sided quote earns nothing while still carrying fill risk.
    """
    plan = QuotePlan(ticker=ticker)

    refs: dict[str, ReferencePrice] = {}
    for side in ("yes", "no"):
        ref = book.reference_price(side, target_size)
        if ref is None:
            plan.skipped_reason = (
                f"{side} book never reaches 1/5 of Target Size ({target_size})"
            )
            return plan
        refs[side] = ref

    for side, ref in refs.items():
        best = book.levels(side)[0][0]
        plan.quotes.append(IntendedQuote(
            side=side,
            price_cents=ref.price,
            size=size,
            reference_price=ref.price,
            ticks_below_top=best - ref.price,
        ))
    return plan


class RequoteTracker:
    """
    Requote hysteresis (safety rule 7).

    The rule is "only move a quote if the Reference Price moved by >= 1 tick",
    and the substance is what it compares. Book deltas arrive constantly; the
    Reference Price changes rarely. Tracking the price we are resting at rather
    than the state of the book is what stops a requote on every delta from
    burning the rate limit and churning fees.
    """

    def __init__(self, threshold_ticks: int = REQUOTE_THRESHOLD_TICKS):
        self.threshold_ticks = threshold_ticks
        self._resting: dict[tuple[str, str], int] = {}

    def should_requote(self, ticker: str, side: str, price_cents: int) -> bool:
        current = self._resting.get((ticker, side))
        if current is None:
            return True
        return abs(price_cents - current) >= self.threshold_ticks

    def record(self, ticker: str, side: str, price_cents: int) -> None:
        self._resting[(ticker, side)] = price_cents

    def resting_price(self, ticker: str, side: str) -> int | None:
        return self._resting.get((ticker, side))


class PaperQuoter:
    """
    Reads books, logs intended quotes, places nothing.

    Requires a ReadOnlyClient. Passing an ordinary KalshiClient raises at
    construction rather than at the first order — because there is no first
    order, and a mistake here should surface before any polling starts.
    """

    def __init__(self, client, *, size: int = DEFAULT_SIZE,
                 target_size: int = TARGET_SIZE_ASSUMPTION,
                 log_path: Path = DEFAULT_LOG_PATH,
                 requote_threshold_ticks: int = REQUOTE_THRESHOLD_TICKS):
        if not isinstance(client, ReadOnlyClient):
            raise TypeError(
                "PaperQuoter requires a ReadOnlyClient. Paper mode reads "
                "production books, and a client that merely intends not to "
                "write is not a safety property. Use build_readonly_prod_client()."
            )
        self.client = client
        self.size = size
        self.target_size = target_size
        self.log_path = Path(log_path)
        self.requotes = RequoteTracker(requote_threshold_ticks)
        self.polls = 0

    # --- reading ---------------------------------------------------------

    def fetch_book(self, ticker: str) -> OrderBook:
        return OrderBook.from_api(
            self.client.get(f"/markets/{ticker}/orderbook", params={"depth": 20})
        )

    def snapshot(self, ticker: str) -> dict:
        """One market, one moment: what the book was and what we would do."""
        book = self.fetch_book(ticker)
        plan = plan_quotes(book, self.target_size, self.size, ticker)

        moved = [q.side for q in plan.quotes
                 if self.requotes.should_requote(ticker, q.side, q.price_cents)]
        for quote in plan.quotes:
            self.requotes.record(ticker, quote.side, quote.price_cents)

        yes_ref = book.reference_price("yes", self.target_size)
        no_ref = book.reference_price("no", self.target_size)

        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "target_size": self.target_size,
            "target_size_is_an_assumption": True,
            "size": self.size,
            "reference_price": {
                "yes": {"price": yes_ref.price, "depth": yes_ref.depth} if yes_ref else None,
                "no": {"price": no_ref.price, "depth": no_ref.depth} if no_ref else None,
            },
            "quotes": [asdict(q) for q in plan.quotes],
            "requoted_sides": moved,
            "skipped_reason": plan.skipped_reason,
            "best": {
                "yes_bid": book.yes[0][0] if book.yes else None,
                "no_bid": book.no[0][0] if book.no else None,
                "spread": spread_cents(book),
                "mid": mid_cents(book),
            },
            "qualifying_depth": {
                "yes": qualifying_depth(book, "yes", self.target_size),
                "no": qualifying_depth(book, "no", self.target_size),
            },
            "two_sided_ok": two_sided_ok(book, self.target_size),
            # Full book, on purpose: the replay can only answer questions the
            # snapshot preserved.
            "book": {"yes": book.yes, "no": book.no},
        }

    # --- logging ---------------------------------------------------------

    def write(self, record: dict) -> None:
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def poll(self, tickers: list[str]) -> list[dict]:
        records = []
        for ticker in tickers:
            try:
                record = self.snapshot(ticker)
            except KalshiAPIError as exc:
                log(f"  {ticker}: {exc.status} — skipped this poll")
                continue
            self.write(record)
            records.append(record)
        self.polls += 1
        return records

    def run(self, tickers: list[str], *, interval: float = DEFAULT_INTERVAL_S,
            duration: float | None = None) -> int:
        """
        Poll until `duration` seconds have passed, or forever if None.

        Ctrl-C is a clean stop: there is nothing resting to cancel, which is
        the whole point of paper mode.
        """
        log(f"paper mode: {len(tickers)} market(s), every {interval:.0f}s, "
            f"logging to {self.log_path}")
        log(f"Target Size ASSUMED {self.target_size} — not published per market")
        log(f"env={self.client.creds.env} via {type(self.client).__name__} "
            f"(writes raise; nothing can be placed)")

        started = time.monotonic()
        try:
            while duration is None or time.monotonic() - started < duration:
                records = self.poll(tickers)
                quotable = sum(1 for r in records if not r["skipped_reason"])
                requoted = sum(1 for r in records if r["requoted_sides"])
                log(f"poll {self.polls}: {quotable}/{len(records)} quotable, "
                    f"{requoted} requoted")
                if duration is not None and time.monotonic() - started >= duration:
                    break
                time.sleep(interval)
        except KeyboardInterrupt:
            log("stopped (nothing was resting)")
        log(f"{self.polls} polls written to {self.log_path}")
        return 0


def main() -> int:
    import argparse

    from .auth import build_readonly_prod_client

    parser = argparse.ArgumentParser(
        description="Paper mode: read prod books, log intended quotes, place nothing.",
    )
    parser.add_argument("--ticker", action="append", dest="tickers",
                        help="market to quote; repeatable")
    parser.add_argument("--top", type=int,
                        help="instead of --ticker, quote the top N markets the "
                             "scanner ranks on prod")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE_ASSUMPTION,
                        help="ASSUMED; Kalshi does not publish it per market")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    parser.add_argument("--duration", type=float,
                        help="seconds to run; omit to run until Ctrl-C")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    args = parser.parse_args()

    # The only client paper mode is allowed to hold.
    client = build_readonly_prod_client()

    tickers = args.tickers or []
    if args.top:
        from .scanner import scan_markets
        ranked = scan_markets(client, my_size=args.size, hours=24.0,
                              probe=max(args.top * 3, 30),
                              target_size=args.target_size)
        tickers = [s.ticker for s in ranked if s.quotable][:args.top]
        if not tickers:
            log("scanner found no quotable market at this Target Size; "
                "nothing to paper-trade")
            return 1
    if not tickers:
        parser.error("pass --ticker TICKER (repeatable) or --top N")

    quoter = PaperQuoter(client, size=args.size, target_size=args.target_size,
                         log_path=args.log)
    return quoter.run(tickers, interval=args.interval, duration=args.duration)


if __name__ == "__main__":
    import sys
    sys.exit(main())
