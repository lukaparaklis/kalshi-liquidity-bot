"""
scanner.py
----------
Rank markets by reward-versus-fill-risk. READ-ONLY: it places nothing, cancels
nothing, and never constructs a RiskManager.

    python -m src.scanner scan
    python -m src.scanner scan --size 200 --hours 12 --probe 60
    python -m src.scanner book KXSOMETHING-26AUG21-T60

What the ranking is for
-----------------------
It answers "which markets are worth measuring", not "which markets are
profitable". Reward estimates rest on a Target Size nobody publishes and a
daily pool that is not in the API; fill cost is modelled rather than measured.
CLAUDE.md is blunt that fill rate is the metric that decides everything and
that it has to be measured. So this narrows a few hundred markets down to a
shortlist worth pointing paper mode at.

Every run prints its assumptions. A ranking that hides them invites exactly the
overconfidence the published backtests warn about — 695/1000 strategies
profitable in one 30-day window, 102/4904 a week later, same series.

Pools
-----
Daily pools are published at kalshi.com/incentives, not through the API. Paste
them into pools.json as {"TICKER_OR_PREFIX": dollars_per_day, ...}. Without it
the scanner still ranks, using reward share per dollar of modelled fill cost,
and says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .auth import KalshiAPIError, build_client, log
from .book import OrderBook
from .economics import (
    TARGET_SIZE_ASSUMPTION,
    assumptions,
    capital_required_dollars,
    estimate_fill_cost_dollars,
    estimate_reward,
    qualifying_depth,
    spread_cents,
    two_sided_ok,
)

POOLS_FILE = Path("pools.json")

MARKET_PAGE_SIZE = 200
MARKET_PAGES = 4
DEFAULT_PROBE = 30      # orderbook fetches, one request each
DEFAULT_SIZE = 100      # contracts rested per side
DEFAULT_HOURS = 24.0


# ---------------------------------------------------------------------------
# POOLS
# ---------------------------------------------------------------------------

def load_pools() -> dict[str, float]:
    if not POOLS_FILE.exists():
        return {}
    try:
        return {k: float(v) for k, v in json.loads(POOLS_FILE.read_text()).items()}
    except (ValueError, TypeError) as exc:
        log(f"{POOLS_FILE} is unreadable ({exc}); continuing without pools")
        return {}


def match_pool(ticker: str, pools: dict[str, float]) -> float | None:
    """Exact ticker first, then longest matching prefix."""
    if ticker in pools:
        return pools[ticker]
    matches = [(k, v) for k, v in pools.items() if ticker.startswith(k)]
    if not matches:
        return None
    return max(matches, key=lambda kv: len(kv[0]))[1]


# ---------------------------------------------------------------------------
# MARKET DATA
# ---------------------------------------------------------------------------

def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def fetch_open_markets(client, pages: int = MARKET_PAGES) -> list[dict]:
    """
    Open markets across several pages.

    MVE (multi-leg combination) markets are dropped. They dominate the first
    page, are not directly orderable — POSTing one returns 404 market_not_found
    — and would otherwise fill the entire shortlist.
    """
    markets: list[dict] = []
    cursor = None
    for _ in range(pages):
        params = {"status": "open", "limit": MARKET_PAGE_SIZE}
        if cursor:
            params["cursor"] = cursor
        data = client.get("/markets", params=params)
        markets += data.get("markets") or []
        cursor = data.get("cursor") or ""
        if not cursor:
            break

    tradeable = [m for m in markets
                 if m.get("ticker") and not m.get("mve_collection_ticker")]
    log(f"{len(markets)} open markets, {len(markets) - len(tradeable)} MVE skipped, "
        f"{len(tradeable)} tradeable")
    return tradeable


def fetch_book(client, ticker: str) -> OrderBook:
    return OrderBook.from_api(
        client.get(f"/markets/{ticker}/orderbook", params={"depth": 20})
    )


# ---------------------------------------------------------------------------
# SCAN
# ---------------------------------------------------------------------------

@dataclass
class Scan:
    ticker: str
    volume: float
    book: OrderBook
    target_size: int
    my_size: int
    hours: float
    pool: float | None

    def __post_init__(self):
        self.yes_ref = self.book.reference_price("yes", self.target_size)
        self.no_ref = self.book.reference_price("no", self.target_size)
        self.yes_depth = qualifying_depth(self.book, "yes", self.target_size)
        self.no_depth = qualifying_depth(self.book, "no", self.target_size)
        self.two_sided = two_sided_ok(self.book, self.target_size)
        self.spread = spread_cents(self.book)
        self.reward = estimate_reward(self.book, self.my_size, self.target_size,
                                      self.pool)
        self.fill_cost = estimate_fill_cost_dollars(
            self.book, self.my_size, self.hours, self.volume)
        self.capital = capital_required_dollars(self.book, self.my_size)

    @property
    def quotable(self) -> bool:
        """
        Both sides need a Reference Price to rest at. Without one the book is
        too thin to reach a fifth of Target Size, so there is nowhere to earn
        full credit and the snapshot would not count anyway.
        """
        return self.yes_ref is not None and self.no_ref is not None

    @property
    def net_dollars(self) -> float | None:
        if self.reward.dollars_per_day is None:
            return None
        return self.reward.dollars_per_day - self.fill_cost

    @property
    def rank_key(self) -> float:
        """Net dollars when the pool is known, else share per dollar of fill risk."""
        if not self.quotable:
            return float("-inf")
        if self.net_dollars is not None:
            return self.net_dollars
        return self.reward.share / (self.fill_cost + 0.01)


def scan_markets(client, *, my_size: int, hours: float, probe: int,
                 target_size: int) -> list[Scan]:
    pools = load_pools()
    if not pools:
        log(f"no {POOLS_FILE} — ranking by reward share per dollar of fill risk")

    markets = fetch_open_markets(client)
    markets.sort(key=lambda m: _number(m.get("volume_24h_fp")) or _number(m.get("volume_fp")),
                 reverse=True)

    scans: list[Scan] = []
    for market in markets[:probe]:
        ticker = market["ticker"]
        try:
            book = fetch_book(client, ticker)
        except KalshiAPIError as exc:
            log(f"  {ticker}: orderbook unavailable ({exc.status})")
            continue
        volume = _number(market.get("volume_24h_fp")) or _number(market.get("volume_fp"))
        scans.append(Scan(ticker=ticker, volume=volume, book=book,
                          target_size=target_size, my_size=my_size, hours=hours,
                          pool=match_pool(ticker, pools)))

    scans.sort(key=lambda s: s.rank_key, reverse=True)
    return scans


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def print_assumptions(target_size: int) -> None:
    print()
    print("ASSUMPTIONS — every number below depends on these:")
    for line in assumptions(target_size):
        print(f"  * {line}")
    print()


def print_table(scans: list[Scan], target_size: int) -> None:
    header = (f"  {'ticker':<40} {'vol24':>9} {'ref_y':>5} {'ref_n':>5} "
              f"{'qual_y':>7} {'qual_n':>7} {'2sd':>3} {'share':>6} {'vld':>5} "
              f"{'rwd/d':>7} {'fill/d':>7} {'net/d':>7} {'cap$':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for scan in scans:
        reward = scan.reward.dollars_per_day
        net = scan.net_dollars
        print(
            f"  {scan.ticker[:40]:<40} {scan.volume:>9.0f} "
            f"{(scan.yes_ref.price if scan.yes_ref else '-'):>5} "
            f"{(scan.no_ref.price if scan.no_ref else '-'):>5} "
            f"{scan.yes_depth:>7} {scan.no_depth:>7} "
            f"{('Y' if scan.two_sided else 'n'):>3} "
            f"{scan.reward.share * 100:>5.1f}% "
            f"{scan.reward.valid_fraction:>5.2f} "
            f"{(f'{reward:.2f}' if reward is not None else '-'):>7} "
            f"{scan.fill_cost:>7.2f} "
            f"{(f'{net:.2f}' if net is not None else '-'):>7} "
            f"{scan.capital:>8.2f}"
        )

    quotable = sum(1 for s in scans if s.quotable)
    two_sided = sum(1 for s in scans if s.two_sided)
    print()
    log(f"{quotable}/{len(scans)} markets have a Reference Price on both sides; "
        f"{two_sided} would produce a qualifying snapshot at Target Size {target_size}")
    if quotable == 0 and scans:
        log("None are quotable at this Target Size. If the real Target Size is "
            "lower, more would qualify — which is exactly why the assumption matters.")


def cmd_scan(client, args) -> int:
    print_assumptions(args.target_size)
    scans = scan_markets(client, my_size=args.size, hours=args.hours,
                         probe=args.probe, target_size=args.target_size)
    if not scans:
        log("nothing to rank")
        return 1
    print_table(scans, args.target_size)
    return 0


def cmd_book(client, args) -> int:
    """Detail for one market: the book, and where the Reference Price falls."""
    print_assumptions(args.target_size)
    book = fetch_book(client, args.ticker)

    for side in ("yes", "no"):
        ref = book.reference_price(side, args.target_size)
        depth = qualifying_depth(book, side, args.target_size)
        print(f"  {side.upper()} side — qualifying depth {depth}/{args.target_size}")
        cumulative = 0
        for price, size in book.levels(side):
            cumulative += size
            marker = "  <-- REFERENCE PRICE (rest here)" if ref and price == ref.price else ""
            print(f"    {price:>3}c  size {size:>6}  cum {cumulative:>7}{marker}")
        if ref is None:
            print("    no Reference Price: book never reaches 1/5 of Target Size")
        print()

    print(f"  spread {spread_cents(book)}  two-sided ok: {two_sided_ok(book, args.target_size)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE_ASSUMPTION,
                        help=f"ASSUMED Target Size (default {TARGET_SIZE_ASSUMPTION}); "
                             "Kalshi does not publish it per market")
    sub = parser.add_subparsers(dest="mode", required=True)

    scan = sub.add_parser("scan", help="rank markets by reward vs fill risk")
    scan.add_argument("--size", type=int, default=DEFAULT_SIZE,
                      help="contracts rested per side")
    scan.add_argument("--hours", type=float, default=DEFAULT_HOURS)
    scan.add_argument("--probe", type=int, default=DEFAULT_PROBE,
                      help="how many books to fetch (one request each)")

    detail = sub.add_parser("book", help="show one book and its Reference Price")
    detail.add_argument("ticker")

    args = parser.parse_args()
    client = build_client()
    log(f"env={client.creds.env} (read-only; this command places no orders)")

    return {"scan": cmd_scan, "book": cmd_book}[args.mode](client, args)


if __name__ == "__main__":
    sys.exit(main())
