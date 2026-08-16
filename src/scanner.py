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

from .auth import (KalshiAPIError, build_client,
                   build_readonly_prod_client, log)
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

# Anchored to the repo, NOT the working directory. As a relative path this
# silently resolved against wherever the command happened to be run from, found
# nothing, loaded no pools, and left the scanner ranking arbitrary discovered
# series — which looks exactly like a wrong pools.json rather than a missing one.
REPO_ROOT = Path(__file__).resolve().parent.parent
POOLS_FILE = REPO_ROOT / "pools.json"

MARKET_PAGE_SIZE = 200
SERIES_LIMIT = 40       # series probed, one /markets request each
DEFAULT_PROBE = 30      # orderbook fetches, one request each

# CLAUDE.md rejects fast series outright: the underlying trades faster on other
# venues, so our quotes are stale and every fill is adversely selected.
FAST_FREQUENCIES = frozenset({"fifteen_min", "hourly"})
DEFAULT_SIZE = 100      # contracts rested per side
DEFAULT_HOURS = 24.0


# ---------------------------------------------------------------------------
# POOLS
# ---------------------------------------------------------------------------

def _load_raw_pools() -> dict:
    """Entries from pools.json, minus "_" comment keys."""
    if not POOLS_FILE.exists():
        return {}
    try:
        raw = json.loads(POOLS_FILE.read_text())
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    except (ValueError, TypeError, AttributeError) as exc:
        log(f"{POOLS_FILE} is unreadable ({exc}); continuing without pools")
        return {}


def load_pools() -> dict[str, float]:
    """
    Daily pool per event ticker.

    An entry is either a bare number or an object carrying the incentives
    page's liquidity rating alongside it:

        "KXH200WS-26SEP11": 165.0
        "KXH200WS-26SEP11": {"pool": 165.0, "liquidity": "Low"}
    """
    pools = {}
    for key, value in _load_raw_pools().items():
        try:
            pools[key] = float(value["pool"] if isinstance(value, dict) else value)
        except (TypeError, ValueError, KeyError):
            log(f"pools.json: skipping {key}, no usable pool value")
    return pools


def load_liquidity() -> dict[str, str]:
    """
    The incentives page's Low/Medium/High rating per event, where given.

    It is a competition proxy, and a coarse one — the scanner measures actual
    resting depth, which is the thing the rating approximates. Kept because it
    is the exchange's own view and disagreement with measured depth is worth
    seeing.
    """
    ratings = {}
    for key, value in _load_raw_pools().items():
        if isinstance(value, dict) and value.get("liquidity"):
            ratings[key] = str(value["liquidity"]).strip().title()
    return ratings


def match_pool(event_ticker: str, pools: dict[str, float]) -> float | None:
    """
    The daily pool for an EVENT. Exact event ticker first, then longest prefix.

    Prefixes matter because events expire — KXB200MS-27MAR is gone next April —
    so a file of exact event tickers goes stale almost immediately. A prefix
    like "KXB200" keeps working across rollovers; an exact entry pinned from the
    incentives page beats it when both match.
    """
    if not event_ticker:
        return None
    if event_ticker in pools:
        return pools[event_ticker]
    matches = [(k, v) for k, v in pools.items() if event_ticker.startswith(k)]
    if not matches:
        return None
    return max(matches, key=lambda kv: len(kv[0]))[1]


def series_of_event(event_ticker: str) -> str:
    """
    The series an event belongs to: KXB200MS-26AUG -> KXB200MS.

    Event tickers are SERIES-PERIOD, and /markets can only be queried by
    series, so this is how a pooled event turns into a fetchable request.
    """
    return str(event_ticker or "").rsplit("-", 1)[0]


def events_from_pools(pools: dict[str, float]) -> list[str]:
    """
    Pool keys that name an actual event, i.e. contain a period suffix.

    A bare prefix like "KXB200" cannot be fetched — it is not a series — so it
    is usable for looking a pool UP but not for discovering markets.
    """
    return [key for key in pools if "-" in key]


def count_markets_per_event(markets: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for market in markets:
        event = market.get("event_ticker")
        if event:
            counts[event] = counts.get(event, 0) + 1
    return counts


def pool_per_market(market: dict, pools: dict[str, float],
                    markets_per_event: dict[str, int]) -> tuple[float | None, int]:
    """
    (pool share for this market, markets in its event).

    The incentives page lists one figure per EVENT row, and an event holds every
    strike in it — 1 to 33 of them here, averaging 10.5. Treating that figure as
    per-market multiplies the implied programme by the number of strikes: these
    five chip families alone came to $171,045/day, which is not a plausible
    incentive budget. Splitting evenly across the event's markets is the
    conservative reading. Kalshi may well weight it by activity instead, which
    would favour the busier strikes within an event — worth checking before any
    of these numbers are trusted.
    """
    event = market.get("event_ticker") or ""
    pool = match_pool(event, pools)
    count = markets_per_event.get(event, 1) or 1
    if pool is None:
        return None, count
    return pool / count, count


# ---------------------------------------------------------------------------
# MARKET DATA
# ---------------------------------------------------------------------------

def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def is_mve(ticker: str) -> bool:
    """MVE = multi-variate event: multi-leg combination markets."""
    return str(ticker or "").upper().startswith("KXMVE")


def select_series(series: list[dict], *, categories: set[str] | None = None,
                  include_fast: bool = False, limit: int = SERIES_LIMIT) -> list[dict]:
    """
    Choose which series to look for markets in.

    Two filters, for two different reasons:

    MVE series are dropped because their markets are not directly orderable —
    POSTing one returns 404 market_not_found. There are only 14 such series out
    of ~13,000, but on prod they hold tens of thousands of open markets and
    /markets returns them first, so a blind sweep sees nothing else.

    Fast series (fifteen_min, hourly) are dropped because CLAUDE.md rejects
    them on the merits: "the underlying trades faster on a hundred other
    exchanges, so our quotes are always stale and every fill is adversely
    selected. Target SLOW markets." Using the frequency field makes that a
    property of discovery rather than a rule someone has to remember.
    """
    chosen = []
    for entry in series:
        ticker = entry.get("ticker")
        if not ticker or is_mve(ticker):
            continue
        if not include_fast and entry.get("frequency") in FAST_FREQUENCIES:
            continue
        if categories and entry.get("category") not in categories:
            continue
        chosen.append(entry)
        if len(chosen) >= limit:
            break
    return chosen


def fetch_open_markets(client, *, series_tickers: list[str] | None = None,
                       events: list[str] | None = None,
                       categories: set[str] | None = None,
                       series_limit: int = SERIES_LIMIT,
                       include_fast: bool = False) -> list[dict]:
    """
    Open, directly-tradeable markets, discovered BY SERIES.

    Paging /markets blindly does not work on prod. Two MVE series hold so many
    open markets that 5,000 rows across 25 pages contained not one ordinary
    market — the sweep never reaches them, at any page budget worth spending.
    Demo hides this: it has few enough MVE markets that the first page or two
    look normal.

    So discovery goes through /series (one request, ~13k series, of which 14
    are MVE) and then asks for each series' markets by name. Bounded by
    series_limit, since that is one request per series.
    """
    wanted_events = set(events or [])

    if wanted_events:
        chosen = [{"ticker": t} for t in sorted({series_of_event(e) for e in wanted_events})]
        log(f"scanning {len(wanted_events)} pooled event(s) across "
            f"{len(chosen)} series: {', '.join(sorted(wanted_events))}")
    elif series_tickers:
        chosen = [{"ticker": t} for t in series_tickers]
        log(f"scanning {len(chosen)} explicitly requested series")
    else:
        all_series = client.get("/series").get("series") or []
        chosen = select_series(all_series, categories=categories,
                               include_fast=include_fast, limit=series_limit)
        log(f"{len(all_series)} series available; probing {len(chosen)} "
            f"(MVE and {'no ' if include_fast else ''}fast series excluded"
            f"{', categories ' + ','.join(sorted(categories)) if categories else ''})")
        if len(all_series) > series_limit:
            log(f"NOTE: only the first {len(chosen)} matching series are probed. "
                f"Use --series or --category to target deliberately.")

    markets: list[dict] = []
    for entry in chosen:
        ticker = entry["ticker"]
        try:
            data = client.get("/markets", params={"series_ticker": ticker,
                                                  "status": "open",
                                                  "limit": MARKET_PAGE_SIZE})
        except KalshiAPIError as exc:
            log(f"  series {ticker}: {exc.status}")
            continue
        markets += data.get("markets") or []

    if wanted_events:
        # A series holds many events; keep only the ones carrying a pool.
        markets = [m for m in markets if m.get("event_ticker") in wanted_events]

    # Belt and braces: drop MVE at the market level too, in case a series slips
    # through under a name that does not start with KXMVE.
    tradeable = [m for m in markets
                 if m.get("ticker") and not m.get("mve_collection_ticker")]
    log(f"{len(markets)} open markets found, {len(markets) - len(tradeable)} MVE "
        f"dropped, {len(tradeable)} tradeable")
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
    pool: float | None            # this market's share of its event's pool
    event_ticker: str = ""
    event_pool: float | None = None
    markets_in_event: int = 1
    liquidity: str = ""           # the incentives page's Low/Medium/High

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
    def total_depth(self) -> int:
        """Every resting contract on both sides: the competition, measured."""
        return (sum(size for _, size in self.book.yes)
                + sum(size for _, size in self.book.no))

    @property
    def pool_per_depth(self) -> float | None:
        """
        Daily pool dollars per resting contract already on the book.

        This is the reward-density measure: a big pool on a thin book beats a
        bigger pool on a mobbed one. It is what the page's Low/Medium/High
        rating gestures at, computed from the actual book instead.
        """
        if self.pool is None:
            return None
        return self.pool / max(self.total_depth, 1)

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
        if self.pool_per_depth is not None:
            return self.pool_per_depth
        return self.reward.share / (self.fill_cost + 0.01)


def scan_markets(client, *, my_size: int, hours: float, probe: int,
                 target_size: int, series_tickers: list[str] | None = None,
                 categories: set[str] | None = None,
                 series_limit: int = SERIES_LIMIT,
                 include_fast: bool = False,
                 from_pools: bool = False) -> list[Scan]:
    pools = load_pools()
    ratings = load_liquidity()
    if not pools:
        log(f"WARNING: no pools loaded from {POOLS_FILE}")
        log("  Without pools there is no reward estimate, so the ranking falls "
            "back to share per dollar of fill risk over whatever series "
            "discovery happens to return.")
    else:
        log(f"{len(pools)} pool entr(ies) from {POOLS_FILE}: "
            f"{', '.join(sorted(pools))}")

    events = None
    if from_pools:
        events = events_from_pools(pools)
        prefixes = [k for k in pools if "-" not in k]
        if prefixes:
            log(f"ignoring {len(prefixes)} prefix-only pool key(s) for discovery "
                f"({', '.join(sorted(prefixes))}); they still resolve pools but "
                f"name no event to fetch")
        if not events:
            sys.exit(f"--from-pools needs event tickers in {POOLS_FILE}, "
                     f"e.g. KXB200MS-26AUG. Only prefixes were found.")

    markets = fetch_open_markets(client, series_tickers=series_tickers,
                                 events=events,
                                 categories=categories, series_limit=series_limit,
                                 include_fast=include_fast)
    # Counted over EVERY open market found, not just the ones probed: an
    # event's pool is split across all its strikes, whether or not we look at
    # them.
    markets_per_event = count_markets_per_event(markets)
    if pools:
        log(f"{len(markets_per_event)} events; pools are per-event, split across "
            f"each event's markets (mean "
            f"{len(markets) / max(len(markets_per_event), 1):.1f} markets/event)")

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
        share, count = pool_per_market(market, pools, markets_per_event)
        event = market.get("event_ticker") or ""
        scans.append(Scan(ticker=ticker, volume=volume, book=book,
                          target_size=target_size, my_size=my_size, hours=hours,
                          pool=share, event_ticker=event,
                          event_pool=match_pool(event, pools),
                          markets_in_event=count,
                          liquidity=ratings.get(event, "")))

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
    header = (f"  {'ticker':<34} {'liq':>6} {'depth':>9} {'pool/mkt':>9} "
              f"{'$/1k depth':>11} {'ref_y':>5} {'ref_n':>5} {'2sd':>3} "
              f"{'share':>6} {'rwd/d':>7} {'fill/d':>7} {'net/d':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for scan in scans:
        reward = scan.reward.dollars_per_day
        net = scan.net_dollars
        density = scan.pool_per_depth
        print(
            f"  {scan.ticker[:34]:<34} {(scan.liquidity or '-'):>6} "
            f"{scan.total_depth:>9,} "
            f"{(f'{scan.pool:.3f}' if scan.pool is not None else '-'):>9} "
            f"{(f'{density * 1000:.3f}' if density is not None else '-'):>11} "
            f"{(scan.yes_ref.price if scan.yes_ref else '-'):>5} "
            f"{(scan.no_ref.price if scan.no_ref else '-'):>5} "
            f"{('Y' if scan.two_sided else 'n'):>3} "
            f"{scan.reward.share * 100:>5.2f}% "
            f"{(f'{reward:.3f}' if reward is not None else '-'):>7} "
            f"{scan.fill_cost:>7.2f} "
            f"{(f'{net:+.3f}' if net is not None else '-'):>7}"
        )

    quotable = sum(1 for s in scans if s.quotable)
    two_sided = sum(1 for s in scans if s.two_sided)
    priced = [s for s in scans if s.net_dollars is not None]
    positive = [s for s in priced if s.net_dollars > 0]
    print()
    log(f"{quotable}/{len(scans)} markets have a Reference Price on both sides; "
        f"{two_sided} would produce a qualifying snapshot at Target Size {target_size}")
    if priced:
        best = max(priced, key=lambda s: s.net_dollars)
        log(f"NET POSITIVE: {len(positive)}/{len(priced)} markets with a known pool")
        log(f"best: {best.ticker} at {best.net_dollars:+.3f}/day "
            f"(reward {best.reward.dollars_per_day:.3f} - fill {best.fill_cost:.2f})")
        total = sum(s.net_dollars for s in positive)
        log(f"sum of every net-positive market: {total:+.2f}/day across "
            f"{len(positive)} markets, needing "
            f"${sum(s.capital for s in positive):,.0f} of capital")
    if quotable == 0 and scans:
        log("None are quotable at this Target Size. If the real Target Size is "
            "lower, more would qualify — which is exactly why the assumption matters.")


def cmd_scan(client, args) -> int:
    print_assumptions(args.target_size)

    # Rewards only exist where a pool exists, so if pools.json names events,
    # scan those unless told otherwise. Blind discovery returns series in
    # arbitrary order — MLB strikeouts, restaurant closures — which is a
    # confusing default for a tool whose whole subject is reward pools.
    if (not args.from_pools and not args.all_series
            and not args.series and not args.category):
        if events_from_pools(load_pools()):
            args.from_pools = True
            log("targeting the events in pools.json (--all-series to scan "
                "discovered series instead)")
    scans = scan_markets(client, my_size=args.size, hours=args.hours,
                         probe=args.probe, target_size=args.target_size,
                         series_tickers=args.series,
                         categories=set(args.category) if args.category else None,
                         series_limit=args.series_limit,
                         include_fast=args.include_fast,
                         from_pools=args.from_pools)
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
    parser.add_argument("--prod", action="store_true",
                        help="read PROD books with the read-only prod credential. "
                             "The scanner never writes; demo has no liquidity.")
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
    scan.add_argument("--series", action="append",
                      help="series ticker to scan, e.g. KXHIGHNY; repeatable. "
                           "Bypasses discovery entirely.")
    scan.add_argument("--category", action="append",
                      help="restrict discovery to a category, e.g. "
                           "'Climate and Weather'; repeatable")
    scan.add_argument("--series-limit", type=int, default=SERIES_LIMIT,
                      help=f"series to probe when discovering (default {SERIES_LIMIT})")
    scan.add_argument("--from-pools", action="store_true",
                      help=f"scan exactly the event tickers listed in {POOLS_FILE}. "
                           "Rewards only exist where there is a pool, so this is "
                           "usually what you want.")
    scan.add_argument("--all-series", action="store_true",
                      help="scan discovered series instead of the pooled events")
    scan.add_argument("--include-fast", action="store_true",
                      help="include fifteen_min/hourly series, which CLAUDE.md "
                           "rejects as structurally adversely selected")

    detail = sub.add_parser("book", help="show one book and its Reference Price")
    detail.add_argument("ticker")

    args = parser.parse_args()

    # Scanning demo is close to useless: it carries the same market listings as
    # prod but almost no resting liquidity, so every book reads empty and every
    # market looks unquotable. CLAUDE.md marks the scanner read-only and safe
    # against prod. The choice is explicit rather than inferred — silently
    # picking an environment is how a "prod" scan ends up reading demo.
    if args.prod:
        client = build_readonly_prod_client()
        log(f"env=prod via {type(client).__name__} — READ-ONLY, writes raise")
    else:
        client = build_client()
        log(f"env={client.creds.env} (read-only; this command places no orders)")
        log("demo has the market listings but no resting liquidity, so books "
            "will read empty. Pass --prod to scan real books.")

    return {"scan": cmd_scan, "book": cmd_book}[args.mode](client, args)


if __name__ == "__main__":
    sys.exit(main())
