#!/usr/bin/env python3
"""
kalshi_liquidity_scanner.py
---------------------------
Finds Kalshi markets where providing resting liquidity is likely to pay more
in incentive rewards than it costs you in adverse fills.

THE THESIS
    Kalshi's Liquidity Incentive Program pays you for resting orders even if
    they never fill. That is rent, not a trading edge -- it does not require
    you to predict anything. But your orders CAN fill, and the fills are
    adversely selected (you trade with whoever knew first). So the question
    is purely:

        expected reward share  >  expected loss from fills ?

    This tool estimates both sides and ranks markets by the ratio.

THE KEY MECHANIC (why this can work at all)
    Rewards are NOT paid for being top-of-book. Kalshi finds a "Reference
    Price" by walking down from the best bid to the first level where
    cumulative size reaches 1/5 of Target Size. Orders at or better than the
    Reference Price get FULL credit (1.0x); orders below it decay by
    DiscountFactor^ticks.

    So the optimal resting spot is AT the Reference Price -- deep enough to
    rarely fill, good enough to earn full score. That is the whole game.

WHAT MAKES A MARKET GOOD
    HIGH  reward pool           -> more rent
    LOW   existing qualifying depth -> less dilution (your share is bigger)
    LOW   volume relative to depth  -> fewer fills = less adverse selection
    LOW   price volatility          -> your quotes go stale slowly
    WIDE  spread                    -> you also capture spread when filled
    SHORT time to close             -> capital freed sooner

    Note the tension: high-volume markets are where Kalshi puts the biggest
    pools, and they are also where you get run over. The sweet spot is a
    decent pool in a SLOW market. Bitcoin 15-minute is the opposite of this.

REWARD DATA
    I could not verify a public API endpoint that exposes incentive pools.
    The tool tries a plausible one, and if that fails you paste pool sizes
    from https://kalshi.com/incentives into pools.json:

        {"KXHIGHNY-26AUG13": 50.0, "KXFED-26SEP": 200.0}

    (ticker or event prefix -> dollars per day)

USAGE
    python kalshi_liquidity_scanner.py scan --min-pool 10
    python kalshi_liquidity_scanner.py plan KXHIGHNY-26AUG13-B88 --capital 500
    python kalshi_liquidity_scanner.py monitor      # track fills vs rewards

NOT INVESTMENT ADVICE. Every number below is an estimate built on assumptions
you should check. Rewards can be changed or ended by Kalshi at any time, and
resting orders are real exposure, not free money.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from kalshi_btc15m_bot import KalshiClient, log, parse_ts
except ImportError:
    sys.exit("Keep this next to kalshi_btc15m_bot.py.")

POOLS_FILE = Path("pools.json")
FILL_LOG = Path("liquidity_fills.csv")

# Program parameters. Kalshi sets Target Size per time period (>100, <20000)
# and does not publish it per-market, so we assume a value and flag it.
ASSUMED_TARGET_SIZE = 1000
REFERENCE_FRACTION = 0.2        # ref price = level where cum. size hits 1/5 target

TAKER_MULT = 0.07
MAKER_MULT = 0.0175


# ---------------------------------------------------------------------------
# ORDER BOOK ANALYSIS
# ---------------------------------------------------------------------------

@dataclass
class BookView:
    ticker: str
    yes_levels: list[tuple[int, int]]   # (price_cents, size), best first
    no_levels: list[tuple[int, int]]
    spread: int
    mid: float
    volume: int
    close: datetime | None

    def reference_price(self, side: str, target: int) -> tuple[int, int] | None:
        """
        Walk down from the best price until cumulative size reaches
        REFERENCE_FRACTION * target. Return (price, cumulative_depth_to_there).
        This is the deepest price that still earns full reward credit.
        """
        need = REFERENCE_FRACTION * target
        levels = self.yes_levels if side == "yes" else self.no_levels
        cum = 0
        for price, size in levels:
            cum += size
            if cum >= need:
                return price, cum
        return None

    def qualifying_depth(self, side: str, target: int) -> int:
        """Total resting size counted toward Target Size on this side."""
        levels = self.yes_levels if side == "yes" else self.no_levels
        return min(sum(s for _, s in levels), target)

    def two_sided_ok(self, target: int) -> bool:
        """Snapshots only count when BOTH sides meet Target Size."""
        return (self.qualifying_depth("yes", target) >= target
                and self.qualifying_depth("no", target) >= target)


def fetch_book(client: KalshiClient, ticker: str) -> BookView | None:
    try:
        ob = client._request("GET", f"/markets/{ticker}/orderbook",
                             params={"depth": 40}).get("orderbook", {})
        m = client.market(ticker)
    except RuntimeError as exc:
        log(f"  {ticker}: {str(exc)[:90]}")
        return None

    def levels(raw):
        if not raw:
            return []
        # Kalshi returns [[price, size], ...] ascending; best bid is highest.
        out = [(int(p), int(s)) for p, s in raw if s]
        return sorted(out, key=lambda x: -x[0])

    yes = levels(ob.get("yes"))
    no = levels(ob.get("no"))
    if not yes or not no:
        return None

    yb, nb = yes[0][0], no[0][0]
    spread = 100 - yb - nb          # implied spread between the two sides
    close_s = m.get("close_time") or m.get("expiration_time")
    return BookView(
        ticker=ticker, yes_levels=yes, no_levels=no,
        spread=max(0, spread), mid=(yb + (100 - nb)) / 2,
        volume=int(m.get("volume") or 0),
        close=parse_ts(close_s) if close_s else None,
    )


# ---------------------------------------------------------------------------
# ECONOMICS
# ---------------------------------------------------------------------------

def estimate_daily_reward(book: BookView, pool_per_day: float,
                          my_size: int, target: int) -> float:
    """
    Your snapshot score is your SHARE of qualifying liquidity on each side.
    Score per snapshot = share_yes + share_no (max 2.0 across all traders).
    Reward = your share of total score x pool x (fraction of valid snapshots).
    """
    if not book.two_sided_ok(target):
        # We could be the one who fixes that -- but assume partial credit.
        valid_fraction = 0.5
    else:
        valid_fraction = 0.95

    share = 0.0
    for side in ("yes", "no"):
        existing = book.qualifying_depth(side, target)
        # our size adds to the pool of qualifying liquidity
        share += my_size / max(1.0, existing + my_size)
    share /= 2.0        # normalize: snapshot total is 2.0 across both sides
    return pool_per_day * share * valid_fraction


def estimate_fill_cost(book: BookView, my_size: int, hours: float) -> float:
    """
    Rough adverse-selection cost.

    Fills scale with how much volume churns relative to book depth. When you
    are filled, assume you lose roughly half the spread to adverse selection
    (the trade happened because someone knew something), plus the maker fee.

    This is the single most uncertain number in the tool. Measure your real
    fill rate with `monitor` and replace this estimate.
    """
    total_depth = sum(s for _, s in book.yes_levels) + sum(s for _, s in book.no_levels)
    if total_depth <= 0:
        return float("inf")

    # fraction of our resting size we expect to trade over the period
    hourly_turnover = book.volume / max(1.0, total_depth) / 24.0
    expected_filled = min(my_size, my_size * hourly_turnover * hours)

    adverse_c = max(1.0, book.spread / 2.0)       # cents lost per filled contract
    p = book.mid / 100.0
    fee_c = MAKER_MULT * p * (1 - p) * 100
    return expected_filled * (adverse_c + fee_c) / 100.0


def capital_required(book: BookView, my_size: int) -> float:
    """Cash tied up resting my_size on both sides."""
    yes_px = book.yes_levels[0][0] / 100.0
    no_px = book.no_levels[0][0] / 100.0
    return my_size * (yes_px + no_px)


# ---------------------------------------------------------------------------
# COMMANDS
# ---------------------------------------------------------------------------

def load_pools() -> dict[str, float]:
    if POOLS_FILE.exists():
        return {k: float(v) for k, v in json.loads(POOLS_FILE.read_text()).items()}
    log(f"No {POOLS_FILE}. Paste pools from https://kalshi.com/incentives as")
    log('  {"TICKER_OR_PREFIX": dollars_per_day, ...}')
    return {}


def match_pool(ticker: str, pools: dict[str, float]) -> float:
    if ticker in pools:
        return pools[ticker]
    for k, v in pools.items():
        if ticker.startswith(k):
            return v
    return 0.0


def cmd_scan(client: KalshiClient, args) -> None:
    pools = load_pools()
    if not pools:
        sys.exit("Nothing to scan without pool data.")

    log(f"scanning markets against {len(pools)} pool entries "
        f"(assumed Target Size {ASSUMED_TARGET_SIZE})")
    rows = []
    seen = set()

    for prefix in pools:
        try:
            data = client._request("GET", "/markets",
                                   params={"status": "open", "limit": 200,
                                           "event_ticker": prefix.split("-")[0]})
            markets = data.get("markets", [])
        except RuntimeError:
            markets = []
        if not markets:
            markets = [{"ticker": prefix}]

        for m in markets[:12]:
            tk = m["ticker"]
            if tk in seen:
                continue
            seen.add(tk)
            pool = match_pool(tk, pools)
            if pool < args.min_pool:
                continue

            book = fetch_book(client, tk)
            if not book:
                continue

            hours = 24.0
            if book.close:
                hours = min(24.0, max(0.5,
                    (book.close - datetime.now(timezone.utc)).total_seconds() / 3600))

            reward = estimate_daily_reward(book, pool, args.size, ASSUMED_TARGET_SIZE)
            cost = estimate_fill_cost(book, args.size, hours)
            cap = capital_required(book, args.size)
            net = reward - cost
            yield_pct = (net / cap * 100) if cap > 0 else 0.0

            rows.append({
                "ticker": tk, "pool": pool, "reward": reward, "cost": cost,
                "net": net, "cap": cap, "yield": yield_pct,
                "spread": book.spread, "vol": book.volume,
                "depth_y": book.qualifying_depth("yes", ASSUMED_TARGET_SIZE),
                "depth_n": book.qualifying_depth("no", ASSUMED_TARGET_SIZE),
                "two_sided": book.two_sided_ok(ASSUMED_TARGET_SIZE),
            })
            time.sleep(0.15)

    if not rows:
        sys.exit("No markets matched. Check pools.json tickers.")

    rows.sort(key=lambda r: -r["yield"])
    print(f"\n{'ticker':28} {'pool/d':>7} {'reward':>7} {'fillcost':>8} "
          f"{'net/d':>7} {'capital':>8} {'yield/d':>8} {'sprd':>5} {'2sided':>7}")
    print("-" * 100)
    for r in rows[:25]:
        print(f"{r['ticker'][:28]:28} ${r['pool']:6.0f} ${r['reward']:6.2f} "
              f"${r['cost']:7.2f} ${r['net']:+6.2f} ${r['cap']:7.2f} "
              f"{r['yield']:+7.2f}% {r['spread']:5d} {str(r['two_sided']):>7}")

    good = [r for r in rows if r["net"] > 0]
    print(f"\n{len(good)}/{len(rows)} markets show positive net.")
    print("\nREAD THIS BEFORE BELIEVING THE TABLE:")
    print("  - Target Size is ASSUMED at", ASSUMED_TARGET_SIZE, "- Kalshi sets it per")
    print("    period and does not publish it. Wrong value = wrong dilution math.")
    print("  - fillcost is a crude model. Run `monitor` for a week and replace it.")
    print("  - yield assumes you keep quotes up continuously. Downtime = zero reward.")
    print("  - Your share shrinks as others join. Today's yield is not tomorrow's.")


def cmd_plan(client: KalshiClient, args) -> None:
    """Where exactly to rest orders in one market."""
    book = fetch_book(client, args.ticker)
    if not book:
        sys.exit("could not read book")

    target = ASSUMED_TARGET_SIZE
    print(f"\n{book.ticker}")
    print(f"  spread {book.spread}c   mid {book.mid:.1f}c   volume {book.volume}")
    print(f"  two-sided target met: {book.two_sided_ok(target)}")

    for side in ("yes", "no"):
        levels = book.yes_levels if side == "yes" else book.no_levels
        ref = book.reference_price(side, target)
        best = levels[0][0]
        print(f"\n  {side.upper()} side")
        print(f"    best price      : {best}c")
        if ref:
            print(f"    REFERENCE PRICE : {ref[0]}c  (cum depth {ref[1]})")
            print(f"    -> rest AT {ref[0]}c for full credit, {best-ref[0]} ticks "
                  f"back from the top.")
            print(f"    -> that is {best-ref[0]} ticks of protection against being "
                  f"picked off, at no reward cost.")
        else:
            print(f"    REFERENCE PRICE : book too thin to reach "
                  f"{REFERENCE_FRACTION*target:.0f} contracts")
            print(f"    -> you would be SETTING the reference price. Full credit,")
            print(f"       but you are also the only liquidity. High fill risk.")
        print(f"    depth counted   : {book.qualifying_depth(side, target)}/{target}")

    size = int(args.capital / (book.mid / 100.0) / 2)
    print(f"\n  With ${args.capital:.0f} you could rest ~{size} contracts per side.")
    print(f"  Capital at risk if BOTH sides fill: ${capital_required(book, size):.2f}")
    print("\n  Start at a tenth of that. The first week is a measurement exercise,")
    print("  not an earnings exercise.")


def cmd_monitor(client: KalshiClient, args) -> None:
    """Track actual fills so you can replace the fill-cost model with data."""
    log("polling fills every 60s. Ctrl-C to stop.")
    seen = set()
    new = not FILL_LOG.exists()
    fh = FILL_LOG.open("a")
    if new:
        fh.write("ts,ticker,side,count,price,is_taker\n")
    try:
        while True:
            try:
                fills = client._request("GET", "/portfolio/fills",
                                        params={"limit": 200}).get("fills", [])
            except RuntimeError as exc:
                log(f"fills error: {str(exc)[:100]}")
                time.sleep(60)
                continue
            for f in fills:
                fid = f.get("trade_id") or f.get("fill_id")
                if fid in seen:
                    continue
                seen.add(fid)
                fh.write(f"{f.get('created_time','')},{f.get('ticker','')},"
                         f"{f.get('side','')},{f.get('count',0)},"
                         f"{f.get('yes_price',0)},{f.get('is_taker','')}\n")
                fh.flush()
                log(f"FILL {f.get('ticker')} {f.get('side')} x{f.get('count')} "
                    f"@{f.get('yes_price')}c")
            time.sleep(60)
    except KeyboardInterrupt:
        fh.close()
        log("stopped")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan")
    s.add_argument("--min-pool", type=float, default=5.0)
    s.add_argument("--size", type=int, default=200, help="contracts per side")

    p = sub.add_parser("plan")
    p.add_argument("ticker")
    p.add_argument("--capital", type=float, default=500.0)

    sub.add_parser("monitor")

    args = ap.parse_args()
    client = KalshiClient.from_env()

    {"scan": cmd_scan, "plan": cmd_plan, "monitor": cmd_monitor}[args.cmd](client, args)


if __name__ == "__main__":
    main()
