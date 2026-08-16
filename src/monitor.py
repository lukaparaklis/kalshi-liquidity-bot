"""
monitor.py
----------
Replays paper logs and estimates the metric that decides everything.

CLAUDE.md: "Fill rate. Every other number in this project is an estimate. Fill
rate is measured." This module is where that measurement happens. It reads the
JSONL written by paper mode and asks, for each market, how often the market
traded through the price we said we would rest at.

How a would-be fill is detected
-------------------------------
We rest a BID on each side, in that side's own currency. A no bid at Q is a yes
ask at 100-Q, so the two sides share one price axis and either side's book tells
us what the other could have traded at.

A resting bid of P on side S fills when someone is willing to sell to us at or
below P — that is, when the implied ask on side S drops to <= P:

    implied ask on S = 100 - (best bid on the other side)

    so we are crossed when   best_bid_other >= 100 - P

Each consecutive pair of snapshots for a market is checked. A crossing between
two snapshots means the market traded through our price while we sat there.

What this is and is not
-----------------------
It is an upper bound on fill rate, and deliberately so:

  - A crossing is counted as filling our WHOLE quote. In reality we might be
    behind others in the queue at that price and take only part.
  - Polling samples the book; anything that crossed and reverted between two
    polls is invisible. Shorter intervals see more.

Both push the same way — this over-counts fills rather than under-counting, so
a market that looks quiet here is quiet, while one that looks busy deserves a
closer look before it is dismissed.

Adverse selection is reported alongside, because a fill is only bad if the
price kept going. CLAUDE.md asks specifically how the mid moved in the 10s
after each fill; that is what `adverse_cents` measures.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .quoter import DEFAULT_LOG_PATH

# How long after a would-be fill to measure the mid move against us.
ADVERSE_HORIZON_S = 10.0


def load_records(path: Path) -> list[dict]:
    """Read a JSONL paper log. Malformed lines are skipped, not fatal."""
    path = Path(path)
    if not path.exists():
        sys.exit(f"no log at {path}. Run paper mode first.")

    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def crossed(quote_side: str, quote_price: int, best: dict) -> bool:
    """
    Would a resting bid of `quote_price` on `quote_side` have been traded through?

    True when the other side's best bid implies an ask at or below our price.
    """
    other = "no" if quote_side == "yes" else "yes"
    other_bid = best.get(f"{other}_bid")
    if other_bid is None:
        return False
    return other_bid >= 100 - quote_price


@dataclass
class MarketStats:
    ticker: str
    snapshots: int = 0
    quotable_snapshots: int = 0
    two_sided_snapshots: int = 0
    fills: dict[str, int] = field(default_factory=lambda: {"yes": 0, "no": 0})
    adverse_moves: list[float] = field(default_factory=list)
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    size: int = 0

    @property
    def hours(self) -> float:
        if not self.first_ts or not self.last_ts:
            return 0.0
        return max((self.last_ts - self.first_ts).total_seconds() / 3600.0, 0.0)

    @property
    def total_fills(self) -> int:
        return self.fills["yes"] + self.fills["no"]

    @property
    def fills_per_hour(self) -> float:
        return self.total_fills / self.hours if self.hours else 0.0

    @property
    def fills_per_hour_per_100(self) -> float:
        """Normalised so markets quoted at different sizes are comparable."""
        if not self.hours or not self.size:
            return 0.0
        return self.fills_per_hour * (100.0 / self.size)

    @property
    def quotable_fraction(self) -> float:
        return self.quotable_snapshots / self.snapshots if self.snapshots else 0.0

    @property
    def earning_fraction(self) -> float:
        """
        Snapshots where we would have quoted AND the market qualified. This is
        the rent side of the ledger: no qualifying snapshot, no reward.
        """
        return self.two_sided_snapshots / self.snapshots if self.snapshots else 0.0

    @property
    def mean_adverse_cents(self) -> float | None:
        if not self.adverse_moves:
            return None
        return sum(self.adverse_moves) / len(self.adverse_moves)

    @property
    def verdict(self) -> str:
        """
        CLAUDE.md: quotes mostly untouched means it works; quotes filling
        constantly means we are being run over.
        """
        if self.quotable_snapshots == 0:
            return "never quotable"
        if self.earning_fraction < 0.5:
            return "rarely qualifies"
        if self.fills_per_hour_per_100 > 10:
            return "RUN OVER"
        if self.fills_per_hour_per_100 > 2:
            return "busy"
        return "quiet — collecting rent"


def _adverse_after(records: list[dict], index: int, side: str) -> float | None:
    """
    Mid move against us in the ADVERSE_HORIZON_S after a would-be fill.

    Positive means the market moved against the position we just acquired: we
    bought yes and the mid fell, or we bought no (short yes) and the mid rose.
    """
    fill = records[index]
    mid_at_fill = (fill.get("best") or {}).get("mid")
    fill_ts = _parse_ts(fill.get("ts", ""))
    if mid_at_fill is None or fill_ts is None:
        return None

    for later in records[index + 1:]:
        later_ts = _parse_ts(later.get("ts", ""))
        later_mid = (later.get("best") or {}).get("mid")
        if later_ts is None or later_mid is None:
            continue
        if (later_ts - fill_ts).total_seconds() >= ADVERSE_HORIZON_S:
            move = mid_at_fill - later_mid      # bought yes: a falling mid hurts
            return move if side == "yes" else -move
    return None


def analyse(records: list[dict]) -> dict[str, MarketStats]:
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("ticker"):
            by_ticker[record["ticker"]].append(record)

    stats: dict[str, MarketStats] = {}
    for ticker, series in by_ticker.items():
        series.sort(key=lambda r: r.get("ts") or "")
        entry = MarketStats(ticker=ticker)

        # Fills are edge-triggered: counted when the market crosses THROUGH our
        # price, not for every snapshot it spends beyond it. An order fills once
        # and is gone; a market that sits crossed for twenty polls did not fill
        # us twenty times. Once it moves back and crosses again, that is a new
        # fill, because by then we have requoted.
        was_crossed = {"yes": False, "no": False}

        for index, record in enumerate(series):
            entry.snapshots += 1
            entry.size = record.get("size") or entry.size

            timestamp = _parse_ts(record.get("ts", ""))
            if timestamp:
                entry.first_ts = entry.first_ts or timestamp
                entry.last_ts = timestamp

            if record.get("two_sided_ok"):
                entry.two_sided_snapshots += 1

            quotes = record.get("quotes") or []
            if not quotes:
                # Nothing resting, so nothing to be crossed; the next quote
                # starts fresh rather than inheriting a stale crossed state.
                was_crossed = {"yes": False, "no": False}
                continue
            entry.quotable_snapshots += 1

            # Did the NEXT snapshot show the market trading through us?
            if index + 1 >= len(series):
                continue
            following = series[index + 1]
            for quote in quotes:
                side = quote["side"]
                now_crossed = crossed(side, quote["price_cents"],
                                      following.get("best") or {})
                if now_crossed and not was_crossed[side]:
                    entry.fills[side] += 1
                    move = _adverse_after(series, index + 1, side)
                    if move is not None:
                        entry.adverse_moves.append(move)
                was_crossed[side] = now_crossed

        stats[ticker] = entry
    return stats


def print_report(stats: dict[str, MarketStats]) -> None:
    header = (f"  {'ticker':<40} {'snaps':>6} {'hours':>6} {'quotable':>9} "
              f"{'earning':>8} {'fills':>6} {'f/h/100':>8} {'adverse':>8}  verdict")
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))

    for entry in sorted(stats.values(), key=lambda s: s.fills_per_hour_per_100):
        adverse = entry.mean_adverse_cents
        print(
            f"  {entry.ticker[:40]:<40} {entry.snapshots:>6} {entry.hours:>6.2f} "
            f"{entry.quotable_fraction * 100:>8.0f}% {entry.earning_fraction * 100:>7.0f}% "
            f"{entry.total_fills:>6} {entry.fills_per_hour_per_100:>8.2f} "
            f"{(f'{adverse:+.2f}c' if adverse is not None else '-'):>8}  {entry.verdict}"
        )

    total_snaps = sum(s.snapshots for s in stats.values())
    total_fills = sum(s.total_fills for s in stats.values())
    print()
    print(f"  {len(stats)} markets, {total_snaps} snapshots, {total_fills} would-be fills")
    print()
    print("  READ THIS AS AN UPPER BOUND ON FILL RATE:")
    print("    * a crossing is counted as filling the whole quote; queue position is ignored")
    print("    * anything that crossed and reverted between two polls is invisible")
    print("    * Target Size was assumed, so 'quotable' depends on a guess")
    print("  A market that looks quiet here is quiet. A busy one deserves a closer look.")


def cmd_analyse(args) -> int:
    records = load_records(args.log)
    if not records:
        print(f"no usable records in {args.log}")
        return 1
    print_report(analyse(records))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH,
                        help=f"paper log to replay (default {DEFAULT_LOG_PATH})")
    return cmd_analyse(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
