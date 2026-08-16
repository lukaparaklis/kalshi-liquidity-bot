"""
book.py
-------
Order book model and Reference Price computation. This is the core mechanic;
see tests/test_reference_price.py, which was written first and defines the
contract this module satisfies.

The rule (CLAUDE.md, "The core mechanic"):

    Walk down from the best bid to the first level where cumulative resting
    size reaches 1/5 of Target Size. That level is the Reference Price. Orders
    at or better than it earn full credit (1.0x); orders below it earn
    DiscountFactor ^ ticks_away.

Resting AT the Reference Price earns the same reward as resting at top of book
while filling materially less often. That gap is the entire edge, so any change
here that moves the returned price toward the top of book is a regression, not
an improvement.

Book model
    Each side is its own book of bids denominated in its own currency: a "no"
    bid at 38c is a "yes" ask at 62c. This is exactly how Kalshi's orderbook
    endpoint returns them, and it means "walk down from the best bid" is simply
    descending price within one side. The two sides are scored independently
    (the program pays a share per side), so they never interact here.

ASSUMPTIONS FLAGGED (per CLAUDE.md)
    - Target Size is set per period and is NOT published per market. There is
      no default here: target_size is a required argument so the assumption
      stays visible at every call site.
    - REFERENCE_FRACTION = 1/5 and "reaches" is inclusive (>=). Both are read
      off the program description, not a Kalshi worked example. If rewards ever
      reconcile badly against actuals, suspect these two before anything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, NamedTuple

# Fraction of Target Size that cumulative depth must reach for a level to be
# the Reference Price.
#
# The comparison below is `cumulative >= REFERENCE_FRACTION * target_size` in
# plain float, per an explicit decision: at a Target Size not divisible by 5
# (say 1001, threshold 200.2) a cumulative depth of 200 does NOT qualify. That
# is the conservative reading — when ambiguous it quotes one tick deeper, which
# errs toward fewer fills.
#
# Float is safe here, not merely convenient: 0.2 * t was checked against exact
# rational arithmetic for every integer target in 1..20000 against every
# integer cumulative depth at and adjacent to the threshold, and the verdicts
# agree everywhere. No representation error reaches the comparison.
REFERENCE_FRACTION = 0.2

SIDES = ("yes", "no")


class ReferencePrice(NamedTuple):
    """
    The deepest price still earning full reward credit, and the cumulative
    resting size down to and including that level.

    A NamedTuple so call sites can read `ref.price` while tests can compare
    against a plain `(price, depth)` tuple.
    """

    price: int
    depth: int


def _normalize(levels: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Collapse raw levels into one entry per price, best (highest) first.

    Three things happen here, and the walk below is only correct because they
    do:

    1. Entries at the same price are summed. Our own order rests alongside
       other participants' at a level, and reward credit is level-granular —
       an order cannot be half at the Reference Price — so a walk that stopped
       partway through a level would report a depth that does not exist.
    2. Levels are sorted descending. Kalshi returns them ascending, and a walk
       over unsorted input silently double-counts a price.
    3. Non-positive sizes are dropped. The API can carry a 0-size level, and a
       phantom level must never be returned as the Reference Price.
    """
    totals: dict[int, int] = {}
    for price, size in levels:
        size = int(size)
        if size > 0:
            totals[int(price)] = totals.get(int(price), 0) + size
    return sorted(totals.items(), key=lambda level: -level[0])


def _parse_levels(raw, *, dollars: bool) -> list[tuple[int, int]]:
    """
    Raw API levels -> [(price_cents, size)]. Malformed rows are skipped: a
    scanner sweeping hundreds of markets must not die on one bad row.
    """
    out: list[tuple[int, int]] = []
    for entry in raw or []:
        try:
            price, size = entry[0], entry[1]
            cents = round(float(price) * 100) if dollars else int(float(price))
            out.append((cents, int(float(size))))
        except (TypeError, ValueError, IndexError, KeyError):
            continue
    return out


@dataclass
class OrderBook:
    """
    A single snapshot of one market's book.

    Levels are (price_cents, size). Input may be unsorted, may repeat a price,
    and may contain zero-size entries; it is normalized on construction, so
    `yes` and `no` are always aggregated and descending after __init__.
    """

    yes: list[tuple[int, int]] = field(default_factory=list)
    no: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.yes = _normalize(self.yes)
        self.no = _normalize(self.no)

    @classmethod
    def from_api(cls, payload: dict) -> "OrderBook":
        """
        Build from a GET /markets/{ticker}/orderbook response.

        v2 nests the book under "orderbook_fp" as yes_dollars/no_dollars, with
        dollar-string prices and fixed-point string sizes:

            {"orderbook_fp": {"yes_dollars": [["0.0100", "2.00"]], ...}}

        The v1 "orderbook" -> yes/no shape used integer cents and is still read
        as a fallback. Which key was present decides how the numbers are
        interpreted — 60 (v1 cents) and "0.6000" (v2 dollars) are the same
        price, so guessing from the value alone would be wrong by 100x.

        Reading the wrong key returns an empty book rather than raising, which
        is exactly how the scanner once concluded the whole exchange was dead.
        Hence one parser, here, with tests.
        """
        payload = payload or {}

        book = payload.get("orderbook_fp")
        if book is not None:
            return cls(yes=_parse_levels(book.get("yes_dollars"), dollars=True),
                       no=_parse_levels(book.get("no_dollars"), dollars=True))

        book = payload.get("orderbook") or {}
        return cls(yes=_parse_levels(book.get("yes"), dollars=False),
                   no=_parse_levels(book.get("no"), dollars=False))

    def levels(self, side: str) -> list[tuple[int, int]]:
        """
        Levels for one side, best first.

        Unknown sides raise rather than resolving to a default. The obvious
        spelling — `self.yes if side == "yes" else self.no` — silently returns
        the no book for "YES" or a typo, and quoting the wrong side of a market
        is a directional position taken by accident.
        """
        if side == "yes":
            return self.yes
        if side == "no":
            return self.no
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")

    def reference_price(self, side: str, target_size: int) -> ReferencePrice | None:
        """
        The Reference Price for one side, or None when the book cannot reach
        the threshold.

        `target_size` is required and deliberately unvalidated. CLAUDE.md gives
        the range as >100 and <20,000, but Kalshi does not publish the value
        per market, so it is a guess either way; rejecting a boundary value
        would break scanning if Kalshi ever set one. Callers own that policy.

        None means "no Reference Price exists", and callers must not soften it
        into a price. There is deliberately no fallback to the best or the
        deepest level: falling back to the best bid would quote top of book,
        the exact behaviour this project exists to avoid. A book this thin also
        cannot meet full Target Size, so the snapshot is excluded from scoring
        and the reward is zero regardless of where we rest.
        """
        threshold = REFERENCE_FRACTION * target_size

        cumulative = 0
        for price, size in self.levels(side):
            cumulative += size
            if cumulative >= threshold:
                return ReferencePrice(price=price, depth=cumulative)
        return None
