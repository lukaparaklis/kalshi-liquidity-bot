"""
economics.py
------------
Fees, reward estimation, and the fill-cost model. Read-only arithmetic; nothing
here talks to the exchange.

Two very different classes of number live in this file and they must not be
confused:

  VERIFIED   The fee formula. Checked against Kalshi's published schedule AND
             against a real filled order on demo (1 contract at 29c paid
             taker_fees_dollars 0.014500, which is 0.07 x 0.29 x 0.71 rounded
             up to a centicent). Trust it.

  ASSUMED    Everything about the reward program. Target Size is not published
             per market. The discount factor is not published. The fraction of
             snapshots that qualify is a guess. The fill-cost model is the
             roughest estimate in the project. Every one of these is a named
             constant below, and assumptions() surfaces them so no output can
             quietly imply more confidence than exists.

The fill-cost estimate in particular is a placeholder for a measurement.
CLAUDE.md is explicit that fill rate is the metric that decides everything and
that it must be measured, not estimated. Treat the ranking here as "which
markets are worth measuring", never as "which markets are profitable".
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal

from .book import OrderBook

# ---------------------------------------------------------------------------
# FEES — verified
# ---------------------------------------------------------------------------

# fees = round_up_to_centicent(M x rate x C x P x (1-P)), P in dollars.
TAKER_FEE_RATE = Decimal("0.07")
MAKER_FEE_RATE = Decimal("0.0175")      # exactly one quarter of the taker rate
CENTICENT = Decimal("0.0001")           # $0.0001; the fee rounding increment


def trade_fee_dollars(price_cents: int, contracts: int, *, maker: bool,
                      multiplier: float = 1.0) -> float:
    """
    Fee in dollars for `contracts` at `price_cents`.

    `maker` has no default on purpose. The taker rate is 4x the maker rate, and
    a caller who forgets which one they meant would understate cost by 300%.

    Rounding is UP to the next centicent ($0.0001) — not to the next cent.
    CLAUDE.md says "rounded up to the next cent per order", and that is wrong:
    a 1-contract fill at 29c is charged $0.0145, not $0.02. Rounding to whole
    cents overstates the fee by up to 100x on the small orders this bot rests,
    which would make every market look unprofitable.

    Computed in Decimal, not float. `0.07 * 0.5 * 0.5` is 0.017500000000000002
    in binary, so a float ceiling charges 0.0176 for a fee that is exactly
    0.0175 — an overcharge on every price that lands on the centicent grid.

    `multiplier` is Kalshi's per-contract M, default 1. Some series (crypto
    among them, per CLAUDE.md) carry a higher one; verify per series.
    """
    if contracts <= 0:
        return 0.0

    price = Decimal(price_cents) / 100
    rate = MAKER_FEE_RATE if maker else TAKER_FEE_RATE
    raw = Decimal(str(multiplier)) * rate * Decimal(contracts) * price * (Decimal(1) - price)
    return float(raw.quantize(CENTICENT, rounding=ROUND_CEILING))


def trade_fee_cents(price_cents: int, contracts: int, *, maker: bool,
                    multiplier: float = 1.0) -> float:
    """Same fee, expressed in cents. Convenience for cent-denominated output."""
    return trade_fee_dollars(price_cents, contracts, maker=maker,
                             multiplier=multiplier) * 100


# ---------------------------------------------------------------------------
# PROGRAM PARAMETERS — assumed, not published
# ---------------------------------------------------------------------------

# Kalshi sets Target Size per period, >100 and <20,000, and does NOT publish it
# per market. Everything downstream of this number is a guess of the same size.
# Every function taking it requires it explicitly — no defaults — so the
# assumption stays visible at each call site rather than hiding in a signature.
TARGET_SIZE_ASSUMPTION = 1000

# Orders below the Reference Price earn DiscountFactor ^ ticks_away. The factor
# is not published. It does not affect this bot's own estimates, which quote AT
# the Reference Price (0 ticks, multiplier 1.0 for any factor), so it only
# matters if a strategy ever chooses to rest deeper.
DISCOUNT_FACTOR_ASSUMPTION = 0.75

# A snapshot is excluded unless BOTH sides meet Target Size, market-wide. These
# are pure guesses at how often that holds; they scale the reward estimate
# linearly, so a wrong guess here moves every number in the same direction.
VALID_FRACTION_TWO_SIDED = 0.95     # already qualifies without us
VALID_FRACTION_MARGINAL = 0.50      # qualifies only once our size is added
VALID_FRACTION_UNREACHABLE = 0.0    # cannot qualify even with our size

# Kept as an alias: the reference implementation's name for the marginal case.
VALID_FRACTION_ONE_SIDED = VALID_FRACTION_MARGINAL


def assumptions(target_size: int) -> list[str]:
    """
    Human-readable list of everything unverified feeding the estimates. Any
    command that prints a ranking must print this too.
    """
    return [
        f"Target Size assumed {target_size} — NOT published per market; Kalshi "
        f"sets it per period, >100 and <20,000. Every reward number scales with it.",
        f"Discount factor assumed {DISCOUNT_FACTOR_ASSUMPTION} — not published. "
        f"Unused while quoting at the Reference Price.",
        f"Valid-snapshot fraction assumed {VALID_FRACTION_TWO_SIDED} two-sided / "
        f"{VALID_FRACTION_ONE_SIDED} otherwise — a guess.",
        "Daily pool is per market and comes from kalshi.com/incentives; it is "
        "not in the API. Markets with no known pool show reward share only.",
        "Fill cost is modelled, not measured. CLAUDE.md: fill rate is the metric "
        "that decides everything, and it must be measured. Rank = what to measure.",
    ]


# ---------------------------------------------------------------------------
# BOOK-DERIVED QUANTITIES
# ---------------------------------------------------------------------------

def total_depth(book: OrderBook, side: str) -> int:
    """
    All resting size on one side. UNCAPPED — this is the competition.

    Use this wherever the question is "how much liquidity am I sharing the
    reward with", which is every share calculation. Capping it destroys the
    only signal that distinguishes a market with 1,100 contracts resting from
    one with 22,000.
    """
    return sum(size for _, size in book.levels(side))


def qualifying_depth(book: OrderBook, side: str, target_size: int) -> int:
    """
    Resting size counted toward the Target Size THRESHOLD, capped at it.

    The cap belongs here and only here: the program asks whether the market
    reaches Target Size, so depth beyond it does not help meet the requirement.
    That makes this the right input to the two-sided snapshot test and the
    wrong input to any share calculation — see total_depth.
    """
    return min(total_depth(book, side), target_size)


def two_sided_ok(book: OrderBook, target_size: int) -> bool:
    """
    Whether a snapshot of this book would count at all.

    The requirement is market-wide, not about our own orders: if either side
    falls short of Target Size the snapshot is excluded and everybody earns
    nothing from it.
    """
    return all(qualifying_depth(book, side, target_size) >= target_size
               for side in ("yes", "no"))


def two_sided_reachable(book: OrderBook, my_size: int, target_size: int) -> bool:
    """
    Whether the market could reach Target Size on both sides once our own size
    is added.

    If it cannot, no snapshot qualifies however we quote, so the reward is zero
    no matter how large a share of the book we would own.
    """
    return all(qualifying_depth(book, side, target_size) + my_size >= target_size
               for side in ("yes", "no"))


def spread_cents(book: OrderBook) -> int | None:
    """
    Implied spread between the two sides. A no bid at P is a yes ask at 100-P,
    so the spread is 100 - yes_bid - no_bid. None if either side is empty.
    """
    if not book.yes or not book.no:
        return None
    return max(0, 100 - book.yes[0][0] - book.no[0][0])


def mid_cents(book: OrderBook) -> float | None:
    """Midpoint in yes terms. None if either side is empty."""
    if not book.yes or not book.no:
        return None
    return (book.yes[0][0] + (100 - book.no[0][0])) / 2


def credit_multiplier(ticks_away: int,
                      discount_factor: float = DISCOUNT_FACTOR_ASSUMPTION) -> float:
    """
    Reward credit for an order `ticks_away` below the Reference Price.

    At or better than the Reference Price earns full credit. This is the whole
    edge: resting AT the Reference Price earns the same 1.0x as resting at top
    of book, while filling far less often.
    """
    if ticks_away <= 0:
        return 1.0
    return discount_factor ** ticks_away


# ---------------------------------------------------------------------------
# ESTIMATES
# ---------------------------------------------------------------------------

@dataclass
class RewardEstimate:
    share: float                 # our fraction of total qualifying liquidity, 0..1
    valid_fraction: float        # guessed fraction of snapshots that qualify
    dollars_per_day: float | None  # None when the market's pool is unknown

    @property
    def two_sided(self) -> bool:
        return self.valid_fraction >= VALID_FRACTION_TWO_SIDED


def estimate_reward(book: OrderBook, my_size: int, target_size: int,
                    pool_dollars_per_day: float | None = None) -> RewardEstimate:
    """
    Our share of the reward pool for resting `my_size` on BOTH sides.

    Score per snapshot is our share of qualifying liquidity on each side, and
    the snapshot total is 2.0 across all participants (1.0 per side), so our
    fraction of the whole is the mean of the two side shares.

    The denominator is TOTAL resting depth, not depth capped at Target Size.
    The capped version was an outright bug: it put a capped denominator under
    an uncapped numerator, so every market with more than Target Size resting
    returned the identical share — 100/(1000+100) = 9.1% — whatever was really
    on the book. Measured against prod, that overstated our share by 8x to 17x
    and, worse, made the estimate blind: 22,000 contracts of competition and
    1,100 scored the same, so the ranking collapsed onto whatever pool numbers
    happened to be in pools.json.

    Raw depth is still the conservative choice. Orders resting below the
    Reference Price earn DiscountFactor ^ ticks_away, so the true
    credit-weighted denominator is smaller than the raw sum and our real share
    is at least this. Refining that needs the discount factor, which Kalshi
    does not publish.

    Ported from reference/kalshi_liquidity_scanner.py with one other deliberate
    divergence. The reference gave any market that failed the two-sided test a
    0.5 valid fraction, which on an EMPTY book produced a 100% share at half
    credit — ranking dead markets top, since owning all of nothing looks like
    owning everything. A market that cannot reach Target Size even with our
    size added earns nothing at all, so that case is now zero.
    """
    if two_sided_ok(book, target_size):
        valid_fraction = VALID_FRACTION_TWO_SIDED
    elif two_sided_reachable(book, my_size, target_size):
        valid_fraction = VALID_FRACTION_MARGINAL
    else:
        valid_fraction = VALID_FRACTION_UNREACHABLE

    share = 0.0
    for side in ("yes", "no"):
        existing = total_depth(book, side)
        share += my_size / max(1.0, existing + my_size)
    share /= 2.0

    dollars = None if pool_dollars_per_day is None else (
        pool_dollars_per_day * share * valid_fraction
    )
    return RewardEstimate(share=share, valid_fraction=valid_fraction,
                          dollars_per_day=dollars)


def estimate_fill_cost_dollars(book: OrderBook, my_size: int, hours: float,
                               daily_volume: float) -> float:
    """
    Rough adverse-selection cost of resting `my_size` for `hours`.

    Fills scale with how much volume churns relative to book depth. On a fill,
    assume we lose about half the spread — the trade happened because someone
    wanted it to — plus the maker fee.

    THE SINGLE MOST UNCERTAIN NUMBER IN THIS PROJECT. It exists so the ranking
    is not purely reward-chasing. Replace it with measured fill rate from
    monitor.py as soon as there is data; until then treat the ordering it
    produces as a hypothesis.
    """
    total_depth = (sum(size for _, size in book.yes)
                   + sum(size for _, size in book.no))
    if total_depth <= 0:
        return float("inf")

    spread = spread_cents(book)
    mid = mid_cents(book)
    if spread is None or mid is None:
        return float("inf")

    hourly_turnover = daily_volume / total_depth / 24.0
    expected_filled = min(my_size, my_size * hourly_turnover * hours)

    adverse_cents = max(1.0, spread / 2.0)
    fee_cents = trade_fee_cents(round(mid), 1, maker=True)
    return expected_filled * (adverse_cents + fee_cents) / 100.0


def capital_required_dollars(book: OrderBook, my_size: int) -> float:
    """Cash tied up resting my_size on both sides at the current best prices."""
    if not book.yes or not book.no:
        return 0.0
    return my_size * (book.yes[0][0] + book.no[0][0]) / 100.0
