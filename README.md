# Kalshi Liquidity Incentive Program: is it worth it at retail size?

**Question.** Can a retail-sized market maker make money from Kalshi's
Liquidity Incentive Program by leaving resting orders on the book?

**Answer: no.** I measured 408 markets, and the best one came to about
**+$0.44/day**. That is below Kalshi's $1.00/day payout minimum, so the
program would pay it nothing. The main cause is how pools are set up. Each pool
belongs to an *event*, not a market. It is split across every market in that
event, and then each participant gets a share based on how much of the depth
they provide. After those splits, very little of the headline pool is left for
a small participant.

## Strategies tested and why each was rejected

| Strategy | Result | Why it fails |
|---|---|---|
| BTC 15-min momentum (KXBTC15M) | ~55.8% direction persistence 30s into the window | Breakeven after spread and fees is about 60%. The 55.8% comes from Brownian sign persistence, `1/2 + arcsin(sqrt(t/T))/pi` with t=30s and T=900s, and the market already prices it in. |
| Momentum with a stop-loss | -1.65¢/contract across 500k simulated paths | Market prices are roughly a martingale, so by optional stopping no exit rule can change expected value. The stop just adds a second fee and a second spread crossing. The simulation agrees with the theory. |
| Taking liquidity | Structurally negative | The taker fee is 4x the maker fee, and a taker pays the spread instead of earning it. |
| Liquidity provision (this repo) | Best market +$0.44/day | Measured directly. Below the $1/day payout minimum. |

## The key mechanic

The program does not reward orders at the top of the book. For each snapshot,
Kalshi computes a **Reference Price**: it walks down from the best bid until
the cumulative resting size reaches **1/5 of Target Size**. Orders at or better
than that price get full credit. Orders below it are discounted by
`DiscountFactor ^ ticks_away`.

So an order can rest several ticks behind the best bid and still earn the same
reward as one at the top, while getting filled much less often. The whole
strategy depends on this. `src/book.py` implements it, and
`tests/test_reference_price.py` pins down its behavior.

## Data

- Three paper-trading sessions against production order books, with snapshots
  every 10 seconds: **1,715**, **3,430**, and **4,840** snapshots.
- **Zero fills** at the Reference Price in any session.
- Reward estimates came from per-event pools published at kalshi.com/incentives,
  split across the markets in each event and then scaled by our share of
  qualifying depth.

## Limitations

- **Target Size was assumed.** Kalshi does not publish it per market. The code
  uses 1000 and prints that assumption on every run. A different Target Size
  moves both the Reference Price and our share of depth.
- **The market survey was partial.** 408 markets is not the whole exchange.
- **Most sessions ran off-peak.** Fill rates and competing depth may be
  different during active hours.
- **Zero fills does not mean the strategy is safe.** Orders were small and
  sessions were short. Fills were also inferred from book snapshots, not placed
  orders. Zero fills in this sample means the fill cost was too small to
  measure. It is not evidence that adverse selection is absent.

## Architecture

```
src/
  auth.py       RSA-PSS request signing, retrying REST client, and a ReadOnlyClient
                whose write path raises before a request is built
  book.py       Order book model and Reference Price computation
  economics.py  Fee formulas (checked against a real demo fill), reward estimates,
                fill-cost model. All assumed constants are named and printed
  scanner.py    Read-only. Ranks markets by estimated reward against fill risk
                to decide which ones are worth measuring
  quoter.py     Paper mode. Reads prod books, logs intended quotes at the
                Reference Price, places nothing
  monitor.py    Replays paper logs. Reports would-be fill rate (an upper bound)
                and adverse selection
  risk.py       Safety layer (below)
scripts/
  smoke_balance.py   Signed GET /portfolio/balance, the first auth check
  verify_safety.py   Demo-only manual check of shutdown cancellation
reference/           Earlier standalone prototypes, including the BTC momentum bot
```

**Safety layer (`src/risk.py`).** `RiskManager` is the only path for placing
orders, so every order goes through its checks. It provides:

- cancel-all on SIGINT, SIGTERM, and `atexit`;
- on startup, cancellation of any orders left resting by an earlier process;
- a 60-second dead-man's switch;
- hard position limits per market and in aggregate, checked before every order;
- a daily loss limit.

Paper mode can only run with a `ReadOnlyClient`, and it contains no code that
places orders. There is no live trading mode, because the paper data never
justified building one.

**Tests:** 331, all passing (`pytest`).

## Setup

Tested on Python 3.14.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Create `.env` in the repo root. It is gitignored, along with `*.pem` and
`pools.json`. Do not commit any of them.

```bash
# Trading credential. Demo by default.
KALSHI_ENV=demo                 # demo | prod
KALSHI_KEY_ID=your-key-id
KALSHI_PRIVATE_KEY_PATH=/path/to/demo-key.pem

# Separate read-only credential, used for reading prod books in scanner/paper mode.
KALSHI_PROD_KEY_ID=your-readonly-prod-key-id
KALSHI_PROD_PRIVATE_KEY_PATH=/path/to/prod-readonly-key.pem
```

**Demo vs prod.** `KALSHI_ENV` defaults to demo, and that is where anything
that writes should run. Demo books are almost empty, though, so measurement has
to read production. That uses a *separate* credential (`KALSHI_PROD_*`) which
is only ever given to a `ReadOnlyClient`. Changing `KALSHI_ENV` cannot point
the paper path at a writable prod client. Use a read-only API key for it as
well.

```bash
# Check that signing works (demo)
python scripts/smoke_balance.py

# Rank markets on prod books (read-only)
python -m src.scanner --prod scan
python -m src.scanner --prod book <TICKER>     # one book and its Reference Price

# Paper mode: log intended quotes every 10s, place nothing
python -m src.quoter --top 5 --pools-only --duration 3600
python -m src.quoter --ticker <TICKER> --size 100

# Replay the paper log: fill rate and adverse selection
python -m src.monitor
```

Daily pools are not available through the API. Copy them from
kalshi.com/incentives into `pools.json` as `{"EVENT_OR_PREFIX": dollars_per_day}`.
Without that file, the scanner still ranks markets but cannot estimate dollar
rewards.
