# Kalshi Liquidity Provision Bot

## What this project is

An automated market-maker that earns **Kalshi Liquidity Incentive Program rewards**
by keeping resting orders on the book. It is NOT a directional trading bot.

The bot makes no prediction about any outcome. It earns rent for providing
depth, and its only job is to keep that rent larger than the losses it takes
when its orders get filled.

## What this project is NOT — read before proposing changes

Do not build, suggest, or "improve toward" any of the following. Each was
analyzed and rejected with math, not vibes:

- **Directional momentum strategies.** On KXBTC15M, checking price direction
  30s into a 15-min window gives ~55.8% persistence (Brownian sign-persistence:
  `1/2 + arcsin(sqrt(t/T))/pi`). The market prices this exactly. After ~3c
  spread + fees you need ~60% to break even. Structurally -EV.
- **Stop-losses to "cut risk".** Market prices are approximately a martingale.
  By optional stopping, NO exit rule changes expected value. A stop only adds
  a second fee and a second spread crossing. Simulated cost: -1.65c/contract.
- **Taking liquidity (crossing the spread).** Taker fee is 4x maker fee, plus
  you pay the spread instead of earning it. Worth ~4c/contract. If a proposed
  feature crosses the spread, it needs an extremely good reason.
- **Bitcoin 15-minute markets.** Worst possible venue for a slow participant:
  the underlying trades faster on a hundred other exchanges, so our quotes are
  always stale and every fill is adversely selected. Target SLOW markets.

If asked to add any of these, push back and reference this section.

## The core mechanic (the reason this can work)

Rewards are **not** paid for top-of-book. Kalshi computes a **Reference Price**
per snapshot by walking down from the best bid to the first level where
cumulative resting size reaches **1/5 of Target Size**.

- Orders at or better than the Reference Price → **full credit (1.0x)**
- Orders below it → `DiscountFactor ^ ticks_away`

**Therefore: rest AT the Reference Price, not at the top of book.** Same reward,
materially lower fill probability. This is the entire edge. Any change that
moves quotes closer to the top of book without a reward justification is a
regression.

## Program rules (verify at help.kalshi.com — these change)

- Snapshot of the book **once per second**, at a random moment within the second
- Score = your **share** of qualifying liquidity, per side. Snapshot total is
  2.0 across all participants (yes side + no side)
- Your reward = `your_share × pool × (valid_snapshots / total_snapshots)`
- A snapshot is **excluded** unless BOTH sides meet Target Size — market-wide,
  not just our orders
- Target Size: set per period, >100 and <20,000. **Not published per market.**
  Currently assumed at 1000 in code — flag this assumption in any output
- Daily pool: $1–$1,000 per market per day
- Minimum payout $1.00, rounded down
- US members only; excludes designated market makers, IBs, FCMs
- Program end date: 2027-01-01, modifiable or cancellable at any time

## Fee formulas

```
taker_fee_cents = 0.07   × price × (1 - price) × 100   # price in dollars
maker_fee_cents = 0.0175 × price × (1 - price) × 100   # 1/4 of taker
```

Rounded up to the next cent **per order**, not per contract. Fee peaks at 50c
and decays toward both ends — working 85-95c costs ~1/5 what 50c costs.
Crypto series may carry a higher multiplier; verify per series.

## API essentials

- Prod: `https://api.elections.kalshi.com/trade-api/v2`
- Demo: `https://external-api.demo.kalshi.co/trade-api/v2`
- Auth: RSA-**PSS** (not PKCS#1v1.5, not HMAC) over
  `timestamp_ms + UPPERCASE_METHOD + path_without_query`
- Headers: `KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-TIMESTAMP`, `KALSHI-ACCESS-SIGNATURE`
- Timestamp in **milliseconds**. Wrong units → unhelpful 401
- Rate limits: separate read/write buckets by tier; check `GET /account/limits`
- WebSocket for book deltas — strongly preferred over polling for requote logic

## SAFETY RULES — non-negotiable

These protect real money. Do not relax them to make a feature work.

1. **Cancel-all on shutdown.** If the process dies, orders stay live on the
   exchange. Register SIGINT/SIGTERM handlers AND an atexit hook that cancels
   every resting order. This is the single most important safety property.
2. **Dead-man's switch.** If the main loop hasn't completed an iteration in
   60s, cancel everything. A hung bot with live quotes is the worst state.
3. **Hard position limits** per market and in aggregate, checked before every
   order, not after.
4. **Daily loss limit** that halts new quoting for the session.
5. **Demo first.** `KALSHI_ENV` defaults to demo. Live mode requires an
   explicit typed confirmation, never a flag alone.
6. **Never log or commit** the private key, key ID, or any `.pem`. Add
   `*.pem`, `.env`, `pools.json` to `.gitignore`.
7. **Requote hysteresis.** Only move a quote if the Reference Price moved by
   >= 1 tick. Naive requoting on every book delta will burn the rate limit and
   churn fees.

## Architecture

```
src/
  auth.py        RSA-PSS signing, client, retry/backoff
  book.py        order book model, Reference Price computation
  economics.py   fees, reward estimation, fill-cost model
  quoter.py      quote placement, requote hysteresis, inventory awareness
  risk.py        limits, kill switches, cancel-all, dead-man's switch
  scanner.py     rank markets by reward-vs-fill-risk
  monitor.py     fill logging, reward reconciliation
  main.py        entrypoint: scan | plan | paper | live | monitor
tests/
  test_reference_price.py    ← highest-value tests; this is the core mechanic
  test_economics.py
  test_risk_limits.py
```

## Build order

1. `auth.py` + a passing signed request against **demo**. Nothing works until
   signing is right, and it's the most common failure point.
2. `book.py` + Reference Price, with unit tests against hand-built books.
   Test the edge cases: book too thin to reach 1/5 target, one-sided book,
   ties at a price level.
3. `risk.py` including cancel-all. **Before** anything places an order.
4. `scanner.py` — read-only, safe to run against prod.
5. `quoter.py` in paper mode. Log intended quotes, place nothing.
6. `monitor.py`. Run a week in paper.
7. Live, tiny size, only after paper data justifies it.

## The metric that decides everything

**Fill rate.** Every other number in this project is an estimate. Fill rate is
measured. Track it from day one:

- fills per hour per 100 contracts resting
- realized adverse selection: how did the mid move in the 10s after each fill?
- rewards earned vs. fill losses, per market, per day

If quotes mostly sit untouched → collecting rent, it works.
If quotes fill constantly → being run over, stop.

Any session that adds a feature without improving observability of this metric
is probably a wasted session.

## Honest expectation

The most likely outcome is that this proves the edge is too small or too
crowded, and the correct move is to stop. That is a successful project — it
cost paper-trading time instead of money. Build it to answer the question
cheaply, not to justify a conclusion already reached.

Published backtests on Kalshi crypto series found 695/1000 strategies
profitable in one 30-day window and 102/4904 profitable a week later. Same
series. That instability is the finding. Treat any positive result from a
short sample with deep suspicion, and always report confidence intervals
alongside win rates.
