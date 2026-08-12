#!/usr/bin/env python3
"""
kalshi_btc15m_bot.py
--------------------
Momentum bot for Kalshi's KXBTC15M (Bitcoin 15-minute up/down) markets.

Strategy
  At each quarter-hour boundary, wait ENTRY_DELAY seconds, read the market's
  own price, and take the side the market is leaning. Hold to settlement.
  (Holding to expiry means no exit fill and no exit fee.)

Modes
  inspect  - print the live market objects, place nothing. RUN THIS FIRST.
  paper    - run the full loop, log intended trades, place nothing.
  live     - actually place orders.
  score    - read the trade log, fetch settlement results, print P&L.

Setup
  pip install requests cryptography
  export KALSHI_KEY_ID="..."
  export KALSHI_PRIVATE_KEY_PATH="/path/to/kalshi_private_key.pem"
  export KALSHI_ENV="demo"        # or "prod"

  python kalshi_btc15m_bot.py inspect
  python kalshi_btc15m_bot.py paper
  python kalshi_btc15m_bot.py score

VERIFY EVERYTHING against https://docs.kalshi.com before going live. Endpoint
shapes, field names, and fee multipliers change. `inspect` exists so you can
see the real payloads instead of trusting my assumptions about them.
"""

from __future__ import annotations

import base64
import csv
import json
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

# --------------------------------------------------------------------------
# CONFIG - tune here
# --------------------------------------------------------------------------

SERIES_TICKER = "KXBTC15M"

LOCAL_TZ = ZoneInfo("America/Los_Angeles")
START_HOUR = 1          # inclusive, local time
END_HOUR = 5            # exclusive, local time

ENTRY_DELAY_SEC = 30    # how long after the window opens to read the market
CONTRACTS = 5           # contracts per trade
MAX_ENTRY_PRICE = 70    # cents; skip if the side we want costs more than this
MIN_EDGE_CENTS = 3      # skip if mid is within this many cents of 50 (no signal)
MAX_SPREAD_CENTS = 4    # skip if the book is wider than this (thin overnight)

MAX_TRADES_PER_SESSION = 16
DAILY_LOSS_LIMIT_USD = 25.0     # stop for the night if realized loss exceeds this
CONSECUTIVE_LOSS_LIMIT = 4      # stop for the night after this many losses in a row

FILL_TIMEOUT_SEC = 20   # cancel an unfilled order after this long
TAKER_FEE_MULTIPLIER = 0.07     # verify against kalshi.com/fee-schedule for crypto

TRADE_LOG = Path("kalshi_btc15m_trades.csv")

PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"
DEMO_BASE = "https://external-api.demo.kalshi.co/trade-api/v2"


# --------------------------------------------------------------------------
# AUTH / CLIENT
# --------------------------------------------------------------------------

class KalshiClient:
    """Minimal Kalshi REST client with RSA-PSS request signing."""

    def __init__(self, key_id: str, private_key: rsa.RSAPrivateKey, base_url: str):
        self.key_id = key_id
        self.private_key = private_key
        self.base_url = base_url.rstrip("/")
        self.path_prefix = "/" + base_url.split("/", 3)[3].rstrip("/")  # "/trade-api/v2"
        self.session = requests.Session()

    @classmethod
    def from_env(cls) -> "KalshiClient":
        key_id = os.environ.get("KALSHI_KEY_ID")
        key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        if not key_id or not key_path:
            sys.exit("Set KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH.")
        with open(key_path, "rb") as fh:
            private_key = serialization.load_pem_private_key(fh.read(), password=None)
        env = os.environ.get("KALSHI_ENV", "demo").lower()
        base = PROD_BASE if env in ("prod", "production") else DEMO_BASE
        log(f"env={env}  base={base}")
        return cls(key_id, private_key, base)

    def _headers(self, method: str, path: str) -> dict:
        """
        Signature = RSA-PSS(SHA256) over the concatenation of
        timestamp_ms + UPPERCASE_METHOD + path.
        The path is the full path from the API root WITHOUT query parameters.
        """
        ts = str(int(time.time() * 1000))
        message = (ts + method.upper() + path).encode()
        sig = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256().digest_size,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(self, method: str, route: str, params=None, body=None) -> dict:
        path = self.path_prefix + route
        url = self.base_url + route
        for attempt in range(4):
            try:
                resp = self.session.request(
                    method,
                    url,
                    headers=self._headers(method, path),
                    params=params,
                    data=json.dumps(body) if body is not None else None,
                    timeout=10,
                )
            except requests.RequestException as exc:
                log(f"network error ({exc}); retrying")
                time.sleep(2 ** attempt)
                continue

            if resp.status_code == 429:
                log("rate limited; backing off")
                time.sleep(2 ** attempt)
                continue
            if resp.status_code >= 500:
                log(f"server error {resp.status_code}; retrying")
                time.sleep(2 ** attempt)
                continue
            if not resp.ok:
                raise RuntimeError(f"{method} {route} -> {resp.status_code}: {resp.text[:400]}")
            return resp.json() if resp.text else {}
        raise RuntimeError(f"{method} {route} failed after retries")

    # --- endpoints -------------------------------------------------------

    def open_markets(self, series_ticker: str) -> list[dict]:
        data = self._request(
            "GET", "/markets",
            params={"series_ticker": series_ticker, "status": "open", "limit": 200},
        )
        return data.get("markets", [])

    def market(self, ticker: str) -> dict:
        return self._request("GET", f"/markets/{ticker}").get("market", {})

    def balance(self) -> dict:
        return self._request("GET", "/portfolio/balance")

    def create_order(self, ticker: str, side: str, count: int, price_cents: int) -> dict:
        body = {
            "ticker": ticker,
            "client_order_id": str(uuid.uuid4()),
            "action": "buy",
            "side": side,               # "yes" or "no"
            "count": count,
            "type": "limit",
        }
        body["yes_price" if side == "yes" else "no_price"] = price_cents
        return self._request("POST", "/portfolio/orders", body=body)

    def get_order(self, order_id: str) -> dict:
        return self._request("GET", f"/portfolio/orders/{order_id}").get("order", {})

    def cancel_order(self, order_id: str) -> dict:
        return self._request("DELETE", f"/portfolio/orders/{order_id}")


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------

def log(msg: str) -> None:
    stamp = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def next_quarter_boundary(now: datetime) -> datetime:
    """The next :00 / :15 / :30 / :45 at or after `now`."""
    minute = (now.minute // 15 + 1) * 15
    base = now.replace(second=0, microsecond=0, minute=0)
    return base + timedelta(minutes=minute)


def in_trading_window(now_local: datetime) -> bool:
    return START_HOUR <= now_local.hour < END_HOUR


def taker_fee_usd(contracts: int, price_cents: int) -> float:
    """round_up_to_cent(multiplier * C * P * (1-P)), P in dollars."""
    p = price_cents / 100.0
    raw = TAKER_FEE_MULTIPLIER * contracts * p * (1 - p)
    return math.ceil(raw * 100) / 100.0


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def pick_current_market(markets: list[dict], target_close: datetime) -> dict | None:
    """Pick the open KXBTC15M market closing at the current window's boundary."""
    candidates = []
    for m in markets:
        close = m.get("close_time") or m.get("expiration_time")
        if not close:
            continue
        try:
            if abs((parse_ts(close) - target_close).total_seconds()) < 60:
                candidates.append(m)
        except ValueError:
            continue
    if not candidates:
        return None
    if len(candidates) > 1:
        log(f"WARNING: {len(candidates)} markets match this window: "
            f"{[c.get('ticker') for c in candidates]} - using highest volume. "
            f"Run `inspect` and confirm this is what you want.")
        candidates.sort(key=lambda m: m.get("volume", 0), reverse=True)
    return candidates[0]


# --------------------------------------------------------------------------
# SIGNAL
# --------------------------------------------------------------------------

@dataclass
class Signal:
    side: str | None        # "yes", "no", or None to skip
    entry_price: int        # cents we'd pay
    mid: float
    spread: int
    reason: str


def read_signal(market: dict) -> Signal:
    """
    The market's own price IS the 'above or below' read: KXBTC15M opens with
    the strike at spot, so mid > 50 means BTC has moved above the strike and
    mid < 50 means below. No external price feed needed, and no risk of
    disagreeing with the venue's own index.
    """
    yes_bid = market.get("yes_bid") or 0
    yes_ask = market.get("yes_ask") or 0
    if not yes_bid or not yes_ask:
        return Signal(None, 0, 0, 0, "empty book")

    mid = (yes_bid + yes_ask) / 2
    spread = yes_ask - yes_bid

    if spread > MAX_SPREAD_CENTS:
        return Signal(None, 0, mid, spread, f"spread {spread}c too wide")
    if abs(mid - 50) < MIN_EDGE_CENTS:
        return Signal(None, 0, mid, spread, f"mid {mid:.1f}c too close to 50")

    if mid > 50:
        side, entry = "yes", yes_ask                # buy YES at the ask
    else:
        side, entry = "no", 100 - yes_bid           # buy NO at the NO ask

    if entry > MAX_ENTRY_PRICE:
        return Signal(None, entry, mid, spread, f"entry {entry}c above cap")
    if entry < 1 or entry > 99:
        return Signal(None, entry, mid, spread, "entry out of range")

    return Signal(side, entry, mid, spread, "ok")


# --------------------------------------------------------------------------
# LOGGING / SCORING
# --------------------------------------------------------------------------

LOG_FIELDS = [
    "timestamp_utc", "mode", "ticker", "close_time", "side",
    "entry_price_cents", "contracts", "mid", "spread", "est_fee_usd", "order_id",
]


def append_trade(row: dict) -> None:
    new = not TRADE_LOG.exists()
    with TRADE_LOG.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def score(client: KalshiClient) -> None:
    """Fetch settlement results for every logged trade and print P&L."""
    if not TRADE_LOG.exists():
        sys.exit(f"No trade log at {TRADE_LOG}.")

    rows = list(csv.DictReader(TRADE_LOG.open()))
    wins = losses = pending = 0
    gross = fees = 0.0

    for row in rows:
        try:
            m = client.market(row["ticker"])
        except RuntimeError as exc:
            log(f"could not fetch {row['ticker']}: {exc}")
            continue

        result = (m.get("result") or "").lower()
        if result not in ("yes", "no"):
            pending += 1
            continue

        n = int(row["contracts"])
        entry = int(row["entry_price_cents"])
        fee = float(row["est_fee_usd"] or 0)
        cost = n * entry / 100.0
        payout = n * 1.0 if result == row["side"] else 0.0

        gross += payout - cost
        fees += fee
        if payout > cost:
            wins += 1
        else:
            losses += 1

    settled = wins + losses
    print("\n--- RESULTS " + "-" * 48)
    print(f"settled trades : {settled}   (pending: {pending})")
    if settled:
        print(f"win rate       : {wins / settled:.1%}  ({wins}W / {losses}L)")
    print(f"gross P&L      : ${gross:+.2f}")
    print(f"fees           : ${fees:.2f}")
    print(f"net P&L        : ${gross - fees:+.2f}")
    if settled:
        print(f"net per trade  : ${(gross - fees) / settled:+.3f}")
    print("-" * 60)
    print("A positive net over fewer than ~200 trades is not evidence of edge.")


# --------------------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------------------

@dataclass
class SessionState:
    trades: int = 0
    consecutive_losses: int = 0
    realized_loss: float = 0.0
    seen_windows: set = field(default_factory=set)

    def halted(self) -> str | None:
        if self.trades >= MAX_TRADES_PER_SESSION:
            return "trade cap reached"
        if self.consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
            return f"{self.consecutive_losses} losses in a row"
        if self.realized_loss >= DAILY_LOSS_LIMIT_USD:
            return f"daily loss limit (${self.realized_loss:.2f})"
        return None


def inspect(client: KalshiClient) -> None:
    """Print raw market objects so you can verify field names before trading."""
    markets = client.open_markets(SERIES_TICKER)
    log(f"{len(markets)} open {SERIES_TICKER} markets")
    now = datetime.now(timezone.utc)
    target = next_quarter_boundary(now)
    log(f"current window closes at {target.isoformat()}")
    for m in markets[:6]:
        print(json.dumps(m, indent=2)[:1400])
        print("-" * 60)
    chosen = pick_current_market(markets, target)
    if chosen:
        log(f"would trade: {chosen.get('ticker')}  "
            f"yes_bid={chosen.get('yes_bid')} yes_ask={chosen.get('yes_ask')}")
        print(f"signal -> {read_signal(chosen)}")
    else:
        log("no market matched the current window - check close_time field name")


def execute(client: KalshiClient, market: dict, sig: Signal, live: bool) -> None:
    ticker = market["ticker"]
    fee = taker_fee_usd(CONTRACTS, sig.entry_price)
    cost = CONTRACTS * sig.entry_price / 100.0

    log(f"SIGNAL {ticker}: mid={sig.mid:.1f}c spread={sig.spread}c -> "
        f"BUY {CONTRACTS} {sig.side.upper()} @ {sig.entry_price}c "
        f"(cost ${cost:.2f}, fee ~${fee:.2f}, breakeven {(cost + fee) / CONTRACTS:.1f}c)")

    order_id = ""
    if live:
        resp = client.create_order(ticker, sig.side, CONTRACTS, sig.entry_price)
        order = resp.get("order", resp)
        order_id = order.get("order_id", "")
        log(f"order placed: {order_id}")

        deadline = time.time() + FILL_TIMEOUT_SEC
        while time.time() < deadline:
            time.sleep(3)
            status = client.get_order(order_id).get("status", "")
            if status in ("executed", "filled"):
                log("filled")
                break
            if status in ("canceled", "cancelled"):
                log("canceled by exchange")
                return
        else:
            log("not filled in time - canceling")
            try:
                client.cancel_order(order_id)
            except RuntimeError as exc:
                log(f"cancel failed: {exc}")
            return
    else:
        log("PAPER MODE - no order sent")

    append_trade({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "live" if live else "paper",
        "ticker": ticker,
        "close_time": market.get("close_time", ""),
        "side": sig.side,
        "entry_price_cents": sig.entry_price,
        "contracts": CONTRACTS,
        "mid": f"{sig.mid:.1f}",
        "spread": sig.spread,
        "est_fee_usd": f"{fee:.2f}",
        "order_id": order_id,
    })


def run(client: KalshiClient, live: bool) -> None:
    state = SessionState()
    log(f"starting in {'LIVE' if live else 'PAPER'} mode; "
        f"window {START_HOUR:02d}:00-{END_HOUR:02d}:00 {LOCAL_TZ}")
    if live:
        try:
            log(f"balance: {client.balance()}")
        except RuntimeError as exc:
            sys.exit(f"auth check failed: {exc}")

    while True:
        now_utc = datetime.now(timezone.utc)
        now_local = now_utc.astimezone(LOCAL_TZ)

        if not in_trading_window(now_local):
            state = SessionState()   # reset for the next night
            time.sleep(30)
            continue

        halt = state.halted()
        if halt:
            log(f"HALTED for this session: {halt}. Sleeping until the window closes.")
            time.sleep(60)
            continue

        boundary = next_quarter_boundary(now_utc)
        window_open = boundary - timedelta(minutes=15)
        entry_at = window_open + timedelta(seconds=ENTRY_DELAY_SEC)
        key = boundary.isoformat()

        if key in state.seen_windows or now_utc > entry_at + timedelta(seconds=45):
            # Already handled, or we missed the entry slot for this window.
            time.sleep(min(5, max(1, (boundary - now_utc).total_seconds())))
            continue

        wait = (entry_at - now_utc).total_seconds()
        if wait > 0:
            log(f"waiting {wait:.0f}s for entry on window closing {boundary:%H:%M}")
            time.sleep(wait)

        state.seen_windows.add(key)
        try:
            markets = client.open_markets(SERIES_TICKER)
            market = pick_current_market(markets, boundary)
            if not market:
                log("no market for this window - skipping")
                continue
            market = client.market(market["ticker"])   # refresh for a current book
            sig = read_signal(market)
            if sig.side is None:
                log(f"skip {market['ticker']}: {sig.reason}")
                continue
            execute(client, market, sig, live)
            state.trades += 1
        except RuntimeError as exc:
            log(f"error this window: {exc}")
        except Exception as exc:  # noqa: BLE001 - never let one window kill the loop
            log(f"unexpected error: {exc!r}")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    if mode not in ("inspect", "paper", "live", "score"):
        sys.exit("usage: kalshi_btc15m_bot.py [inspect|paper|live|score]")

    client = KalshiClient.from_env()

    if mode == "inspect":
        inspect(client)
    elif mode == "score":
        score(client)
    else:
        if mode == "live":
            env = os.environ.get("KALSHI_ENV", "demo").lower()
            target = "PRODUCTION (real money)" if env.startswith("prod") else "the demo environment"
            confirm = input(f"Place live orders against {target}? Type YES: ")
            if confirm.strip() != "YES":
                sys.exit("aborted")
        try:
            run(client, live=(mode == "live"))
        except KeyboardInterrupt:
            log("stopped by user")


if __name__ == "__main__":
    main()
