"""
auth.py
-------
RSA-PSS request signing and a minimal retrying REST client for Kalshi.

Nothing else in this project works until signing is right, so this module is
deliberately small and deliberately paranoid.

Signing contract (verify at https://docs.kalshi.com if a 401 shows up):

    message   = timestamp_ms + UPPERCASE_METHOD + path_without_query
    signature = base64( RSA-PSS-SHA256( message ) )   # PSS, salt = digest size

    KALSHI-ACCESS-KEY        <key id>
    KALSHI-ACCESS-TIMESTAMP  <timestamp_ms>
    KALSHI-ACCESS-SIGNATURE  <signature>

Two failure modes account for most 401s and neither returns a useful message:
timestamps in seconds instead of milliseconds, and signing the path with the
query string attached. Both are handled here; don't "simplify" them away.
"""

from __future__ import annotations

import base64
import json
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"
DEMO_BASE = "https://external-api.demo.kalshi.co/trade-api/v2"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = REPO_ROOT / ".env"

# Retry tuning. Backoff is exponential with full jitter — lockstep retries from
# several coroutines hitting the same rate limit bucket is its own problem.
MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 0.5
BACKOFF_CAP_S = 8.0
REQUEST_TIMEOUT_S = 10.0


# --------------------------------------------------------------------------
# ERRORS
# --------------------------------------------------------------------------

class KalshiError(Exception):
    """Base for everything this module raises."""


class KalshiConfigError(KalshiError):
    """Missing or unusable credentials/config."""


class ReadOnlyError(KalshiError):
    """A write was attempted through a client that cannot write."""


class KalshiAPIError(KalshiError):
    """Non-retryable HTTP error, or retries exhausted."""

    def __init__(self, method: str, route: str, status: int | None, body: str):
        self.method = method
        self.route = route
        self.status = status
        self.body = body
        status_txt = status if status is not None else "no response"
        super().__init__(f"{method} {route} -> {status_txt}: {body[:400]}")


# --------------------------------------------------------------------------
# LOGGING
# --------------------------------------------------------------------------

def log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print(f"[{stamp}] {msg}", flush=True)


def encode_order(side: str, price_cents: int) -> tuple[str, str]:
    """
    Translate this project's yes/no vocabulary into the v2 single-book shape.
    Returns (book_side, price_as_dollar_string).

    v2 replaced "buy yes / buy no" with one book quoted in yes terms and a
    bid/ask direction. Kalshi's changelog calls outcome_side (yes/no) and
    book_side (bid/ask) "the same directional bit in market vocabulary", so:

        buy yes @ 60c  ->  bid @ "0.6000"
        buy no  @ 38c  ->  ask @ "0.6200"      (selling yes at 62c)

    VERIFIED against demo on 2026-08-15. Posting side=ask price="0.9900" came
    back from GET /portfolio/orders as:

        {"book_side": "ask", "outcome_side": "no",
         "yes_price_dollars": "0.9900", "no_price_dollars": "0.0100"}

    An ask at 0.99 is therefore buying NO at 1c, which is exactly what
    encode_order("no", 1) produces. The bid leg matched too: side=bid
    price="0.0100" returned outcome_side "yes" at yes_price_dollars "0.0100".
    """
    if side == "yes":
        return "bid", f"{price_cents / 100:.4f}"
    if side == "no":
        return "ask", f"{(100 - price_cents) / 100:.4f}"
    raise ValueError(f"side must be 'yes' or 'no', got {side!r}")


def _clean(params: dict) -> dict | None:
    """Drop unset query params. Query strings are never part of the signature."""
    present = {key: value for key, value in params.items() if value is not None}
    return present or None


def mask(secret: str) -> str:
    """Render a credential for logs. Never print one raw."""
    if not secret:
        return "<empty>"
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]}"


# --------------------------------------------------------------------------
# .env LOADING
# --------------------------------------------------------------------------

def load_dotenv(path: Path | str = DEFAULT_ENV_FILE, *, override: bool = False) -> dict:
    """
    Read a .env file into os.environ.

    Small hand-rolled parser so the project keeps a two-package dependency
    footprint. Handles `KEY=value`, `KEY="value"`, `KEY='value'`, `export KEY=`,
    blank lines and `#` comments. Real values already in the environment win
    unless override=True, so an exported var can shadow the file.
    """
    path = Path(path)
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


# --------------------------------------------------------------------------
# CREDENTIALS
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Credentials:
    key_id: str
    private_key: rsa.RSAPrivateKey
    env: str
    base_url: str

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"

    def __repr__(self) -> str:  # keep key material out of tracebacks
        return f"Credentials(env={self.env!r}, key_id={mask(self.key_id)!r})"


def _normalize_env(raw: str | None) -> str:
    """Anything that isn't explicitly production is demo. Fail safe."""
    value = (raw or "demo").strip().strip("\"'").lower()
    return "prod" if value in ("prod", "production", "live") else "demo"


def _read_private_key(key_path_raw: str) -> rsa.RSAPrivateKey:
    key_path = Path(key_path_raw).expanduser()
    if not key_path.exists():
        raise KalshiConfigError(f"private key not found at {key_path}")

    key_bytes = key_path.read_bytes()
    try:
        private_key = serialization.load_pem_private_key(key_bytes, password=None)
    except TypeError as exc:
        raise KalshiConfigError(
            f"private key at {key_path} is encrypted; this client expects an "
            "unencrypted PEM"
        ) from exc
    except ValueError as exc:
        raise KalshiConfigError(f"could not parse private key at {key_path}: {exc}") from exc
    finally:
        del key_bytes

    if not isinstance(private_key, rsa.RSAPrivateKey):
        raise KalshiConfigError(
            f"key at {key_path} is {type(private_key).__name__}, not RSA. Kalshi "
            "signs with RSA-PSS."
        )
    return private_key


def _env_value(name: str) -> str:
    return os.environ.get(name, "").strip().strip("\"'")


def load_credentials(env_file: Path | str | None = DEFAULT_ENV_FILE) -> Credentials:
    """Load key id, private key and target environment from .env / os.environ."""
    if env_file is not None:
        load_dotenv(env_file)

    key_id = _env_value("KALSHI_KEY_ID")
    key_path_raw = _env_value("KALSHI_PRIVATE_KEY_PATH")

    if not key_id:
        raise KalshiConfigError("KALSHI_KEY_ID is not set (check .env)")
    if not key_path_raw:
        raise KalshiConfigError("KALSHI_PRIVATE_KEY_PATH is not set (check .env)")

    env = _normalize_env(os.environ.get("KALSHI_ENV"))
    return Credentials(
        key_id=key_id,
        private_key=_read_private_key(key_path_raw),
        env=env,
        base_url=PROD_BASE if env == "prod" else DEMO_BASE,
    )


def load_readonly_prod_credentials(
        env_file: Path | str | None = DEFAULT_ENV_FILE) -> Credentials:
    """
    Credentials for READING prod. Separate variables, separate key, on purpose.

    KALSHI_PROD_KEY_ID / KALSHI_PROD_PRIVATE_KEY_PATH, and deliberately NOT
    KALSHI_ENV. Paper mode needs prod books because demo has no liquidity worth
    quoting against, but pointing the ordinary trading client at prod by
    flipping one environment variable is exactly the accident this project
    cannot afford. The two paths cannot be confused because they read different
    variables, and this one is only ever handed to a ReadOnlyClient.

    Give the key trade permissions only if you want to find out the hard way
    that a scope grant is not a safety mechanism — the read-only guarantee here
    comes from the client type, but a read-only API key is a good second layer.
    """
    if env_file is not None:
        load_dotenv(env_file)

    key_id = _env_value("KALSHI_PROD_KEY_ID")
    key_path_raw = _env_value("KALSHI_PROD_PRIVATE_KEY_PATH")

    if not key_id or not key_path_raw:
        raise KalshiConfigError(
            "paper mode reads prod books and needs its own read-only credential.\n"
            "Set KALSHI_PROD_KEY_ID and KALSHI_PROD_PRIVATE_KEY_PATH in .env.\n"
            "Use a key with read-only permissions; the client refuses writes "
            "regardless, but defence in depth is the point."
        )

    return Credentials(
        key_id=key_id,
        private_key=_read_private_key(key_path_raw),
        env="prod",
        base_url=PROD_BASE,
    )


# --------------------------------------------------------------------------
# CLIENT
# --------------------------------------------------------------------------

class KalshiClient:
    """
    Signed Kalshi REST client with bounded retries.

    Retries cover transport errors, 429 and 5xx. Writes are only retried when
    the caller marks them idempotent: a POST that created an order and then
    timed out on the response must not be blindly replayed.
    """

    def __init__(self, creds: Credentials, *, timeout: float = REQUEST_TIMEOUT_S,
                 max_attempts: int = MAX_ATTEMPTS):
        self.creds = creds
        self.base_url = creds.base_url.rstrip("/")
        # Sign the full path from the API root, e.g. "/trade-api/v2/portfolio/balance".
        self.path_prefix = urlsplit(self.base_url).path.rstrip("/")
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.session = requests.Session()

    @classmethod
    def from_env(cls, env_file: Path | str | None = DEFAULT_ENV_FILE, **kwargs) -> "KalshiClient":
        return cls(load_credentials(env_file), **kwargs)

    def __repr__(self) -> str:
        return f"KalshiClient(env={self.creds.env!r}, base_url={self.base_url!r})"

    # --- signing ---------------------------------------------------------

    def sign(self, timestamp_ms: str, method: str, path: str) -> str:
        """base64 RSA-PSS(SHA-256) over timestamp_ms + METHOD + path."""
        message = (timestamp_ms + method.upper() + path).encode()
        signature = self.creds.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256().digest_size,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    def _headers(self, method: str, path: str) -> dict:
        # Milliseconds. Seconds yields a 401 with no explanation of why.
        timestamp_ms = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self.creds.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": self.sign(timestamp_ms, method, path),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # --- transport -------------------------------------------------------

    def _sleep_for(self, attempt: int, resp: requests.Response | None) -> float:
        """Honor Retry-After when present, else exponential backoff, full jitter."""
        if resp is not None:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(float(retry_after), BACKOFF_CAP_S * 2)
                except ValueError:
                    pass
        return random.uniform(0, min(BACKOFF_CAP_S, BACKOFF_BASE_S * (2 ** attempt)))

    def request(self, method: str, route: str, *, params: dict | None = None,
                body: dict | None = None, idempotent: bool | None = None) -> dict:
        """
        Send a signed request. `route` is relative to the API root, e.g.
        "/portfolio/balance".

        idempotent defaults to True for GET/DELETE/HEAD and False otherwise.
        Pass idempotent=True on a POST only when the payload carries a
        deduplicating key such as client_order_id.
        """
        method = method.upper()
        if idempotent is None:
            idempotent = method in ("GET", "HEAD", "DELETE")

        # Query params are excluded from the signed path on purpose; requests
        # appends them to the URL only.
        path = self.path_prefix + route
        url = self.base_url + route
        payload = json.dumps(body) if body is not None else None

        last_status: int | None = None
        last_text = ""

        for attempt in range(self.max_attempts):
            resp = None
            try:
                # Headers are rebuilt per attempt: the timestamp is part of the
                # signature and a stale one will be rejected.
                resp = self.session.request(
                    method, url,
                    headers=self._headers(method, path),
                    params=params,
                    data=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_text = f"network error: {exc}"
                if not idempotent:
                    raise KalshiAPIError(method, route, None, last_text) from exc
                log(f"{method} {route}: {last_text}; retrying")
            else:
                last_status, last_text = resp.status_code, resp.text or ""

                if resp.ok:
                    return resp.json() if resp.text else {}

                retryable = resp.status_code == 429 or resp.status_code >= 500
                if not retryable or not idempotent:
                    raise KalshiAPIError(method, route, resp.status_code, last_text)

                reason = "rate limited" if resp.status_code == 429 else f"server error {resp.status_code}"
                log(f"{method} {route}: {reason}; retrying")

            if attempt < self.max_attempts - 1:
                time.sleep(self._sleep_for(attempt, resp))

        raise KalshiAPIError(method, route, last_status,
                             f"failed after {self.max_attempts} attempts: {last_text}")

    def get(self, route: str, **kwargs) -> dict:
        return self.request("GET", route, **kwargs)

    def post(self, route: str, **kwargs) -> dict:
        return self.request("POST", route, **kwargs)

    def delete(self, route: str, **kwargs) -> dict:
        return self.request("DELETE", route, **kwargs)

    # --- endpoints -------------------------------------------------------

    def balance(self) -> dict:
        """GET /portfolio/balance -> {"balance": <cents>, ...}"""
        return self.get("/portfolio/balance")

    def exchange_status(self) -> dict:
        """GET /exchange/status. Unsigned in practice, but harmless to sign."""
        return self.get("/exchange/status")

    # The four risk.py calls. Do not call create_order from anywhere else:
    # RiskManager is the gateway that enforces limits before an order is sent.

    def create_order(self, ticker: str, side: str, count: int, price_cents: int,
                     *, post_only: bool = True, client_order_id: str | None = None,
                     time_in_force: str = "good_till_canceled",
                     self_trade_prevention_type: str = "taker_at_cross",
                     retry: bool = False) -> dict:
        """
        POST /portfolio/events/orders — a resting limit order, v2.

        The v1 endpoint now returns 410 deprecated_v1_order_endpoint. v2 takes
        a bid/ask side with fixed-point dollar strings; see encode_order() for
        the yes/no translation and the assumption it rests on.

        post_only=True by default. A post-only order rests or is rejected — it
        can never cross the spread. CLAUDE.md rates taking liquidity at ~4c a
        contract and says crossing needs an extremely good reason, so the
        exchange enforcing that is strictly better than us remembering to. Pass
        post_only=False only where crossing is the deliberate point.

        retry=False by default, and that is a money decision, not an oversight.
        A create_order that times out may or may not have reached the matching
        engine, so replaying it can double the position. A client_order_id is
        always sent, which is what would MAKE a retry safe — but Kalshi's
        deduplication behaviour is not verified here, and a dropped quote costs
        one requote cycle while a duplicate costs real money.

        Returns the v2 response, which is flat: order_id, client_order_id,
        fill_count, remaining_count, ts_ms — NOT wrapped in {"order": ...}.
        """
        if count <= 0:
            raise ValueError(f"count must be positive, got {count!r}")
        if not 1 <= price_cents <= 99:
            raise ValueError(f"price_cents must be 1..99, got {price_cents!r}")

        book_side, price = encode_order(side, price_cents)
        body = {
            "ticker": ticker,
            "client_order_id": client_order_id or str(uuid.uuid4()),
            "side": book_side,
            "count": str(count),
            "price": price,
            "time_in_force": time_in_force,
            "self_trade_prevention_type": self_trade_prevention_type,
            "post_only": post_only,
        }
        return self.post("/portfolio/events/orders", body=body, idempotent=retry)

    def cancel_order(self, order_id: str) -> dict:
        """
        DELETE /portfolio/events/orders/{id}, the v2 endpoint. Retried on
        5xx/429, being idempotent.

        A 404 counts as success, which is a deliberate reversal of the earlier
        conservative reading. Two things observed on demo forced it:

          - Cancelling an already-cancelled order returns 404 not_found.
          - GET /portfolio/orders lags order state by a beat. Immediately after
            placing two orders it reported none resting; immediately after
            cancelling them it reported both.

        Together those mean reconciliation routinely lists an order that is
        already gone. Treating that as a failed cancel left the bot permanently
        blocked at startup — a safety rule defeating itself. The goal of cancel
        is "this order is not resting", and a 404 already satisfies it. Order
        ids only ever come from the exchange or from our own place_order, so a
        404 here cannot mean "wrong id".
        """
        try:
            return self.delete(f"/portfolio/events/orders/{order_id}")
        except KalshiAPIError as exc:
            if exc.status == 404:
                log(f"cancel {order_id}: already gone (404)")
                return {"order_id": order_id, "already_gone": True}
            raise

    def get_orders(self, *, ticker: str | None = None, status: str | None = None,
                   cursor: str | None = None, limit: int | None = None) -> dict:
        """
        GET /portfolio/orders -> {"orders": [...], "cursor": "..."}

        Still current — only the v1 order *mutation* endpoints returned 410, so
        there is no /portfolio/events/orders GET to move to. Kalshi's changelog
        does note that legacy portfolio endpoints now cost 10x the v2 rate-limit
        budget, which matters for reconciliation frequency, not correctness.

        Order objects use v2 field names: outcome_side (yes/no) rather than
        side, remaining_count_fp rather than remaining_count, and
        yes_price_dollars/no_price_dollars rather than integer cents.
        RiskManager._parse_order reads both spellings.

        Callers must follow the cursor. Reading only the first page is the
        silent failure mode: cancel what you saw, report success, leave the
        rest resting on the exchange.
        """
        params = {"ticker": ticker, "status": status, "cursor": cursor, "limit": limit}
        return self.get("/portfolio/orders", params=_clean(params))

    def get_positions(self, *, ticker: str | None = None, cursor: str | None = None,
                      limit: int | None = None,
                      settlement_status: str | None = None) -> dict:
        """
        GET /portfolio/positions -> {"market_positions": [...], "cursor": "..."}

        Each market position carries a signed `position`: positive is long yes,
        negative is long no. Paginated, same as orders.

        settlement_status defaults to the server's choice. Filtering to
        unsettled positions would be more precise, but including settled ones
        only over-states exposure, which fails safe — so this stays unset until
        the parameter's accepted values are confirmed against the live API.
        """
        params = {"ticker": ticker, "cursor": cursor, "limit": limit,
                  "settlement_status": settlement_status}
        return self.get("/portfolio/positions", params=_clean(params))


class ReadOnlyClient(KalshiClient):
    """
    A Kalshi client that cannot trade.

    Not a flag, not a mode, not a check the caller has to remember: the write
    path raises before any request is built. `request` is overridden rather
    than only the order helpers, so a method added to KalshiClient later is
    refused too without anyone remembering to update this class.

    This is what makes paper mode structurally safe. Paper reads prod books,
    and prod is where real money lives; a client that merely promises not to
    write is one bug away from writing.
    """

    WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

    def request(self, method: str, route: str, **kwargs) -> dict:
        if method.upper() in self.WRITE_METHODS:
            raise ReadOnlyError(
                f"ReadOnlyClient refused {method.upper()} {route}. This client "
                f"exists so paper mode cannot trade against {self.creds.env}."
            )
        return super().request(method, route, **kwargs)

    def create_order(self, *args, **kwargs):
        raise ReadOnlyError("ReadOnlyClient cannot place orders")

    def cancel_order(self, *args, **kwargs):
        raise ReadOnlyError("ReadOnlyClient cannot cancel orders")


def build_client(**kwargs) -> KalshiClient:
    """from_env() with a readable message instead of a traceback on misconfig."""
    try:
        return KalshiClient.from_env(**kwargs)
    except KalshiConfigError as exc:
        sys.exit(f"config error: {exc}")


def build_readonly_prod_client(env_file: Path | str | None = DEFAULT_ENV_FILE,
                               **kwargs) -> ReadOnlyClient:
    """A ReadOnlyClient pointed at prod, for reading books in paper mode."""
    try:
        creds = load_readonly_prod_credentials(env_file)
    except KalshiConfigError as exc:
        sys.exit(f"config error: {exc}")
    return ReadOnlyClient(creds, **kwargs)
