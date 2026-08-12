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


def load_credentials(env_file: Path | str | None = DEFAULT_ENV_FILE) -> Credentials:
    """Load key id, private key and target environment from .env / os.environ."""
    if env_file is not None:
        load_dotenv(env_file)

    key_id = os.environ.get("KALSHI_KEY_ID", "").strip().strip("\"'")
    key_path_raw = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip().strip("\"'")

    if not key_id:
        raise KalshiConfigError("KALSHI_KEY_ID is not set (check .env)")
    if not key_path_raw:
        raise KalshiConfigError("KALSHI_PRIVATE_KEY_PATH is not set (check .env)")

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

    env = _normalize_env(os.environ.get("KALSHI_ENV"))
    return Credentials(
        key_id=key_id,
        private_key=private_key,
        env=env,
        base_url=PROD_BASE if env == "prod" else DEMO_BASE,
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
    # Only what step 1 needs. Trading endpoints land with quoter.py/risk.py.

    def balance(self) -> dict:
        """GET /portfolio/balance -> {"balance": <cents>, ...}"""
        return self.get("/portfolio/balance")

    def exchange_status(self) -> dict:
        """GET /exchange/status. Unsigned in practice, but harmless to sign."""
        return self.get("/exchange/status")


def build_client(**kwargs) -> KalshiClient:
    """from_env() with a readable message instead of a traceback on misconfig."""
    try:
        return KalshiClient.from_env(**kwargs)
    except KalshiConfigError as exc:
        sys.exit(f"config error: {exc}")
