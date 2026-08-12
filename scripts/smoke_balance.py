#!/usr/bin/env python3
"""
smoke_balance.py
----------------
Build order step 1's acceptance test: prove RSA-PSS signing works by making one
authenticated call to GET /portfolio/balance.

    python scripts/smoke_balance.py

Reads .env. Refuses to run against prod unless --allow-prod is passed, since a
smoke test has no business touching the live account by accident.

If this returns 401, the cause is almost always one of:
  - timestamp in seconds rather than milliseconds
  - query string included in the signed path
  - PKCS#1v1.5 padding instead of PSS
  - key id paired with a private key from the other environment
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.auth import KalshiAPIError, build_client, log, mask  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-prod", action="store_true",
                        help="permit running when KALSHI_ENV=prod")
    args = parser.parse_args()

    client = build_client()
    creds = client.creds

    log(f"env={creds.env}  base={client.base_url}  key_id={mask(creds.key_id)}")

    if creds.is_prod and not args.allow_prod:
        log("KALSHI_ENV=prod. Re-run with --allow-prod if that is intentional.")
        return 2

    try:
        result = client.balance()
    except KalshiAPIError as exc:
        log(f"FAILED: {exc}")
        if exc.status in (401, 403):
            log("401/403 means the signature or key id was rejected — see the "
                "checklist at the top of this file.")
        return 1

    log(f"raw response: {result}")

    balance_cents = result.get("balance")
    if isinstance(balance_cents, int):
        log(f"balance: {balance_cents} cents (${balance_cents / 100:,.2f})")
    else:
        log("no integer 'balance' field in the response — field names may have "
            "changed; trust the raw response above.")

    log("OK: signed request succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
