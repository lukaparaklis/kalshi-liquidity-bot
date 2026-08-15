"""
test_client.py
--------------
Transport-level behaviour of KalshiClient that risk.py depends on but cannot
express with a duck-typed fake: HTTP status handling.

The 404-on-cancel case is the one that matters operationally. Verified on demo:
cancelling an already-cancelled order returns 404 not_found, and
GET /portfolio/orders lags order state by a beat — right after placing two
orders it reported none resting, and right after cancelling them it reported
both. So reconciliation routinely tries to cancel something already gone. If
that counted as a failed cancel, startup would block permanently.
"""

from __future__ import annotations

import json

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from src.auth import DEMO_BASE, Credentials, KalshiAPIError, KalshiClient

# One key for the module: generating RSA per test is slow and buys nothing.
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class StubResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.text = json.dumps(payload) if payload is not None else ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> dict:
        return json.loads(self.text)


class StubSession:
    """Replays a fixed sequence of responses and records what was sent."""

    def __init__(self, *responses: StubResponse):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def request(self, method, url, **_kwargs):
        self.calls.append((method, url))
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def make_client(*responses: StubResponse) -> KalshiClient:
    creds = Credentials(key_id="key", private_key=KEY, env="demo", base_url=DEMO_BASE)
    client = KalshiClient(creds, max_attempts=2)
    client.session = StubSession(*responses)
    return client


class TestCancelOrder:
    def test_a_normal_cancel_returns_the_payload(self):
        client = make_client(StubResponse(200, {"order_id": "o1", "reduced_by": "1.00"}))

        result = client.cancel_order("o1")

        assert result["order_id"] == "o1"
        assert result.get("already_gone") is not True

    def test_404_counts_as_cancelled(self):
        """The order is not resting, which is all cancel had to achieve."""
        client = make_client(StubResponse(404, {"error": {"code": "not_found"}}))

        result = client.cancel_order("gone")

        assert result["already_gone"] is True
        assert result["order_id"] == "gone"

    def test_it_uses_the_v2_endpoint(self):
        client = make_client(StubResponse(200, {"order_id": "o1"}))

        client.cancel_order("o1")

        method, url = client.session.calls[0]
        assert method == "DELETE"
        assert url.endswith("/portfolio/events/orders/o1")

    def test_other_errors_still_raise(self):
        """Only 'already gone' is forgiven; a 403 must not read as success."""
        client = make_client(StubResponse(403, {"error": {"code": "forbidden"}}))

        with pytest.raises(KalshiAPIError) as excinfo:
            client.cancel_order("o1")

        assert excinfo.value.status == 403


class TestCreateOrder:
    def test_it_posts_the_v2_shape_to_the_v2_endpoint(self):
        client = make_client(StubResponse(201, {"order_id": "o1"}))

        client.create_order("MKT-A", "no", count=3, price_cents=38)

        method, url = client.session.calls[0]
        assert method == "POST"
        assert url.endswith("/portfolio/events/orders")

    def test_the_response_is_flat(self):
        """v2 returns order_id at the top level, with no {"order": ...} wrapper."""
        client = make_client(StubResponse(201, {"order_id": "o1", "ts_ms": 1}))

        assert client.create_order("MKT-A", "yes", 1, 60)["order_id"] == "o1"

    def test_a_server_error_is_not_retried_by_default(self):
        """
        A create that timed out may already have reached the matching engine.
        One attempt, then raise — a duplicate order costs money.
        """
        client = make_client(StubResponse(500, {"error": "boom"}))

        with pytest.raises(KalshiAPIError):
            client.create_order("MKT-A", "yes", 1, 60)

        assert len(client.session.calls) == 1

    @pytest.mark.parametrize("price_cents", [0, 100, -1])
    def test_prices_outside_the_tick_range_are_rejected_locally(self, price_cents):
        client = make_client(StubResponse(201, {"order_id": "o1"}))

        with pytest.raises(ValueError):
            client.create_order("MKT-A", "yes", 1, price_cents)

        assert client.session.calls == []
