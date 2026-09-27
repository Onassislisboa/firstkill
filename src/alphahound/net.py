"""HTTP plumbing shared by every data source and venue adapter.

Two things live here that nothing else should reimplement: a per-host token
bucket, and retry-with-backoff that respects Retry-After. Free-tier data APIs
will 429 you constantly, and a bot that treats a 429 as a hard failure will
skip exactly the candidates that everyone else is also looking at - which is
to say, the interesting ones.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any
from urllib.parse import urlsplit

import httpx

from .log import get

log = get("net")

RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class HttpError(RuntimeError):
    def __init__(self, status: int, body: str, url: str) -> None:
        super().__init__(f"HTTP {status} for {url}: {body[:300]}")
        self.status = status
        self.body = body
        self.url = url


class TokenBucket:
    def __init__(self, rate_per_sec: float, burst: int) -> None:
        self.rate = rate_per_sec
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def take(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                await asyncio.sleep((1.0 - self.tokens) / self.rate)


class Http:
    """Shared async client. One instance per process; pass it around."""

    def __init__(
        self,
        timeout: float = 8.0,
        default_rate: float = 8.0,
        default_burst: int = 16,
    ) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            headers={"user-agent": "FirstKill/0.1 (+https://github.com/Onassislisboa/firstkill)"},
            follow_redirects=True,
        )
        self._buckets: dict[str, TokenBucket] = {}
        self._default = (default_rate, default_burst)
        self._overrides: dict[str, tuple[float, int]] = {}

    def limit(self, host: str, rate_per_sec: float, burst: int) -> None:
        """Meter a host, or a path prefix under it ("host/orders")."""
        self._overrides[host] = (rate_per_sec, burst)
        self._buckets.pop(host, None)

    def _bucket(self, url: str) -> TokenBucket:
        parts = urlsplit(url)
        target = parts.netloc + parts.path
        # Longest registered prefix wins, so one host can hold several buckets.
        # Dexscreener meters /latest/dex at 300/min but /orders and /token-*
        # at 60/min; on a shared bucket the cheap endpoints spend the budget
        # and the quote loop is the one that gets the 429.
        key = max(
            (k for k in self._overrides if target.startswith(k)),
            key=len,
            default=parts.netloc,
        )
        bucket = self._buckets.get(key)
        if bucket is None:
            rate, burst = self._overrides.get(key, self._default)
            bucket = TokenBucket(rate, burst)
            self._buckets[key] = bucket
        return bucket

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        content: bytes | str | None = None,
        headers: dict[str, str] | None = None,
        retries: int = 3,
    ) -> Any:
        attempt = 0
        while True:
            await self._bucket(url).take()
            try:
                resp = await self._client.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    content=content,
                    headers=headers,
                )
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if attempt >= retries:
                    raise HttpError(0, str(exc), url) from exc
                await self._backoff(attempt, None)
                attempt += 1
                continue

            if resp.status_code in RETRY_STATUS and attempt < retries:
                await self._backoff(attempt, resp.headers.get("retry-after"))
                attempt += 1
                continue
            if resp.status_code >= 400:
                raise HttpError(resp.status_code, resp.text, url)
            if not resp.content:
                return None
            try:
                return resp.json()
            except ValueError:
                return resp.text

    async def get(self, url: str, **kw: Any) -> Any:
        return await self.request("GET", url, **kw)

    async def post(self, url: str, **kw: Any) -> Any:
        return await self.request("POST", url, **kw)

    @staticmethod
    async def _backoff(attempt: int, retry_after: str | None) -> None:
        if retry_after:
            try:
                await asyncio.sleep(min(float(retry_after), 10.0))
                return
            except ValueError:
                pass
        await asyncio.sleep(min(0.35 * (2**attempt), 6.0))

    async def aclose(self) -> None:
        await self._client.aclose()


async def gather_ok(*aws: Any) -> list[Any]:
    """asyncio.gather that logs and drops failures instead of poisoning the
    whole batch. Used when enriching N candidates: one dead RPC call should
    cost you one candidate, not the tick."""
    results = await asyncio.gather(*aws, return_exceptions=True)
    out = []
    for r in results:
        if isinstance(r, BaseException):
            log.debug("subtask failed", extra={"error": str(r)})
            continue
        out.append(r)
    return out
