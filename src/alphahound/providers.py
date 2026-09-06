"""Third-party read-only data providers.

Dexscreener is the backbone because it is free, keyless, multi-chain and
indexes new pairs within seconds. Helius and Birdeye are optional accelerators
for the two things Dexscreener does not expose: an exact holder count and
candle history.

Every method here degrades to None or an empty list instead of raising. A dead
provider should cost you one feature, not the tick - and critically, a missing
feature must read as "unknown" downstream, never as a favourable value. Silently
substituting zero for an unknown holder count is how a bot ends up buying a
token with four holders.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from .log import get
from .models import Candidate, Candle, Chain, now_ms
from .net import Http, HttpError

log = get("providers")

DEXSCREENER = "https://api.dexscreener.com"

# Dexscreener chain slugs. Robinhood Chain is `robinhood` (not the numeric 4663).
DEX_CHAIN_SLUG: dict[Chain, str] = {
    Chain.SOLANA: "solana",
    Chain.BNB: "bsc",
    Chain.BASE: "base",
    Chain.ROBINHOOD_CHAIN: "robinhood",
}
SLUG_TO_CHAIN = {v: k for k, v in DEX_CHAIN_SLUG.items()}


@dataclass(slots=True)
class PairSnapshot:
    chain: Chain
    pair_address: str
    token_address: str
    symbol: str
    name: str
    price_usd: float
    liquidity_usd: float
    mcap_usd: float
    volume_m5: float
    volume_h1: float
    buys_m5: int
    sells_m5: int
    price_change_m5: float
    price_change_h1: float
    created_at_ms: int
    dex_id: str = ""
    quote_symbol: str = ""
    twitter: str = ""
    blurb: str = ""
    dex_paid: bool = False
    dex_photo: bool = False
    dex_aligned: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def to_candidate(self, source: str) -> Candidate:
        return Candidate(
            chain=self.chain,
            address=self.token_address,
            symbol=self.symbol,
            name=self.name,
            created_at_ms=self.created_at_ms,
            price_usd=self.price_usd,
            liquidity_usd=self.liquidity_usd,
            mcap_usd=self.mcap_usd,
            volume_5m_usd=self.volume_m5,
            pool_address=self.pair_address,
            quote_asset=self.quote_symbol,
            source=source,
            dex_id=self.dex_id,
            ret_5m=self.price_change_m5,
            dex_paid=self.dex_paid or source == "dexscreener_boosts",
            dex_photo=self.dex_photo,
            dex_aligned=self.dex_aligned,
        )

    def stamp(self, candidate: Candidate) -> None:
        candidate.dex_paid = (
            candidate.dex_paid or self.dex_paid or candidate.source == "dexscreener_boosts"
        )
        candidate.dex_photo = self.dex_photo
        candidate.dex_aligned = self.dex_aligned
        candidate.symbol = candidate.symbol or self.symbol
        candidate.name = candidate.name or self.name
        # Pair birth, not visor arrival. hood_stream stamps now_ms until DS has pairCreatedAt.
        if self.created_at_ms > 0 and (
            not candidate.created_at_ms or candidate.created_at_ms > self.created_at_ms
        ):
            candidate.created_at_ms = self.created_at_ms


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pair_created_ms(value: Any) -> int:
    n = int(_f(value))
    if 0 < n < 10**12:
        n *= 1000
    return n


def orders_mark_paid(data: Any) -> bool:
    """Dexscreener /orders/v1: approved profile or boost = they paid."""
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = list(data.get("orders") or []) + list(data.get("boosts") or [])
    else:
        return False
    return any(
        isinstance(row, dict) and str(row.get("status") or "").lower() == "approved"
        for row in rows
    )


def pair_dex_flags(pair: dict[str, Any]) -> tuple[bool, bool, bool]:
    """Paid Dexscreener boost + photo + branded profile. Rugs rarely pay."""
    info = pair.get("info") or {}
    boosts = pair.get("boosts") or {}
    paid = (
        _f(boosts.get("active")) > 0
        or _f(boosts.get("amount")) > 0
        or bool(str(info.get("header") or "").strip())
    )
    photo = bool(str(info.get("imageUrl") or info.get("header") or "").strip())
    handle, blurb = pair_socials(pair)
    websites = bool(info.get("websites"))
    ticker = str((pair.get("baseToken") or {}).get("symbol") or "").lower().lstrip("$")
    name = str((pair.get("baseToken") or {}).get("name") or "").lower()
    blob = f"{blurb} {handle}".lower()
    named = bool(ticker) and (
        ticker in blob or (len(name) >= 3 and name.split()[0] in blob)
    )
    aligned = photo and bool(handle or websites) and bool(blurb.strip()) and named
    return paid, photo, aligned


_X_SKIP = frozenset(
    {
        "home",
        "search",
        "i",
        "intent",
        "share",
        "hashtag",
        "explore",
        "compose",
        "login",
        "signup",
        "communities",
        "status",
        "statuses",
        "tweet",
        "jobs",
        "privacy",
        "tos",
        "settings",
    }
)
_HANDLE = re.compile(r"^[A-Za-z0-9_]{1,15}$")


def twitter_handle(url: str) -> str:
    """Profile handle from an X/Twitter URL. Tweet ids and /i/communities 404."""
    raw = (url or "").strip()
    if not raw:
        return ""
    low = raw.lower()
    if "twitter.com/" not in low and "x.com/" not in low:
        h = raw.lstrip("@")
        return h if _HANDLE.fullmatch(h) and not h.isdigit() else ""
    parsed = urlparse(raw if "://" in raw else "https://" + raw)
    parts = [p for p in parsed.path.split("/") if p]
    if not parts or parts[0].lower() in _X_SKIP:
        return ""
    h = parts[0].lstrip("@")
    return h if _HANDLE.fullmatch(h) and not h.isdigit() else ""


def pair_socials(pair: dict[str, Any]) -> tuple[str, str]:
    """Dexscreener profile twitter handle + blurb. Free, no X key."""
    info = pair.get("info") or {}
    blurb = str(info.get("description") or "")[:240]
    handle = ""
    for row in info.get("socials") or []:
        got = twitter_handle(str((row or {}).get("url") or ""))
        if got:
            handle = got
            break
    return handle, blurb


def inst_weight(followers: int, verified_type: str = "") -> float:
    """0–1. Business/gov and 50k+ CT accounts are the shill that moves a mint."""
    v = (verified_type or "").lower()
    score = 0.85 if v in {"business", "government"} else 0.0
    if followers >= 500_000:
        score = max(score, 1.0)
    elif followers >= 100_000:
        score = max(score, 0.75)
    elif followers >= 50_000:
        score = max(score, 0.55)
    elif followers >= 10_000:
        score = max(score, 0.25)
    return score


def utility_hint(*texts: str) -> str:
    blob = " ".join(t for t in texts if t).lower()
    if not blob:
        return ""
    if any(w in blob for w in ("stake", "game", "nft", "defi", "dao", "ai agent", "utility")):
        return "claims utility"
    if any(w in blob for w in ("meme", "just a", "culture")):
        return "meme"
    return ""


def parse_pair(pair: dict[str, Any]) -> PairSnapshot | None:
    slug = pair.get("chainId", "")
    chain = SLUG_TO_CHAIN.get(slug)
    if chain is None:
        return None
    base = pair.get("baseToken") or {}
    txns = (pair.get("txns") or {}).get("m5") or {}
    twitter, blurb = pair_socials(pair)
    paid, photo, aligned = pair_dex_flags(pair)
    return PairSnapshot(
        chain=chain,
        pair_address=pair.get("pairAddress", ""),
        token_address=base.get("address", ""),
        symbol=base.get("symbol", ""),
        name=base.get("name", ""),
        price_usd=_f(pair.get("priceUsd")),
        liquidity_usd=_f((pair.get("liquidity") or {}).get("usd")),
        mcap_usd=_f(pair.get("marketCap") or pair.get("fdv")),
        volume_m5=_f((pair.get("volume") or {}).get("m5")),
        volume_h1=_f((pair.get("volume") or {}).get("h1")),
        buys_m5=int(_f(txns.get("buys"))),
        sells_m5=int(_f(txns.get("sells"))),
        price_change_m5=_f((pair.get("priceChange") or {}).get("m5")) / 100.0,
        price_change_h1=_f((pair.get("priceChange") or {}).get("h1")) / 100.0,
        created_at_ms=pair_created_ms(pair.get("pairCreatedAt")),
        dex_id=pair.get("dexId", ""),
        quote_symbol=(pair.get("quoteToken") or {}).get("symbol", ""),
        twitter=twitter,
        blurb=blurb,
        dex_paid=paid,
        dex_photo=photo,
        dex_aligned=aligned,
        raw=pair,
    )


class Dexscreener:
    def __init__(self, http: Http, cache_seconds: float = 2.0) -> None:
        self.http = http
        # A tick asks for the same token several times over: the enricher
        # refreshes it, then the router quotes a buy, then it quotes the sell
        # back. Those are the same snapshot - prices do not move inside one
        # tick - so caching for slightly less than a tick cuts the request rate
        # by 3-4x. Without it the free tier 429s during exactly the launches
        # worth trading.
        self.cache_seconds = cache_seconds
        self._cache: dict[str, tuple[float, PairSnapshot]] = {}
        # Documented as 300 req/min for the pairs endpoints. Staying under it
        # deliberately, because being rate-limited during a launch is the one
        # moment the data is worth anything.
        http.limit("api.dexscreener.com", rate_per_sec=5.0, burst=8)
        self._paid: dict[str, tuple[float, bool]] = {}

    async def _get(self, path: str, **kw: Any) -> Any:
        try:
            return await self.http.get(f"{DEXSCREENER}{path}", **kw)
        except HttpError as exc:
            log.warning("dexscreener failed", extra={"path": path, "status": exc.status})
            return None

    async def token_pairs(self, addresses: list[str]) -> list[PairSnapshot]:
        """Best pair per token, for up to 30 tokens in one call."""
        if not addresses:
            return []
        now = time.monotonic()
        out: dict[str, PairSnapshot] = {}
        missing: list[str] = []
        for address in addresses[:30]:
            cached = self._cache.get(address)
            if cached is not None and now - cached[0] < self.cache_seconds:
                out[address] = cached[1]
            else:
                missing.append(address)

        if missing:
            data = await self._get("/latest/dex/tokens/" + ",".join(missing))
            for raw in (data or {}).get("pairs") or []:
                snap = parse_pair(raw)
                if snap is None or not snap.token_address:
                    continue
                current = out.get(snap.token_address)
                if current is None or snap.liquidity_usd > current.liquidity_usd:
                    out[snap.token_address] = snap
            for address in missing:
                snap = out.get(address)
                if snap is not None:
                    self._cache[address] = (now, snap)

        if len(self._cache) > 1024:
            self._cache = {
                k: v for k, v in self._cache.items() if now - v[0] < self.cache_seconds
            }
        return list(out.values())

    async def search(self, query: str) -> list[PairSnapshot]:
        data = await self._get("/latest/dex/search", params={"q": query})
        return [s for s in (parse_pair(p) for p in (data or {}).get("pairs") or []) if s]

    async def new_profiles(self) -> list[str]:
        """Tokens that just created a Dexscreener profile.

        A profile means somebody bothered to fill in socials, which is a weak
        but real filter against pure noise launches - and it lands earlier than
        the token shows up in any trending list.
        """
        data = await self._get("/token-profiles/latest/v1")
        out = []
        for item in data or []:
            slug, address = item.get("chainId"), item.get("tokenAddress")
            if slug in SLUG_TO_CHAIN and address:
                out.append(address)
        return out

    async def token_is_paid(self, chain: Chain, address: str) -> bool:
        """Profile/boost order, cached. Pair.boosts.active misses paid listings."""
        now = time.monotonic()
        hit = self._paid.get(address)
        if hit is not None and now - hit[0] < 45:
            return hit[1]
        slug = DEX_CHAIN_SLUG.get(chain)
        if not slug or not address:
            return False
        data = await self._get(f"/orders/v1/{slug}/{address}")
        if data is None:
            return False
        paid = orders_mark_paid(data)
        self._paid[address] = (now, paid)
        return paid

    async def boosted(self) -> list[str]:
        """Paid-boost tokens. Included as a signal about the *promoter*, not the
        token: someone spending on visibility expects an audience, and the
        audience is what the strategy is trying to arrive before."""
        data = await self._get("/token-boosts/latest/v1")
        return [
            item["tokenAddress"]
            for item in data or []
            if item.get("chainId") in SLUG_TO_CHAIN and item.get("tokenAddress")
        ]

    async def pair(self, chain: Chain, pair_address: str) -> PairSnapshot | None:
        slug = DEX_CHAIN_SLUG.get(chain)
        if not slug:
            return None
        data = await self._get(f"/latest/dex/pairs/{slug}/{pair_address}")
        pairs = (data or {}).get("pairs") or []
        return parse_pair(pairs[0]) if pairs else None


class Helius:
    """Optional. Used for the one number nothing else gives away for free: the
    exact holder count."""

    def __init__(self, http: Http, api_key: str) -> None:
        self.http = http
        self.api_key = api_key
        self.url = f"https://mainnet.helius-rpc.com/?api-key={api_key}"

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def holder_count(self, mint: str) -> int | None:
        if not self.enabled:
            return None
        try:
            data = await self.http.post(
                self.url,
                json_body={
                    "jsonrpc": "2.0",
                    "id": "holders",
                    "method": "getTokenAccounts",
                    "params": {"mint": mint, "limit": 1, "options": {"showZeroBalance": False}},
                },
            )
        except HttpError as exc:
            log.debug("helius holder_count failed", extra={"status": exc.status})
            return None
        result = (data or {}).get("result") or {}
        total = result.get("total")
        return int(total) if isinstance(total, (int, float)) else None


class Birdeye:
    """Optional. Candle history for tokens old enough to have any."""

    def __init__(self, http: Http, api_key: str) -> None:
        self.http = http
        self.api_key = api_key
        http.limit("public-api.birdeye.so", rate_per_sec=1.0, burst=3)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def candles(
        self, address: str, *, chain: Chain = Chain.SOLANA, minutes: int = 60
    ) -> list[Candle]:
        if not self.enabled:
            return []
        now_s = now_ms() // 1000
        try:
            data = await self.http.get(
                "https://public-api.birdeye.so/defi/ohlcv",
                params={
                    "address": address,
                    "type": "1m",
                    "time_from": now_s - minutes * 60,
                    "time_to": now_s,
                },
                headers={
                    "X-API-KEY": self.api_key,
                    "x-chain": DEX_CHAIN_SLUG.get(chain, "solana"),
                },
            )
        except HttpError as exc:
            log.debug("birdeye candles failed", extra={"status": exc.status})
            return []
        items = ((data or {}).get("data") or {}).get("items") or []
        out: list[Candle] = []
        for item in items:
            out.append(
                Candle(
                    ts=int(_f(item.get("unixTime")) * 1000),
                    open=_f(item.get("o")),
                    high=_f(item.get("h")),
                    low=_f(item.get("l")),
                    close=_f(item.get("c")),
                    volume=_f(item.get("v")),
                )
            )
        out.sort(key=lambda c: c.ts)
        return out


class FomoGraph:
    """Cope Capital maps Fomo handles to wallets. Research only — never a venue.

    Without COPE_API_KEY this is a no-op and labeled `source=fomo` wallets in
    config/whales.toml are the only Fomo profiles we know.
    """

    BASE = "https://api.cope.capital"

    def __init__(self, http: Http, api_key: str = "", strategy: Any = None) -> None:
        self.http = http
        self.api_key = api_key
        self.min_win_rate = 0.55
        self.min_trades = 8
        if strategy is not None:
            self.min_win_rate = float(strategy.get("whales.min_win_rate", 0.55))
            self.min_trades = int(strategy.get("whales.min_trades", 8))
        http.limit("api.cope.capital", rate_per_sec=0.2, burst=2)
        self._elite: set[str] = set()
        self._elite_at = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self.enabled:
            return None
        try:
            return await self.http.get(
                f"{self.BASE}{path}",
                params=params or {},
                headers=self._headers(),
            )
        except HttpError as exc:
            log.debug("cope failed", extra={"status": exc.status, "path": path})
            return None

    def _worth_chasing(self, row: dict[str, Any]) -> bool:
        wr = _f(row.get("win_rate") or row.get("winRate") or row.get("accuracy"), -1.0)
        if wr > 1.0:
            wr /= 100.0
        n = _f(row.get("trades") or row.get("trade_count") or row.get("n"), 0.0)
        if wr < 0:
            return True
        return wr >= self.min_win_rate and n >= self.min_trades

    def _rows(self, data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        if isinstance(data, dict):
            for key in ("data", "traders", "leaderboard", "results", "items"):
                inner = data.get(key)
                if isinstance(inner, list):
                    return [r for r in inner if isinstance(r, dict)]
            return [data]
        return []

    async def elite_wallets(self) -> set[str]:
        if not self.enabled:
            return set()
        now = time.time()
        if self._elite and now - self._elite_at < 1800:
            return self._elite
        data = await self._get("/v1/leaderboard", {"timeframe": "7d", "limit": "50"})
        from .signals.whales import wallets_in

        wallets: set[str] = set()
        for row in self._rows(data):
            if self._worth_chasing(row):
                wallets |= wallets_in(row)
        self._elite = wallets
        self._elite_at = now
        return wallets

    async def token_wallets(self, mint: str, chain: Chain) -> set[str]:
        if not self.enabled:
            return set()
        data = await self._get(f"/v1/tokens/{mint}/thesis", {"chain": chain.value})
        from .signals.whales import wallets_in

        return wallets_in(data, skip={mint})


# Cashtags that would drown a "is anyone talking about this mint?" search.
_TWITTER_SKIP = frozenset(
    {
        "SOL",
        "BTC",
        "ETH",
        "USDC",
        "USDT",
        "WETH",
        "BNB",
        "WBNB",
        "USD",
        "PEPE",
        "DOGE",
        "WIF",
        "BONK",
    }
)


@dataclass(slots=True)
class TweetRead:
    mentions: float = 0.0
    inst: float = 0.0
    fresh: float = 0.0
    official: str = ""
    official_age_min: float | None = None
    posts: list[dict[str, Any]] = field(default_factory=list)
    utility: str = ""

    def as_visor(self) -> dict[str, Any]:
        return {
            "official": self.official,
            "official_age_min": self.official_age_min,
            "inst": round(self.inst, 2),
            "fresh": round(self.fresh, 2),
            "mentions": int(self.mentions),
            "utility": self.utility,
            "posts": self.posts[:5],
        }


def _tweet_age_min(iso: str) -> float | None:
    if not iso:
        return None
    try:
        ts = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return max(0.0, (now_ms() - ts.timestamp() * 1000) / 60_000.0)


class Twitter:
    """Light X search: CA / $ticker / official handle. Needs TWITTER_BEARER_TOKEN.

    Few trades a day, so one recent-search per scored mint is the budget.
    """

    def __init__(self, http: Http, bearer: str = "") -> None:
        self.http = http
        self.bearer = bearer
        http.limit("api.x.com", rate_per_sec=0.15, burst=2)
        self._cache: dict[str, tuple[float, TweetRead | None]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.bearer)

    def _query(self, symbol: str, address: str, handle: str = "") -> str:
        parts: list[str] = []
        if address and len(address) >= 32:
            parts.append(address)
        sym = (symbol or "").lstrip("$").strip()
        if sym and 3 <= len(sym) <= 12 and sym.upper() not in _TWITTER_SKIP and sym.isascii():
            parts.append(f"${sym}")
        h = handle.lstrip("@")
        if h and h.replace("_", "").isalnum() and len(h) <= 15:
            parts.append(f"from:{h}")
        return " OR ".join(parts)

    async def mentions(self, symbol: str, address: str) -> float | None:
        read = await self.scan(symbol, address)
        return None if read is None else read.mentions

    async def scan(
        self, symbol: str, address: str, *, handle: str = "", blurb: str = ""
    ) -> TweetRead | None:
        if not self.enabled:
            return None
        q = self._query(symbol, address, handle)
        if not q:
            return TweetRead(official=handle.lstrip("@"), utility=utility_hint(blurb))
        now = time.monotonic()
        cached = self._cache.get(q)
        if cached and now - cached[0] < 90:
            return cached[1]
        try:
            data = await self.http.get(
                "https://api.x.com/2/tweets/search/recent",
                params={
                    "query": f"({q}) -is:retweet",
                    "max_results": "20",
                    "expansions": "author_id",
                    "tweet.fields": "created_at,text",
                    "user.fields": "verified,verified_type,public_metrics,username",
                },
                headers={"Authorization": f"Bearer {self.bearer}"},
            )
        except HttpError as exc:
            log.debug("twitter search failed", extra={"status": exc.status})
            return None
        users = {
            str(u.get("id")): u
            for u in ((data or {}).get("includes") or {}).get("users") or []
        }
        official = handle.lstrip("@").lower()
        seen: set[str] = set()
        n = 0.0
        inst = 0.0
        fresh = 0.0
        official_age: float | None = None
        posts: list[dict[str, Any]] = []
        texts: list[str] = [blurb]
        for tw in (data or {}).get("data") or []:
            uid = str(tw.get("author_id") or "")
            user = users.get(uid) or {}
            uname = str(user.get("username") or "")
            followers = int((user.get("public_metrics") or {}).get("followers_count") or 0)
            vtype = str(user.get("verified_type") or "")
            verified = bool(user.get("verified")) or bool(vtype)
            age = _tweet_age_min(str(tw.get("created_at") or ""))
            text = str(tw.get("text") or "").replace("\n", " ")[:140]
            if official and uname.lower() == official and age is not None:
                official_age = age if official_age is None else min(official_age, age)
                if age < 30:
                    fresh = max(fresh, 1.0)
                elif age < 90:
                    fresh = max(fresh, 0.5)
                if text and not any(p.get("handle", "").lower() == official for p in posts):
                    posts.append(
                        {
                            "handle": uname,
                            "followers": followers,
                            "age_min": round(age, 1),
                            "text": text,
                        }
                    )
            if uid and uid not in seen and verified and followers >= 1000:
                seen.add(uid)
                n += 1.0
                inst = max(inst, inst_weight(followers, vtype))
                if age is not None and age < 30:
                    fresh = max(fresh, 1.0)
                elif age is not None and age < 90:
                    fresh = max(fresh, 0.5)
                if not any(p.get("handle", "").lower() == uname.lower() for p in posts):
                    posts.append(
                        {
                            "handle": uname,
                            "followers": followers,
                            "age_min": None if age is None else round(age, 1),
                            "text": text,
                        }
                    )
            texts.append(text)
        posts.sort(key=lambda p: -(p.get("followers") or 0))
        read = TweetRead(
            mentions=n,
            inst=inst,
            fresh=fresh,
            official=handle.lstrip("@"),
            official_age_min=official_age,
            posts=posts[:5],
            utility=utility_hint(*texts),
        )
        self._cache[q] = (now, read)
        return read


class Bubblemaps:
    """Nova-style cluster/bundle overlay. Optional, 25 credits/call.

    Without a key we already compute a one-hop funding cluster on-chain.
    """

    CHAIN = {
        Chain.SOLANA: "solana",
        Chain.BNB: "bsc",
        Chain.ROBINHOOD_CHAIN: "robinhood",
    }

    def __init__(self, http: Http, api_key: str = "") -> None:
        self.http = http
        self.api_key = api_key
        http.limit("api.bubblemaps.io", rate_per_sec=0.2, burst=2)
        self._cache: dict[str, tuple[float, float]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def cluster_pct(self, chain: Chain, address: str) -> float | None:
        slug = self.CHAIN.get(chain)
        if not self.enabled or not slug or not address:
            return None
        key = f"{slug}:{address}"
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached and now - cached[0] < 300:
            return cached[1]
        try:
            data = await self.http.get(
                f"https://api.bubblemaps.io/v0/tokens/metrics/{slug}/{address}",
                headers={"X-ApiKey": self.api_key},
            )
        except HttpError as exc:
            log.debug("bubblemaps failed", extra={"status": exc.status})
            return None
        stats = (data or {}).get("supply_stats") or {}
        pct = _f(stats.get("bundles"))
        self._cache[key] = (now, pct)
        return pct
