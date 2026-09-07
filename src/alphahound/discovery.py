"""Candidate discovery.

Being early is a latency problem before it is an analysis problem: a token you
hear about ten minutes late is a token whose retail wave you are joining rather
than preceding. So discovery is ordered by how fast each source is, not how
convenient:

    pump.fun websocket   sub-second, push
    dexscreener profiles ~seconds, poll
    dexscreener boosts   ~seconds, poll, and a signal about the promoter
    watchlist            whenever you say so

Everything is deduplicated and thrown away after `max_candidate_age_minutes`,
because an old candidate is not a candidate, it is a chart.

Honest ceiling: a public REST/websocket feed puts you in the same cohort as
every other bot on the same feeds. The latency ladder above this is a Geyser /
Yellowstone gRPC stream, and above that co-location with a validator. If you
are competing for the first block of a launch, you need those. This layer
targets the 1-10 minute window instead, where analysis quality still decides
the outcome.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field

from .log import get
from .models import Candidate, Chain, now_ms
from .net import Http
from .origin import launchpad_origin
from .providers import DEX_CHAIN_SLUG, Dexscreener
from .settings import Config, Settings

log = get("discovery")


@dataclass(slots=True)
class DiscoveryStats:
    seen: int = 0
    emitted: int = 0
    dropped_stale: int = 0
    dropped_duplicate: int = 0
    by_source: dict[str, int] = field(default_factory=dict)


class Discovery:
    def __init__(
        self,
        settings: Settings,
        strategy: Config,
        dexscreener: Dexscreener,
        http: Http | None = None,
    ) -> None:
        self.settings = settings
        self.strategy = strategy
        self.dex = dexscreener
        self.http = http
        self.max_age_minutes = float(strategy.get("loop.max_candidate_age_minutes", 180))
        self.stats = DiscoveryStats()

        self._seen: dict[str, int] = {}
        self._seen_addr: dict[str, int] = {}
        self._queue: asyncio.Queue[Candidate] = asyncio.Queue(maxsize=2000)
        self._tasks: list[asyncio.Task] = []
        self._watchlist: set[str] = set()

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        if self.settings.pumpportal_ws_url and Chain.SOLANA in self.settings.enabled_chains:
            self._tasks.append(asyncio.create_task(self._pump_stream(), name="pump-stream"))
        if (
            Chain.ROBINHOOD_CHAIN in self.settings.enabled_chains
            and (self.settings.rpc_urls.get(Chain.ROBINHOOD_CHAIN) or "").strip()
        ):
            self._tasks.append(asyncio.create_task(self._hood_stream(), name="hood-stream"))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

    def watch(self, address: str) -> None:
        a = address.strip()
        if a:
            self._watchlist.add(a)

    # -- polling -----------------------------------------------------------
    async def poll(self) -> list[Candidate]:
        """One round of the polled sources, merged and deduplicated."""
        addresses: list[str] = []
        sources: dict[str, str] = {}

        fetched = await asyncio.gather(
            self.dex.new_profiles(),
            self.dex.boosted(),
            return_exceptions=True,
        )
        for source, found in (
            ("dexscreener_profiles", fetched[0]),
            ("dexscreener_boosts", fetched[1]),
        ):
            if isinstance(found, Exception):
                log.warning("source failed", extra={"source": source, "error": str(found)})
                continue
            for address in found:
                if address not in sources:
                    sources[address] = source
                    addresses.append(address)

        for address in self._watchlist:
            if address and address not in sources:
                sources[address] = "inspect"
                addresses.append(address)

        out: list[Candidate] = []
        # Repeat profile CAs don't need token_pairs again this minute; that
        # call was starving the visor quote loop on the same Dexscreener bucket.
        addresses = [
            a
            for a in addresses
            if sources.get(a) in {"dexscreener_boosts", "inspect"} or self._pair_unknown(a)
        ]
        # token_pairs takes 30 addresses per call, so this is len/30 requests
        # rather than len.
        for chunk_start in range(0, len(addresses), 30):
            chunk = addresses[chunk_start : chunk_start + 30]
            try:
                snaps = await self.dex.token_pairs(chunk)
            except Exception as exc:  # noqa: BLE001
                log.warning("token_pairs failed", extra={"error": str(exc)})
                continue
            for snap in snaps:
                if snap.chain not in self.settings.enabled_chains:
                    continue
                candidate = snap.to_candidate(sources.get(snap.token_address, "dexscreener"))
                if self._accept(candidate):
                    out.append(candidate)

        while not self._queue.empty():
            candidate = self._queue.get_nowait()
            if self._accept(candidate):
                out.append(candidate)
        return out

    def _accept(self, candidate: Candidate) -> bool:
        self.stats.seen += 1
        if candidate.chain not in self.settings.enabled_chains:
            return False
        if not candidate.address:
            return False
        # Operator paste: keep even if the mint is older than the playbook window.
        if candidate.source != "inspect":
            allowed, _reason = launchpad_origin(candidate, self.strategy)
            if not allowed:
                return False
            cap = float(self.strategy.get("loop.max_candidate_age_minutes", 180))
            if not candidate.created_at_ms or candidate.age_minutes > cap:
                self.stats.dropped_stale += 1
                return False

        last = self._seen.get(candidate.key, 0)
        # Re-emit a known candidate at most once a minute: the engine wants
        # fresh feature vectors on tokens it is watching, not a stampede of
        # duplicates from three sources.
        if now_ms() - last < 60_000:
            self.stats.dropped_duplicate += 1
            return False
        self._seen[candidate.key] = now_ms()
        if candidate.address:
            self._seen_addr[candidate.address.lower()] = now_ms()

        self.stats.emitted += 1
        self.stats.by_source[candidate.source] = self.stats.by_source.get(candidate.source, 0) + 1
        return True

    def _pair_unknown(self, address: str) -> bool:
        ts = self._seen_addr.get((address or "").lower(), 0)
        return now_ms() - ts >= 60_000

    def prune(self) -> None:
        cutoff = now_ms() - int(self.max_age_minutes * 60_000) * 2
        for key, ts in list(self._seen.items()):
            if ts < cutoff:
                del self._seen[key]
        for addr, ts in list(self._seen_addr.items()):
            if ts < cutoff:
                del self._seen_addr[addr]

    # -- streaming ---------------------------------------------------------
    async def _pump_stream(self) -> None:
        """pump.fun new-token firehose via PumpPortal.

        Reconnects forever with backoff. A discovery source that dies quietly is
        worse than one that never existed, because the bot keeps running and you
        conclude the market went quiet.
        """
        try:
            import websockets
        except ImportError:
            log.warning("pip install 'alphahound[stream]' to enable the pump.fun stream")
            return

        url = self.settings.pumpportal_ws_url
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    await ws.send(json.dumps({"method": "subscribeNewToken"}))
                    log.info("pump stream connected")
                    backoff = 1.0
                    async for raw in ws:
                        candidate = self._parse_pump_event(raw)
                        if candidate is None:
                            continue
                        try:
                            self._queue.put_nowait(candidate)
                        except asyncio.QueueFull:
                            # Backpressure: dropping the newest is wrong, but so
                            # is blocking the socket. The engine is the
                            # bottleneck here, and it will see the token again
                            # via polling.
                            pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("pump stream dropped", extra={"error": str(exc), "retry_in": backoff})
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    @staticmethod
    def _parse_pump_event(raw: str | bytes) -> Candidate | None:
        try:
            event = json.loads(raw)
        except (ValueError, TypeError):
            return None
        mint = event.get("mint") or event.get("ca")
        if not mint:
            return None
        return Candidate(
            chain=Chain.SOLANA,
            address=mint,
            symbol=event.get("symbol", ""),
            name=event.get("name", ""),
            created_at_ms=now_ms(),
            pool_address=event.get("bondingCurveKey", "") or event.get("pool", ""),
            deployer=event.get("traderPublicKey", "") or event.get("creator", ""),
            source="pumpfun_stream",
        )

    def _push(self, candidate: Candidate) -> None:
        try:
            self._queue.put_nowait(candidate)
        except asyncio.QueueFull:
            pass

    async def _hood_stream(self) -> None:
        """Pons TokenLaunched only. Handmade Uniswap V3 pools are ignored.

        eth_subscribe when the RPC actually speaks JSON-RPC over WebSocket
        (Alchemy/QuickNode). The public Hood RPC is HTTP-only (wss → 400), so
        we poll eth_getLogs instead. Dexscreener profiles/boosts stay parallel.
        """
        http_rpc = (self.settings.rpc_urls.get(Chain.ROBINHOOD_CHAIN) or "").strip()
        weth = (self.settings.rh_chain_weth or "").lower()
        ws_url = _hood_jsonrpc_ws(http_rpc)
        if ws_url:
            await self._hood_subscribe(ws_url, weth)
            return
        await self._hood_poll_logs(http_rpc, weth)

    async def _hood_subscribe(self, ws_url: str, weth: str) -> None:
        try:
            import websockets
        except ImportError:
            log.warning("pip install 'alphahound[stream]' to enable hood factory subscribe")
            return

        filt = {
            "address": [PONS_FACTORY, PONS_V2_FACTORY],
            "topics": [[TOPIC_TOKEN_LAUNCHED, TOPIC_TOKEN_LAUNCHED_V2]],
        }
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(ws_url, ping_interval=20, open_timeout=15) as ws:
                    await ws.send(
                        json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": 1,
                                "method": "eth_subscribe",
                                "params": ["logs", filt],
                            }
                        )
                    )
                    log.info("hood stream connected")
                    backoff = 1.0
                    async for raw in ws:
                        msg = _json_obj(raw)
                        if not msg:
                            continue
                        if msg.get("error"):
                            log.warning("hood subscribe error", extra={"error": msg.get("error")})
                            continue
                        result = (msg.get("params") or {}).get("result")
                        if not isinstance(result, dict):
                            continue
                        for candidate in candidates_from_hood_log(result, weth=weth):
                            self._push(candidate)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("hood stream dropped", extra={"error": str(exc), "retry_in": backoff})
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _hood_poll_logs(self, http_rpc: str, weth: str) -> None:
        # ponytail: public Hood RPC has no eth_subscribe; ~1s getLogs. Use a
        # JSON-RPC websocket URL (Alchemy) to drop this poll.
        from .execution.evm import EvmRpc

        owned = self.http is None
        http = self.http or Http()
        rpc = EvmRpc(http, http_rpc)
        seen: dict[str, None] = {}
        backoff = 1.0
        from_block: int | None = None
        log.info("hood stream polling factory logs")
        try:
            while True:
                caught_up = True
                try:
                    head = int(await rpc.call("eth_blockNumber", []), 16)
                    if from_block is None:
                        from_block = max(0, head - HOOD_LOG_CHUNK)
                    if head < from_block:
                        from_block = head
                    if head >= from_block:
                        to_block = min(head, from_block + HOOD_LOG_CHUNK - 1)
                        caught_up = to_block >= head
                        logs = await rpc.get_logs(
                            {
                                "address": [PONS_FACTORY, PONS_V2_FACTORY],
                                "topics": [[TOPIC_TOKEN_LAUNCHED, TOPIC_TOKEN_LAUNCHED_V2]],
                                "fromBlock": hex(from_block),
                                "toBlock": hex(to_block),
                            }
                        )
                        stamps: dict[int, int] = {}
                        for raw in logs:
                            if not isinstance(raw, dict):
                                continue
                            key = f"{raw.get('transactionHash')}:{raw.get('logIndex')}"
                            if key in seen:
                                continue
                            seen[key] = None
                            bn = raw.get("blockNumber")
                            try:
                                block_n = int(str(bn), 16) if str(bn).startswith("0x") else int(bn)
                            except (TypeError, ValueError):
                                block_n = 0
                            if block_n and block_n not in stamps:
                                try:
                                    blk = await rpc.call("eth_getBlockByNumber", [hex(block_n), False])
                                    stamps[block_n] = int((blk or {}).get("timestamp") or "0x0", 16)
                                except Exception:  # noqa: BLE001
                                    stamps[block_n] = 0
                            if stamps.get(block_n):
                                raw = {**raw, "blockTimestamp": hex(stamps[block_n])}
                            for candidate in candidates_from_hood_log(raw, weth=weth):
                                self._push(candidate)
                        if len(seen) > 4000:
                            for old in list(seen)[:2000]:
                                del seen[old]
                        from_block = to_block + 1
                    backoff = 1.0
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    log.warning("hood log poll failed", extra={"error": str(exc), "retry_in": backoff})
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    continue
                if caught_up:
                    await asyncio.sleep(1.0)
        finally:
            if owned:
                await http.aclose()


# Uniswap V3 factory + Pons launch factory, chain 4663.
UNI_V3_FACTORY = "0x1f7d7550b1b028f7571e69a784071f0205fd2efa"
PONS_FACTORY = "0xa5aab3f0c6eeadf30ef1d3eb997108e976351feb"
PONS_V2_FACTORY = "0x7ed598bcef8bd9edd8c97a195c6d13f40801ec7e"
TOPIC_POOL_CREATED = "0x783cca1c0412dd0d695e784568c96da2e9c22ff989357a2e8b1d9b2b4e6b7118"
TOPIC_TOKEN_LAUNCHED = "0xdb51ea9ad51ab453a65a4cb7e60c3cb378c9501bb002609f8f97778fb6c4235a"
TOPIC_TOKEN_LAUNCHED_V2 = "0x8d4aad4953d0ca700d468f3753aa14432d1b35b43ec6409f051fb6aa43a89607"
HOOD_LOG_CHUNK = 400


def _hood_jsonrpc_ws(http_rpc: str) -> str | None:
    """None on the public Hood RPC (HTTP only). Alchemy/QuickNode speak eth_subscribe."""
    raw = (http_rpc or "").strip()
    if not raw:
        return None
    host = raw.lower()
    if raw.startswith("wss://") or raw.startswith("ws://"):
        return raw
    if "alchemy.com" in host or "quiknode.pro" in host or "drpc.org" in host:
        return _hood_ws_url(raw)
    return None


def _hood_ws_url(http_rpc: str) -> str:
    if http_rpc.startswith("https://"):
        return "wss://" + http_rpc[len("https://") :]
    if http_rpc.startswith("http://"):
        return "ws://" + http_rpc[len("http://") :]
    return http_rpc


def _log_created_ms(log: dict) -> int:
    """Pair birth if the node stamps the log; else subscribe arrival (≈ block)."""
    raw = log.get("blockTimestamp") or log.get("timestamp") or log.get("timeStamp")
    if raw is not None:
        try:
            n = int(str(raw), 16) if str(raw).startswith("0x") else int(raw)
            if n < 10**12:
                n *= 1000
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
    return now_ms()


def _json_obj(raw: str | bytes) -> dict | None:
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _topic_addr(topic: str) -> str:
    raw = (topic or "").replace("0x", "").lower()
    return ("0x" + raw[-40:]) if len(raw) >= 40 else ""


def _is_quote(address: str, weth: str) -> bool:
    a = (address or "").lower()
    if not a or a == "0x" + "0" * 40:
        return True
    if weth and a == weth.lower():
        return True
    return False


def candidates_from_hood_log(log: dict, *, weth: str = "") -> list[Candidate]:
    """Parse Pons TokenLaunched. PoolCreated (own LP) is dropped."""
    topics = [str(t).lower() for t in (log.get("topics") or [])]
    if not topics:
        return []
    emitter = str(log.get("address") or "").lower()
    ts = _log_created_ms(log)
    out: list[Candidate] = []

    def emit(address: str, dex_id: str, pool: str = "") -> None:
        if not address or _is_quote(address, weth):
            return
        out.append(
            Candidate(
                chain=Chain.ROBINHOOD_CHAIN,
                address=address,
                created_at_ms=ts,
                pool_address=pool,
                source="hood_stream",
                dex_id=dex_id,
            )
        )

    if (
        topics[0] in {TOPIC_TOKEN_LAUNCHED, TOPIC_TOKEN_LAUNCHED_V2}
        and emitter in {PONS_FACTORY, PONS_V2_FACTORY}
        and len(topics) >= 2
    ):
        emit(_topic_addr(topics[1]), "pons")
    return out


def chain_supported_by_dexscreener(chain: Chain) -> bool:
    return chain in DEX_CHAIN_SLUG
