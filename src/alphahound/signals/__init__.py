"""Feature assembly.

One entry point - `Enricher.enrich` - turns a cheap Candidate into the full
feature vector the model scores. It is the only place that knows which
providers exist, so adding a data source does not ripple through the engine.

Two rules hold throughout:

1. A measurement that failed is recorded as unknown, never as a favourable
   default. `Enrichment.unknown` is what the gate layer reads to decide whether
   it is allowed to have an opinion at all.
2. Cheap features first. Nothing calls the chain until the free Dexscreener
   snapshot has already had a chance to disqualify the candidate.
"""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ..log import get
from ..models import Candidate, Candle, Chain, Features, RoundTrip, now_ms
from ..origin import known_holder_share
from ..playbook import copy_signal as copy_flag
from ..settings import Config, crowd_addresses, whale_addresses
from ..store import Store
from . import chart, flow, terminals, whales
from .distribution import (
    TRANSFER_TOPIC0,
    Holder,
    HolderStats,
    analyze,
    decode_transfer_log,
    first_pool_buyers,
    holder_growth,
    holders_from_moves,
    unmeasured_from_stats,
    cluster_share_by_root,
    log_topic_addr,
    ZERO_ADDR,
)
from .terminals import Attribution, ShareTracker, TerminalRegistry
from .whales import who_inside

if TYPE_CHECKING:
    # Providers and the chain reader pull in httpx. They are only ever passed
    # in, never constructed here, so importing them lazily keeps the pure
    # feature maths usable without the HTTP stack installed.
    from ..providers import Birdeye, Bubblemaps, Dexscreener, FomoGraph, Helius, PairSnapshot, Twitter
    from .solana import MintInfo, SolanaReader

log = get("signals")

CostProbe = Callable[[Candidate, float], Awaitable[RoundTrip]]


@dataclass(slots=True)
class Enrichment:
    candidate: Candidate
    features: Features
    holders: HolderStats = field(default_factory=HolderStats)
    chain_probed: bool = False
    mint: MintInfo | None = None
    attribution: Attribution = field(default_factory=Attribution)
    round_trip: RoundTrip = field(default_factory=lambda: RoundTrip(ok=False, note="not probed"))
    snap: PairSnapshot | None = None
    candles: list[Candle] = field(default_factory=list)
    trades: list[flow.Trade] = field(default_factory=list)
    snipers: set[str] = field(default_factory=set)
    # Unique buy wallets in the attribution window. Fed to the smart-money
    # learner when the position closes.
    buyers: list[str] = field(default_factory=list)
    # Names of features that could not be measured. Gates that depend on an
    # unknown feature abstain rather than pass.
    unknown: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    crowd: dict = field(default_factory=dict)
    twitter: dict = field(default_factory=dict)


class Enricher:
    def __init__(
        self,
        *,
        store: Store,
        strategy: Config,
        dexscreener: Dexscreener,
        registry: TerminalRegistry,
        solana: SolanaReader | None = None,
        helius: Helius | None = None,
        birdeye: Birdeye | None = None,
        probe: CostProbe | None = None,
        fomo: FomoGraph | None = None,
        whale_rows: list | None = None,
        twitter: Twitter | None = None,
        bubbles: Bubblemaps | None = None,
        evm_rpcs: dict | None = None,
    ) -> None:
        self.store = store
        self.strategy = strategy
        self.dex = dexscreener
        self.registry = registry
        self.solana = solana
        self.helius = helius
        self.birdeye = birdeye
        self.probe = probe
        self.fomo = fomo
        self.whale_rows: list = whale_rows or []
        self.twitter = twitter
        self.bubbles = bubbles
        self.evm_rpcs: dict = evm_rpcs or {}

        self.shares = ShareTracker()
        self._holder_history: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._launch_slots: dict[str, int] = {}
        self._smart_cache: dict[Chain, set[str]] = {}
        self._whale_fp: tuple[str, ...] | None = None
        self._evm_cache: dict[str, tuple[int, dict[str, Any]]] = {}

    # -- public ------------------------------------------------------------
    async def refresh(self, candidate: Candidate) -> PairSnapshot | None:
        """Update price/liquidity in place from the free provider."""
        snaps = await self.dex.token_pairs([candidate.address])
        snap = next((s for s in snaps if s.token_address == candidate.address), None)
        if snap is None:
            return None
        candidate.price_usd = snap.price_usd or candidate.price_usd
        candidate.liquidity_usd = snap.liquidity_usd
        candidate.mcap_usd = snap.mcap_usd or candidate.mcap_usd
        candidate.volume_5m_usd = snap.volume_m5
        candidate.pool_address = candidate.pool_address or snap.pair_address
        candidate.created_at_ms = candidate.created_at_ms or snap.created_at_ms
        candidate.symbol = candidate.symbol or snap.symbol
        candidate.dex_id = candidate.dex_id or snap.dex_id
        candidate.ret_5m = snap.price_change_m5
        snap.stamp(candidate)
        return snap

    @staticmethod
    def free_enrichment(candidate: Candidate) -> Enrichment:
        """What the last free snapshot alone can say; everything else unknown.

        Used to throw a candidate out before spending RPC calls on it. The
        liquidity gate alone rejects the large majority of new tokens and costs
        nothing to evaluate, while a full enrichment costs one signature fetch
        plus a transaction fetch per signature - so enriching first and gating
        afterwards spends the entire RPC budget on tokens that were disqualified
        by a number already in hand.
        """
        result = Enrichment(candidate=candidate, features=Features())
        measured = {"liquidity_usd", "liq_to_mcap", "dex_profile"}
        if candidate.created_at_ms:
            measured.add("token_age_minutes")
        pack = {
            "is_vamp": 1.0 if candidate.pack_role == "vamp" else 0.0,
            "is_beta": 1.0 if candidate.pack_role == "beta" else 0.0,
            "is_main": 1.0 if candidate.pack_role == "main" else 0.0,
            "main_ret_5m": candidate.main_ret_5m,
        }
        if candidate.pack_role:
            measured.update(pack)
        result.unknown.update(set(Features.names()) - measured)
        result.features = Features(
            liquidity_usd=candidate.liquidity_usd,
            liq_to_mcap=(
                candidate.liquidity_usd / candidate.mcap_usd if candidate.mcap_usd > 0 else 0.0
            ),
            token_age_minutes=candidate.age_minutes,
            dex_profile=candidate.dex_profile,
            **pack,
        )
        return result

    async def enrich(self, candidate: Candidate, probe_size_usd: float) -> Enrichment:
        result = Enrichment(candidate=candidate, features=Features())
        snap = await self.refresh(candidate)
        result.snap = snap
        if snap is None:
            result.unknown.update({"price_usd", "liquidity_usd"})
            result.notes.append("no market data from dexscreener")

        values: dict[str, float] = {}
        chart_f, chain_f, cost_f = await asyncio.gather(
            self._chart_features(candidate, snap, result),
            self._onchain_features(candidate, result),
            self._cost_features(candidate, probe_size_usd, result),
        )
        values.update(chart_f)
        values.update(chain_f)
        values.update(cost_f)

        values["token_age_minutes"] = candidate.age_minutes
        if not candidate.created_at_ms:
            result.unknown.add("token_age_minutes")
        values["liquidity_usd"] = candidate.liquidity_usd
        values["liq_to_mcap"] = (
            candidate.liquidity_usd / candidate.mcap_usd if candidate.mcap_usd > 0 else 0.0
        )
        values.update(await self._narrative_features(candidate, result, values))
        values["is_vamp"] = 1.0 if candidate.pack_role == "vamp" else 0.0
        values["is_beta"] = 1.0 if candidate.pack_role == "beta" else 0.0
        values["is_main"] = 1.0 if candidate.pack_role == "main" else 0.0
        values["main_ret_5m"] = candidate.main_ret_5m
        if not candidate.pack_role:
            result.unknown.update({"is_vamp", "is_beta", "is_main", "main_ret_5m"})

        known = set(Features.names())
        result.features = Features(**{k: v for k, v in values.items() if k in known})
        return result

    # -- chart -------------------------------------------------------------
    async def _chart_features(
        self, candidate: Candidate, snap: PairSnapshot | None, result: Enrichment
    ) -> dict[str, float]:
        cfg = self.strategy.section("chart")

        candles: list[Candle] = []
        if self.birdeye and self.birdeye.enabled:
            candles = await self.birdeye.candles(
                candidate.address,
                chain=candidate.chain,
                minutes=int(cfg.get("lookback_candles", 30)),
            )
        result.candles = candles
        if candles:
            return chart.extract(candles, cfg)

        # No candle provider. On Solana the trade stream gives us real candles
        # a few lines further down; everywhere else we fall back to the
        # aggregate price changes, which are coarse but not wrong.
        if snap is not None:
            result.unknown.update({"vwap_dev", "atr_pct", "breakout", "volume_z", "body_ratio"})
            result.notes.append("chart from aggregate price changes, no candles")
            return {
                "ret_5m": snap.price_change_m5,
                "ret_15m": snap.price_change_h1,
                "parabolic": min(
                    1.0,
                    max(0.0, snap.price_change_h1 / float(cfg.get("parabolic_return_threshold", 1.5))),
                ),
            }
        result.unknown.add("chart")
        return {}

    # -- chain -------------------------------------------------------------
    async def _onchain_features(
        self, candidate: Candidate, result: Enrichment
    ) -> dict[str, float]:
        if candidate.chain is Chain.SOLANA and self.solana is not None:
            values = await self._solana_features(candidate, result)
            result.chain_probed = True
            return values
        values = self._aggregate_only_features(candidate, result)
        values.update(await self._evm_launch_distribution(candidate, result))
        values.update(await self._evm_labeled_crowd(candidate, result))
        result.chain_probed = True
        return values

    def _evm_cached(self, key: str) -> dict[str, Any] | None:
        hit = self._evm_cache.get(key)
        if hit is None or hit[0] <= now_ms():
            return None
        return hit[1]

    def _evm_put(self, key: str, payload: dict[str, Any]) -> None:
        ttl = int(float(self.strategy.get("gates.bundle_cache_seconds", 60)) * 1000)
        self._evm_cache[key] = (now_ms() + max(0, ttl), payload)

    async def _evm_launch_distribution(
        self, candidate: Candidate, result: Enrichment
    ) -> dict[str, float]:
        """Launch bundle + token-source cluster from ERC-20 Transfer logs."""
        rpc = self.evm_rpcs.get(candidate.chain)
        if rpc is None or not candidate.address:
            return {}
        cached = self._evm_cached(f"dist:{candidate.key}")
        if cached is not None:
            stats = cached.get("stats")
            if stats is not None:
                result.holders = stats
            missing = set(cached.get("unknown") or [])
            if stats is not None:
                missing |= unmeasured_from_stats(stats)
            for key in (
                "bundle_pct",
                "cluster_pct",
                "top10_pct",
                "top1_pct",
                "gini",
                "dev_holding_pct",
                "fresh_wallet_pct",
                "holder_count",
            ):
                if key in missing:
                    result.unknown.add(key)
                else:
                    result.unknown.discard(key)
            if cached.get("note"):
                result.notes.append(str(cached["note"]))
            values = dict(cached.get("values") or {})
            for key in missing:
                values.pop(key, None)
            return values
        lookback = int(self.strategy.get("gates.bundle_log_lookback_blocks", 40_000))
        first_n = int(self.strategy.get("gates.bundle_first_n", 40))
        window = int(self.strategy.get("gates.bundle_slot_window", 3))
        try:
            latest = int(await rpc.call("eth_blockNumber", []), 16)
            start = max(0, latest - lookback)
            raw = await rpc.get_logs(
                {
                    "fromBlock": hex(start),
                    "toBlock": hex(latest),
                    "address": candidate.address,
                    "topics": [TRANSFER_TOPIC0],
                }
            )
        except Exception as exc:  # noqa: BLE001
            result.notes.append(f"transfer logs failed: {exc}")
            return {}
        moves = [m for m in (decode_transfer_log(x) for x in raw) if m]
        if not moves:
            result.notes.append("transfer logs empty in lookback")
            return {}
        pools = {candidate.pool_address.lower()} if candidate.pool_address else set()
        holders, launch = holders_from_moves(
            moves, pools=pools, deployer=candidate.deployer, first_n=first_n
        )
        if not holders:
            return {}
        stats = analyze(holders, now_ms=now_ms(), launch_slot=launch, bundle_slot_window=window)
        token_cluster = stats.largest_funding_cluster_pct
        quote_cluster, quote_note = await self._quote_two_hop_cluster(
            rpc,
            candidate,
            result,
            holders,
            moves,
            pools,
            first_n,
            start,
            latest,
        )
        if quote_cluster > token_cluster:
            stats.largest_funding_cluster_pct = quote_cluster
            stats.notes.append(quote_note)
        elif quote_note:
            stats.notes.append(quote_note)
        result.holders = stats
        values = {
            "holder_count": float(stats.holder_count),
            "top10_pct": stats.top10_pct,
            "top1_pct": stats.top1_pct,
            "gini": stats.gini,
            "fresh_wallet_pct": stats.fresh_wallet_pct,
            "bundle_pct": stats.bundle_pct,
            "dev_holding_pct": stats.dev_holding_pct,
            "cluster_pct": stats.largest_funding_cluster_pct,
        }
        missing = unmeasured_from_stats(stats)
        for key in (
            "bundle_pct",
            "cluster_pct",
            "top10_pct",
            "top1_pct",
            "gini",
            "dev_holding_pct",
            "fresh_wallet_pct",
            "holder_count",
        ):
            if key in missing:
                result.unknown.add(key)
                values.pop(key, None)
            else:
                result.unknown.discard(key)
        result.notes.append(
            f"evm dist: bundle {stats.bundle_pct:.0%} cluster {stats.largest_funding_cluster_pct:.0%} "
            f"(token {token_cluster:.0%} quote {quote_cluster:.0%}) n={stats.holder_count}"
            + (" ages unknown" if "fresh_wallet_pct" in missing else "")
        )
        self._evm_put(
            f"dist:{candidate.key}",
            {
                "values": values,
                "stats": stats,
                "note": result.notes[-1],
                "unknown": sorted(missing),
            },
        )
        return values

    async def _quote_two_hop_cluster(
        self,
        rpc,
        candidate: Candidate,
        result: Enrichment,
        holders: list,
        moves: list,
        pools: set[str],
        first_n: int,
        start: int,
        latest: int,
    ) -> tuple[float, str]:
        """2-hop funding on the quote token (GLD/WETH/…). Complements token 1-hop.

        Pool buyers have no token-hop funder. Same-source quote inbound is the
        GI pattern. Hop-1: first_n pool buyers. Hop-2: unique hop-1 parents only.
        """
        snap = result.snap
        quote = str(((snap.raw.get("quoteToken") if snap else None) or {}).get("address") or "")
        if not quote.startswith("0x"):
            return 0.0, "quote cluster skipped: no quote token"
        quote = quote.lower()
        buyers = first_pool_buyers(
            moves,
            pools,
            first_n,
            {h.address.lower(): h.balance for h in holders},
        )[:40]
        if not buyers:
            return 0.0, "quote cluster: no pool buyers"
        skip = {
            quote,
            candidate.address.lower(),
            ZERO_ADDR,
            *(p.lower() for p in pools),
        }
        for extra in (os.environ.get("RH_CHAIN_SWAP_ROUTER"), os.environ.get("RH_CHAIN_WETH")):
            if extra and extra.lower().startswith("0x"):
                skip.add(extra.lower())

        sem = asyncio.Semaphore(8)

        async def inbound(wallet: str) -> str:
            async with sem:
                try:
                    raw = await rpc.get_logs(
                        {
                            "fromBlock": hex(start),
                            "toBlock": hex(latest),
                            "address": quote,
                            "topics": [TRANSFER_TOPIC0, None, log_topic_addr(wallet)],
                        }
                    )
                except Exception:  # noqa: BLE001
                    return ""
            decoded = [m for m in (decode_transfer_log(x) for x in raw) if m]
            if not decoded:
                return ""
            frm = min(decoded, key=lambda m: m.block).frm.lower()
            return "" if frm in skip else frm

        hop1: dict[str, str] = {}
        found = await asyncio.gather(*(inbound(w) for w in buyers))
        for wallet, sender in zip(buyers, found):
            if sender:
                hop1[wallet] = sender
        parents = list(dict.fromkeys(hop1.values()))
        hop2: dict[str, str] = {}
        if parents:
            found2 = await asyncio.gather(*(inbound(p) for p in parents))
            for parent, src in zip(parents, found2):
                if src:
                    hop2[parent] = src
        roots = {wallet: hop2.get(sender, sender) for wallet, sender in hop1.items()}
        circ = sum(h.balance for h in holders if not (h.is_lp or h.is_burn))
        share = cluster_share_by_root(holders, roots, circ)
        return share, f"quote 2-hop cluster {share:.0%} ({len(hop1)}/{len(buyers)} buyers)"

    async def _solana_features(
        self, candidate: Candidate, result: Enrichment
    ) -> dict[str, float]:
        assert self.solana is not None
        values: dict[str, float] = {}
        cfg_t = self.strategy.section("terminals")
        window_min = float(cfg_t.get("attribution_window_minutes", 10))
        max_txs = int(cfg_t.get("max_txs_inspected", 250))

        pool_addresses = {candidate.pool_address} if candidate.pool_address else set()

        async def mint_job():
            try:
                return await self.solana.mint_info(candidate.address)
            except Exception as exc:  # noqa: BLE001
                result.notes.append(f"mint_info failed: {exc}")
                return None

        async def launch_job():
            cached = self._launch_slots.get(candidate.address)
            if cached is not None:
                return cached
            try:
                slot = await self.solana.launch_slot(candidate.address)
            except Exception:  # noqa: BLE001
                slot = 0
            self._launch_slots[candidate.address] = slot
            return slot

        async def holders_job():
            try:
                return await self.solana.largest_holders(
                    candidate.address,
                    pool_addresses=pool_addresses,
                    deployer=candidate.deployer,
                    resolve_ages=10,
                )
            except Exception as exc:  # noqa: BLE001
                result.notes.append(f"holders failed: {exc}")
                return None

        mint, launch_slot, holders_or_none = await asyncio.gather(
            mint_job(), launch_job(), holders_job()
        )
        result.mint = mint
        if mint is None:
            result.unknown.update({"mint_authority", "freeze_authority"})
        if not launch_slot:
            result.unknown.add("bundle_pct")
        holders = holders_or_none or []
        if holders_or_none is None:
            result.unknown.update(
                {
                    "top10_pct",
                    "top1_pct",
                    "gini",
                    "fresh_wallet_pct",
                    "known_holder_pct",
                    "dev_holding_pct",
                    "cluster_pct",
                    "lp_locked_pct",
                    "bundle_pct",
                }
            )

        stats = analyze(holders, now_ms=now_ms(), launch_slot=launch_slot or 0)
        result.holders = stats

        exact_count = None
        if self.helius and self.helius.enabled:
            exact_count = await self.helius.holder_count(candidate.address)
        if exact_count is None:
            result.unknown.add("holder_count")
            result.notes.append("holder count unknown (no Helius key)")
        else:
            stats.holder_count = exact_count
            history = self._holder_history[candidate.key]
            history.append((now_ms(), exact_count))
            del history[:-40]
            values["holder_growth_5m"] = holder_growth(history, 300_000)

        trades: list[flow.Trade] = []
        buys: list[terminals.BuyerTx] = []
        source = candidate.pool_address or candidate.address
        try:
            trades, buys = await self.solana.recent_activity(
                source, candidate.address, candidate.price_usd, max_txs=max_txs
            )
        except Exception as exc:  # noqa: BLE001
            result.notes.append(f"activity failed: {exc}")
            result.unknown.update({"retail_share", "bot_share", "axiom_share"})
        result.trades = trades

        if not result.candles and trades:
            result.candles = chart.candles_from_trades(
                [(t.ts_ms, t.price, t.size_usd) for t in trades],
                int(self.strategy.get("chart.candle_seconds", 60)),
            )
            if result.candles:
                values.update(chart.extract(result.candles, self.strategy.section("chart")))
                result.unknown -= {"vwap_dev", "atr_pct", "breakout", "volume_z", "body_ratio"}

        attribution = terminals.attribute(buys, self.registry)
        result.attribution = attribution
        values.update(
            terminals.extract(
                attribution,
                self.shares,
                candidate.key,
                now_ms(),
                int(window_min * 60_000),
            )
        )
        # Zero detectable terminals means the attribution features are noise,
        # not evidence. Flagging it here is what stops the strategy from
        # silently degenerating into "buy anything with volume".
        if not self.registry.attributable_labels:
            result.unknown.update({"retail_share", "retail_share_delta_5m", "axiom_share"})
            result.notes.append(
                "no terminal fee accounts labeled; run `alphahound discover-terminals`"
            )

        smart = self._known_wallets(candidate.chain)
        values.update(flow.extract(trades, now_ms(), smart=smart))
        bought: dict[str, float] = {}
        for t in trades:
            if t.wallet and t.side is flow.Side.BUY:
                bought[t.wallet] = bought.get(t.wallet, 0.0) + t.size_usd
        result.buyers = sorted(bought, key=bought.get, reverse=True)[:12]

        values.update(
            {
                "holder_count": float(stats.holder_count),
                "top10_pct": stats.top10_pct,
                "top1_pct": stats.top1_pct,
                "gini": stats.gini,
                "fresh_wallet_pct": stats.fresh_wallet_pct,
                "bundle_pct": stats.bundle_pct,
                "dev_holding_pct": stats.dev_holding_pct,
                "lp_locked_pct": stats.burned_pct,
                "known_holder_pct": known_holder_share(holders, smart),
            }
        )
        for key in unmeasured_from_stats(stats):
            result.unknown.add(key)
            values.pop(key, None)
        values.setdefault("holder_growth_5m", 0.0)
        values.update(await self._crowd_features(candidate, holders, trades, result))
        if "top10_pct" in result.unknown:
            result.unknown.update(
                {
                    "fomo_inside",
                    "fomo_net_flow",
                    "fomo_supply_pct",
                    "whale_hold_pct",
                    "whale_net_flow",
                }
            )
            for key in (
                "fomo_inside",
                "fomo_net_flow",
                "fomo_supply_pct",
                "whale_hold_pct",
                "whale_net_flow",
            ):
                values.pop(key, None)
        return values

    def _known_wallets(self, chain: Chain) -> set[str]:
        fp = tuple(sorted(str(r.get("address") or "") for r in self.whale_rows))
        if fp != self._whale_fp:
            self._smart_cache.clear()
            self._whale_fp = fp
        cached = self._smart_cache.get(chain)
        if cached is not None:
            return cached
        seeded = {str(a) for a in (self.strategy.get("flow.kol_wallets") or []) if a}
        known = self.store.smart_wallets(chain) | seeded | whale_addresses(self.whale_rows)
        self._smart_cache[chain] = known
        return known

    async def _crowd_features(
        self,
        candidate: Candidate,
        holders: list,
        trades: list[flow.Trade],
        result: Enrichment,
        size_pct: float | None = None,
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        whale_set = crowd_addresses(self.whale_rows, "whale")
        if size_pct is None:
            size_pct = float(self.strategy.get("whales.size_pct", 0.02))
        whale = whales.crowd_read(holders, trades, whale_set, size_pct=size_pct)
        kol_map: dict[str, str] = {}
        fomo_map: dict[str, str] = {}
        for row in self.whale_rows:
            addr = str(row.get("address") or "").strip()
            if not addr:
                continue
            key = addr.lower() if addr.startswith("0x") else addr
            name = (
                str(row.get("name") or row.get("handle") or "").strip().lstrip("@") or addr[:6]
            )
            source = str(row.get("source") or "").lower()
            klass = str(row.get("class") or "kol").lower()
            if source == "fomo" or klass == "fomo":
                fomo_map[key] = name
            elif klass != "whale":
                kol_map[key] = name
        result.crowd = {
            "whale_n": whale.inside,
            "whale_pct": round(whale.hold_pct, 4),
            "whale_usd": round(whale.hold_pct * max(0.0, candidate.mcap_usd)),
            "kols": who_inside(holders, trades, kol_map),
            "fomo": who_inside(holders, trades, fomo_map),
            "wallets": list(dict.fromkeys([*whale.wallets, *kol_map, *fomo_map]))[:16],
        }
        ignore_mcap = float(self.strategy.get("whales.ignore_mcap_usd", 50_000_000))
        if candidate.mcap_usd > ignore_mcap > 0:
            values["fomo_inside"] = 0.0
            values["fomo_net_flow"] = 0.0
            values["fomo_supply_pct"] = 0.0
            values["whale_hold_pct"] = 0.0
            values["whale_net_flow"] = 0.0
            result.notes.append("labeled flow ignored: mcap above accumulation floor")
            return values
        fomo_set = crowd_addresses(self.whale_rows, "fomo")
        if self.fomo and self.fomo.enabled:
            in_token = await self.fomo.token_wallets(candidate.address, candidate.chain)
            elite = await self.fomo.elite_wallets()
            if elite:
                in_token &= elite
            fomo_set |= in_token
        if fomo_set:
            fomo = whales.crowd_read(holders, trades, fomo_set)
            values["fomo_inside"] = float(fomo.inside)
            values["fomo_net_flow"] = fomo.net_flow
            values["fomo_supply_pct"] = fomo.hold_pct
            result.notes.append(
                f"fomo: {fomo.inside} inside {fomo.hold_pct:.0%} net {fomo.net_flow:+.2f}"
            )
        else:
            result.unknown.update({"fomo_inside", "fomo_net_flow", "fomo_supply_pct"})

        values["whale_hold_pct"] = whale.hold_pct
        values["whale_net_flow"] = whale.net_flow
        result.notes.append(
            f"whales: {whale.inside} hold {whale.hold_pct:.0%} net {whale.net_flow:+.2f}"
        )
        return values

    async def _narrative_features(
        self, candidate: Candidate, result: Enrichment, values: dict[str, float]
    ) -> dict[str, float]:
        """Twitter chatter, bubble/funding cluster, copy-trade flag.

        Copy is a boost on a young small launch with labeled flow — never a
        reason to chase a whale stacking an old $SOL bag.
        """
        out: dict[str, float] = {}
        holders_ok = "top10_pct" not in result.unknown
        cluster = result.holders.largest_funding_cluster_pct if holders_ok else 0.0
        if self.bubbles and self.bubbles.enabled:
            bm = await self.bubbles.cluster_pct(candidate.chain, candidate.address)
            if bm is not None:
                cluster = max(cluster, bm)
                holders_ok = True
        if holders_ok:
            out["cluster_pct"] = cluster
        else:
            result.unknown.add("cluster_pct")

        from ..providers import utility_hint

        handle = result.snap.twitter if result.snap else ""
        blurb = result.snap.blurb if result.snap else ""
        if self.twitter and self.twitter.enabled:
            read = await self.twitter.scan(
                candidate.symbol, candidate.address, handle=handle, blurb=blurb
            )
            if read is None:
                result.unknown.update({"twitter_mentions", "twitter_inst", "twitter_fresh"})
                result.twitter = {"official": handle, "utility": utility_hint(blurb), "posts": []}
            else:
                out["twitter_mentions"] = read.mentions
                out["twitter_inst"] = read.inst
                out["twitter_fresh"] = read.fresh
                result.twitter = read.as_visor()
                if read.mentions or read.official:
                    result.notes.append(
                        f"twitter: {read.mentions:.0f} ct"
                        + (f" @{read.official}" if read.official else "")
                        + (f" inst {read.inst:.2f}" if read.inst else "")
                    )
        else:
            result.unknown.update({"twitter_mentions", "twitter_inst", "twitter_fresh"})
            result.twitter = {"official": handle, "utility": utility_hint(blurb), "posts": []}

        out["copy_signal"] = copy_flag(
            age_minutes=candidate.age_minutes,
            mcap_usd=candidate.mcap_usd,
            smart_buys=values.get("smart_money_buys", 0.0),
            fomo_inside=values.get("fomo_inside", 0.0),
            fomo_net_flow=values.get("fomo_net_flow", 0.0),
            whale_net_flow=values.get("whale_net_flow", 0.0),
            strategy=self.strategy,
            chain=candidate.chain,
        )
        if result.snap is not None:
            result.snap.stamp(candidate)
        out["dex_profile"] = candidate.dex_profile
        if candidate.dex_paid or candidate.dex_photo:
            result.notes.append(
                "dex: "
                + ("paid" if candidate.dex_paid else "unpaid")
                + (" · foto" if candidate.dex_photo else "")
                + (" · alinhada" if candidate.dex_aligned else "")
            )
        return out

    def _aggregate_only_features(
        self, candidate: Candidate, result: Enrichment
    ) -> dict[str, float]:
        """The aggregate view only: every EVM chain, and Solana without an RPC.

        Holder distribution, wallet ages and terminal attribution all require a
        per-chain indexer we do not have, so those features are marked unknown
        rather than estimated. The honeypot and tax risk that distribution
        would have caught is instead caught by the round-trip probe, which is
        chain-agnostic and, for that specific risk, strictly better evidence.
        """
        result.unknown.update(
            {
                "holder_count",
                "top10_pct",
                "gini",
                "fresh_wallet_pct",
                "bundle_pct",
                "dev_holding_pct",
                "retail_share",
                "retail_share_delta_5m",
                "bot_share",
                "axiom_share",
                "known_holder_pct",
                "top1_pct",
                "mint_authority",
                "freeze_authority",
                "fomo_inside",
                "fomo_net_flow",
                "whale_hold_pct",
                "whale_net_flow",
            }
        )
        result.notes.append(f"{candidate.chain.value}: aggregate-only enrichment")

        snap = result.snap
        if snap is None:
            result.unknown.update({"buy_sell_ratio", "unique_buyers_5m", "net_inflow_usd_5m"})
            return {}

        # Transaction counts, not dollars. Weaker than the USD-weighted ratio
        # the Solana path computes, and marked as such so the model does not
        # treat the two as interchangeable.
        total = snap.buys_m5 + snap.sells_m5
        ratio = min(10.0, snap.buys_m5 / snap.sells_m5) if snap.sells_m5 else (3.0 if snap.buys_m5 else 1.0)
        avg = snap.volume_m5 / total if total else 0.0
        result.unknown.add("net_inflow_usd_5m")
        # ponytail: Dexscreener m5 buys are tx counts, not unique wallets.
        result.unknown.add("unique_buyers_5m")
        return {
            "buy_sell_ratio": ratio,
            "unique_buyers_5m": float(snap.buys_m5),
            "avg_buy_size_usd": avg,
            "net_inflow_usd_5m": snap.volume_m5 * (ratio - 1.0) / (ratio + 1.0) if ratio else 0.0,
        }

    async def _evm_labeled_crowd(
        self, candidate: Candidate, result: Enrichment
    ) -> dict[str, float]:
        """balanceOf on labeled KOL/Fomo wallets. Full holder lists need an indexer."""
        rpc = self.evm_rpcs.get(candidate.chain)
        addrs = []
        seen: set[str] = set()
        for row in self.whale_rows:
            addr = str(row.get("address") or "").strip()
            key = addr.lower()
            if not addr.startswith("0x") or key in seen:
                continue
            seen.add(key)
            addrs.append(addr)
        if rpc is None or not addrs:
            result.notes.append(
                f"{candidate.chain.value}: KOL check skipped ({'no rpc' if rpc is None else 'no labeled evm wallets'})"
            )
            return {}
        cached = self._evm_cached(f"crowd:{candidate.key}")
        if cached is not None:
            if cached.get("crowd"):
                result.crowd = dict(cached["crowd"])
            result.unknown.discard("fomo_inside")
            result.unknown.discard("fomo_supply_pct")
            result.unknown.update({"whale_hold_pct", "whale_net_flow", "fomo_net_flow"})
            if cached.get("note"):
                result.notes.append(str(cached["note"]))
            return dict(cached.get("values") or {})

        async def one(addr: str):
            try:
                bal = await rpc.balance_of(candidate.address, addr)
            except Exception as exc:  # noqa: BLE001
                result.notes.append(f"balanceOf {addr[:8]}: {exc}")
                return None
            if bal > 0:
                return Holder(address=addr, balance=float(bal))
            return None

        found = await asyncio.gather(*(one(a) for a in addrs))
        holders = [h for h in found if h is not None]
        extra = await self._crowd_features(candidate, holders, [], result, size_pct=0.0)
        result.unknown.discard("fomo_inside")
        extra["fomo_inside"] = float(len((result.crowd or {}).get("fomo") or []))
        extra.pop("whale_hold_pct", None)
        extra.pop("whale_net_flow", None)
        extra.pop("fomo_net_flow", None)
        result.unknown.update({"whale_hold_pct", "whale_net_flow", "fomo_net_flow"})
        fomo_set = {a.lower() for a in crowd_addresses(self.whale_rows, "fomo") if a.startswith("0x")}
        fomo_bal = sum(h.balance for h in holders if h.address.lower() in fomo_set)
        try:
            supply = float(await rpc.total_supply(candidate.address))
        except Exception as exc:  # noqa: BLE001
            result.notes.append(f"totalSupply: {exc}")
            supply = 0.0
        if supply > 0:
            extra["fomo_supply_pct"] = min(1.0, fomo_bal / supply)
            result.unknown.discard("fomo_supply_pct")
        else:
            extra.pop("fomo_supply_pct", None)
            result.unknown.add("fomo_supply_pct")
        result.notes.append(
            f"labeled holders: {len(holders)}/{len(addrs)} KOL/fomo wallets have a bag"
            + (f" fomo_supply {extra.get('fomo_supply_pct', 0):.0%}" if "fomo_supply_pct" in extra else "")
        )
        self._evm_put(
            f"crowd:{candidate.key}",
            {
                "values": extra,
                "crowd": dict(result.crowd or {}),
                "note": result.notes[-1],
            },
        )
        return extra

    # -- cost --------------------------------------------------------------
    async def _cost_features(
        self, candidate: Candidate, probe_size_usd: float, result: Enrichment
    ) -> dict[str, float]:
        if self.probe is None or probe_size_usd <= 0:
            result.unknown.update({"price_impact", "round_trip_cost", "sellable"})
            return {}
        try:
            rt = await self.probe(candidate, probe_size_usd)
        except Exception as exc:  # noqa: BLE001
            result.round_trip = RoundTrip(ok=False, note=str(exc))
            result.unknown.update({"price_impact", "round_trip_cost", "sellable"})
            return {}
        result.round_trip = rt
        if not rt.ok:
            result.unknown.add("sellable")
        return {"price_impact": rt.price_impact, "round_trip_cost": rt.total_cost_pct}

    def forget(self, key: str) -> None:
        self.shares.forget(key)
        self._holder_history.pop(key, None)
        self._evm_cache.pop(f"dist:{key}", None)
        self._evm_cache.pop(f"crowd:{key}", None)
