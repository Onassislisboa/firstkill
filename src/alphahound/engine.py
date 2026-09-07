"""The engine.

One loop, in a fixed order that encodes the priorities:

    1. manage open positions   money already at risk outranks money not yet at
                               risk, always
    2. update shadow tracking  measure what the filters cost
    3. discover                pull new candidates
    4. score and enter         spend the RPC budget on the best candidates only
    5. learn                   periodically, never inside the hot path

Open positions are persisted to state/positions.json on every change. A bot
that forgets its positions on restart is a bot that leaves bags on chain, and
restarts happen at exactly the moments you least want that.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import IO, Any

from . import learning
from .discovery import Discovery
from .execution import ExecutionError, Router, build_router
from .execution.evm import EvmRpc
from .log import get
from .models import (
    Action,
    Candidate,
    Chain,
    Decision,
    ErrorClass,
    ExitReason,
    Features,
    Position,
    Score,
    TradeRecord,
    VenueId,
    describe_exit,
    now_ms,
)
from .net import Http
from .portfolio import ExitOrder, PositionManager
from .playbook import gate as pb_gate
from .playbook import section as pb_section
from .preview import write_preview
from .providers import Birdeye, Bubblemaps, Dexscreener, FomoGraph, Helius, Twitter, twitter_handle
from .risk import RiskEngine
from .scoring import Model, Scorer, hide_from_visor, hold_cut, wait_on_visor, watch_call
from .settings import (
    PUBLIC_SOLANA_RPC,
    Config,
    Settings,
    apply_aggressive_learning,
    chase_rows,
    load_strategy,
    load_terminals,
    score_floors,
)
from .signals import Enricher, Enrichment
from .signals.pack import apply_tags, dump_beta_keys
from .signals.solana import SolanaReader
from .signals.terminals import TerminalRegistry
from .store import Store, features_from_json, lock_state_dir
from .verdict import security_cert

log = get("engine")

# RPC/network may fail a pass. These mean the code itself is wrong — retrying
# forever looks "alive" on the visor while scan is frozen (NameError mcap_is_dead).
LOOP_BUGS = (
    NameError,
    AttributeError,
    TypeError,
    SyntaxError,
    ImportError,
    IndentationError,
)


def loop_bug(exc: BaseException) -> bool:
    return isinstance(exc, LOOP_BUGS)


def enrich_due(last_scored_ms: int, now: int, ttl_ms: int) -> bool:
    """First look always; afterwards wait ttl so RPC isn't spent re-reading the same mint."""
    return last_scored_ms <= 0 or ttl_ms <= 0 or now - last_scored_ms >= ttl_ms


def mcap_is_dead(mcap_usd: float, floor: float) -> bool:
    return floor > 0 and 0 < mcap_usd < floor


def below_scan_mcap(mcap_usd: float, floor: float) -> bool:
    """No visor/enrich under the scan floor, including unquoted mcap=0."""
    return floor > 0 and mcap_usd < floor


def drop_for_scan_mcap(candidate: Candidate, floor: float) -> bool:
    """Dead-mcap prune. hood_stream at mcap 0 waits on Dexscreener off-visor; 4k is dead."""
    if candidate.source == "inspect":
        return False
    if observe_early(candidate.source) and candidate.mcap_usd <= 0:
        return False
    return below_scan_mcap(candidate.mcap_usd, floor)


def unpaid_for_scan(candidate: Candidate) -> bool:
    """No visor/enrich/watch without a paid Dexscreener listing. Inspect still goes through."""
    return candidate.source != "inspect" and not candidate.dex_paid


def on_scan_visor(candidate: Candidate, floor: float) -> bool:
    if candidate.source == "inspect":
        return True
    if unpaid_for_scan(candidate):
        return False
    return not below_scan_mcap(candidate.mcap_usd, floor)


def visor_card(candidate: Candidate, reads: dict, floor: float) -> bool:
    """Skip/bundle/LP never occupy a card. WAIT floors still do."""
    if not on_scan_visor(candidate, floor):
        return False
    return (reads.get(candidate.key) or {}).get("call") != "skip"


def observe_early(source: str) -> bool:
    """On the visor before buy floors. Must still fail prefilter on enter."""
    return source == "hood_stream"


def absorb_watch(dst: Candidate, src: Candidate) -> None:
    """Keep the visor object. Discovery re-emits a blank mint and would freeze mcap."""
    dst.symbol = src.symbol or dst.symbol
    dst.name = src.name or dst.name
    # ponytail: Dexscreener relabels Pons graduates as uniswap; keep origin.
    if (dst.dex_id or "").lower() != "pons":
        dst.dex_id = src.dex_id or dst.dex_id
    dst.pool_address = src.pool_address or dst.pool_address
    dst.dex_paid = dst.dex_paid or src.dex_paid
    dst.dex_photo = dst.dex_photo or src.dex_photo
    dst.dex_aligned = dst.dex_aligned or src.dex_aligned
    if src.mcap_usd > 0:
        dst.price_usd = src.price_usd or dst.price_usd
        dst.mcap_usd = src.mcap_usd
        dst.volume_5m_usd = src.volume_5m_usd
        dst.liquidity_usd = src.liquidity_usd
        dst.ret_5m = src.ret_5m
    if src.created_at_ms and (not dst.created_at_ms or src.created_at_ms < dst.created_at_ms):
        dst.created_at_ms = src.created_at_ms


_FLOOR_KINDS = frozenset({"mcap", "volume", "liquidity"})


def stamp_live_floors(read: dict, floors: list[str]) -> dict:
    """Keep cluster/etc vetoes; rewrite mcap/vol/liq from the live quote."""
    kept = [
        v
        for v in (read.get("vetoes") or [])
        if str(v).split(":", 1)[0] not in _FLOOR_KINDS
    ]
    out = {**read, "vetoes": floors + kept}
    kind = str(out.get("why") or "").split(":", 1)[0]
    if floors and kind in _FLOOR_KINDS | {"", "ok"}:
        out["why"] = floors[0]
        if not out.get("p"):
            out["explain"] = floors[0]
    elif kind in _FLOOR_KINDS:
        out["why"] = kept[0] if kept else "ok"
        if not out.get("p"):
            out["explain"] = out["why"]
    return out


def best_setup_p(reads: dict[str, dict], keys: list[str]) -> float:
    best = 0.0
    for key in keys:
        rec = reads.get(key) or {}
        if rec.get("call") == "skip":
            continue
        best = max(best, float(rec.get("p") or 0))
    return best


def early_observe_floors(candidate: Candidate, strategy: Config) -> list[str]:
    """Treat unmeasured mcap/vol as below the buy floor. Detect ≠ eligible.

    evaluate_gates skips mcap when it is 0 and volume when Dexscreener has not
    stamped a snap yet. A factory log has both at zero, so without this the
    token could walk into enrich/enter. Same numbers as the playbook, not a
    lower floor.
    """
    pb = pb_section(strategy, candidate.chain)
    copy_min = float(pb.get("copy_min_mcap_usd", 100_000))
    min_vol = float(pb.get("min_volume_5m", 0) or 0)
    min_liq = pb_gate(strategy, candidate.chain, "min_liquidity_usd", 15_000.0)
    out: list[str] = []
    if copy_min > 0 and candidate.mcap_usd < copy_min:
        out.append(f"mcap: {candidate.mcap_usd:.0f} below {copy_min:.0f} floor")
    if min_vol > 0 and candidate.volume_5m_usd < min_vol:
        out.append(f"volume: {candidate.volume_5m_usd:.0f} < {min_vol:.0f} (5m)")
    if min_liq > 0 and candidate.liquidity_usd < min_liq:
        out.append(f"liquidity: {candidate.liquidity_usd:.0f} < {min_liq:.0f}")
    return out


def sell_queued_for(queued: set[str], key: str, address: str) -> bool:
    """True if this open bag was asked to flatten from the visor/CLI file."""
    if not queued:
        return False
    if key in queued or address in queued:
        return True
    want = {key.lower(), address.lower()}
    return any(q.lower() in want for q in queued)


def _crowd_sponsors(crowd: dict | None) -> list[str]:
    out: list[str] = []
    for x in list((crowd or {}).get("kols") or []) + list((crowd or {}).get("wallets") or []):
        s = str(x)
        if s and s not in out:
            out.append(s)
    return out[:16]


def build_registry(store: Store, terminals: Config) -> TerminalRegistry:
    entries = terminals.get("terminal", []) or []
    venues = terminals.section("venues")
    return TerminalRegistry(entries, venues, store.terminal_labels())


class Engine:
    def __init__(
        self,
        settings: Settings,
        strategy: Config | None = None,
        terminals: Config | None = None,
    ) -> None:
        self.settings = settings
        self.strategy = strategy or load_strategy()
        self.terminals = terminals or load_terminals()

        self.store = Store(settings.state_dir)
        self.strategy = apply_aggressive_learning(self.strategy, self.store)
        self.http = Http()
        self.dex = Dexscreener(
            self.http,
            cache_seconds=max(0.3, float(self.strategy.get("loop.quote_seconds", 1.0))),
        )
        self.helius = Helius(self.http, settings.helius_api_key)
        self.birdeye = Birdeye(self.http, settings.birdeye_api_key)
        self.fomo = FomoGraph(self.http, settings.cope_api_key, self.strategy)
        self.twitter = Twitter(self.http, settings.twitter_bearer)
        self.bubbles = Bubblemaps(self.http, settings.bubblemaps_api_key)
        # Without an RPC the whole distribution and terminal-attribution half of
        # the model is unmeasured, and an unmeasured feature contributes zero -
        # so the bot would quietly score every token on aggregates alone and
        # reject all of them, looking like it was being disciplined. In paper
        # mode fall back to the public endpoint so the features exist; live mode
        # refuses it outright in Settings.validate().
        rpc = settings.solana_rpc_url
        if not rpc and not settings.live and Chain.SOLANA in settings.enabled_chains:
            rpc = PUBLIC_SOLANA_RPC
            log.warning(
                "no SOLANA_RPC_URL: falling back to the public endpoint for paper mode. "
                "It is heavily rate-limited, so holder and attribution features will be "
                "patchy. A paid RPC is required before live trading."
            )
        self.solana = (
            SolanaReader(self.http, rpc)
            if rpc and Chain.SOLANA in settings.enabled_chains
            else None
        )
        evm_rpcs = {
            chain: EvmRpc(self.http, url)
            for chain, url in settings.rpc_urls.items()
            if url
        }

        self.router: Router = build_router(
            settings, self.strategy, self.dex, self.http, store=self.store
        )
        self.registry = build_registry(self.store, self.terminals)
        self.enricher = Enricher(
            store=self.store,
            strategy=self.strategy,
            dexscreener=self.dex,
            registry=self.registry,
            solana=self.solana,
            helius=self.helius,
            birdeye=self.birdeye,
            probe=self.router.round_trip,
            fomo=self.fomo,
            whale_rows=chase_rows(settings.state_dir),
            twitter=self.twitter,
            bubbles=self.bubbles,
            evm_rpcs=evm_rpcs,
        )
        self.scorer = Scorer(Model.load(self.store), self.strategy, self.store, live=settings.live)
        self.risk = RiskEngine(self.strategy, self.store)
        self.exits = PositionManager(self.strategy, self.store)
        self.discovery = Discovery(settings, self.strategy, self.dex, http=self.http)

        self.positions: dict[str, Position] = {}
        self.watching: dict[str, Candidate] = {}
        self._positions_path = settings.state_dir / "positions.json"
        self._closed_since_learn = 0
        self._lock: IO[bytes] | None = None
        self._tick_counts: Counter[str] = Counter()
        self._last_heartbeat_ms = now_ms()
        self._watch_in = 0
        self._watch_out = 0
        self._skip_ban: dict[str, int] = {}
        self._reads: dict[str, dict] = {}
        self._inflight: set[str] = set()
        self._manual_sell: set[str] = set()
        self._label_fail: set[str] = set()
        self._stop = asyncio.Event()
        self._loop_faults: dict[str, str] = {}
        self._loop_dead: set[str] = set()
        self._load_positions()

    # -- lifecycle ---------------------------------------------------------
    async def run(self) -> None:
        problems = self.settings.validate()
        if problems:
            for problem in problems:
                log.error("configuration problem", extra={"problem": problem})
            raise SystemExit("refusing to start with an invalid live configuration")

        try:
            self._lock = lock_state_dir(self.settings.state_dir)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

        min_p, min_ev = score_floors(self.strategy, self.store)
        log.info(
            "starting",
            extra={
                "mode": self.settings.mode,
                "chains": [c.value for c in self.settings.enabled_chains],
                "weights_version": self.scorer.model.version,
                "equity_usd": self.risk.equity(),
                "open_positions": len(self.positions),
                "shadow_track_minutes": int(
                    float(self.strategy.get("learning.shadow_track_minutes", 60))
                ),
                "min_expected_value": min_ev,
                "min_probability": min_p,
                "max_concurrent_positions": int(
                    self.strategy.get("risk.max_concurrent_positions", 1)
                ),
                "aggressive_learning": bool(self.strategy.get("aggressive_learning._active")),
                "quote_cluster_hops": 2,
                "fresh_unmeasured_neutral": True,
                "max_candidate_age_minutes": int(
                    float(self.strategy.get("loop.max_candidate_age_minutes", 180))
                ),
            },
        )
        if (
            not self.registry.attributable_labels
            and Chain.SOLANA in self.settings.enabled_chains
        ):
            log.warning(
                "no terminal fee accounts labeled: attribution features are inert. "
                "Run `alphahound discover-terminals` to populate them."
            )

        self._assert_loop_helpers()
        await self.discovery.start()
        try:
            await asyncio.gather(self._risk_loop(), self._scan_loop(), self._quote_loop())
        finally:
            await self.shutdown()

    def _assert_loop_helpers(self) -> None:
        if not mcap_is_dead(1.0, 40_000) or mcap_is_dead(0.0, 40_000):
            raise RuntimeError("mcap_is_dead helper is broken")
        if not enrich_due(0, 1, 15_000):
            raise RuntimeError("enrich_due helper is broken")
        # Boot fails if ingest would hide a wait-class coin again.
        if hide_from_visor(["chase: 5m ripped, wait dip"]):
            raise RuntimeError("chase must stay on the visor")
        if hide_from_visor(["mcap: 80000 below 100000 floor"]):
            raise RuntimeError("buy-floor mcap must stay on the visor")
        if hide_from_visor(["volume: 2000 < 5000 (5m)"]):
            raise RuntimeError("buy-floor volume must stay on the visor")
        if not hide_from_visor(["lp_unlocked: 100% da liquidez livre"]):
            raise RuntimeError("hard skip must still hide at ingest")
        if not hide_from_visor(["volume: 819 < 5000 (5m)", "cluster: 37% linked supply"]):
            raise RuntimeError("bundle must hide even when a wait floor is also true")

    async def _every(self, seconds: float, body, label: str) -> None:
        while not self._stop.is_set():
            started = now_ms()
            try:
                await body()
                self._loop_faults.pop(label, None)
            except Exception as exc:  # noqa: BLE001
                log.exception(f"{label} failed")
                self._loop_faults[label] = f"{type(exc).__name__}: {exc}"[:240]
                self._write_preview()
                if loop_bug(exc):
                    self._loop_dead.add(label)
                    log.error(
                        "loop dead",
                        extra={"loop": label, "error": self._loop_faults[label]},
                    )
                    return
            elapsed = (now_ms() - started) / 1000.0
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=max(0.1, seconds - elapsed))

    async def _risk_loop(self) -> None:
        """Exits and shadow tracking, on their own schedule.

        Separate from the entry scan on purpose. Enrichment latency belongs to
        somebody else's rate limiter, and one slow candidate used to push the
        whole loop past two minutes - during which a stop that should have
        filled at -28% instead fills wherever the token drifted to. Opportunity
        can wait for data; risk cannot.
        """

        async def body() -> None:
            await self.manage_positions()
            await self.update_shadows()
            self._write_preview()
            self._heartbeat()

        await self._every(float(self.strategy.get("loop.tick_seconds", 3.0)), body, "risk pass")

    async def _quote_loop(self) -> None:
        async def body() -> None:
            await self._refresh_watch_quotes(probe_paid=False)
            self._prune_watching()
            self._write_preview()

        await self._every(float(self.strategy.get("loop.quote_seconds", 1.0)), body, "quotes")

    async def _scan_loop(self) -> None:
        async def body() -> None:
            self.enricher.whale_rows = chase_rows(self.settings.state_dir)
            self._drain_inspect()
            found = await self.discovery.poll()
            before = set(self.watching)
            dead_floor = float(self.strategy.get("loop.dead_mcap_usd", 50_000))
            probed = 0
            for candidate in found:
                if unpaid_for_scan(candidate):
                    if probed >= 8:  # ponytail: 8 orders/v1 per scan; unpaid never sit on visor
                        continue
                    probed += 1
                    try:
                        if await self.dex.token_is_paid(candidate.chain, candidate.address):
                            candidate.dex_paid = True
                    except Exception:  # noqa: BLE001
                        continue
                    if unpaid_for_scan(candidate):
                        continue
                ban_ms = int(float(self.strategy.get("loop.skip_ban_minutes", 120)) * 60_000)
                banned = self._skip_ban.get(candidate.key)
                if (
                    ban_ms > 0
                    and candidate.source != "inspect"
                    and banned
                    and now_ms() - banned < ban_ms
                ):
                    continue
                prev = self.watching.get(candidate.key)
                if prev is not None:
                    absorb_watch(prev, candidate)
                    candidate = prev
                if candidate.pack_role == "vamp":
                    continue
                if candidate.source != "inspect" and mcap_is_dead(candidate.mcap_usd, dead_floor):
                    continue
                if candidate.source != "inspect" and not observe_early(candidate.source):
                    cheap = self.enricher.free_enrichment(candidate)
                    vetoes = [
                        v
                        for v in self.scorer.prefilter(cheap)
                        if not v.startswith("age:") and not v.startswith("priced:")
                    ]
                    if vetoes:
                        if hide_from_visor(vetoes):
                            self._tick_counts[f"free_veto:{vetoes[0].split(':')[0]}"] += 1
                            continue
                self.watching[candidate.key] = candidate
                self._reads.setdefault(
                    candidate.key,
                    {"call": "wait" if observe_early(candidate.source) else "scan"},
                )
            self.discovery.prune()
            self._drop_below_scan_mcap()
            self._drop_unpaid()
            await self.score_and_enter()
            self._prune_watching()
            self._watch_in = sum(1 for k in self.watching if k not in before)
            self._watch_out = sum(1 for k in before if k not in self.watching)
            await self.maybe_learn()
            self._write_preview()

        await self._every(float(self.strategy.get("loop.scan_seconds", 3.0)), body, "scan pass")

    def request_stop(self) -> None:
        self._stop.set()

    async def shutdown(self) -> None:
        await self.discovery.stop()
        await self.http.aclose()
        self._save_positions()
        self.store.close()
        if self._lock is not None:
            self._lock.close()
            self._lock = None
        log.info("stopped")

    # -- the loop ----------------------------------------------------------

    def _write_preview(self) -> None:
        self._retag()
        halted, reason = self.risk.halted()
        holds = []
        for position in self.positions.values():
            mark = position.candidate.price_usd or position.entry_price
            remaining = (
                position.tokens_remaining / position.tokens if position.tokens else 0.0
            )
            held_usd = round(position.tokens_remaining * mark, 2)
            holds.append(
                {
                    "key": position.candidate.key,
                    "symbol": position.candidate.symbol or "",
                    "chain": position.candidate.chain.value,
                    "address": position.candidate.address,
                    "size_usd": round(position.size_usd, 2),
                    "held_usd": held_usd,
                    "unrealized_usd": round(position.unrealized_usd(mark), 2),
                    "unrealized_pct": round(position.gain(mark), 4),
                    "remaining_pct": round(remaining, 4),
                    "ladder": position.ladder_filled,
                    "age_min": int((now_ms() - position.opened_at_ms) / 60_000),
                    "role": position.candidate.pack_role or "",
                    "entry_rubric": round(position.entry_rubric, 1),
                    "hold_rubric": round(position.last_hold_rubric, 1),
                    "hold_why": position.last_hold_why,
                    "hold_strikes": position.hold_strikes,
                    "mcap": round(position.candidate.mcap_usd),
                    "mcap_entry": round(position.entry_mcap_usd),
                }
            )
        scan_floor = float(self.strategy.get("loop.dead_mcap_usd", 50_000))
        visor = [c for c in self.watching.values() if visor_card(c, self._reads, scan_floor)]
        visor_keys = [c.key for c in visor]
        write_preview(
            self.settings.state_dir,
            {
                "ts_ms": now_ms(),
                "mode": self.settings.mode,
                "halted": halted,
                "halt_reason": reason,
                "equity_usd": round(self.risk.equity(), 2),
                "watching": len(visor),
                "watch_in": self._watch_in,
                "watch_out": self._watch_out,
                "watch": [
                    {
                        "symbol": c.symbol or "",
                        "name": c.name or "",
                        "chain": c.chain.value,
                        "address": c.address,
                        "age_min": round(c.age_minutes, 1),
                        "mcap": round(c.mcap_usd),
                        "vol5m": round(c.volume_5m_usd),
                        "ret_5m": round(c.ret_5m, 4),
                        "dex": c.dex_id,
                        "source": c.source,
                        "role": c.pack_role or "solo",
                        "stem": c.pack_stem,
                        "pack": c.pack_size,
                        "dex_paid": c.dex_paid,
                        "dex_photo": c.dex_photo,
                        "dex_aligned": c.dex_aligned,
                        "liq": round(c.liquidity_usd),
                        **stamp_live_floors(
                            dict(self._reads.get(c.key) or {"call": "scan"}),
                            early_observe_floors(c, self.strategy),
                        ),
                    }
                    for c in sorted(
                        visor,
                        key=lambda x: (
                            {"scan": 0, "trade": 1, "wait": 2, "skip": 3}.get(
                                (self._reads.get(x.key) or {}).get("call") or "scan", 4
                            ),
                            0 if x.dex_paid else 1,
                            {"main": 0, "beta": 1, "solo": 2, "vamp": 3}.get(x.pack_role, 2),
                            x.age_minutes,
                        ),
                    )
                ],
                "best_probability": round(best_setup_p(self._reads, visor_keys), 3),
                "tick": dict(self._tick_counts),
                "holds": holds,
                "faults": dict(self._loop_faults),
                "dead_loops": sorted(self._loop_dead),
            },
        )

    def _heartbeat(self) -> None:
        """Periodic proof of life, with the reason nothing was bought.

        A selective bot is silent for long stretches, and silence is
        indistinguishable from a hung loop or a dead data feed. Reporting what
        was seen and what rejected it is the difference between "working as
        intended" and an operator restarting a healthy process.
        """
        interval = float(self.strategy.get("loop.heartbeat_seconds", 60.0))
        if interval <= 0 or now_ms() - self._last_heartbeat_ms < interval * 1000:
            return
        self._last_heartbeat_ms = now_ms()
        scan_floor = float(self.strategy.get("loop.dead_mcap_usd", 50_000))
        visor = [c for c in self.watching.values() if visor_card(c, self._reads, scan_floor)]
        log.info(
            "heartbeat",
            extra={
                "watching": len(visor),
                "watch_in": self._watch_in,
                "watch_out": self._watch_out,
                "open": len(self.positions),
                "equity_usd": round(self.risk.equity(), 2),
                "since_last": dict(self._tick_counts),
                "best_probability": round(best_setup_p(self._reads, [c.key for c in visor]), 3),
                "dead_loops": sorted(self._loop_dead),
            },
        )
        self._tick_counts.clear()

    # -- positions ---------------------------------------------------------
    async def manage_positions(self) -> None:
        self._drain_sells()
        if not self.positions:
            self._manual_sell.clear()
            return
        halted, reason = self.risk.halted()
        self._retag()

        for key, position in list(self.positions.items()):
            await self.enricher.refresh(position.candidate)
            price = position.candidate.price_usd
            if price <= 0:
                log.warning("no price for open position", extra={"key": key})
                continue

            orders = self.exits.evaluate(position, price, position.candidate.liquidity_usd)
            main = self.watching.get(position.candidate.main_key)
            main_ret = main.ret_5m if main is not None else position.candidate.main_ret_5m
            if position.candidate.pack_role == "beta" and main_ret <= -0.20:
                orders = [
                    ExitOrder(1.0, ExitReason.THESIS_CUT, "beta: main runner dumping")
                ]
            if halted:
                # Flatten everything. A TP rung in the same tick must not leave a stub.
                orders = [ExitOrder(1.0, ExitReason.KILL_SWITCH, reason)]
            elif sell_queued_for(
                self._manual_sell, position.candidate.key, position.candidate.address
            ):
                orders = [ExitOrder(1.0, ExitReason.MANUAL, "sold from preview")]
            full = any(o.fraction >= 1.0 for o in orders)
            if not full:
                cut = await self._stage3(position)
                if cut is not None:
                    orders = [cut]
            for order in orders:
                await self._exit(position, order.fraction, order.reason, order.note)
                if position.tokens_remaining <= 1e-12:
                    break
        self._manual_sell = {
            q
            for q in self._manual_sell
            if any(
                sell_queued_for({q}, p.candidate.key, p.candidate.address)
                for p in self.positions.values()
            )
        }

    async def _stage3(self, position: Position) -> ExitOrder | None:
        now = now_ms()
        grace = float(self.strategy.get("hold.grace_seconds", 90))
        if now - position.opened_at_ms < int(grace * 1000):
            return None
        every = float(self.strategy.get("hold.rescore_seconds", 15))
        if position.last_hold_ms and now - position.last_hold_ms < int(every * 1000):
            return None
        probe = max(10.0, float(self.strategy.get("risk.min_position_usd", 10)))
        try:
            enr = await self.enricher.enrich(position.candidate, probe)
        except Exception as exc:  # noqa: BLE001
            log.debug("hold enrich failed", extra={"error": str(exc)})
            return None
        score = self.scorer.score(enr)
        review = hold_cut(position, enr, score, self.strategy)
        position.last_hold_ms = now
        position.last_hold_rubric = review.rubric
        position.last_hold_why = review.cut or review.why
        prev = self._reads.get(position.candidate.key) or {}
        self._reads[position.candidate.key] = {
            **prev,
            **(enr.crowd or {}),
            "call": "hold",
            "why": review.why,
            "rubric": score.rubric or {},
            "tw": enr.twitter or {},
        }
        if not review.cut:
            return None
        return ExitOrder(1.0, ExitReason.THESIS_CUT, review.cut)

    async def _exit(self, position: Position, fraction: float, reason: ExitReason, note: str) -> None:
        tokens = position.tokens_remaining * max(0.0, min(1.0, fraction))
        if tokens <= 0:
            return
        try:
            fill = await self.router.sell(position, tokens)
        except ExecutionError as exc:
            log.error(
                "exit failed",
                extra={"key": position.candidate.key, "reason": reason.value, "error": str(exc)},
            )
            return

        cost_basis = position.size_usd * (tokens / position.tokens) if position.tokens else 0.0
        position.realized_usd += fill.amount_out - cost_basis
        position.fees_usd += fill.fee_usd
        position.tokens_remaining = max(0.0, position.tokens_remaining - tokens)
        position.last_exit_price = fill.price
        position.last_exit_reason = reason.value
        position.exit_legs.append(
            {
                "ts_ms": now_ms(),
                "usd_out": round(fill.amount_out, 2),
                "size_usd": round(cost_basis, 2),
                "pnl_usd": round(fill.amount_out - cost_basis, 2),
                "mcap": round(position.candidate.mcap_usd),
                "reason": reason.value,
                "why": describe_exit(reason),
                "note": note,
                "tokens": tokens,
                **self.enricher.exit_tape(position.candidate),
            }
        )

        log.info(
            "exit",
            extra={
                "symbol": position.candidate.symbol or position.candidate.address,
                "reason": reason.value,
                "note": note,
                "tokens": tokens,
                "usd_out": round(fill.amount_out, 2),
                "realized_usd": round(position.realized_usd, 2),
                "remaining": position.tokens_remaining,
            },
        )

        if position.tokens_remaining <= 1e-12:
            self._close(position, reason, note)
        self._save_positions()

    def _close(self, position: Position, reason: ExitReason, note: str = "") -> None:
        mfe, mae = PositionManager.excursions(position)
        trade = TradeRecord(
            key=position.candidate.key,
            chain=position.candidate.chain,
            venue=position.venue,
            opened_at_ms=position.opened_at_ms,
            closed_at_ms=now_ms(),
            entry_price=position.entry_price,
            exit_price=position.last_exit_price or position.candidate.price_usd,
            signal_price=position.signal_price,
            size_usd=position.size_usd,
            pnl_usd=position.realized_usd,
            fees_usd=position.fees_usd,
            exit_reason=reason,
            error_class=ErrorClass.WIN,
            features=position.entry_features,
            unknown=set(position.entry_unknown),
            weights_version=self.scorer.model.version,
            max_favorable_excursion=mfe,
            max_adverse_excursion=mae,
            entry_slippage=0.0,
            symbol=position.candidate.symbol or "",
            mcap_entry_usd=position.entry_mcap_usd or position.candidate.mcap_usd,
            mcap_exit_usd=position.candidate.mcap_usd,
            exit_legs=list(position.exit_legs),
            notes=(note or "")[:400],
        )
        trade.error_class = learning.classify(trade, self.strategy)
        self.store.record_trade(trade)
        deployer = position.candidate.deployer
        for wallet in position.entry_buyers:
            if wallet and wallet != deployer:
                self.store.record_buyer_outcome(wallet, position.candidate.chain, trade.pnl_usd)
        self.enricher._smart_cache.pop(position.candidate.chain, None)
        self.positions.pop(position.candidate.key, None)
        self.enricher.forget(position.candidate.key)
        self.risk.note_trade_closed(trade.won)
        self._closed_since_learn += 1
        log.info(
            "closed",
            extra={
                "symbol": position.candidate.symbol or position.candidate.address,
                "pnl_usd": round(trade.pnl_usd, 2),
                "pnl_pct": round(trade.pnl_pct, 4),
                "error_class": trade.error_class.value,
                "exit_reason": reason.value,
                "mfe": round(mfe, 3),
                "mcap_entry": round(trade.mcap_entry_usd),
                "mcap_exit": round(trade.mcap_exit_usd),
            },
        )
        self._log_trade_file(trade)

    def _log_trade_file(self, trade: TradeRecord) -> None:
        # ponytail: one JSON line per close; sqlite is source of truth, this is grep.
        path = self.settings.state_dir / "trades.jsonl"
        rec = {
            "ts": trade.closed_at_ms,
            "symbol": trade.symbol or trade.key,
            "chain": trade.chain.value,
            "key": trade.key,
            "size_usd": round(trade.size_usd, 2),
            "pnl_usd": round(trade.pnl_usd, 2),
            "pnl_pct": round(trade.pnl_pct, 4),
            "mcap_entry": round(trade.mcap_entry_usd),
            "mcap_exit": round(trade.mcap_exit_usd),
            "exit": trade.exit_reason.value,
            "note": trade.notes,
            "venue": trade.venue.value,
        }
        last = (trade.exit_legs or [{}])[-1] if trade.exit_legs else {}
        if "vol5m" in last:
            rec["vol5m"] = last["vol5m"]
        if "holder_growth_5m" in last:
            rec["holder_growth_5m"] = last["holder_growth_5m"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    # -- shadow tracking ---------------------------------------------------
    async def update_shadows(self) -> None:
        """Track rejected candidates so the cost of our own filters is a number
        rather than an opinion."""
        rows = self.store.open_shadows()
        if not rows:
            return
        horizon_ms = int(float(self.strategy.get("learning.shadow_track_minutes", 60)) * 60_000)
        stale = [r for r in rows if now_ms() - r["opened_at_ms"] >= horizon_ms]
        stale_ids = {r["decision_id"] for r in stale}
        live_rows = [r for r in rows if r["decision_id"] not in stale_ids][:30]

        for row in stale:
            entry = float(row["price_at_decision"]) or 0.0
            best = float(row["best_price"] or entry)
            self.store.resolve_shadow(
                row["decision_id"], (best / entry - 1.0) if entry > 0 else 0.0
            )

        if not live_rows:
            return
        addresses = []
        by_address: dict[str, list[int]] = {}
        for row in live_rows:
            _, _, address = row["key"].partition(":")
            if not address:
                continue
            by_address.setdefault(address, []).append(row["decision_id"])
            addresses.append(address)
        for start in range(0, len(addresses), 30):
            snaps = await self.dex.token_pairs(addresses[start : start + 30])
            for snap in snaps:
                for decision_id in by_address.get(snap.token_address, []):
                    self.store.update_shadow(decision_id, snap.price_usd)

    def _drain_inspect(self) -> None:
        path = self.settings.state_dir / "inspect.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            data = []
        path.unlink(missing_ok=True)
        addrs = data if isinstance(data, list) else [data]
        for raw in addrs:
            addr = str(raw).strip()
            if addr:
                self.discovery.watch(addr)

    def _drain_sells(self) -> None:
        # ponytail: same file queue as inspect.json; preview and the engine are
        # separate processes. Duplicate keys no-op if the bag already closed.
        path = self.settings.state_dir / "sell.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            data = []
        path.unlink(missing_ok=True)
        items = data if isinstance(data, list) else [data]
        for raw in items:
            key = str(raw).strip()
            if key:
                self._manual_sell.add(key)

    # -- scoring and entry -------------------------------------------------
    async def score_and_enter(self) -> None:
        if not self.watching:
            return

        now = now_ms()

        def _attention(c: Candidate) -> tuple:
            if c.pack_role == "vamp":
                return (3, 0.0, 0.0)
            unseen = 0 if c.last_scored_ms == 0 else 1
            stale = -(now - c.last_scored_ms)
            vol = -c.volume_5m_usd
            return (unseen, 0 if c.dex_paid else 1, stale, vol)

        # Every visor card gets a note. Halt only blocks the fill, not the grade.
        ttl_ms = int(float(self.strategy.get("loop.rescore_seconds", 15)) * 1000)
        scan_floor = float(self.strategy.get("loop.dead_mcap_usd", 50_000))
        ranked = [
            c
            for c in sorted(self.watching.values(), key=_attention)
            if c.key not in self.positions
            and c.key not in self._inflight
            and self.router.has_venue(c.chain)
            and (c.source == "inspect" or enrich_due(c.last_scored_ms, now, ttl_ms))
            and on_scan_visor(c, scan_floor)
        ]

        probe_size = max(
            float(self.strategy.get("risk.min_position_usd", 15.0)),
            self.risk.equity() * float(self.strategy.get("risk.max_position_pct", 0.05)),
        )

        n = max(1, int(self.strategy.get("loop.enrich_concurrency", 4)))
        sem = asyncio.Semaphore(n)

        async def guarded(candidate: Candidate):
            async with sem:
                return await self._score_one(candidate, probe_size)

        scored = await asyncio.gather(*(guarded(c) for c in ranked), return_exceptions=True)
        halted, _reason = self.risk.halted()
        picks: list[tuple[Decision, list[str], list[str]]] = []
        for item in scored:
            if isinstance(item, Exception):
                log.debug("score failed", extra={"error": str(item)})
                continue
            if item:
                picks.append(item)
        if halted or not picks:
            return
        picks.sort(
            key=lambda x: (x[0].score.expected_value, x[0].score.probability),
            reverse=True,
        )
        decision, buyers, sponsors = picks[0]
        decision_id = self.store.record_decision(decision)
        await self._enter(decision, decision_id, buyers=buyers, sponsors=sponsors)

    def _buy_floors(self) -> dict[str, float]:
        min_p, min_ev = score_floors(self.strategy, self.store)
        return {
            "min_p": round(min_p, 4),
            "min_ev": round(min_ev, 4),
            "min_rubric": round(float(self.strategy.get("scoring.min_rubric", 7.0)), 1),
        }

    def _score_read(
        self,
        candidate: Candidate,
        score: Score,
        call: str,
        why: str,
        enrichment: Enrichment | None,
    ) -> dict:
        mint = None if enrichment is None else enrichment.mint
        crowd = {} if enrichment is None else (enrichment.crowd or {})
        tw = {} if enrichment is None else (enrichment.twitter or {})
        holders = None
        rt = None
        if enrichment is not None:
            if "holder_count" not in enrichment.unknown:
                holders = int(enrichment.features.holder_count or 0)
            rt = round(float(enrichment.features.round_trip_cost or 0.0), 4)
        crowd_ui = {
            k: crowd[k]
            for k in ("kols", "fomo", "whale_n", "whale_pct", "whale_usd")
            if k in crowd
        }
        explain = why if score.veto_reasons else (
            self.scorer.explain(score) if score.probability else why
        )
        return {
            **(score.dist or {}),
            **crowd_ui,
            "call": call,
            "why": why,
            "p": round(score.probability, 4),
            "ev": round(score.expected_value, 4),
            **self._buy_floors(),
            "holders": holders,
            "rt": rt,
            "vetoes": list(score.veto_reasons),
            "explain": explain,
            "cert": security_cert(
                (score.dist or {}).get("label") or "",
                score.veto_reasons,
                mint.authorities_revoked if mint is not None else None,
            ),
            "tw": tw,
            "rubric": score.rubric or {},
        }

    async def _score_one(
        self, candidate: Candidate, probe_size: float
    ) -> tuple[Decision, list[str], list[str]] | None:
        self._inflight.add(candidate.key)
        painted = False
        try:
            if candidate.source != "inspect" and below_scan_mcap(
                candidate.mcap_usd, float(self.strategy.get("loop.dead_mcap_usd", 50_000))
            ):
                if drop_for_scan_mcap(
                    candidate, float(self.strategy.get("loop.dead_mcap_usd", 50_000))
                ):
                    self._drop_watch(candidate, "dead_mcap")
                return None
            if candidate.pack_role == "vamp":
                self._drop_watch(candidate, "vamp")
                return None

            cheap = self.enricher.free_enrichment(candidate)
            free_vetoes = [
                v
                for v in self.scorer.prefilter(cheap)
                if not v.startswith("age:") and not v.startswith("priced:")
            ]
            if observe_early(candidate.source):
                seen = {v.split(":", 1)[0] for v in free_vetoes}
                for veto in early_observe_floors(candidate, self.strategy):
                    kind = veto.split(":", 1)[0]
                    if kind not in seen:
                        free_vetoes.append(veto)
                        seen.add(kind)
            if free_vetoes and candidate.source != "inspect":
                if wait_on_visor(free_vetoes):
                    try:
                        rug = await self.enricher.rug_probe(candidate)
                    except Exception:  # noqa: BLE001
                        rug = cheap
                    extra = [
                        v
                        for v in self.scorer.prefilter(rug)
                        if v.startswith("cluster:") or v.startswith("bundle:")
                    ]
                    merged = list(free_vetoes)
                    seen = {v.split(":", 1)[0] for v in merged}
                    for veto in extra:
                        kind = veto.split(":", 1)[0]
                        if kind not in seen:
                            merged.append(veto)
                            seen.add(kind)
                    free_vetoes = merged
                    cheap = rug
                self._tick_counts[f"free_veto:{free_vetoes[0].split(':')[0]}"] += 1
                cheap_score = Score(
                    probability=0.0,
                    expected_value=0.0,
                    veto_reasons=free_vetoes,
                )
                if not candidate.last_scored_ms:
                    self._record(
                        candidate,
                        cheap.features,
                        cheap_score,
                        Action.REJECT_GATE,
                        0.0,
                        free_vetoes[0],
                        cheap.unknown,
                    )
                if hide_from_visor(free_vetoes):
                    self._ban_skip(candidate, free_vetoes[0].split(":")[0])
                    return None
                call = "wait"
                self._reads[candidate.key] = self._score_read(
                    candidate, cheap_score, call, free_vetoes[0], cheap
                )
                painted = True
                return None

            try:
                enrichment = await self.enricher.enrich(candidate, probe_size)
            except Exception as exc:  # noqa: BLE001
                self._tick_counts["enrich_failed"] += 1
                log.debug("enrich failed", extra={"key": candidate.key, "error": str(exc)})
                self._paint_watch(candidate, call="scan", why=f"enrich: {exc}"[:80])
                return None

            self._tick_counts["enriched"] += 1
            score = self.scorer.score(enrichment)
            ok, why = self.scorer.passes(score)
            call = watch_call(vetoed=score.vetoed, ok=ok, reasons=score.veto_reasons)
            self._reads[candidate.key] = self._score_read(
                candidate, score, call, why, enrichment
            )
            painted = True
            if not ok:
                action = Action.REJECT_GATE if score.vetoed else Action.REJECT_SCORE
                self._tick_counts[
                    f"veto:{score.veto_reasons[0].split(':')[0]}" if score.vetoed else "low_score"
                ] += 1
                self._record(
                    candidate, enrichment.features, score, action, 0.0, why, enrichment.unknown
                )
                if hide_from_visor(score.veto_reasons) or call == "skip":
                    self._ban_skip(
                        candidate,
                        (score.veto_reasons[0].split(":")[0] if score.veto_reasons else "skip"),
                    )
                    return None
                return None

            sizing = self.risk.size(candidate, score, self.scorer.payoff, list(self.positions.values()))
            if not sizing.allowed:
                self._record(
                    candidate,
                    enrichment.features,
                    score,
                    Action.REJECT_RISK,
                    0.0,
                    sizing.reason,
                    enrichment.unknown,
                )
                return None

            decision = Decision(
                candidate=candidate,
                features=enrichment.features,
                score=score,
                action=Action.ENTER,
                size_usd=sizing.size_usd,
                reason=sizing.reason,
                weights_version=self.scorer.model.version,
                unknown=enrichment.unknown,
            )
            return decision, enrichment.buyers, _crowd_sponsors(enrichment.crowd)
        finally:
            if painted:
                candidate.last_scored_ms = now_ms()
            self._inflight.discard(candidate.key)

    def _paint_watch(
        self,
        candidate: Candidate,
        *,
        call: str,
        why: str,
        rubric: dict | None = None,
    ) -> None:
        rec = dict(self._reads.get(candidate.key) or {})
        rec["call"] = call
        rec["why"] = why
        if rubric is not None:
            rec["rubric"] = rubric
        self._reads[candidate.key] = rec

    def _record(
        self,
        candidate: Candidate,
        features: Features,
        score,
        action: Action,
        size_usd: float,
        reason: str,
        unknown: set[str] | None = None,
    ) -> None:
        self.store.record_decision(
            Decision(
                candidate=candidate,
                features=features,
                score=score,
                action=action,
                size_usd=size_usd,
                reason=reason,
                weights_version=self.scorer.model.version,
                unknown=unknown or set(),
            )
        )

    async def _enter(
        self,
        decision: Decision,
        decision_id: int,
        buyers: list[str] | None = None,
        sponsors: list[str] | None = None,
    ) -> None:
        candidate = decision.candidate
        signal_price = candidate.price_usd
        try:
            fill = await self.router.buy(candidate, decision.size_usd)
        except ExecutionError as exc:
            # A failed submit still costs gas and, more importantly, is evidence
            # about execution quality. Recording it as a zero-size loss keeps it
            # in the error taxonomy instead of vanishing into the log.
            log.warning("entry failed", extra={"key": candidate.key, "error": str(exc)})
            self.store.record_trade(
                TradeRecord(
                    key=candidate.key,
                    chain=candidate.chain,
                    venue=VenueId.PAPER,
                    opened_at_ms=now_ms(),
                    closed_at_ms=now_ms(),
                    entry_price=signal_price,
                    exit_price=signal_price,
                    signal_price=signal_price,
                    size_usd=decision.size_usd,
                    pnl_usd=0.0,
                    fees_usd=0.0,
                    exit_reason=ExitReason.MANUAL,
                    error_class=ErrorClass.EXECUTION_FAIL,
                    features=decision.features,
                    unknown=set(decision.unknown),
                    weights_version=decision.weights_version,
                    notes=str(exc)[:400],
                )
            )
            return

        if fill.amount_out <= 0:
            log.error("entry filled zero tokens", extra={"key": candidate.key})
            return

        position = Position(
            candidate=candidate,
            venue=fill.venue,
            entry_price=fill.price,
            size_usd=decision.size_usd,
            tokens=fill.amount_out,
            tokens_remaining=fill.amount_out,
            fees_usd=fill.fee_usd,
            decision_id=decision_id,
            entry_features=decision.features,
            entry_unknown=sorted(decision.unknown),
            signal_price=signal_price,
            entry_buyers=list(buyers or []),
            entry_sponsors=list(sponsors or []),
            entry_rubric=float((decision.score.rubric or {}).get("total") or 0.0),
            entry_mcap_usd=candidate.mcap_usd,
            entry_vol_score=float(decision.features.volatility_volume_score or 0.0),
        )
        self.positions[candidate.key] = position
        self._save_positions()
        log.info(
            "entered",
            extra={
                "symbol": candidate.symbol or candidate.address,
                "chain": candidate.chain.value,
                "venue": fill.venue.value,
                "size_usd": decision.size_usd,
                "entry_price": fill.price,
                "drift_from_signal": round(fill.price / signal_price - 1.0, 4)
                if signal_price
                else 0.0,
                "score": self.scorer.explain(decision.score),
                "sizing": decision.reason,
                "tx": fill.tx_id,
                "mcap_usd": round(candidate.mcap_usd),
            },
        )

    # -- learning ----------------------------------------------------------
    async def maybe_learn(self) -> None:
        if not self.strategy.get("learning.enabled", True):
            return
        cadence = int(self.strategy.get("learning.retrain_every_closed_trades", 10))
        if self._closed_since_learn < cadence:
            return
        self._closed_since_learn = 0

        rolled_back = learning.check_rollback(self.store, self.strategy)
        report = learning.run_postmortem(self.store, self.strategy)
        relaxed = learning.relax_costly_gates(self.store, self.strategy)
        result = learning.train(self.store, self.strategy)

        if result.promoted or rolled_back:
            self.scorer.model = Model.load(self.store)
        log.info(
            "learning cycle",
            extra={
                "postmortem": report.counts,
                "applied": report.applied,
                "relaxed": relaxed,
                "training": result.note,
                "rollback": rolled_back,
                "active_weights": self.scorer.model.version,
            },
        )

    # -- housekeeping ------------------------------------------------------
    async def _refresh_watch_quotes(self, *, probe_paid: bool = True) -> None:
        """Re-price everything on the visor. Discovery only re-emits a mint
        about once a minute, so without this the mcap/vol/5m freeze at first sight.
        """
        addrs = list({c.address for c in self.watching.values() if c.address})
        by: dict[str, Any] = {}
        for i in range(0, len(addrs), 30):
            try:
                snaps = await self.dex.token_pairs(addrs[i : i + 30])
            except Exception:  # noqa: BLE001
                continue
            by.update({s.token_address.lower(): s for s in snaps})
        unpaid = []
        unlabeled = []
        for candidate in self.watching.values():
            was_mcap = candidate.mcap_usd
            snap = by.get(candidate.address.lower())
            was_paid = candidate.dex_paid
            if snap is not None:
                candidate.price_usd = snap.price_usd or candidate.price_usd
                candidate.mcap_usd = snap.mcap_usd or candidate.mcap_usd
                candidate.volume_5m_usd = snap.volume_m5
                candidate.liquidity_usd = snap.liquidity_usd
                candidate.ret_5m = snap.price_change_m5
                snap.stamp(candidate)
                if observe_early(candidate.source) and was_mcap <= 0 and candidate.mcap_usd > 0:
                    log.info(
                        "hood indexed",
                        extra={
                            "symbol": candidate.symbol or candidate.address[:10],
                            "wait_s": round((now_ms() - candidate.discovered_at_ms) / 1000.0, 1),
                            "mcap": round(candidate.mcap_usd),
                        },
                    )
                if snap.twitter:
                    rec = self._reads.setdefault(candidate.key, {"call": "scan"})
                    tw = dict(rec.get("tw") or {})
                    if not twitter_handle(str(tw.get("official") or "")):
                        tw["official"] = snap.twitter
                        rec["tw"] = tw
            if not candidate.dex_paid:
                unpaid.append(candidate)
            elif not was_paid:
                candidate.last_scored_ms = 0
                read = self._reads.get(candidate.key)
                if read is not None:
                    read["call"] = "scan"
            if not candidate.symbol and not candidate.name and candidate.key not in self._label_fail:
                unlabeled.append(candidate)
        if probe_paid:
            for candidate in unpaid[:8]:
                try:
                    paid = await self.dex.token_is_paid(candidate.chain, candidate.address)
                except Exception:  # noqa: BLE001
                    continue
                if not paid:
                    continue
                candidate.dex_paid = True
                candidate.last_scored_ms = 0
                read = self._reads.get(candidate.key)
                if read is not None:
                    read["call"] = "scan"
        for candidate in unlabeled[:8]:
            rpc = self.enricher.evm_rpcs.get(candidate.chain)
            if rpc is None:
                continue
            try:
                symbol, name = await rpc.erc20_labels(candidate.address)
            except Exception:  # noqa: BLE001
                continue
            candidate.symbol = candidate.symbol or symbol
            candidate.name = candidate.name or name
            if not candidate.symbol and not candidate.name:
                self._label_fail.add(candidate.key)

    def _retag(self) -> dict:
        by_key = {c.key: c for c in self.watching.values()}
        extras = [
            p.candidate for p in self.positions.values() if p.candidate.key not in by_key
        ]
        tags = apply_tags(list(by_key.values()) + extras)
        for position in self.positions.values():
            tag = tags.get(position.candidate.key)
            if tag is None:
                continue
            position.candidate.pack_role = tag.role
            position.candidate.pack_stem = tag.stem
            position.candidate.main_key = tag.main_key
            position.candidate.main_ret_5m = tag.main_ret_5m
            position.candidate.pack_size = tag.pack_size
        return tags

    def _drop_below_scan_mcap(self) -> None:
        floor = float(self.strategy.get("loop.dead_mcap_usd", 50_000))
        for candidate in list(self.watching.values()):
            if candidate.key in self.positions or candidate.source == "inspect":
                continue
            if drop_for_scan_mcap(candidate, floor):
                self._drop_watch(candidate, "dead_mcap")

    def _drop_unpaid(self) -> None:
        for candidate in list(self.watching.values()):
            if candidate.key in self.positions or candidate.source == "inspect":
                continue
            if unpaid_for_scan(candidate):
                self._drop_watch(candidate, "unpaid")

    def _drop_watch(self, candidate: Candidate, tag: str) -> None:
        if candidate.key in self.positions:
            return
        self.watching.pop(candidate.key, None)
        self._reads.pop(candidate.key, None)
        self.enricher.forget(candidate.key)
        self._tick_counts[tag] += 1

    def _ban_skip(self, candidate: Candidate, tag: str) -> None:
        if candidate.source != "inspect":
            self._skip_ban[candidate.key] = now_ms()
        self._drop_watch(candidate, tag)

    def _prune_watching(self) -> None:
        tags = self._retag()
        dying = dump_beta_keys(tags)
        cap = int(self.strategy.get("loop.max_watching", 24))
        ignore_mcap = float(self.strategy.get("whales.ignore_mcap_usd", 50_000_000))
        now = now_ms()
        for key, candidate in list(self.watching.items()):
            if key in self.positions:
                continue
            if candidate.source == "inspect":
                if candidate.last_scored_ms and now - candidate.last_scored_ms > 180_000:
                    del self.watching[key]
                    self.enricher.forget(key)
                continue
            if ignore_mcap > 0 and candidate.mcap_usd > ignore_mcap:
                self._drop_watch(candidate, "fat_mcap")
                continue
            if drop_for_scan_mcap(
                candidate, float(self.strategy.get("loop.dead_mcap_usd", 50_000))
            ):
                self._drop_watch(candidate, "dead_mcap")
                continue
            if unpaid_for_scan(candidate):
                self._drop_watch(candidate, "unpaid")
                continue
            if (self._reads.get(key) or {}).get("call") == "skip":
                self._ban_skip(candidate, "skip")
                continue
            if candidate.pack_role == "vamp":
                del self.watching[key]
                self.enricher.forget(key)
                self._tick_counts["vamp"] += 1
                continue
            if key in dying:
                del self.watching[key]
                self.enricher.forget(key)
                self._tick_counts["beta_dump"] += 1
                continue
            visor_age = float(self.strategy.get("loop.max_candidate_age_minutes", 180))
            if candidate.created_at_ms and candidate.age_minutes > visor_age:
                del self.watching[key]
                self.enricher.forget(key)

        overflow = [c for c in self.watching.values() if c.key not in self.positions]
        if len(self.watching) > cap:
            overflow.sort(
                key=lambda c: (
                    0 if c.source == "inspect" or observe_early(c.source) else 1,
                    0 if c.dex_paid else 1,
                    0 if c.pack_role == "main" else 1 if c.pack_role == "beta" else 2,
                    3 if (self._reads.get(c.key) or {}).get("call") == "skip" else 0,
                    c.age_minutes,
                    -c.volume_5m_usd,
                )
            )
            slots = max(0, cap - len(self.positions))
            for key in {c.key for c in overflow[slots:]}:
                del self.watching[key]
                self.enricher.forget(key)
        self._reads = {k: v for k, v in self._reads.items() if k in self.watching}

    def _save_positions(self) -> None:
        payload = []
        for position in self.positions.values():
            data = asdict(position)
            data["candidate"]["chain"] = position.candidate.chain.value
            data["venue"] = position.venue.value
            data["entry_features"] = position.entry_features.as_dict()
            payload.append(data)
        tmp = self._positions_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._positions_path)

    def _load_positions(self) -> None:
        if not self._positions_path.exists():
            return
        try:
            payload = json.loads(self._positions_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            log.error("could not read persisted positions", extra={"error": str(exc)})
            return
        for data in payload:
            try:
                candidate_data = dict(data.pop("candidate"))
                candidate_data["chain"] = Chain(candidate_data["chain"])
                candidate = Candidate(
                    **{
                        k: v
                        for k, v in candidate_data.items()
                        if k in Candidate.__dataclass_fields__
                    }
                )
                features = features_from_json(json.dumps(data.pop("entry_features", {})))
                position = Position(
                    candidate=candidate,
                    venue=VenueId(data.pop("venue")),
                    entry_features=features,
                    **{
                        k: v
                        for k, v in data.items()
                        if k in Position.__dataclass_fields__
                        and k not in {"candidate", "venue", "entry_features"}
                    },
                )
                self.positions[candidate.key] = position
                self.watching[candidate.key] = candidate
                # Drain vs a peak from a previous process is a false rug: the
                # bot was off, liq moved, first tick looks like LP pulling.
                position.peak_liquidity_usd = 0.0
            except (KeyError, TypeError, ValueError) as exc:
                log.error("skipping unreadable position", extra={"error": str(exc)})
        if self.positions:
            log.info("restored positions", extra={"count": len(self.positions)})


async def run_forever(settings: Settings) -> None:
    engine = Engine(settings)
    await engine.run()
