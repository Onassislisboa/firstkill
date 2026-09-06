"""FirstKill CLI (`python -m alphahound`).

`run` is the bot. Everything else exists to answer a question you will have
at 2am: what did it decide, why, what did that cost, and what has it changed
about itself since.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys

from . import __version__, backtest, learning, log
from .discovery import Discovery
from .engine import Engine, build_registry
from .models import Chain
from .net import Http
from .playbook import gate as pb_gate
from .preview import read_preview
from .providers import Dexscreener
from .risk import RiskEngine
from .scoring import Model
from .settings import (
    Settings,
    aggressive_closes_since_on,
    apply_aggressive_learning,
    chase_rows,
    load_strategy,
    load_terminals,
    score_floors,
)
from .signals.solana import SolanaReader
from .signals.terminals import discover_fee_accounts
from .store import Store


def _bootstrap() -> tuple[Settings, object]:
    settings = Settings.from_env()
    log.setup(settings.log_level)
    return settings, load_strategy()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    settings, strategy = _bootstrap()
    if args.paper:
        settings.mode = "paper"
    engine = Engine(settings, strategy)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, engine.request_stop)
        except (NotImplementedError, ValueError):
            # Windows does not support add_signal_handler for SIGTERM; KeyboardInterrupt
            # below covers the interactive case.
            pass
    try:
        loop.run_until_complete(engine.run())
    except KeyboardInterrupt:
        engine.request_stop()
        loop.run_until_complete(engine.shutdown())
    finally:
        loop.close()
    return 0


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    settings, strategy = _bootstrap()
    terminals = load_terminals()
    store = Store(settings.state_dir)
    strategy = apply_aggressive_learning(strategy, store)
    registry = build_registry(store, terminals)

    print(f"FirstKill {__version__}")
    print(f"  mode              {settings.mode}")
    print(f"  chains            {', '.join(c.value for c in settings.enabled_chains)}")
    print(f"  state             {store.path}")
    print(f"  closed trades     {store.trade_count()}")
    print(f"  realized pnl      {store.realized_pnl():+.2f} USD")

    model = Model.load(store)
    print(f"  weights version   {model.version} ({'learned' if model.version else 'priors'})")
    print(
        f"  open slots        {int(strategy.get('risk.max_concurrent_positions', 1))}"
        f" concurrent / {int(strategy.get('risk.max_positions_per_chain', 1))} per chain"
    )
    min_p, min_ev = score_floors(strategy, store)
    equity = RiskEngine(strategy, store).equity()
    min_pct = float(strategy.get("risk.min_position_pct", 0))
    max_pct = float(strategy.get("risk.max_position_pct", 0))
    print(f"  score floors      p>={min_p:.3f}  ev>={min_ev:.3f}")
    print(
        f"  size band         {min_pct:.1%}–{max_pct:.1%} equity"
        f" (~${equity * min_pct:.0f}–${equity * max_pct:.0f})"
    )
    ag = strategy.section("aggressive_learning")
    if strategy.get("aggressive_learning._active"):
        need = int(ag.get("graduate_at_closes", 40))
        since = aggressive_closes_since_on(store)
        print(
            f"  learning mode     aggressive"
            f" ({store.trade_count()}/{need} closes, {since} since on)"
        )
        tuned_ev = store.param(
            "scoring.min_expected_value",
            float(strategy.get("scoring.min_expected_value", 0)),
        )
        if tuned_ev > min_ev + 1e-9:
            print(
                f"     self-tune wants ev {tuned_ev:.3f} vs aggressive {min_ev:.3f}"
                " — fighting (aggressive wins until graduate)"
            )
    elif ag.get("enabled"):
        print(
            f"  learning mode     production"
            f" (aggressive graduated at {int(ag.get('graduate_at_closes', 40))} closes)"
        )
    print(
        f"  shadow track      {int(float(strategy.get('learning.shadow_track_minutes', 60)))} min"
    )
    chain0 = settings.enabled_chains[0] if settings.enabled_chains else None
    if chain0 is not None:
        print(
            f"  bundle/cluster    max_bundle {pb_gate(strategy, chain0, 'max_bundle_pct', 0.20, store):.0%}"
            f"  max_cluster {pb_gate(strategy, chain0, 'max_cluster_pct', 0.20, store):.0%}"
            " (hard veto)"
        )

    engaged = store.get_kv("kill_switch") == "1"
    print(f"  kill switch       {'ENGAGED - ' + store.get_kv('kill_switch_reason') if engaged else 'clear'}")
    live = read_preview(settings.state_dir)
    dead = live.get("dead_loops") or []
    if dead:
        print(f"  loop faults       DEAD {', '.join(dead)} {live.get('faults')}")

    labels = registry.attributable_labels
    print(
        f"  terminal labels   {len(labels)} attributable"
        + (": " + ", ".join(sorted(labels)) if labels else "")
    )
    if not labels:
        print(
            "     -> retail/bot attribution is inert, so retail_share and axiom_share\n"
            "        contribute nothing. Run: alphahound discover-terminals"
        )

    rows = chase_rows(settings.state_dir)
    fomo_n = sum(
        1
        for r in rows
        if str(r.get("source", "")).lower() == "fomo" or str(r.get("class", "")).lower() == "fomo"
    )
    whale_n = len(rows) - fomo_n
    print(f"  labeled wallets   {whale_n} whale/moby, {fomo_n} fomo (whales.toml + visor)")
    print(
        "  fomo research     "
        + ("Cope key set" if settings.cope_api_key else "no COPE_API_KEY; labeled fomo only")
    )

    overrides = store.all_params()
    if overrides:
        print("  self-tuned params")
        for name, value in sorted(overrides.items()):
            base = strategy.get(name, "?")
            print(f"     {name}: {value:.4g} (config {base})")

    problems = settings.validate()
    if problems:
        print("\nblocking problems:")
        for problem in problems:
            print(f"  ! {problem}")
    else:
        print("\nconfiguration looks usable" + (" (live)" if settings.live else " (paper)"))
    print(
        "  account floor     none — daily loss halt "
        f"{float(strategy.get('risk.daily_loss_limit_pct', 0)):.0%} of base equity;"
        " no $15–20 equity stop"
    )

    # Connectivity, only when asked, because it costs requests.
    if args.check_network:
        asyncio.run(_check_network(settings))
    store.close()
    return 1 if problems else 0


async def _check_network(settings: Settings) -> None:
    http = Http()
    dex = Dexscreener(http)
    print("\nnetwork:")
    try:
        snaps = await dex.token_pairs(["So11111111111111111111111111111111111111112"])
        price = next((s.price_usd for s in snaps), 0.0)
        print(f"  dexscreener       ok (SOL ${price:,.2f})")
    except Exception as exc:  # noqa: BLE001
        print(f"  dexscreener       FAILED: {exc}")

    if settings.solana_rpc_url:
        try:
            slot = await SolanaReader(http, settings.solana_rpc_url).get_slot()
            print(f"  solana rpc        ok (slot {slot})")
        except Exception as exc:  # noqa: BLE001
            print(f"  solana rpc        FAILED: {exc}")
    await http.aclose()


# ---------------------------------------------------------------------------
# learning / inspection
# ---------------------------------------------------------------------------


def cmd_learn(args: argparse.Namespace) -> int:
    settings, strategy = _bootstrap()
    store = Store(settings.state_dir)

    rolled_back = learning.check_rollback(store, strategy)
    if rolled_back:
        print(f"rollback: {rolled_back}")

    report = learning.run_postmortem(store, strategy)
    print(f"postmortem over {report.trades} closed trades")
    for klass, count in sorted(report.counts.items(), key=lambda kv: -kv[1]):
        pnl = report.pnl_by_class.get(klass, 0.0)
        print(f"  {klass:<20} {count:>4}  {pnl:>+9.2f} USD")
    for applied in report.applied:
        print(f"  applied: {applied}")
    for skipped in report.skipped:
        print(f"  skipped: {skipped}")

    relaxed = learning.relax_costly_gates(store, strategy)
    for note in relaxed:
        print(f"  relaxed: {note}")

    if not args.no_train:
        result = learning.train(store, strategy)
        print(f"\ntraining: {result.note}")
        if result.frozen_features:
            print(
                f"  {len(result.frozen_features)} features frozen for lack of observations: "
                + ", ".join(result.frozen_features[:8])
                + ("..." if len(result.frozen_features) > 8 else "")
            )
    store.close()
    return 0


def cmd_weights(args: argparse.Namespace) -> int:
    settings, _ = _bootstrap()
    store = Store(settings.state_dir)
    rows = learning.feature_report(store)
    print(f"{'feature':<26} {'active':>8} {'prior':>8} {'delta':>8} {'obs':>6}")
    for name, active, prior, observations in rows:
        print(f"{name:<26} {active:>8.3f} {prior:>8.3f} {active - prior:>+8.3f} {observations:>6}")
    if args.export:
        from pathlib import Path

        print(learning.export_weights(store, Path(args.export)))
    store.close()
    return 0


def cmd_trades(args: argparse.Namespace) -> int:
    settings, _ = _bootstrap()
    store = Store(settings.state_dir)
    trades = store.trades(limit=args.limit)
    if not trades:
        print("no closed trades yet")
        store.close()
        return 0
    print(
        f"{'symbol/key':<26} {'venue':<10} {'size':>8} {'pnl':>9} {'pnl%':>8} "
        f"{'mcap in':>8} {'mcap out':>8} {'exit':<16} {'class':<18}"
    )
    for trade in trades:
        _, _, address = trade.key.partition(":")
        label = (trade.symbol or address)[:24]
        print(
            f"{label:<26} {trade.venue.value:<10} {trade.size_usd:>8.2f} "
            f"{trade.pnl_usd:>+9.2f} {trade.pnl_pct:>+8.1%} "
            f"{trade.mcap_entry_usd:>8.0f} {trade.mcap_exit_usd:>8.0f} "
            f"{trade.exit_reason.value:<16} {trade.error_class.value:<18}"
        )
    total = sum(t.pnl_usd for t in trades)
    wins = sum(1 for t in trades if t.won)
    print(f"\n{len(trades)} trades, {wins} wins ({wins / len(trades):.0%}), {total:+.2f} USD")
    store.close()
    return 0


def cmd_filter_cost(args: argparse.Namespace) -> int:
    settings, _ = _bootstrap()
    store = Store(settings.state_dir)
    costs = learning.filter_cost(store)
    if not costs:
        print("no resolved shadow rejections yet - let it run for an hour")
        store.close()
        return 0
    print("what the gates cost (rejections that subsequently moved)\n")
    print(f"{'gate':<24} {'rejected':>9} {'would win':>10} {'median move':>12}")
    for cost in costs:
        print(
            f"{cost.gate:<24} {cost.rejected:>9} {cost.would_have_won:>10} "
            f"{cost.median_counterfactual:>+12.1%}"
        )
    store.close()
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    settings, strategy = _bootstrap()
    store = Store(settings.state_dir)
    if args.compare:
        results = backtest.compare_prior_vs_learned(store, strategy)
    else:
        results = backtest.sweep(store, strategy)
    if not results:
        print("nothing to replay yet: no closed trades and no resolved shadows")
        store.close()
        return 0
    for result in results:
        print(result.line())
    if args.json:
        print(backtest.dump(results))
    for note in results[0].notes:
        print(f"\nnote: {note}")
    store.close()
    return 0


# ---------------------------------------------------------------------------
# terminal attribution
# ---------------------------------------------------------------------------


def cmd_discover_terminals(args: argparse.Namespace) -> int:
    settings, strategy = _bootstrap()
    if not settings.solana_rpc_url:
        print("SOLANA_RPC_URL is required to mine transactions")
        return 1
    return asyncio.run(_discover_terminals(settings, strategy, args))


async def _discover_terminals(settings: Settings, strategy, args) -> int:
    http = Http()
    dex = Dexscreener(http)
    store = Store(settings.state_dir)
    registry = build_registry(store, load_terminals())
    reader = SolanaReader(http, settings.solana_rpc_url)
    discovery = Discovery(settings, strategy, dex)

    print(f"sampling up to {args.tokens} recent Solana tokens...")
    candidates = [c for c in await discovery.poll() if c.chain is Chain.SOLANA][: args.tokens]
    if not candidates:
        print("no candidates found; try again in a minute")
        await http.aclose()
        store.close()
        return 1

    all_buys = []
    for candidate in candidates:
        source = candidate.pool_address or candidate.address
        try:
            _, buys = await reader.recent_activity(
                source, candidate.address, candidate.price_usd, max_txs=args.txs
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {candidate.symbol or candidate.address[:8]}: {exc}")
            continue
        all_buys.extend(buys)
        print(f"  {candidate.symbol or candidate.address[:8]}: {len(buys)} buys")

    ignore = {candidate.address for candidate in candidates} | {
        candidate.pool_address for candidate in candidates if candidate.pool_address
    }
    found = discover_fee_accounts(all_buys, registry, ignore=ignore, limit=args.limit)

    path = settings.state_dir / "terminal_candidates.json"
    path.write_text(
        json.dumps(
            [
                {
                    "address": c.address,
                    "distinct_buyers": c.distinct_buyers,
                    "distinct_txs": c.distinct_txs,
                    "score": round(c.score, 2),
                    "label": "",
                    "class": "",
                }
                for c in found
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n{len(all_buys)} buys inspected. Top recurring unlabeled addresses:\n")
    print(f"{'address':<46} {'buyers':>7} {'txs':>6} {'score':>8}")
    for c in found[:20]:
        print(f"{c.address:<46} {c.distinct_buyers:>7} {c.distinct_txs:>6} {c.score:>8.1f}")
    print(f"\nwritten to {path}")
    print(
        "Identify each one (paste it into a Solana explorer and look at what pays it), "
        "then label it:\n  alphahound label-terminal <address> axiom retail"
    )
    await http.aclose()
    store.close()
    return 0


def cmd_label_terminal(args: argparse.Namespace) -> int:
    settings, _ = _bootstrap()
    if args.klass not in {"retail", "bot", "neutral"}:
        print("class must be one of: retail, bot, neutral")
        return 1
    store = Store(settings.state_dir)
    store.label_terminal(args.address, args.label, args.klass)
    print(f"labeled {args.address} as {args.label} ({args.klass})")
    store.close()
    return 0


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------


def cmd_pause(args: argparse.Namespace) -> int:
    settings, strategy = _bootstrap()
    store = Store(settings.state_dir)
    RiskEngine(strategy, store).engage_kill_switch(args.reason or "paused manually")
    print("kill switch engaged; open positions will be closed on the next tick")
    store.close()
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    settings, strategy = _bootstrap()
    store = Store(settings.state_dir)
    RiskEngine(strategy, store).release_kill_switch()
    store.set_kv("cooldown_until_ms", "0")
    store.set_kv("cooldown_level", "0")
    print("kill switch released, cooldown cleared")
    store.close()
    return 0


def cmd_params(args: argparse.Namespace) -> int:
    settings, strategy = _bootstrap()
    store = Store(settings.state_dir)
    if args.set:
        name, _, value = args.set.partition("=")
        store.set_param(name.strip(), float(value), "set manually")
        print(f"{name.strip()} = {float(value)}")
    overrides = store.all_params()
    if not overrides:
        print("no overrides; everything is coming from config/strategy.toml")
    for name, value in sorted(overrides.items()):
        print(f"{name:<44} {value:>12.4g}  (config {strategy.get(name, '?')})")
    if args.history:
        print("\nchange history:")
        for row in store.param_history(limit=args.history):
            print(f"  {row['name']}: {row['old_value']} -> {row['new_value']}  {row['reason']}")
    store.close()
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    from .preview import serve

    settings, strategy = _bootstrap()
    store = Store(settings.state_dir)
    strategy = apply_aggressive_learning(strategy, store)
    equity = RiskEngine(strategy, store).equity()
    serve(settings.state_dir, store, equity, args.port, settings, strategy)
    return 0


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alphahound", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="start the bot")
    run.add_argument("--paper", action="store_true", help="force paper mode regardless of .env")
    run.set_defaults(func=cmd_run)

    doctor = sub.add_parser("doctor", help="check configuration and state")
    doctor.add_argument("--check-network", action="store_true", help="also probe the providers")
    doctor.set_defaults(func=cmd_doctor)

    learn = sub.add_parser("learn", help="run the postmortem and retrain weights")
    learn.add_argument("--no-train", action="store_true", help="postmortem only")
    learn.set_defaults(func=cmd_learn)

    weights = sub.add_parser("weights", help="show learned vs prior feature weights")
    weights.add_argument("--export", help="also write the active weights to a file")
    weights.set_defaults(func=cmd_weights)

    trades = sub.add_parser("trades", help="list closed trades")
    trades.add_argument("-n", "--limit", type=int, default=30)
    trades.set_defaults(func=cmd_trades)

    sub.add_parser("filter-cost", help="what the gates rejected that would have won").set_defaults(
        func=cmd_filter_cost
    )

    bt = sub.add_parser("backtest", help="replay decisions at different thresholds")
    bt.add_argument("--compare", action="store_true", help="prior weights vs learned weights")
    bt.add_argument("--json", action="store_true")
    bt.set_defaults(func=cmd_backtest)

    disc = sub.add_parser(
        "discover-terminals", help="mine recent swaps for terminal fee accounts to label"
    )
    disc.add_argument("--tokens", type=int, default=8)
    disc.add_argument("--txs", type=int, default=150)
    disc.add_argument("--limit", type=int, default=40)
    disc.set_defaults(func=cmd_discover_terminals)

    label = sub.add_parser("label-terminal", help="label a discovered fee account")
    label.add_argument("address")
    label.add_argument("label")
    label.add_argument("klass", metavar="class", choices=["retail", "bot", "neutral"])
    label.set_defaults(func=cmd_label_terminal)

    pause = sub.add_parser("pause", help="engage the kill switch")
    pause.add_argument("--reason", default="")
    pause.set_defaults(func=cmd_pause)

    sub.add_parser("resume", help="release the kill switch and cooldown").set_defaults(
        func=cmd_resume
    )

    params = sub.add_parser("params", help="inspect or override self-tuned parameters")
    params.add_argument("--set", metavar="NAME=VALUE")
    params.add_argument("--history", type=int, default=0)
    params.set_defaults(func=cmd_params)

    prev = sub.add_parser("preview", help="live PnL / holds / sold in a browser tab")
    prev.add_argument("--port", type=int, default=8765)
    prev.set_defaults(func=cmd_preview)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
