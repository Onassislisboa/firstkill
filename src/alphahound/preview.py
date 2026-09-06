"""Live operator preview.

The engine writes `state/preview.json` every risk tick. `alphahound preview`
serves that file plus closed trades from sqlite on localhost so a browser tab
stays current without another data API.
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .models import Chain, now_ms, describe_error, describe_exit, legs_for_display
from .playbook import max_age_minutes
from .risk import RiskEngine, utc_day_start_ms
from .settings import (
    Config,
    Settings,
    aggressive_closes_since_on,
    load_fomo,
    load_kols,
    save_fomo,
    save_kols,
    score_floors,
)
from .store import Store

PREVIEW_NAME = "preview.json"
SELL_QUEUE = "sell.json"
POSITIONS_FILE = "positions.json"


def match_open_position(rows: list[Any], needle: str) -> dict[str, str] | None:
    needle = (needle or "").strip()
    if not needle:
        return None
    nl = needle.lower()
    for row in rows:
        if not isinstance(row, dict):
            continue
        cand = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
        chain = str(cand.get("chain") or "")
        addr = str(cand.get("address") or "")
        key = f"{chain}:{addr}" if chain and addr else str(row.get("key") or "")
        if needle == key or nl == key.lower() or (addr and nl == addr.lower()):
            return {"key": key, "symbol": str(cand.get("symbol") or "")}
    return None


def enqueue_manual_sell(state_dir: Path, needle: str) -> dict[str, Any]:
    """Queue a single-bag flatten for the engine risk loop. Preview process
    cannot sell itself — the engine holds the router and the position lock."""
    path = state_dir / POSITIONS_FILE
    rows: list[Any] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            loaded = []
        if isinstance(loaded, list):
            rows = loaded
    hit = match_open_position(rows, needle)
    if hit is None:
        return {"ok": False, "error": "posição já fechada"}
    qpath = state_dir / SELL_QUEUE
    pending: list[Any] = []
    if qpath.exists():
        try:
            pending = json.loads(qpath.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pending = []
        if not isinstance(pending, list):
            pending = [pending]
    key = hit["key"]
    if key not in pending:
        pending.append(key)
    qpath.write_text(json.dumps(pending), encoding="utf-8")
    return {"ok": True, "queued": True, "key": key, "symbol": hit["symbol"]}


def write_preview(state_dir: Path, payload: dict[str, Any]) -> None:
    path = state_dir / PREVIEW_NAME
    blob = json.dumps(payload, separators=(",", ":"))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(blob, encoding="utf-8")
    try:
        tmp.replace(path)
    except PermissionError:
        # ponytail: Windows holds preview.json while the HTTP handler reads it.
        path.write_text(blob, encoding="utf-8")
        tmp.unlink(missing_ok=True)


def read_preview(state_dir: Path) -> dict[str, Any]:
    path = state_dir / PREVIEW_NAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def universe(settings: Settings | None, strategy: Config | None) -> dict[str, Any]:
    if settings is None or strategy is None:
        return {"mode": "", "chains": []}
    pubs = settings.wallet_pubkeys()
    rows = []
    for chain in (Chain.SOLANA, Chain.BNB, Chain.ROBINHOOD_CHAIN):
        lp = strategy.section(f"launchpads.{chain.value}")
        pb = strategy.section(f"playbooks.{chain.value}")
        rows.append(
            {
                "chain": chain.value,
                "enabled": chain in settings.enabled_chains,
                "launchpads": list(lp.get("dex_ids") or []),
                "max_age_min": max_age_minutes(strategy, chain),
                "copy_max_age_min": float(pb.get("copy_max_age_minutes", 25)),
                "copy_max_mcap": float(pb.get("copy_max_mcap_usd", 2_000_000)),
                "copy_min_mcap": float(pb.get("copy_min_mcap_usd", 100_000)),
                "min_volume_5m": float(pb.get("min_volume_5m", 0) or 0),
                "min_twitter": float(pb.get("min_twitter_mentions", 0) or 0),
                "wallet": pubs.get(chain.value, ""),
            }
        )
    return {"mode": settings.mode, "chains": rows}


def pnl_curves(trades: list) -> dict[str, Any]:
    chains = ("solana", "bnb", "robinhood_chain")
    acc = {c: 0.0 for c in chains}
    total = 0.0
    series: dict[str, list] = {c: [] for c in chains}
    series["total"] = []
    for t in trades:
        c = t.chain.value
        acc[c] = acc.get(c, 0.0) + t.pnl_usd
        total += t.pnl_usd
        series["total"].append({"t": t.closed_at_ms, "y": round(total, 2)})
        if c in series:
            series[c].append({"t": t.closed_at_ms, "y": round(acc[c], 2)})
    return {
        "total": round(total, 2),
        "by_chain": {c: round(acc[c], 2) for c in chains},
        "series": series,
    }


_DAY_MS = 86_400_000


def pnl_windows(store: Store, now: int | None = None) -> dict[str, Any]:
    now = int(now or now_ms())
    oldest = store.oldest_close_ms()
    hist_days = 0
    if oldest:
        hist_days = max(1, (now - oldest + _DAY_MS - 1) // _DAY_MS)

    def pack(label: str, since_ms: int, span_days: int | None = None) -> dict[str, Any]:
        partial = bool(span_days and hist_days and hist_days < span_days)
        return {
            "label": label,
            "pnl_usd": round(store.realized_pnl(since_ms=since_ms), 2),
            "since_ms": since_ms,
            "partial": partial,
            "history_days": hist_days if partial else (min(hist_days, span_days) if span_days else hist_days),
        }

    return {
        "history_days": hist_days,
        "windows": [
            pack("hoje", utc_day_start_ms(now / 1000.0)),
            pack("7d", now - 7 * _DAY_MS, 7),
            pack("15d", now - 15 * _DAY_MS, 15),
            pack("30d", now - 30 * _DAY_MS, 30),
        ],
    }


_BOOK: tuple[float, dict[str, Any]] | None = None


def _closed_book(store: Store) -> dict[str, Any]:
    """Sqlite barely changes between visor polls. Reuse for ~800ms."""
    global _BOOK
    now = time.monotonic()
    if _BOOK is not None and now - _BOOK[0] < 0.8:
        return _BOOK[1]
    sample = store.trades(limit=500)
    recent = sample[-40:]
    hold_ms = store.avg_hold_ms()
    closed_n = store.trade_count()
    wins = store.win_count()
    book = {
        "sold": [
            {
                "key": t.key,
                "chain": t.chain.value,
                "venue": t.venue.value,
                "size_usd": round(t.size_usd, 2),
                "pnl_usd": round(t.pnl_usd, 2),
                "pnl_pct": round(t.pnl_pct, 4),
                "exit": t.exit_reason.value,
                "exit_why": describe_exit(t.exit_reason),
                "klass": t.error_class.value,
                "klass_why": describe_error(t.error_class),
                "mfe": round(t.max_favorable_excursion, 4),
                "hold_min": int(round((t.closed_at_ms - t.opened_at_ms) / 60_000)),
                "closed_at_ms": t.closed_at_ms,
                "symbol": t.symbol or "",
                "mcap_entry": round(t.mcap_entry_usd),
                "mcap_exit": round(t.mcap_exit_usd),
                "exit_legs": legs_for_display(t),
                "exit_note": t.notes or "",
            }
            for t in reversed(recent)
        ],
        "wins": wins,
        "fees": store.fees_sum(),
        "avg_hold_min": int(round(hold_ms / 60_000)) if hold_ms else None,
        "realized": round(store.realized_pnl(), 2),
        "closed_n": closed_n,
        "pnl_chart": pnl_curves(sample),
        "pnl_windows": pnl_windows(store),
        "sample_n": len(sample),
    }
    _BOOK = (now, book)
    return book


def assemble(
    state_dir: Path,
    store: Store,
    equity: float,
    settings: Settings | None = None,
    strategy: Config | None = None,
) -> dict[str, Any]:
    live = read_preview(state_dir)
    book = _closed_book(store)
    sold = book["sold"]
    wins = book["wins"]
    fees = book["fees"]
    avg_hold_min = book["avg_hold_min"]
    stale_s = 0.0
    if live.get("ts_ms"):
        stale_s = max(0.0, (now_ms() - int(live["ts_ms"])) / 1000.0)
    watch = live.get("watch") or []
    holds = live.get("holds") or []
    unreal = sum(float(h.get("unrealized_usd") or 0) for h in holds)
    realized = book["realized"]
    closed_n = book["closed_n"]
    start_eq = float(strategy.get("risk.equity_usd", equity)) if strategy else equity
    pnl = round(realized + unreal, 2)
    return {
        "ts_ms": now_ms(),
        "bot_ts_ms": live.get("ts_ms"),
        "stale_s": round(stale_s, 1),
        "running": stale_s < 15.0 if live.get("ts_ms") else False,
        "mode": live.get("mode") or (settings.mode if settings else ""),
        "halted": live.get("halted", False),
        "halt_reason": live.get("halt_reason", ""),
        "equity_usd": round(float(live.get("equity_usd", equity)), 2),
        "realized_pnl": realized,
        "unrealized_pnl": round(unreal, 2),
        "pnl": pnl,
        "start_equity": round(start_eq, 2),
        "fees_usd": round(fees, 2),
        "closed": closed_n,
        "wins": wins,
        "win_rate": round(wins / closed_n, 4) if closed_n else None,
        "avg_hold_min": avg_hold_min,
        "holding": len(holds),
        "holding_usd": round(sum(float(h.get("held_usd") or h.get("size_usd") or 0) for h in holds), 2),
        "watching": live.get("watching", len(watch)),
        "watch_in": live.get("watch_in", 0),
        "watch_out": live.get("watch_out", 0),
        "best_probability": live.get("best_probability", 0.0),
        "tick": live.get("tick") or {},
        "holds": holds,
        "watch": watch,
        "sold": sold,
        "universe": universe(settings, strategy),
        "kols": load_kols(state_dir),
        "fomo": load_fomo(state_dir),
        "pnl_chart": book["pnl_chart"],
        "pnl_windows": book["pnl_windows"],
        "aggressive": bool(strategy and strategy.get("aggressive_learning._active")),
        "aggressive_closes": aggressive_closes_since_on(store) if strategy else 0,
        "min_p": score_floors(strategy, store)[0] if strategy else None,
        "min_ev": score_floors(strategy, store)[1] if strategy else None,
        "faults": live.get("faults") or {},
        "dead_loops": live.get("dead_loops") or [],
    }


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>FIRSTKILL</title>
<style>
  :root { color-scheme: dark; --line:#161616; --muted:#6a6a6a; --text:#c8c8c8; --hi:#f5f5f5; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 13px/1.4 ui-monospace, SFMono-Regular, Consolas, monospace;
         background: #000; color: var(--text); }
  header { display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-end;
           padding: 16px 20px 12px; border-bottom: 1px solid var(--line); }
  h1 { font-size: 10px; letter-spacing: .16em; text-transform: uppercase;
       color: var(--muted); font-weight: 600; margin: 0 0 4px; }
  .brand { line-height: .95; }
  .brand .first { font-size: 28px; font-weight: 800; letter-spacing: .18em; color: #fff; }
  .brand .kill { font-size: 28px; font-weight: 800; letter-spacing: .18em; color: #ff3b4e;
                 text-shadow: 0 0 22px #ff3b4e88; }
  .n { font-size: 22px; color: var(--hi); font-variant-numeric: tabular-nums; }
  .up { color: #3dff9a; } .dn { color: #ff3b4e; } .muted { color: var(--muted); }
  .gold { color: #ffd24a; } .cyan { color: #3ad6ff; } .vol { color: #b07dff; }
  .pill { font-size: 10px; padding: 1px 7px; border: 1px solid #2a2a2a; border-radius: 99px; }
  .on { border-color: #3dff9a; color: #3dff9a; }
  .off { opacity: .4; }
  .r-main { border-color: #ffd24a; color: #ffd24a; }
  .r-beta { border-color: #3ad6ff; color: #3ad6ff; }
  .r-vamp { border-color: #ff3b4e; color: #ff3b4e; }
  .verdict { margin: 0 20px; padding: 10px 14px; border: 1px solid var(--line);
             border-radius: 8px; display: flex; gap: 14px; align-items: baseline; }
  .verdict .tag { font-size: 13px; font-weight: 800; letter-spacing: .14em; }
  .verdict.wait .tag { color: #ffd24a; }
  .verdict.up { border-color: #133; background: #03140c; }
  .verdict.dn { border-color: #311; background: #140306; }
  .verdict.wait { border-color: #332; background: #100e04; }
  .tabs { display: flex; gap: 4px; padding: 8px 20px 0; border-bottom: 1px solid var(--line); }
  .tabs button { font: inherit; font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
                 color: var(--muted); background: none; border: 0; border-bottom: 2px solid transparent;
                 padding: 8px 12px; cursor: pointer; }
  .tabs button.on { color: #fff; border-bottom-color: #ff3b4e; }
  #dump-all { font: inherit; font-size: 11px; letter-spacing: .08em; text-transform: uppercase;
              margin: 8px 20px 0 auto; padding: 8px 14px; cursor: pointer;
              background: #2a0008; color: #ff6b7a; border: 1px solid #ff3b4e; border-radius: 4px; }
  .panel { display: none; }
  .panel.on { display: block; }
  .kol-form { display: flex; gap: 8px; flex-wrap: wrap; padding: 14px 20px; }
  .kol-form input, .kol-form select { font: inherit; background: #111; color: var(--hi);
                 border: 1px solid #2a2a2a; padding: 6px 8px; border-radius: 4px; }
  .kol-form input[name="address"] { min-width: 16em; flex: 1 1 16em; }
  .kol-form button { font: inherit; background: #1a1a1a; color: #fff; border: 1px solid #333;
                     padding: 6px 12px; cursor: pointer; }
  #pnl-chart { width: 100%; height: 240px; display: block; }
  .pnl-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
               gap: 10px; padding: 14px 20px; }
  .pnl-windows { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                 gap: 8px; margin: 0; padding: 14px 20px 0; }
  .pnl-windows .pw { border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; background: #050505; }
  .pnl-windows .pw .n { font-size: 18px; font-weight: 800; }
  .exit-legs { margin: 6px 0 0; padding-left: 16px; color: var(--muted); font-size: 11px; }
  .exit-legs li { margin: 2px 0; }
  .v-bundled { color: #ff3b4e; } .v-cabaled { color: #ffb020; }
  .v-organic { color: #5b8cff; } .v-unverified { color: #6a6a6a; }
  tr.pick { cursor: pointer; }
  tr.pick.on td { background: #0c0c0c; }
  .coin-panel { margin: 0 0 14px; padding: 12px 14px; border: 1px solid var(--line); border-radius: 8px; }
  .coin-h { display: flex; gap: 14px; flex-wrap: wrap; align-items: baseline; }
  .coin-h .fit { font-size: 28px; font-weight: 800; color: #fff; }
  .coin-h .mcap { font-size: 22px; font-weight: 800; color: #fff; font-variant-numeric: tabular-nums; }
  .coin-h .age { font-size: 15px; font-weight: 700; color: #ffd24a; }
  .coin-panel ul { margin: 8px 0 0; padding-left: 16px; color: var(--text); word-break: break-word; }
  .coin-ca { word-break: break-all; color: var(--hi); font-size: 12px; }
  .coin-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(148px, 1fr)); gap: 10px 16px; margin: 12px 0; }
  .coin-stats b { display: block; font-size: 10px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
  .coin-stats span { font-size: 15px; color: #fff; font-variant-numeric: tabular-nums; word-break: break-word; }
  .buy-box { margin: 12px 0; padding: 12px; border: 1px solid #333; border-radius: 8px; background: #0a0a0a; }
  .buy-box h2 { margin: 0 0 8px; font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: #ffd24a; }
  .cat-list { display: grid; gap: 6px; margin: 8px 0; }
  .cat-list > div { display: flex; justify-content: space-between; gap: 12px; font-size: 13px; }
  .coin-panel .wrap { white-space: normal; overflow: visible; }
  button.copy, button.sell { font: inherit; font-size: 10px; background: #111; color: #c8c8c8; border: 1px solid #333;
                padding: 2px 8px; border-radius: 4px; cursor: pointer; letter-spacing: .04em; }
  button.copy:hover { color: #fff; border-color: #555; }
  button.sell { color: #ff6b7a; border-color: #5a1520; margin-left: 4px; }
  button.sell:hover { color: #fff; border-color: #ff3b4e; background: #2a0008; }
  .watch-bar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }
  .watch-bar h1 { margin: 0; }
  .watch-bar input { font: inherit; background: #0a0a0a; color: #fff; border: 1px solid #333;
                     padding: 6px 10px; border-radius: 6px; min-width: 220px; flex: 1; }
  .watch-bar button { font: inherit; background: #1a1a1a; color: #fff; border: 1px solid #333;
                      padding: 6px 12px; cursor: pointer; border-radius: 6px; }
  .watch-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 8px; }
  .lane { margin: 0 0 16px; padding: 10px 12px 12px; border: 1px solid var(--line); border-radius: 8px; }
  .lane > h1 { margin: 0 0 8px; }
  .lane-hold { border-color: #5a4a18; background: #0c0a04; }
  .lane-hold > h1 { color: #ffd24a; }
  .lane-scan { border-color: #1a3a4a; background: #040a10; }
  .lane-scan > h1 { color: #3ad6ff; }
  .lane-wait { border-color: #4a3a10; background: #100c04; }
  .lane-wait > h1 { color: #ffb020; }
  .lane-skip { border-color: #3a1518; background: #0c0406; }
  .lane-skip > h1 { color: #ff6b7a; }
  .wcard { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #050505;
           cursor: pointer; min-height: 186px; contain: layout; }
  .wcard.on { border-color: #555; background: #0c0c0c; }
  .wcard-h { display: flex; justify-content: space-between; align-items: center; gap: 6px; margin-bottom: 4px; }
  .wcard .mcap { font-size: 22px; font-weight: 800; color: #fff; letter-spacing: -.03em; line-height: 1.1;
                 font-variant-numeric: tabular-nums; min-width: 6ch; }
  .wcard .age { font-size: 15px; font-weight: 700; color: #ffd24a; font-variant-numeric: tabular-nums; }
  .wcard .meta { font-size: 11px; color: var(--muted); margin: 2px 0; }
  .wcard .kols { font-size: 11px; color: #c8c8c8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dex-paid { color: #3dff9a; font-weight: 800; letter-spacing: .08em; }
  .dex-no { color: #ff3b4e; font-weight: 800; letter-spacing: .08em; }
  a.xbtn { display: inline-block; margin: 4px 0 2px; padding: 2px 8px; border: 1px solid #333;
           border-radius: 99px; color: #3ad6ff; text-decoration: none; font-size: 11px; font-weight: 700; }
  a.xbtn:hover { border-color: #3ad6ff; color: #fff; }
  .score { display: flex; align-items: center; gap: 10px; margin: 6px 0 4px; }
  .score-n { font-size: 28px; font-weight: 800; letter-spacing: -.04em; line-height: 1;
             font-variant-numeric: tabular-nums; min-width: 2.2ch; }
  .score-hi .score-n { color: #3dff9a; } .score-mid .score-n { color: #ffd24a; }
  .score-lo .score-n { color: #ff3b4e; }
  .score-q .score-n { color: #6a6a6a; font-size: 13px; font-weight: 600; letter-spacing: 0; min-width: 0; }
  .score-q .score-bars { display: none; }
  .score-bars { flex: 1; display: grid; gap: 2px; }
  .score-bars b { display: flex; align-items: center; gap: 4px; font-size: 9px; color: #6a6a6a;
                  font-weight: 600; letter-spacing: .04em; }
  .score-bars b em { flex: 1; height: 3px; background: #1a1a1a; border-radius: 2px; overflow: hidden; }
  .score-bars b i { display: block; height: 100%; background: #c8c8c8; }
  .score-hi .score-bars i { background: #3dff9a; }
  .score-mid .score-bars i { background: #ffd24a; }
  .score-lo .score-bars i { background: #ff3b4e; }
  .cert-ok { color: #3dff9a; font-weight: 700; letter-spacing: .06em; }
  .cert-no { color: #ff3b4e; font-weight: 700; letter-spacing: .06em; }
  .cert-q { color: #6a6a6a; font-weight: 700; letter-spacing: .06em; }
  .st-scan { color: #5b8cff; } .st-skip { color: #ff3b4e; }
  .st-wait { color: #ffb020; } .st-trade { color: #3dff9a; }
  table.book td { font-variant-numeric: tabular-nums; }
  table.book .sym { font-size: 15px; font-weight: 700; }
  table.book .pnl { font-size: 16px; font-weight: 800; }
  table.book .mcap-path { font-size: 13px; color: #fff; font-weight: 600; }
  table.book th { position: sticky; top: 0; background: #000; z-index: 1; }
  .rev-bar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; padding: 14px 20px 0; }
  .rev-bar select { font: inherit; background: #111; color: #fff; border: 1px solid #2a2a2a; padding: 6px 8px; }
  tr.rev-feat td { background: #0a0a0a; color: var(--text); font-size: 12px; }
  .out-fumble { color: #ffd24a; }
  .out-rejeicao_correta { color: #6a6a6a; }
  .out-entrada_correta { color: #3ecf8e; }
  .out-entrada_errada { color: #ff3b4e; }
  .out-tracking { color: #5b8cff; }
  main { display: grid; grid-template-columns: 1fr 1fr; }
  main > section:first-child { grid-column: 1 / -1; }
  section { padding: 14px 20px; }
  section + section { border-left: 1px solid var(--line); }
  .full { grid-column: 1 / -1; border-top: 1px solid var(--line); border-left: 0 !important; }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; color: var(--muted); font-weight: 500; font-size: 10px;
       letter-spacing: .08em; text-transform: uppercase; padding: 6px 10px 8px 0; }
  td { padding: 6px 10px 6px 0; border-top: 1px solid #111; vertical-align: top; }
  .sym { color: var(--hi); }
  .ca { color: var(--muted); font-size: 11px; display: flex; gap: 6px; align-items: center; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; }
  .card { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; background: #050505; }
  .card h2 { margin: 0 0 6px; font-size: 13px; color: var(--hi); font-weight: 600; }
  .card p { margin: 0 0 5px; font-size: 12px; color: var(--muted); }
  .addr { font-size: 11px; color: var(--text); word-break: break-all; }
  #fault { display: none; padding: 8px 20px; background: #2a0008; color: #ff6b7a;
           border-bottom: 1px solid #ff3b4e; font-size: 12px; }
  #fault.on { display: block; }
  @media (max-width: 900px) {
    main, .grid { grid-template-columns: 1fr; }
    section + section { border-left: 0; border-top: 1px solid var(--line); }
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div><span class="first">FIRST</span><span class="kill">KILL</span></div>
    <div id="status" class="muted">loading</div>
  </div>
  <div><h1>equity</h1><div class="n" id="equity">—</div></div>
  <div><h1>pnl</h1><div class="n" id="pnl">—</div></div>
  <div><h1>win rate</h1><div class="n" id="winrate">—</div></div>
  <div><h1>avg hold</h1><div class="n" id="avghold">—</div></div>
  <div><h1>holding</h1><div class="n gold" id="holding">—</div></div>
  <div><h1>scanning</h1><div class="n cyan" id="watching">—</div></div>
</header>
<div id="fault"></div>
<div id="verdict" class="verdict wait"><span class="tag">WAITING</span><span class="muted" id="verdict-h">sem trades ainda</span></div>
<nav class="tabs">
  <button type="button" class="on" data-tab="tab-watch">scan</button>
  <button type="button" data-tab="tab-kols">kols</button>
  <button type="button" data-tab="tab-fomo">fomo</button>
  <button type="button" data-tab="tab-review">review</button>
  <button type="button" data-tab="tab-pnl">pnl</button>
  <button type="button" id="dump-all">vender tudo</button>
</nav>
<div id="tab-watch" class="panel on">
<section class="full" style="padding:14px 20px 6px">
  <h1>chains</h1>
  <div class="grid" id="universe"></div>
</section>
<main>
  <section>
    <div class="watch-bar">
      <h1>inspect</h1>
      <input id="ca-in" placeholder="colar CA" autocomplete="off" spellcheck="false"/>
      <button type="button" id="paste-ca">colar</button>
    </div>
    <div id="coin" class="coin-panel" hidden></div>
    <div class="lane lane-hold">
      <h1>hold</h1>
      <div id="watch-hold" class="watch-grid"></div>
      <table id="holds" class="book"><thead><tr>
        <th>token</th><th>held</th><th>pnl</th><th>left</th><th>age</th><th>mcap</th><th>stage 3</th>
      </tr></thead><tbody></tbody></table>
    </div>
    <div class="lane lane-scan">
      <h1>scanning</h1>
      <div id="watch-scan" class="watch-grid"></div>
    </div>
    <div class="lane lane-wait">
      <h1>wait</h1>
      <div id="watch-wait" class="watch-grid"></div>
    </div>
    <div class="lane lane-skip">
      <h1>skip</h1>
      <div id="watch-skip" class="watch-grid"></div>
    </div>
  </section>
  <section>
    <h1>sold</h1>
    <table id="sold" class="book"><thead><tr>
      <th>token</th><th>size</th><th>pnl</th><th>mcap</th><th>hold</th><th>exit</th>
    </tr></thead><tbody></tbody></table>
  </section>
</main>
</div>
<div id="tab-kols" class="panel">
  <p class="muted" style="padding:14px 20px 0">Seguir não é copiar. Cada mint ainda passa nos gates. Só wallets aqui entram no sinal de copy, e só se o token for young/small.</p>
  <form class="kol-form" id="kol-form">
    <input name="address" placeholder="wallet / CA" required size="42"/>
    <input name="handle" placeholder="@handle" size="16"/>
    <select name="klass"><option value="kol">kol</option><option value="whale">whale</option></select>
    <button type="submit">add</button>
  </form>
  <section>
    <table id="kols"><thead><tr>
      <th>handle</th><th>class</th><th>chain</th><th>address</th><th></th>
    </tr></thead><tbody></tbody></table>
  </section>
</div>
<div id="tab-fomo" class="panel">
  <p class="muted" style="padding:14px 20px 0">Perfis da Fomo que você segue. Nome + wallet (várias no mesmo campo, separadas por espaço ou vírgula). O bot não opera na Fomo — copia o sinal on-chain no paper/Jupiter.</p>
  <form class="kol-form" id="fomo-form">
    <input name="handle" placeholder="nome do perfil" required size="18" autocomplete="off"/>
    <input name="address" placeholder="wallets" required autocomplete="off"/>
    <button type="submit">seguir</button>
  </form>
  <p id="fomo-err" class="muted" style="padding:0 20px"></p>
  <section>
    <table id="fomo-rows"><thead><tr>
      <th>nome</th><th>chain</th><th>wallet</th><th></th>
    </tr></thead><tbody></tbody></table>
  </section>
</div>
<div id="tab-review" class="panel">
  <div class="rev-bar">
    <select id="rev-out">
      <option value="">todos</option>
      <option value="fumble">fumble</option>
      <option value="rejeicao_correta">rejeição correta</option>
      <option value="entrada_correta">entrada correta</option>
      <option value="entrada_errada">entrada errada</option>
      <option value="tracking">ainda tracking</option>
    </select>
    <span class="muted" id="rev-meta"></span>
  </div>
  <p class="muted" style="padding:8px 20px 0">MFE do shadow é o pico na janela, não o close. Dump e flat parecem iguais se não bombou. Clique a linha pras features.</p>
  <section>
    <table id="review" class="book"><thead><tr>
      <th>quando</th><th>token</th><th>decisão</th><th>p / ev</th><th>resultado</th><th>outcome</th>
    </tr></thead><tbody></tbody></table>
  </section>
</div>
<div id="tab-pnl" class="panel">
  <div class="pnl-windows" id="pnl-windows"></div>
  <div class="pnl-cards" id="pnl-cards"></div>
  <section>
    <h1>equity curve</h1>
    <svg id="pnl-chart" viewBox="0 0 800 240" preserveAspectRatio="none"></svg>
  </section>
</div>
<script>
const $ = id => document.getElementById(id);
const usd = n => (n<0?'−':'') + '$' + Math.abs(n).toFixed(2);
const usdFull = n => {
  n = Number(n)||0;
  const abs = Math.abs(n);
  const s = abs >= 1000
    ? Math.round(abs).toLocaleString('en-US')
    : abs.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 6});
  return (n<0?'−':'') + '$' + s;
};
const numFull = n => Math.round(Number(n)||0).toLocaleString('en-US');
const pctFull = n => {
  n = Number(n)||0;
  return (n>=0?'+':'') + (n*100).toFixed(2) + '%';
};
const esc = s => String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');
const kM = n => {
  n = Math.abs(Number(n)||0);
  if (n >= 1e6) return (n/1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'k';
  return String(Math.round(n));
};
const mcapTxt = n => {
  n = Math.abs(Number(n)||0);
  if (n >= 1e6) return '$' + (n/1e6).toFixed(2) + 'M';
  if (n >= 1e3) return '$' + (n/1e3).toFixed(1) + 'k';
  return '$' + Math.round(n);
};
const pct = n => (n>=0?'+':'') + (Number(n)*100).toFixed(1) + '%';
const cls = n => n>0?'up':n<0?'dn':'';
const chainShort = c => ({solana:'SOL', bnb:'BNB', robinhood_chain:'HOOD'}[c] || (c||'').toUpperCase());
const ageTxt = m => {
  m = Number(m)||0;
  if (m >= 60) {
    const h = m/60;
    return (h >= 10 ? Math.round(h) : (Math.round(h*10)/10)) + 'h';
  }
  const t = Math.round(m*10)/10;
  return (t % 1 ? t.toFixed(1) : String(Math.round(t))) + ' min';
};
const certHtml = c => {
  const k = c==='ok'?'cert-ok':c==='no'?'cert-no':'cert-q';
  const t = c==='ok'?'CERT OK':c==='no'?'CERT NO':'CERT —';
  return '<span class="'+k+'" data-f="cert">'+t+'</span>';
};
function xHandle(h) {
  h = String(h||'').replace(/^@/,'').trim();
  if (!/^[A-Za-z0-9_]{1,15}$/.test(h) || /^\\d+$/.test(h)) return '';
  return h;
}
function twLine(tw) {
  if (!tw || !(tw.official || (tw.posts||[]).length || tw.utility)) return '';
  const h = xHandle(tw.official);
  const btn = h
    ? '<a class="xbtn" href="https://x.com/'+h+'" target="_blank" rel="noopener">@'+h+'</a>'
    : '';
  const inst = tw.inst ? ' · inst' : '';
  const quiet = tw.official_age_min != null && tw.official_age_min > 360 ? ' quiet' : '';
  if (!btn && !inst && !quiet && !tw.utility) return '';
  return '<div class="meta">'+btn+inst+quiet+(tw.utility?' · '+tw.utility:'')+'</div>';
}
function twBlock(tw) {
  if (!tw) return '';
  const posts = (tw.posts||[]).map(p => '<li>@'+esc(p.handle)+' · '+numFull(p.followers)+' followers · '
    +(p.age_min==null?'':ageTxt(p.age_min)+' · ')+esc(p.text||'')+'</li>').join('');
  return twLine(tw) + (posts ? '<ul>'+posts+'</ul>' : '');
}
function dexLine(w) {
  return w.dex_paid
    ? '<div class="meta"><span class="dex-paid" data-f="dex">DEX PAID</span></div>'
    : '<div class="meta"><span class="dex-no" data-f="dex">NOT DEX</span></div>';
}
function rubricLine(r) {
  if (r == null || r.total == null) {
    return '<div class="score score-q" data-f="score">'
      + '<span class="score-n" data-f="score-n">—</span>'
      + '<div class="score-bars">'
      + '<b>D<em><i data-k="D" style="width:0%"></i></em></b>'
      + '<b>C<em><i data-k="C" style="width:0%"></i></em></b>'
      + '<b>F<em><i data-k="F" style="width:0%"></i></em></b>'
      + '<b>G<em><i data-k="G" style="width:0%"></i></em></b>'
      + '<b>N<em><i data-k="N" style="width:0%"></i></em></b>'
      + '</div></div>';
  }
  const tone = r.total >= 7 ? 'hi' : r.total >= 5.5 ? 'mid' : 'lo';
  const row = (k, v) => '<b>'+k+'<em><i data-k="'+k+'" style="width:'+Math.max(0,Math.min(100,Math.round(((v||0)/10)*100)))+'%"></i></em></b>';
  return '<div class="score score-'+tone+'" data-f="score">'
    + '<span class="score-n" data-f="score-n">'+r.total+'</span>'
    + '<div class="score-bars">'
    + row('D', r.dist)+row('C', r.crowd)+row('F', r.flow)+row('G', r.chart)+row('N', r.narrative)
    + '</div></div>';
}
const mins = n => ageTxt(n);
const caHead = a => {
  a = String(a||'');
  if (a.startsWith('0x') && a.length >= 10) return a.slice(0, 6)+'…'+a.slice(-4);
  return a.slice(0, 6);
};
const role = r => r && r !== 'solo' ? '<span class="pill r-'+r+'">'+r+'</span>' : '';
const copyBtn = a => a ? '<button class="copy" type="button" data-copy="'+a+'" title="copiar CA" aria-label="copiar CA">copiar</button>' : '';
const sellBtn = (key, sym) => key
  ? '<button class="sell" type="button" data-sell="'+esc(key)+'" data-sym="'+esc(sym||'')+'" title="vender" aria-label="vender">vender</button>'
  : '';
let picked = '';
let lastWatch = [];
let inflight = false;
let queued = false;

function watchKey(w) { return w.chain+':'+w.address; }
function watchBody(w) {
  const call = w.call || 'scan';
  const skipHint = (call==='skip' && w.label==='unverified')
    ? 'holders da Solana nao medidos. Precisa Helius/RPC.'
    : '';
  const pill = w.label
    ? '<span class="pill v-'+w.label+'" data-f="pill"'+(skipHint?' title="'+skipHint+'"':'')+'>'+call.toUpperCase()+' '+w.label+'</span>'
    : '<span class="pill st-'+call+'" data-f="pill">'+call.toUpperCase()+'</span>';
  const whales = '<div class="meta" data-f="whales">'
    + (w.whale_n == null ? '' : ('whales '+w.whale_n+' · '+Math.round((w.whale_pct||0)*100)+'% · $'+kM(w.whale_usd||0)))
    + '</div>';
  const kols = (w.kols && w.kols.length) ? w.kols.join(', ') : '—';
  const fomo = (w.fomo && w.fomo.length) ? w.fomo.join(', ') : '—';
  return '<div class="wcard-h"><span class="sym" data-f="sym">'+(w.symbol || w.name || caHead(w.address))+'</span>'+copyBtn(w.address)+'</div>'
    + '<div class="wcard-h"><span class="chain">'+chainShort(w.chain)+'</span><span class="age" data-f="age">'+ageTxt(w.age_min)+'</span></div>'
    + '<div class="mcap" data-f="mcap">'+mcapTxt(w.mcap)+'</div>'
    + certHtml(w.cert)+' '+pill+' '+role(w.role)
    + whales
    + '<div class="kols" data-f="kols">kols '+kols+'</div>'
    + '<div class="kols" data-f="fomo">fomo '+fomo+'</div>'
    + dexLine(w)
    + rubricLine(w.rubric || {})
    + '<div class="meta" data-f="why">'+(w.why && w.why !== 'ok' ? w.why : '')+'</div>'
    + '<div class="meta" data-f="pev">'+(w.p != null ? ('p '+Number(w.p).toFixed(3)+' · EV '+(Number(w.ev)>=0?'+':'')+Number(w.ev).toFixed(3)) : '')+'</div>'
    + '<div data-f="tw" data-h="'+esc(xHandle(w.tw && w.tw.official)||'')+'">'+twLine(w.tw)+'</div>'
    + '<div data-f="ret" class="'+cls(w.ret_5m)+'">'+pct(w.ret_5m||0)+' · vol '+kM(w.vol5m)+'</div>';
}
function setNode(n, t) {
  if (!n) return;
  t = String(t);
  if (n.childNodes.length === 1 && n.firstChild.nodeType === 3) {
    if (n.firstChild.nodeValue !== t) n.firstChild.nodeValue = t;
    return;
  }
  if (n.textContent !== t) n.textContent = t;
}
function cardFp(w) {
  const r = w.rubric || {};
  return [w.call, w.label, w.symbol, w.mcap, w.age_min, w.vol5m, w.ret_5m, w.p, w.ev, r.total,
    w.why, w.cert, w.dex_paid, (w.kols||[]).join(), (w.fomo||[]).join(), w.whale_n].join('|');
}
function fillLive(el, w) {
  const set = (f, t) => setNode(el.querySelector('[data-f="'+f+'"]'), t);
  set('mcap', mcapTxt(w.mcap));
  set('age', ageTxt(w.age_min));
  set('sym', w.symbol || w.name || caHead(w.address));
  set('kols', 'kols '+((w.kols && w.kols.length) ? w.kols.join(', ') : '—'));
  set('fomo', 'fomo '+((w.fomo && w.fomo.length) ? w.fomo.join(', ') : '—'));
  set('why', (w.why && w.why !== 'ok') ? w.why : '');
  set('whales', w.whale_n == null ? '' : ('whales '+w.whale_n+' · '+Math.round((w.whale_pct||0)*100)+'% · $'+kM(w.whale_usd||0)));
  const cert = el.querySelector('[data-f="cert"]');
  if (cert) {
    const ck = w.cert==='ok'?'cert-ok':w.cert==='no'?'cert-no':'cert-q';
    if (cert.className !== ck) cert.className = ck;
    setNode(cert, w.cert==='ok'?'CERT OK':w.cert==='no'?'CERT NO':'CERT —');
  }
  const pill = el.querySelector('[data-f="pill"]');
  if (pill) {
    const call = w.call || 'scan';
    const pk = 'pill ' + (w.label ? ('v-'+w.label) : ('st-'+call));
    if (pill.className !== pk) pill.className = pk;
    setNode(pill, w.label ? (call.toUpperCase()+' '+w.label) : call.toUpperCase());
  }
  const tw = el.querySelector('[data-f="tw"]');
  if (tw) {
    const h = xHandle(w.tw && w.tw.official);
    if (tw.dataset.h !== String(h||'')) {
      tw.dataset.h = h||'';
      tw.innerHTML = twLine(w.tw);
    }
  }
  const dex = el.querySelector('[data-f="dex"]');
  if (dex) {
    const dk = w.dex_paid ? 'dex-paid' : 'dex-no';
    if (dex.className !== dk) dex.className = dk;
    setNode(dex, w.dex_paid ? 'DEX PAID' : 'NOT DEX');
  }
  const r = w.rubric || {};
  const score = el.querySelector('[data-f="score"]');
  if (score) {
    if (r.total == null) {
      if (score.className !== 'score score-q') score.className = 'score score-q';
      set('score-n', '—');
      score.querySelectorAll('i[data-k]').forEach(i => {
        if (i.style.width !== '0%') i.style.width = '0%';
      });
    } else {
      const tone = 'score score-' + (r.total >= 7 ? 'hi' : r.total >= 5.5 ? 'mid' : 'lo');
      if (score.className !== tone) score.className = tone;
      set('score-n', String(r.total));
      const wmap = {D:r.dist, C:r.crowd, F:r.flow, G:r.chart, N:r.narrative};
      score.querySelectorAll('i[data-k]').forEach(i => {
        const pctw = Math.max(0, Math.min(100, Math.round(((wmap[i.dataset.k]||0)/10)*100)));
        if (i.style.width !== pctw+'%') i.style.width = pctw+'%';
      });
    }
  }
  const ret = el.querySelector('[data-f="ret"]');
  if (ret) {
    const rk = cls(w.ret_5m);
    if (ret.className !== rk) ret.className = rk;
    setNode(ret, pct(w.ret_5m||0)+' · vol '+kM(w.vol5m));
  }
  if (w.p != null) {
    set('pev', 'p '+Number(w.p).toFixed(3)+' · EV '+(Number(w.ev)>=0?'+':'')+Number(w.ev).toFixed(3));
  }
}

function laneOf(w) {
  const c = w.call || 'scan';
  return (c === 'hold' || c === 'wait' || c === 'skip') ? c : 'scan';
}
function paintWatch(list, running) {
  const groups = {hold:[], scan:[], wait:[], skip:[]};
  list.forEach(w => groups[laneOf(w)].push(w));
  const keep = new Set(list.map(watchKey));
  const emptyMsg = {
    hold: 'nenhum hold nestes cards',
    scan: running ? 'scanning…' : 'start the bot',
    wait: 'nenhum wait',
    skip: 'nenhum skip',
  };
  ['hold','scan','wait','skip'].forEach(lane => {
    const box = $('watch-'+lane);
    if (!box) return;
    const items = groups[lane];
    if (!items.length) return;
    if (box.dataset.empty) {
      box.dataset.empty = '';
      const muted = box.querySelector(':scope > .muted');
      if (muted) muted.remove();
    }
    items.forEach(w => {
      const key = watchKey(w);
      let el = document.querySelector('#tab-watch .wcard[data-pick="'+key.replace(/"/g,'')+'"]');
      const fp = cardFp(w);
      if (!el) {
        el = document.createElement('div');
        el.className = 'wcard pick';
        el.dataset.pick = key;
        el.innerHTML = watchBody(w);
        el.dataset.fp = fp;
        box.appendChild(el);
      } else {
        if (el.parentNode !== box) box.appendChild(el);
        if (el.dataset.fp !== fp) { fillLive(el, w); el.dataset.fp = fp; }
      }
    });
  });
  document.querySelectorAll('#tab-watch .wcard').forEach(el => {
    if (!keep.has(el.dataset.pick)) el.remove();
  });
  ['hold','scan','wait','skip'].forEach(lane => {
    const box = $('watch-'+lane);
    if (!box) return;
    if (box.querySelector('.wcard')) return;
    const empty = emptyMsg[lane];
    if (box.dataset.empty !== empty) {
      box.dataset.empty = empty;
      box.innerHTML = '<div class="muted">'+empty+'</div>';
    }
  });
}

function fillCoin(el, w) {
  const set = (f, t) => setNode(el.querySelector('[data-f="'+f+'"]'), t);
  const r = w.rubric || {};
  set('mcap', usdFull(w.mcap));
  set('liq', usdFull(w.liq));
  set('vol5m', usdFull(w.vol5m));
  set('ret5m', pctFull(w.ret_5m));
  set('age', ageTxt(w.age_min));
  set('holders', w.holders==null?'—':numFull(w.holders));
  set('rt', w.rt==null?'—':pctFull(w.rt));
  const pending = r.total == null;
  set('c-score', pending ? '—' : String(r.total));
  set('cat-dist', pending ? '—' : r.dist+' / 10');
  set('cat-crowd', pending ? '—' : r.crowd+' / 10');
  set('cat-flow', pending ? '—' : r.flow+' / 10');
  set('cat-chart', pending ? '—' : r.chart+' / 10');
  set('cat-narr', pending ? '—' : r.narrative+' / 10');
  const buy = el.querySelector('.buy-box');
  if (buy) {
    const tone = pending ? 'q' : (r.total >= (w.min_rubric||7) ? 'hi' : r.total >= 5.5 ? 'mid' : 'lo');
    const want = 'buy-box score-'+tone;
    if (buy.className !== want) buy.className = want;
  }
  set('r-cap', pending ? 'sem avaliação ainda' : ('rubric / '+(w.min_rubric==null?'—':w.min_rubric)+' pra passar o piso'));
  const minP = w.min_p == null ? '—' : Number(w.min_p).toFixed(4);
  const minEv = w.min_ev == null ? '—' : Number(w.min_ev).toFixed(4);
  set('pwin', (w.p == null ? '—' : Number(w.p).toFixed(4))+'  (piso '+minP+')');
  set('ev', (w.ev == null ? '—' : ((Number(w.ev)>=0?'+':'')+Number(w.ev).toFixed(4)))+'  (piso '+minEv+')');
  set('why', w.explain || w.why || '');
  set('whales', w.whale_n==null?'—':(numFull(w.whale_n)+' · '+pctFull(w.whale_pct)+' · '+usdFull(w.whale_usd)));
  set('kols', (w.kols&&w.kols.length) ? w.kols.join(', ') : '—');
  set('fomo', (w.fomo&&w.fomo.length) ? w.fomo.join(', ') : '—');
  set('dexname', w.dex||'—');
  set('source', w.source||'—');
  set('dexpaid', w.dex_paid ? 'yes' : 'no');
  const badge = el.querySelector('[data-f="dex"]');
  if (badge) {
    badge.className = w.dex_paid ? 'dex-paid' : 'dex-no';
    setNode(badge, w.dex_paid ? 'DEX PAID' : 'NOT DEX');
  }
  const vetoes = (w.vetoes&&w.vetoes.length) ? w.vetoes : (w.why ? [w.why] : []);
  const vu = el.querySelector('[data-f="vetoes"]');
  if (vu) vu.innerHTML = vetoes.map(s => '<li>'+esc(s)+'</li>').join('');
  const su = el.querySelector('[data-f="signals"]');
  if (su) su.innerHTML = (w.signals||[]).map(s => '<li>'+esc(s)+'</li>').join('');
}

function paintCoin(opts) {
  const box = $('coin');
  if (!box) return;
  const w = lastWatch.find(x => watchKey(x) === picked);
  document.querySelectorAll('#tab-watch .pick').forEach(r => r.classList.toggle('on', r.dataset.pick === picked));
  if (!w) { box.hidden = true; box.innerHTML = ''; box.dataset.pick = ''; return; }
  box.hidden = false;
  const struct = [picked, w.call, w.cert, w.label, w.address, (w.vetoes||[]).join(), w.explain||'', String(w.dex_paid), String(w.holders)].join('|');
  if (box.dataset.pick === picked && box.dataset.struct === struct && box.querySelector('[data-f=mcap]') && !(opts && opts.scroll)) {
    fillCoin(box, w);
    return;
  }
  box.dataset.struct = struct;
  box.dataset.pick = picked;
  const call = (w.call || 'scan').toUpperCase();
  const r = w.rubric || {};
  const stat = (k, f, v) => '<div><b>'+k+'</b><span data-f="'+f+'">'+v+'</span></div>';
  const pTxt = w.p == null ? '—' : Number(w.p).toFixed(4);
  const evTxt = w.ev == null ? '—' : ((Number(w.ev)>=0?'+':'')+Number(w.ev).toFixed(4));
  const minP = w.min_p == null ? '—' : Number(w.min_p).toFixed(4);
  const minEv = w.min_ev == null ? '—' : Number(w.min_ev).toFixed(4);
  const pending = r.total == null;
  const tone = pending ? 'q' : (r.total >= (w.min_rubric||7) ? 'hi' : r.total >= 5.5 ? 'mid' : 'lo');
  const wallets = (w.wallets||[]).join(', ') || '—';
  const kols = (w.kols&&w.kols.length) ? w.kols.join(', ') : '—';
  const fomo = (w.fomo&&w.fomo.length) ? w.fomo.join(', ') : '—';
  const vetoes = (w.vetoes&&w.vetoes.length) ? w.vetoes : (w.why ? [w.why] : []);
  box.innerHTML = '<div class="coin-h">'
    + '<div class="sym">'+(esc(w.symbol||w.name)||'—')+'</div>'
    + '<div class="chain">'+esc(w.chain)+'</div>'
    + certHtml(w.cert)
    + '<span class="pill st-'+(w.call||'scan')+'">'+call+'</span>'
    + (w.label ? '<span class="pill v-'+esc(w.label)+'">'+esc(w.label)+(w.fit!=null?' · fit '+w.fit:'')+'</span>' : '')
    + role(w.role)
    + '</div>'
    + (w.name ? '<p>'+esc(w.name)+'</p>' : '')
    + '<p class="coin-ca">'+esc(w.address)+' '+copyBtn(w.address)+'</p>'
    + '<div class="coin-stats">'
    + stat('mcap', 'mcap', usdFull(w.mcap))
    + stat('liquidity', 'liq', usdFull(w.liq))
    + stat('vol 5m', 'vol5m', usdFull(w.vol5m))
    + stat('ret 5m', 'ret5m', pctFull(w.ret_5m))
    + stat('age', 'age', ageTxt(w.age_min))
    + stat('holders', 'holders', w.holders==null?'—':numFull(w.holders))
    + stat('round trip', 'rt', w.rt==null?'—':pctFull(w.rt))
    + stat('dex', 'dexname', esc(w.dex||'—'))
    + stat('source', 'source', esc(w.source||'—'))
    + stat('dex paid', 'dexpaid', w.dex_paid ? 'yes' : 'no')
    + '</div>'
    + '<div class="buy-box score-'+tone+'">'
    + '<h2>nota desta moeda</h2>'
    + '<p>Régua igual pra todas; os inputs (chart, crowd, chain, narrativa) são desta CA. WAITING no topo é PnL da conta, não nota.</p>'
    + '<div class="coin-h"><span class="score-n" data-f="c-score">'+(pending?'—':r.total)+'</span>'
    + '<span class="muted" data-f="r-cap">'+(pending?'sem avaliação ainda':('rubric / '+(w.min_rubric==null?'—':w.min_rubric)+' pra passar o piso'))+'</span></div>'
    + '<div class="cat-list">'
    + '<div><span>distribution (D)</span><span data-f="cat-dist">'+(pending?'—':r.dist+' / 10')+'</span></div>'
    + '<div><span>crowd (C)</span><span data-f="cat-crowd">'+(pending?'—':r.crowd+' / 10')+'</span></div>'
    + '<div><span>flow (F)</span><span data-f="cat-flow">'+(pending?'—':r.flow+' / 10')+'</span></div>'
    + '<div><span>chart (G)</span><span data-f="cat-chart">'+(pending?'—':r.chart+' / 10')+'</span></div>'
    + '<div><span>narrative (N)</span><span data-f="cat-narr">'+(pending?'—':r.narrative+' / 10')+'</span></div>'
    + '</div>'
    + '<div class="coin-stats">'
    + stat('p (win)', 'pwin', pTxt+'  (piso '+minP+')')
    + stat('EV', 'ev', evTxt+'  (piso '+minEv+')')
    + stat('call', 'call', call)
    + '</div>'
    + '<p data-f="why">'+esc(w.explain || w.why || '')+'</p>'
    + (vetoes.length ? '<ul data-f="vetoes">'+vetoes.map(s => '<li>'+esc(s)+'</li>').join('')+'</ul>' : '<ul data-f="vetoes"></ul>')
    + '</div>'
    + '<p class="wrap">whales <span data-f="whales">'+(w.whale_n==null?'—':(numFull(w.whale_n)+' · '+pctFull(w.whale_pct)+' · '+usdFull(w.whale_usd)))+'</span></p>'
    + '<p class="wrap">kols <span data-f="kols">'+esc(kols)+'</span></p>'
    + '<p class="wrap">fomo <span data-f="fomo">'+esc(fomo)+'</span></p>'
    + '<p class="wrap">wallets '+esc(wallets)+'</p>'
    + dexLine(w)
    + twBlock(w.tw)
    + '<ul data-f="signals">'+(w.signals||[]).map(s => '<li>'+esc(s)+'</li>').join('')+'</ul>'
    + '<p class="muted">Organic não é compra. Bundled = skip. Cabaled na Solana = skip; na Hood o EV desta moeda decide.</p>';
  if (opts && opts.scroll) box.scrollIntoView({behavior:'smooth', block:'nearest'});
}

function setText(id, text, extra) {
  const el = $(id);
  if (!el) return;
  setNode(el, text);
  if (extra != null) el.className = extra;
}

function mcapPath(a, b) {
  if (!a && !b) return '—';
  return '$'+(a?kM(a):'—')+' → $'+(b?kM(b):'—');
}

function holdRowHtml(h) {
  const addr = h.address || (h.key||'').split(':')[1];
  return '<td><div class="sym" data-f="sym">'+esc(h.symbol || caHead(addr))+' '+role(h.role)+'</div>'
    + '<div class="ca">'+chainShort(h.chain||'')+' · '+caHead(addr)+' '+copyBtn(addr)+' '+sellBtn(h.key, h.symbol)+'</div></td>'
    + '<td class="gold" data-f="held"></td>'
    + '<td class="pnl" data-f="pnl"></td>'
    + '<td data-f="left"></td>'
    + '<td data-f="age"></td>'
    + '<td class="mcap-path" data-f="mcap"></td>'
    + '<td data-f="stage"></td>';
}
function fillHoldRow(tr, h) {
  const set = (f, t) => setNode(tr.querySelector('[data-f="'+f+'"]'), t);
  const pnl = tr.querySelector('[data-f="pnl"]');
  const u = h.unrealized_usd != null ? h.unrealized_usd : h.unrealized_pct;
  if (pnl) {
    const want = 'pnl '+cls(u);
    if (pnl.className !== want) pnl.className = want;
    if (pnl.innerHTML !== (usd(h.unrealized_usd||0)+' <span class="muted">'+pct(h.unrealized_pct)+'</span>'))
      pnl.innerHTML = usd(h.unrealized_usd||0)+' <span class="muted">'+pct(h.unrealized_pct)+'</span>';
  }
  set('held', usd(h.held_usd != null ? h.held_usd : h.size_usd));
  set('left', Math.round((h.remaining_pct||0)*100)+'%');
  set('age', mins(h.age_min));
  set('mcap', mcapPath(h.mcap_entry, h.mcap));
  const st = tr.querySelector('[data-f="stage"]');
  if (st) {
    const why = (h.hold_why||'')+(h.hold_strikes?(' · '+h.hold_strikes+' strike'):'');
    const next = (h.entry_rubric ? (h.entry_rubric+' → '+(h.hold_rubric||'—')) : '—')
      + (why ? '<div class="meta">'+esc(why)+'</div>' : '');
    if (st.innerHTML !== next) st.innerHTML = next;
  }
}
function paintHoldTable(items) {
  const body = $('holds').querySelector('tbody');
  if (!body) return;
  if (!items.length) {
    const empty = '<tr class="empty"><td class="muted" colspan="7">none open</td></tr>';
    if (body.innerHTML !== empty) body.innerHTML = empty;
    return;
  }
  const z = body.querySelector('tr.empty');
  if (z) z.remove();
  const have = new Map([...body.querySelectorAll('tr[data-k]')].map(el => [el.dataset.k, el]));
  const keep = new Set();
  items.forEach(h => {
    const k = String(h.key || '');
    keep.add(k);
    let tr = have.get(k);
    if (!tr) {
      tr = document.createElement('tr');
      tr.dataset.k = k;
      tr.innerHTML = holdRowHtml(h);
      body.appendChild(tr);
    }
    fillHoldRow(tr, h);
  });
  have.forEach((el, k) => { if (!keep.has(k)) el.remove(); });
}
function rows(el, items, html, empty, cols) {
  const body = el.querySelector('tbody');
  const next = items.length ? items.map(html).join('')
    : '<tr><td class="muted" colspan="'+(cols||6)+'">'+(empty||'none')+'</td></tr>';
  if (body.innerHTML !== next) body.innerHTML = next;
}

function chainCard(c) {
  const pads = (c.launchpads||[]).join(', ') || '—';
  const extra = [];
  if (c.min_volume_5m) extra.push('vol ≥ '+kM(c.min_volume_5m));
  if (c.min_twitter) extra.push('twitter ≥ '+c.min_twitter);
  extra.push('mcap ≥ '+kM(c.copy_min_mcap||100000)+' · <'+kM(c.copy_max_mcap));
  return `<div class="card ${c.enabled?'':'off'}">
    <h2>${c.chain} ${c.enabled?'<span class="pill on">on</span>':'<span class="pill">off</span>'}</h2>
    <p>${pads} · max ${Math.round(c.max_age_min)} min</p>
    <p>${extra.join(' · ')}</p>
    <div class="addr">${c.wallet || 'no wallet'}</div>
  </div>`;
}

function paintVerdict(d) {
  const box = $('verdict');
  const hint = $('verdict-h');
  const losses = Math.max(0, (d.closed||0) - (d.wins||0));
  if (!d.closed) {
    const p = d.best_probability ? ' melhor setup '+Math.round(d.best_probability*100)+'%' : '';
    box.className = 'verdict wait';
    box.querySelector('.tag').textContent = 'WAITING';
    hint.textContent = 'sem trades ainda — paper caçando' + p;
    return;
  }
  if (d.pnl > 0) {
    box.className = 'verdict up';
    box.querySelector('.tag').textContent = 'UP';
    hint.textContent = usd(d.pnl)+' no lucro · '+d.wins+'W / '+losses+'L';
    return;
  }
  if (d.pnl < 0) {
    box.className = 'verdict dn';
    box.querySelector('.tag').textContent = 'DOWN';
    hint.textContent = usd(d.pnl)+' no prejuízo · '+d.wins+'W / '+losses+'L';
    return;
  }
  box.className = 'verdict wait';
  box.querySelector('.tag').textContent = 'FLAT';
  hint.textContent = 'zero a zero · '+d.wins+'W / '+losses+'L';
}

$('dump-all').addEventListener('click', () => {
  if (!confirm('Vender todas as posições abertas agora?')) return;
  fetch('/api/flatten', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'})
    .then(r => r.json()).then(j => {
      $('status').textContent = j.ok ? 'vendendo tudo…' : (j.error || 'falhou vender');
    }).catch(() => { $('status').textContent = 'falhou vender'; });
});
document.addEventListener('click', e => {
  const tab = e.target.closest('[data-tab]');
  if (tab) {
    document.querySelectorAll('.tabs button').forEach(b => b.classList.toggle('on', b === tab));
    document.querySelectorAll('.panel').forEach(p => p.classList.toggle('on', p.id === tab.dataset.tab));
    if (tab.dataset.tab === 'tab-review') loadReview();
    return;
  }
  const rev = e.target.closest('.rev-row');
  if (rev && !e.target.closest('[data-copy]')) {
    const feat = document.querySelector('[data-rev-feat="'+rev.dataset.rev+'"]');
    if (feat) feat.hidden = !feat.hidden;
    return;
  }
  const pick = e.target.closest('[data-pick]');
  if (pick && !e.target.closest('[data-copy]') && !e.target.closest('a')) {
    picked = pick.dataset.pick;
    paintCoin({scroll: true});
    return;
  }
  const rmFomo = e.target.closest('[data-fomo-rm]');
  if (rmFomo) {
    fetch('/api/fomo', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({action:'remove', address: rmFomo.dataset.fomoRm})}).then(tick);
    return;
  }
  const rm = e.target.closest('[data-remove]');
  if (rm) {
    fetch('/api/kols', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({action:'remove', address: rm.dataset.remove})}).then(tick);
    return;
  }
  const sell = e.target.closest('[data-sell]');
  if (sell) {
    const key = sell.dataset.sell || '';
    const sym = sell.dataset.sym || 'essa moeda';
    if (!confirm('Vender '+sym+' agora?')) return;
    fetch('/api/sell', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({key})})
      .then(r => r.json()).then(j => {
        $('status').textContent = j.ok ? ('venda enviada' + (j.symbol ? ' · '+j.symbol : '')) : (j.error || 'falhou vender');
      }).catch(() => { $('status').textContent = 'falhou vender'; });
    return;
  }
  const b = e.target.closest('[data-copy]');
  if (!b) return;
  const s = b.dataset.copy;
  const done = () => { const old = b.textContent; b.textContent = 'copiado'; setTimeout(() => { b.textContent = old; }, 700); };
  const go = (navigator.clipboard && navigator.clipboard.writeText)
    ? navigator.clipboard.writeText(s) : Promise.reject();
  go.then(done).catch(() => {
    const t = document.createElement('textarea');
    t.value = s; document.body.appendChild(t); t.select();
    document.execCommand('copy'); t.remove(); done();
  });
});

document.getElementById('kol-form').addEventListener('submit', e => {
  e.preventDefault();
  const f = e.target;
  const fd = new FormData(f);
  fetch('/api/kols', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      action: 'add',
      address: String(fd.get('address')||'').trim(),
      handle: String(fd.get('handle')||'').trim(),
      class: String(fd.get('klass')||'kol')
    })}).then(r => r.json()).then(j => { f.reset(); paintKols(j.kols); });
});

document.getElementById('fomo-form').addEventListener('submit', e => {
  e.preventDefault();
  const f = e.target;
  const fd = new FormData(f);
  const err = $('fomo-err');
  err.textContent = '';
  fetch('/api/fomo', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      action: 'add',
      name: String(fd.get('handle')||'').trim(),
      handle: String(fd.get('handle')||'').trim(),
      address: String(fd.get('address')||'').trim()
    })}).then(r => r.json()).then(j => {
      if (!j.ok) { err.textContent = j.error || 'wallet nao gravou — cola o 0x completo'; return; }
      f.reset();
      paintFomo(j.fomo);
    }).catch(() => { err.textContent = 'falhou o save'; });
});

function sendInspect(addr) {
  if (!addr) return;
  fetch('/api/inspect', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({address: addr})}).then(tick);
}
$('paste-ca').addEventListener('click', async () => {
  let v = $('ca-in').value.trim();
  if (!v && navigator.clipboard && navigator.clipboard.readText) {
    try { v = (await navigator.clipboard.readText()).trim(); $('ca-in').value = v; } catch (e) {}
  }
  sendInspect(v);
});
$('ca-in').addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); sendInspect($('ca-in').value.trim()); }
});

function paintKols(list) {
  rows($('kols'), list||[], k => `<tr>
    <td class="sym">${k.handle || '—'}</td>
    <td class="muted">${k.class || 'kol'}</td>
    <td class="muted">${k.chain || '—'}</td>
    <td class="ca">${caHead(k.address)} ${copyBtn(k.address)}</td>
    <td><button class="copy" type="button" data-remove="${k.address}">remove</button></td>
  </tr>`, 'none followed', 5);
}

function paintFomo(list) {
  rows($('fomo-rows'), list||[], k => `<tr>
    <td class="sym">${k.name || k.handle || '—'}</td>
    <td class="muted">${k.chain || '—'}</td>
    <td class="ca">${caHead(k.address)} ${copyBtn(k.address)}</td>
    <td><button class="copy" type="button" data-fomo-rm="${k.address}">remove</button></td>
  </tr>`, 'nenhum perfil', 4);
}

function paintPnlWindows(w) {
  const el = $('pnl-windows');
  if (!el) return;
  const wins = (w && w.windows) || [];
  const sig = wins.map(x => x.label+':'+(x.pnl_usd||0)+':'+(x.partial?'1':'0')).join('|');
  if (el.dataset.sig === sig) return;
  el.dataset.sig = sig;
  el.innerHTML = wins.map(x => {
    const lab = x.partial
      ? x.label+' (parcial, '+x.history_days+' dias de histórico)'
      : x.label;
    return '<div class="pw"><div class="muted">'+esc(lab)+'</div><div class="n '+cls(x.pnl_usd)+'">'+usd(x.pnl_usd||0)+'</div></div>';
  }).join('');
}
const clock = ms => ms ? new Date(ms).toLocaleString(undefined, {year:'numeric', month:'short', day:'numeric', hour:'2-digit', minute:'2-digit', second:'2-digit'}) : '—';
function exitLegsHtml(legs) {
  if (!legs || !legs.length) return '';
  return '<ul class="exit-legs">'+legs.map(l => {
    const amt = l.usd_out != null ? usd(l.usd_out) : (l.size_usd != null ? usd(l.size_usd) : '—');
    const mcap = l.mcap != null ? mcapTxt(l.mcap) : '—';
    const pnl = l.pnl_usd != null ? (' · pnl '+usd(l.pnl_usd)) : '';
    const tape = (l.vol5m != null ? ' · vol5m '+usd(l.vol5m) : '')
      +(l.holder_growth_5m != null ? ' · hg5m '+Number(l.holder_growth_5m).toFixed(2) : '');
    return '<li>'+clock(l.ts_ms)+' · vendeu '+amt+pnl+' · mcap '+mcap+tape+(l.reason ? ' · '+esc(l.reason) : '')+(l.note ? ' · '+esc(l.note) : '')+'</li>';
  }).join('')+'</ul>';
}
let lastPnlSig = '';
function paintPnl(chart) {
  const cards = $('pnl-cards');
  if (!cards) return;
  const sig = String((chart&&chart.total)||0)+':'+( ((chart||{}).series||{}).total||[] ).length;
  if (sig === lastPnlSig) return;
  lastPnlSig = sig;
  const by = (chart && chart.by_chain) || {};
  const tot = chart ? chart.total : 0;
  const cell = (label, n) => `<div class="card"><h2>${label}</h2><div class="n ${cls(n)}">${usd(n||0)}</div></div>`;
  cards.innerHTML = cell('total', tot) + cell('solana', by.solana) + cell('bnb', by.bnb) + cell('robinhood', by.robinhood_chain);
  const svg = $('pnl-chart');
  let pts = ((chart||{}).series||{}).total || [];
  if (!pts.length) {
    svg.innerHTML = '<text x="20" y="120" fill="#6a6a6a" font-size="14">sem trades ainda</text>';
    return;
  }
  if (pts.length === 1) pts = [{t: pts[0].t, y: 0}, pts[0]];
  const ys = pts.map(p => p.y);
  const min = Math.min(0, ...ys), max = Math.max(0, ...ys);
  const span = (max - min) || 1;
  const w = 800, h = 240, pad = 16;
  const xy = pts.map((p,i) => {
    const x = pad + (i/(pts.length-1))*(w-2*pad);
    const y = pad + (1 - (p.y-min)/span)*(h-2*pad);
    return x.toFixed(1)+','+y.toFixed(1);
  });
  const zero = pad + (1 - (0-min)/span)*(h-2*pad);
  const last = pts[pts.length-1].y;
  const color = last>0 ? '#3dff9a' : last<0 ? '#ff3b4e' : '#c8c8c8';
  svg.innerHTML = '<line x1="0" y1="'+zero+'" x2="'+w+'" y2="'+zero+'" stroke="#222" />'
    + '<polyline fill="none" stroke="'+color+'" stroke-width="2" points="'+xy.join(' ')+'"/>';
}

async function tick() {
  if (inflight) { queued = true; return; }
  inflight = true;
  let d;
  try { d = await (await fetch('/api?'+Date.now(), {cache:'no-store'})).json(); }
  catch (err) { $('status').textContent = 'offline'; inflight = false; queued = false; return; }
  const live = d.running ? (d.mode || 'run') : 'stopped';
  const halt = d.halted ? (' · paused ' + (d.halt_reason || '')) : '';
  const learn = d.aggressive ? (' · aggressive ' + (d.aggressive_closes||0) + ' closes') : '';
  const dead = (d.dead_loops || []).length ? (' · loop dead ' + d.dead_loops.join(',')) : '';
  $('status').textContent = live + halt + learn + dead
    + ((d.watch_in||d.watch_out) ? (' · +'+(d.watch_in||0)+'/−'+(d.watch_out||0)) : '')
    + ' · ' + d.stale_s + 's';
  $('status').className = (d.dead_loops || []).length ? 'dn' : 'muted';
  const fault = $('fault');
  const msgs = Object.entries(d.faults || {}).map(([k,v]) => k + ': ' + v);
  if (msgs.length) { fault.textContent = msgs.join(' · '); fault.className = 'on'; }
  else { fault.textContent = ''; fault.className = ''; }
  setText('equity', usd(d.equity_usd), 'n');
  setText('pnl', usd(d.pnl), 'n '+cls(d.pnl));
  setText('winrate', d.win_rate == null ? '—' : Math.round(d.win_rate*100)+'%',
    'n ' + (d.win_rate == null ? '' : d.win_rate >= 0.5 ? 'up' : 'dn'));
  setText('avghold', d.avg_hold_min == null ? '—' : mins(d.avg_hold_min), 'n');
  setText('holding', (d.holding||0) + (d.holding ? '  '+usd(d.holding_usd) : ''), 'n gold');
  setText('watching', String(d.watching||0), 'n cyan');
  paintVerdict(d);
  const uni = $('universe');
  const uniHtml = (d.universe.chains||[]).map(chainCard).join('');
  if (uni.innerHTML !== uniHtml) uni.innerHTML = uniHtml;
  const incoming = d.watch || [];
  if (incoming.length || !lastWatch.length || !d.running || !(d.watching > 0)) lastWatch = incoming;
  paintWatch(lastWatch, d.running);
  paintCoin();
  paintHoldTable(d.holds||[]);
  rows($('sold'), d.sold||[], t => `<tr>
    <td>
      <div class="sym">${t.symbol || caHead((t.key||'').split(':')[1])}</div>
      <div class="ca">${chainShort(t.chain||'')} · ${caHead((t.key||'').split(':')[1])} ${copyBtn((t.key||'').split(':')[1])}</div>
    </td>
    <td>${usd(t.size_usd)}</td>
    <td class="pnl ${cls(t.pnl_usd)}">${usd(t.pnl_usd)} <span class="muted">${pct(t.pnl_pct)}</span></td>
    <td class="mcap-path">${mcapPath(t.mcap_entry, t.mcap_exit)}</td>
    <td>${t.hold_min != null ? mins(t.hold_min) : '—'}</td>
    <td class="muted" title="${esc(t.exit_why||'')}">${t.exit}
      <div class="meta">${esc(t.exit_why||'')}</div>
      ${t.exit_note ? '<div class="meta">'+esc(t.exit_note)+'</div>' : ''}
      <div class="meta">${clock(t.closed_at_ms)}</div>
      ${exitLegsHtml(t.exit_legs)}
    </td></tr>`, 'none closed', 6);
  paintKols(d.kols);
  paintFomo(d.fomo);
  paintPnl(d.pnl_chart);
  paintPnlWindows(d.pnl_windows);
  inflight = false;
  if (queued) { queued = false; tick(); }
}
const OUT_LAB = {fumble:'fumble', rejeicao_correta:'rejeição correta', entrada_correta:'entrada correta',
  entrada_errada:'entrada errada', tracking:'tracking'};
const when = ms => new Date(ms).toLocaleString(undefined, {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'});
let revBusy = false;
async function loadReview() {
  if (revBusy) return;
  revBusy = true;
  const out = $('rev-out').value;
  try {
    const j = await (await fetch('/api/review?outcome='+encodeURIComponent(out)+'&limit=250', {cache:'no-store'})).json();
    $('rev-meta').textContent = (j.open_rows||0)+' shadows abertos · '+(j.open_keys||0)+' tokens no lote de 30 · fumble ≥ '+Math.round((j.fumble_pct||0.2)*100)+'% MFE';
    const body = (j.rows||[]).map((r,i) => {
      const ca = (r.key||'').split(':')[1]||'';
      const res = r.action==='enter'
        ? (usd(r.pnl_usd)+' · '+(r.error_class||''))
        : (r.mfe==null ? '—' : pct(r.mfe));
      const feat = (r.contrib||[]).map(kv => kv[0]+' '+(kv[1]>=0?'+':'')+Number(kv[1]).toFixed(2)).join(' · ') || '—';
      const why = [r.exit_note, r.exit_why, r.error_why, r.reason_why].filter((s,i,a) => s && a.indexOf(s)===i).join(' · ');
      const soldBits = r.action==='enter'
        ? ('<div class="meta">'+clock(r.closed_at_ms || r.ts_ms)+'</div>'+exitLegsHtml(r.exit_legs))
        : '';
      return `<tr class="rev-row" data-rev="${i}"><td>${when(r.ts_ms)}</td>
        <td><div class="sym">${esc(r.symbol||caHead(ca))}</div><div class="ca">${chainShort(r.chain||'')} ${esc(caHead(ca))} ${copyBtn(ca)}</div></td>
        <td>${esc(r.action)}<div class="meta">${esc(r.reason||'')}${r.ticks>1 ? (' · '+r.ticks+' ticks') : ''}</div></td>
        <td>${pct(r.p)} / ${pct(r.ev)}</td>
        <td>${res}${soldBits}</td>
        <td class="out-${esc(r.outcome)}">${OUT_LAB[r.outcome]||r.outcome}</td></tr>
        <tr class="rev-feat" data-rev-feat="${i}" hidden><td colspan="6">${esc(why ? why+' · ' : '')}${esc(feat)}</td></tr>`;
    }).join('') || '<tr><td colspan="6" class="muted">vazio</td></tr>';
    $('review').querySelector('tbody').innerHTML = body;
  } catch (err) {
    $('rev-meta').textContent = 'falhou o review';
  }
  revBusy = false;
}
$('rev-out').addEventListener('change', loadReview);
tick();
setInterval(tick, 400);
</script>
</body>
</html>
"""


def serve(
    state_dir: Path,
    store: Store,
    equity: float,
    port: int,
    settings: Settings | None = None,
    strategy: Config | None = None,
) -> None:
    def snapshot() -> bytes:
        # Reuse the process Store. Opening sqlite on every /api poll made the
        # visor hitch and fight the engine for preview.json.
        return json.dumps(assemble(state_dir, store, equity, settings, strategy)).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/review":
                q = parse_qs(parsed.query)
                try:
                    payload = store.review_history(
                        outcome=(q.get("outcome") or [""])[0],
                        fumble_pct=float((q.get("fumble") or ["0.20"])[0] or 0.20),
                        limit=int((q.get("limit") or ["250"])[0] or 250),
                    )
                    body = json.dumps(payload).encode()
                except Exception as exc:  # noqa: BLE001
                    body = json.dumps({"error": str(exc)}).encode()
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api":
                try:
                    body = snapshot()
                except Exception as exc:  # noqa: BLE001
                    body = json.dumps({"error": str(exc)}).encode()
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(HTML.encode())

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n).decode() if n else "{}")
            except ValueError:
                body = {}
            if path == "/api/sell":
                key = str(body.get("key") or body.get("address") or "").strip()
                result = enqueue_manual_sell(state_dir, key)
                code = 200 if result.get("ok") else 409
                out = json.dumps(result).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(out)
                return
            if path == "/api/flatten":
                if strategy is None:
                    out = json.dumps({"ok": False, "error": "no strategy"}).encode()
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(out)
                    return
                RiskEngine(strategy, store).engage_kill_switch("vender tudo")
                out = json.dumps({"ok": True, "halted": True}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(out)
                return
            if path == "/api/inspect":
                addr = str(body.get("address") or "").strip()
                pending: list = []
                inspect = state_dir / "inspect.json"
                if inspect.exists():
                    try:
                        pending = json.loads(inspect.read_text(encoding="utf-8"))
                    except (ValueError, OSError):
                        pending = []
                    if not isinstance(pending, list):
                        pending = [pending]
                if addr and addr not in pending:
                    pending.append(addr)
                inspect.write_text(json.dumps(pending), encoding="utf-8")
                out = json.dumps({"ok": True, "queued": pending}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(out)
                return
            if path == "/api/fomo":
                rows = load_fomo(state_dir)
                action = str(body.get("action") or "")
                raw = str(body.get("address") or "").strip()
                name = str(body.get("name") or body.get("handle") or "").strip().lstrip("@")
                if action == "add":
                    addrs = [
                        a
                        for a in raw.replace(";", " ").replace(",", " ").split()
                        if len(a) >= 32 or (a.startswith("0x") and len(a) >= 40)
                    ]
                    if action == "add" and not addrs:
                        out = json.dumps({"ok": False, "error": "need a wallet", "fomo": rows}).encode()
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(out)
                        return
                    for addr in addrs:
                        rows = [r for r in rows if str(r.get("address")).lower() != addr.lower()]
                        chain = "evm" if addr.startswith("0x") else "solana"
                        rows.append(
                            {
                                "address": addr,
                                "name": name,
                                "handle": name,
                                "class": "fomo",
                                "source": "fomo",
                                "chain": chain,
                                "chase": True,
                            }
                        )
                elif action == "remove" and raw:
                    rows = [r for r in rows if str(r.get("address")).lower() != raw.lower()]
                save_fomo(state_dir, rows)
                out = json.dumps({"ok": True, "fomo": rows}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(out)
                return
            if path != "/api/kols":
                self.send_error(404)
                return
            rows = load_kols(state_dir)
            action = str(body.get("action") or "")
            addr = str(body.get("address") or "").strip()
            if action == "add" and addr:
                rows = [r for r in rows if str(r.get("address")).lower() != addr.lower()]
                rows.append(
                    {
                        "address": addr,
                        "handle": str(body.get("handle") or "").strip(),
                        "class": str(body.get("class") or "kol"),
                        "source": "manual",
                        "chase": True,
                    }
                )
            elif action == "remove" and addr:
                rows = [r for r in rows if str(r.get("address")).lower() != addr.lower()]
            save_kols(state_dir, rows)
            out = json.dumps({"ok": True, "kols": rows}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"preview  http://127.0.0.1:{port}/   (ctrl+c to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        httpd.server_close()
        store.close()
