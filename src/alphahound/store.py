"""SQLite persistence.

One file, WAL mode, no ORM. The write volume of a bot that takes a handful of
positions an hour does not justify a database server, and a single .db file is
the difference between "I can inspect last Tuesday" and "the logs rotated".

The one non-obvious table is `shadow`: every REJECTED candidate is tracked for
`learning.shadow_track_minutes` so the counterfactual PnL of our own filters is
measurable. A bot that only records trades it took can never discover that its
gates are too tight - it will just get quieter and quieter and call that
discipline.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import IO, Any, Iterable

from .models import (
    Action,
    Chain,
    Decision,
    ErrorClass,
    ExitReason,
    Features,
    TradeRecord,
    VenueId,
    describe_code,
    describe_error,
    describe_exit,
    legs_for_display,
    now_ms,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    key TEXT NOT NULL,
    chain TEXT NOT NULL,
    symbol TEXT,
    action TEXT NOT NULL,
    probability REAL NOT NULL,
    expected_value REAL NOT NULL,
    size_usd REAL NOT NULL,
    signal_price REAL NOT NULL,
    features TEXT NOT NULL,
    contributions TEXT NOT NULL,
    weights_version INTEGER NOT NULL,
    reason TEXT,
    unknown TEXT NOT NULL DEFAULT '[]',
    first_ts_ms INTEGER,
    last_ts_ms INTEGER,
    ticks INTEGER NOT NULL DEFAULT 1,
    first_mcap_usd REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_decisions_key ON decisions(key);
CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts_ms);

CREATE TABLE IF NOT EXISTS shadow (
    decision_id INTEGER PRIMARY KEY,
    key TEXT NOT NULL,
    opened_at_ms INTEGER NOT NULL,
    price_at_decision REAL NOT NULL,
    best_price REAL NOT NULL,
    worst_price REAL NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0,
    counterfactual_pct REAL
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    chain TEXT NOT NULL,
    venue TEXT NOT NULL,
    opened_at_ms INTEGER NOT NULL,
    closed_at_ms INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    signal_price REAL NOT NULL,
    size_usd REAL NOT NULL,
    pnl_usd REAL NOT NULL,
    fees_usd REAL NOT NULL,
    exit_reason TEXT NOT NULL,
    error_class TEXT NOT NULL,
    mfe REAL NOT NULL,
    mae REAL NOT NULL,
    entry_slippage REAL NOT NULL,
    features TEXT NOT NULL,
    weights_version INTEGER NOT NULL,
    notes TEXT,
    unknown TEXT NOT NULL DEFAULT '[]',
    symbol TEXT NOT NULL DEFAULT '',
    mcap_entry_usd REAL NOT NULL DEFAULT 0,
    mcap_exit_usd REAL NOT NULL DEFAULT 0,
    exit_legs TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_trades_closed ON trades(closed_at_ms);

CREATE TABLE IF NOT EXISTS weights (
    version INTEGER PRIMARY KEY,
    ts_ms INTEGER NOT NULL,
    payload TEXT NOT NULL,
    samples INTEGER NOT NULL,
    holdout_logloss REAL,
    active INTEGER NOT NULL DEFAULT 0,
    note TEXT
);

-- Parameters the postmortem loop is allowed to nudge, with an audit trail of
-- why. Nothing changes a live tunable without a row here explaining itself.
CREATE TABLE IF NOT EXISTS param_overrides (
    name TEXT PRIMARY KEY,
    value REAL NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS param_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    name TEXT NOT NULL,
    old_value REAL,
    new_value REAL NOT NULL,
    reason TEXT
);

-- Learned terminal attribution: address -> (label, class).
CREATE TABLE IF NOT EXISTS terminal_map (
    address TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    class TEXT NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0,
    updated_at_ms INTEGER NOT NULL
);

-- Wallets we independently measured, rather than wallets someone tweeted.
CREATE TABLE IF NOT EXISTS wallets (
    address TEXT PRIMARY KEY,
    chain TEXT NOT NULL,
    trades INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    pnl_usd REAL NOT NULL DEFAULT 0,
    is_smart INTEGER NOT NULL DEFAULT 0,
    updated_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def lock_state_dir(state_dir: Path) -> IO[bytes]:
    """Refuse to run two engines against one state directory.

    Two engines sharing a wallet size positions independently, so a 5% cap
    becomes 10% of equity, both believe they own the position, and both try to
    exit it. An OS-level lock rather than a pidfile: the lock dies with the
    process, so a crashed bot can restart without manual cleanup - which is
    what a pidfile gets wrong, and it gets it wrong at 3am.

    Caller keeps the handle open for the lifetime of the process.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    handle = open(state_dir / "engine.lock", "a+b")
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(
            f"another alphahound engine is already running against {state_dir}"
        ) from exc
    return handle


def mcap_from_reason(reason: str) -> float:
    """Parse 'mcap: 71695 below …' from a gate. 0 if that isn't the reason."""
    parts = (reason or "").split()
    if len(parts) < 2 or not parts[0].startswith("mcap"):
        return 0.0
    try:
        return float(parts[1].replace(",", ""))
    except ValueError:
        return 0.0


def episode_kind(action: str, reason: str) -> tuple[str, str]:
    """Gate/score class, not the floats in the message.

    `probability 0.389 < 0.420` and `probability 0.396 < 0.420` are one episode.
    `cluster:` vs `liquidity:` are two.
    """
    r = (reason or "").strip()
    if ":" in r:
        return action, r.split(":", 1)[0].strip().lower()
    return action, (r.split(None, 1)[0].lower() if r else "")


def features_from_json(payload: str) -> Features:
    data = json.loads(payload)
    known = set(Features.names())
    return Features(**{k: float(v) for k, v in data.items() if k in known})


def unknown_from_json(payload: str | None) -> set[str]:
    if not payload:
        return set()
    try:
        return {str(name) for name in json.loads(payload)}
    except ValueError:
        return set()


def _legs_from_row(row: sqlite3.Row) -> list[dict]:
    if "exit_legs" not in row.keys():
        return []
    raw = row["exit_legs"]
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except ValueError:
        return []
    return data if isinstance(data, list) else []


class Store:
    def __init__(self, state_dir: Path) -> None:
        state_dir.mkdir(parents=True, exist_ok=True)
        self.path = state_dir / "alphahound.db"
        self.conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        # CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so a
        # column added later never appears in a database that already exists.
        for table, column, ddl in (
            ("decisions", "unknown", "TEXT NOT NULL DEFAULT '[]'"),
            ("decisions", "first_ts_ms", "INTEGER"),
            ("decisions", "last_ts_ms", "INTEGER"),
            ("decisions", "ticks", "INTEGER NOT NULL DEFAULT 1"),
            ("decisions", "first_mcap_usd", "REAL NOT NULL DEFAULT 0"),
            ("trades", "unknown", "TEXT NOT NULL DEFAULT '[]'"),
            ("trades", "symbol", "TEXT NOT NULL DEFAULT ''"),
            ("trades", "mcap_entry_usd", "REAL NOT NULL DEFAULT 0"),
            ("trades", "mcap_exit_usd", "REAL NOT NULL DEFAULT 0"),
            ("trades", "exit_legs", "TEXT NOT NULL DEFAULT '[]'"),
        ):
            existing = {
                r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if column not in existing:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def close(self) -> None:
        self.conn.close()

    # -- decisions ---------------------------------------------------------
    # Same (key, action, reason) within this gap is one episode. Wider than
    # one missed rescore; narrower than a token leaving the visor and returning.
    EPISODE_GAP_MS = 120_000

    def record_decision(self, decision: Decision) -> int:
        """Insert on state change; bump ticks on the live episode otherwise.

        New episode: action/reason changed, or the last row for this key is
        older than EPISODE_GAP_MS with no open shadow (left the radar).
        ENTER always inserts. One shadow per reject episode, not per tick.
        """
        ts = decision.ts_ms
        key = decision.candidate.key
        action = decision.action.value
        reason = decision.reason or ""
        first_mcap = self._first_seen_mcap(key, decision.candidate.mcap_usd)
        if decision.action is not Action.ENTER:
            prev = self.conn.execute(
                """SELECT id, action, IFNULL(reason,'') AS reason,
                          COALESCE(last_ts_ms, ts_ms) AS last_ts_ms
                   FROM decisions WHERE key = ? ORDER BY id DESC LIMIT 1""",
                (key,),
            ).fetchone()
            if prev is not None and episode_kind(prev["action"], prev["reason"]) == episode_kind(
                action, reason
            ):
                open_shadow = self.conn.execute(
                    "SELECT 1 FROM shadow WHERE key = ? AND resolved = 0 LIMIT 1",
                    (key,),
                ).fetchone()
                if open_shadow is not None or (ts - int(prev["last_ts_ms"])) < self.EPISODE_GAP_MS:
                    self.conn.execute(
                        """UPDATE decisions
                           SET last_ts_ms = ?, ticks = COALESCE(ticks, 1) + 1,
                               first_ts_ms = COALESCE(first_ts_ms, ts_ms)
                           WHERE id = ?""",
                        (ts, int(prev["id"])),
                    )
                    return int(prev["id"])
        cur = self.conn.execute(
            """INSERT INTO decisions (ts_ms, key, chain, symbol, action, probability,
                   expected_value, size_usd, signal_price, features, contributions,
                   weights_version, reason, unknown, first_ts_ms, last_ts_ms, ticks,
                   first_mcap_usd)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ts,
                key,
                decision.candidate.chain.value,
                decision.candidate.symbol,
                action,
                decision.score.probability,
                decision.score.expected_value,
                decision.size_usd,
                decision.candidate.price_usd,
                json.dumps(decision.features.as_dict()),
                json.dumps(decision.score.contributions),
                decision.weights_version,
                decision.reason,
                json.dumps(sorted(decision.unknown)),
                ts,
                ts,
                1,
                first_mcap,
            ),
        )
        decision_id = int(cur.lastrowid or 0)
        if decision.action is not Action.ENTER and decision.candidate.price_usd > 0:
            self.open_shadow(decision_id, key, decision.candidate.price_usd)
        return decision_id

    def _first_seen_mcap(self, key: str, live: float) -> float:
        """Mcap on first visor sight of this mint. Later episodes keep that number."""
        row = self.conn.execute(
            "SELECT first_mcap_usd, reason FROM decisions WHERE key = ? ORDER BY id ASC LIMIT 1",
            (key,),
        ).fetchone()
        if row is None:
            return float(live or 0.0)
        stored = float(row["first_mcap_usd"] or 0.0)
        if stored > 0:
            return stored
        return mcap_from_reason(row["reason"] or "")

    def _first_mcap_by_key(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self.conn.execute(
            "SELECT key, first_mcap_usd, reason FROM decisions ORDER BY id"
        ):
            k = r["key"]
            if k in out:
                continue
            m = float(r["first_mcap_usd"] or 0.0) or mcap_from_reason(r["reason"] or "")
            out[k] = m
        return out

    def recent_decision_keys(self, since_ms: int) -> set[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT key FROM decisions WHERE ts_ms >= ?", (since_ms,)
        ).fetchall()
        return {r["key"] for r in rows}

    # -- shadow tracking ---------------------------------------------------
    def open_shadow(self, decision_id: int, key: str, price: float) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO shadow
               (decision_id, key, opened_at_ms, price_at_decision, best_price, worst_price)
               VALUES (?,?,?,?,?,?)""",
            (decision_id, key, now_ms(), price, price, price),
        )

    def open_shadows(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM shadow WHERE resolved = 0").fetchall()

    def update_shadow(self, decision_id: int, price: float) -> None:
        self.conn.execute(
            """UPDATE shadow
               SET best_price = MAX(best_price, ?), worst_price = MIN(worst_price, ?)
               WHERE decision_id = ?""",
            (price, price, decision_id),
        )

    def resolve_shadow(self, decision_id: int, counterfactual_pct: float) -> None:
        self.conn.execute(
            "UPDATE shadow SET resolved = 1, counterfactual_pct = ? WHERE decision_id = ?",
            (counterfactual_pct, decision_id),
        )

    def filter_cost_report(self, limit: int = 200) -> list[sqlite3.Row]:
        """Rejected candidates that would have won, most profitable first. This
        is the report that tells you which gate is quietly costing you money."""
        return self.conn.execute(
            """SELECT d.action, d.reason, d.probability, d.expected_value,
                      s.counterfactual_pct, d.key, d.symbol, d.features
               FROM shadow s JOIN decisions d ON d.id = s.decision_id
               WHERE s.resolved = 1
               ORDER BY s.counterfactual_pct DESC LIMIT ?""",
            (limit,),
        ).fetchall()

    def review_history(
        self,
        *,
        outcome: str = "",
        fumble_pct: float = 0.20,
        limit: int = 250,
    ) -> dict[str, Any]:
        """Decisions + shadow MFE + closed trades, classified for the visor.

        One row per reject decision (or closed trade). Does not write.
        Default includes unresolved shadows as tracking so the tab matches the session.
        """
        outcome = (outcome or "").strip().lower()
        fumble_pct = float(fumble_pct)
        limit = max(1, min(int(limit), 1000))
        open_n, open_keys = self.conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT key) FROM shadow WHERE resolved = 0"
        ).fetchone()

        def contribs(raw: str) -> list[list]:
            try:
                data = json.loads(raw or "{}")
            except ValueError:
                return []
            if not isinstance(data, dict):
                return []
            items = [(str(k), float(v)) for k, v in data.items() if not str(k).startswith("_")]
            items.sort(key=lambda kv: -abs(kv[1]))
            return [[k, round(v, 4)] for k, v in items[:24]]

        first_mcap = self._first_mcap_by_key()
        rows: list[dict[str, Any]] = []
        want_rej = outcome in ("", "fumble", "rejeicao_correta", "tracking")
        want_ent = outcome in ("", "entrada_correta", "entrada_errada")

        if want_rej:
            sql = """SELECT d.ts_ms, d.key, d.symbol, d.chain, d.action, d.reason,
                            d.probability, d.expected_value, d.contributions,
                            s.resolved, s.counterfactual_pct,
                            COALESCE(d.ticks, 1) AS ticks,
                            COALESCE(d.first_ts_ms, d.ts_ms) AS first_ts_ms,
                            COALESCE(d.last_ts_ms, d.ts_ms) AS last_ts_ms
                     FROM decisions d JOIN shadow s ON s.decision_id = d.id"""
            params: list[Any] = []
            if outcome == "fumble":
                sql += " WHERE s.resolved = 1 AND IFNULL(s.counterfactual_pct, 0) >= ?"
                params.append(fumble_pct)
            elif outcome == "rejeicao_correta":
                sql += " WHERE s.resolved = 1 AND IFNULL(s.counterfactual_pct, 0) < ?"
                params.append(fumble_pct)
            elif outcome == "tracking":
                sql += " WHERE s.resolved = 0"
            else:
                # Default "todos": live tracking + resolved. Resolved-only made
                # the tab look frozen for `shadow_track_minutes` (3h).
                sql += " WHERE 1=1"
            sql += " ORDER BY d.ts_ms DESC LIMIT ?"
            params.append(limit)
            for r in self.conn.execute(sql, params):
                cf = float(r["counterfactual_pct"] or 0.0)
                resolved = int(r["resolved"] or 0)
                if resolved:
                    kind = "fumble" if cf >= fumble_pct else "rejeicao_correta"
                else:
                    kind = "tracking"
                rows.append(
                    {
                        "ts_ms": int(r["ts_ms"]),
                        "key": r["key"],
                        "symbol": r["symbol"] or "",
                        "chain": r["chain"],
                        "action": r["action"],
                        "reason": r["reason"] or "",
                        "reason_why": describe_code(r["reason"] or ""),
                        "ticks": int(r["ticks"] or 1),
                        "first_ts_ms": int(r["first_ts_ms"] or r["ts_ms"]),
                        "last_ts_ms": int(r["last_ts_ms"] or r["ts_ms"]),
                        "p": round(float(r["probability"] or 0.0), 4),
                        "ev": round(float(r["expected_value"] or 0.0), 4),
                        "mfe": round(cf, 4) if resolved else None,
                        "error_class": "",
                        "pnl_usd": None,
                        "outcome": kind,
                        "contrib": contribs(r["contributions"]),
                        "mcap_first": round(first_mcap.get(r["key"], 0.0)),
                    }
                )

        if want_ent:
            enters: dict[str, sqlite3.Row] = {}
            for r in self.conn.execute(
                """SELECT key, probability, expected_value, contributions, reason, ts_ms
                   FROM decisions WHERE action = 'enter' ORDER BY ts_ms"""
            ):
                enters[r["key"]] = r
            for t in self.trades(limit=limit):
                kind = "entrada_correta" if t.pnl_usd > 0 else "entrada_errada"
                if outcome and kind != outcome:
                    continue
                d = enters.get(t.key)
                rows.append(
                    {
                        "ts_ms": int(t.closed_at_ms),
                        "key": t.key,
                        "symbol": t.symbol or "",
                        "chain": t.chain.value,
                        "action": "enter",
                        "reason": (d["reason"] if d else "") or t.exit_reason.value,
                        "reason_why": describe_code(
                            t.exit_reason.value
                            or (d["reason"] if d else "")
                            or t.error_class.value
                        ),
                        "exit_why": describe_exit(t.exit_reason),
                        "error_why": describe_error(t.error_class),
                        "p": round(float(d["probability"] if d else 0.0), 4),
                        "ev": round(float(d["expected_value"] if d else 0.0), 4),
                        "mfe": round(float(t.max_favorable_excursion or 0.0), 4),
                        "error_class": t.error_class.value,
                        "pnl_usd": round(float(t.pnl_usd), 2),
                        "outcome": kind,
                        "contrib": contribs(d["contributions"] if d else "{}"),
                        "closed_at_ms": int(t.closed_at_ms),
                        "mcap_first": round(first_mcap.get(t.key, 0.0)),
                        "mcap_entry": round(t.mcap_entry_usd),
                        "mcap_exit": round(t.mcap_exit_usd),
                        "exit_legs": legs_for_display(t),
                        "exit_note": t.notes or "",
                    }
                )

        rows.sort(key=lambda x: -x["ts_ms"])
        return {
            "rows": rows[:limit],
            "fumble_pct": fumble_pct,
            "open_rows": int(open_n or 0),
            "open_keys": int(open_keys or 0),
        }

    # -- trades ------------------------------------------------------------
    def record_trade(self, trade: TradeRecord) -> int:
        cur = self.conn.execute(
            """INSERT INTO trades (key, chain, venue, opened_at_ms, closed_at_ms,
                   entry_price, exit_price, signal_price, size_usd, pnl_usd, fees_usd,
                   exit_reason, error_class, mfe, mae, entry_slippage, features,
                   weights_version, notes, unknown, symbol, mcap_entry_usd, mcap_exit_usd, exit_legs)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                trade.key,
                trade.chain.value,
                trade.venue.value,
                trade.opened_at_ms,
                trade.closed_at_ms,
                trade.entry_price,
                trade.exit_price,
                trade.signal_price,
                trade.size_usd,
                trade.pnl_usd,
                trade.fees_usd,
                trade.exit_reason.value,
                trade.error_class.value,
                trade.max_favorable_excursion,
                trade.max_adverse_excursion,
                trade.entry_slippage,
                json.dumps(trade.features.as_dict()),
                trade.weights_version,
                trade.notes,
                json.dumps(sorted(trade.unknown)),
                trade.symbol,
                trade.mcap_entry_usd,
                trade.mcap_exit_usd,
                json.dumps(trade.exit_legs or []),
            ),
        )
        return int(cur.lastrowid or 0)

    def trades(self, limit: int | None = None, since_ms: int = 0) -> list[TradeRecord]:
        sql = "SELECT * FROM trades WHERE closed_at_ms >= ? ORDER BY closed_at_ms"
        params: list[Any] = [since_ms]
        if limit:
            sql += " DESC LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        out = [self._row_to_trade(r) for r in rows]
        return list(reversed(out)) if limit else out

    @staticmethod
    def _row_to_trade(row: sqlite3.Row) -> TradeRecord:
        return TradeRecord(
            key=row["key"],
            chain=Chain(row["chain"]),
            venue=VenueId(row["venue"]),
            opened_at_ms=row["opened_at_ms"],
            closed_at_ms=row["closed_at_ms"],
            entry_price=row["entry_price"],
            exit_price=row["exit_price"],
            signal_price=row["signal_price"],
            size_usd=row["size_usd"],
            pnl_usd=row["pnl_usd"],
            fees_usd=row["fees_usd"],
            exit_reason=ExitReason(row["exit_reason"]),
            error_class=ErrorClass(row["error_class"]),
            max_favorable_excursion=row["mfe"],
            max_adverse_excursion=row["mae"],
            entry_slippage=row["entry_slippage"],
            features=features_from_json(row["features"]),
            weights_version=row["weights_version"],
            notes=row["notes"] or "",
            unknown=unknown_from_json(row["unknown"]),
            symbol=row["symbol"] if "symbol" in row.keys() else "",
            mcap_entry_usd=float(row["mcap_entry_usd"] or 0) if "mcap_entry_usd" in row.keys() else 0.0,
            mcap_exit_usd=float(row["mcap_exit_usd"] or 0) if "mcap_exit_usd" in row.keys() else 0.0,
            exit_legs=_legs_from_row(row),
        )

    def trade_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM trades").fetchone()
        return int(row["n"])

    def win_count(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM trades WHERE pnl_usd > 0"
        ).fetchone()
        return int(row["n"])

    def fees_sum(self) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(fees_usd), 0) AS fees FROM trades"
        ).fetchone()
        return float(row["fees"])

    def avg_hold_ms(self) -> float | None:
        row = self.conn.execute(
            "SELECT AVG(closed_at_ms - opened_at_ms) AS ms FROM trades "
            "WHERE closed_at_ms > opened_at_ms"
        ).fetchone()
        return None if row["ms"] is None else float(row["ms"])

    def oldest_close_ms(self) -> int:
        row = self.conn.execute("SELECT MIN(closed_at_ms) AS t FROM trades").fetchone()
        return int(row["t"] or 0)

    def realized_pnl(self, since_ms: int = 0) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(pnl_usd), 0) AS pnl FROM trades WHERE closed_at_ms >= ?",
            (since_ms,),
        ).fetchone()
        return float(row["pnl"])

    def mint_traded_since(self, key: str, since_ms: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM trades WHERE key = ? AND closed_at_ms >= ? LIMIT 1",
            (key, since_ms),
        ).fetchone()
        return row is not None

    def enter_count_since(self, since_ms: int) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM decisions WHERE action = ? AND ts_ms >= ?",
            (Action.ENTER.value, since_ms),
        ).fetchone()
        return int(row["n"])

    def consecutive_losses(self) -> int:
        rows = self.conn.execute(
            "SELECT pnl_usd FROM trades ORDER BY closed_at_ms DESC LIMIT 20"
        ).fetchall()
        n = 0
        for r in rows:
            if r["pnl_usd"] >= 0:
                break
            n += 1
        return n

    def error_class_counts(self, since_ms: int = 0) -> dict[str, int]:
        rows = self.conn.execute(
            """SELECT error_class, COUNT(*) AS n FROM trades
               WHERE closed_at_ms >= ? GROUP BY error_class""",
            (since_ms,),
        ).fetchall()
        return {r["error_class"]: int(r["n"]) for r in rows}

    # -- weights -----------------------------------------------------------
    def save_weights(
        self,
        payload: dict[str, Any],
        samples: int,
        holdout_logloss: float | None,
        note: str = "",
        activate: bool = False,
    ) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(version), 0) AS v FROM weights").fetchone()
        version = int(row["v"]) + 1
        self.conn.execute(
            """INSERT INTO weights (version, ts_ms, payload, samples, holdout_logloss, active, note)
               VALUES (?,?,?,?,?,?,?)""",
            (version, now_ms(), json.dumps(payload), samples, holdout_logloss, 0, note),
        )
        if activate:
            self.activate_weights(version)
        return version

    def activate_weights(self, version: int) -> None:
        self.conn.execute("UPDATE weights SET active = 0")
        self.conn.execute("UPDATE weights SET active = 1 WHERE version = ?", (version,))

    def active_weights(self) -> tuple[int, dict[str, Any]] | None:
        row = self.conn.execute("SELECT version, payload FROM weights WHERE active = 1").fetchone()
        if not row:
            return None
        return int(row["version"]), json.loads(row["payload"])

    def weights_version_row(self, version: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM weights WHERE version = ?", (version,)).fetchone()

    def pnl_by_weights_version(self, version: int) -> tuple[int, float]:
        row = self.conn.execute(
            """SELECT COUNT(*) AS n, COALESCE(SUM(pnl_usd), 0) AS pnl
               FROM trades WHERE weights_version = ?""",
            (version,),
        ).fetchone()
        return int(row["n"]), float(row["pnl"])

    # -- tunable parameters ------------------------------------------------
    def param(self, name: str, default: float) -> float:
        row = self.conn.execute(
            "SELECT value FROM param_overrides WHERE name = ?", (name,)
        ).fetchone()
        return float(row["value"]) if row else default

    def set_param(self, name: str, value: float, reason: str) -> None:
        old = self.conn.execute(
            "SELECT value FROM param_overrides WHERE name = ?", (name,)
        ).fetchone()
        self.conn.execute(
            """INSERT INTO param_overrides (name, value, updated_at_ms, reason)
               VALUES (?,?,?,?)
               ON CONFLICT(name) DO UPDATE SET value=excluded.value,
                   updated_at_ms=excluded.updated_at_ms, reason=excluded.reason""",
            (name, value, now_ms(), reason),
        )
        self.conn.execute(
            """INSERT INTO param_history (ts_ms, name, old_value, new_value, reason)
               VALUES (?,?,?,?,?)""",
            (now_ms(), name, float(old["value"]) if old else None, value, reason),
        )

    def all_params(self) -> dict[str, float]:
        rows = self.conn.execute("SELECT name, value FROM param_overrides").fetchall()
        return {r["name"]: float(r["value"]) for r in rows}

    def param_history(self, limit: int = 50) -> list[sqlite3.Row]:
        # id breaks ties: two changes inside the same millisecond are common
        # during a learning cycle, and ordering by timestamp alone makes their
        # order undefined.
        return self.conn.execute(
            "SELECT * FROM param_history ORDER BY ts_ms DESC, id DESC LIMIT ?", (limit,)
        ).fetchall()

    # -- terminal attribution ---------------------------------------------
    def terminal_labels(self) -> dict[str, tuple[str, str]]:
        rows = self.conn.execute("SELECT address, label, class FROM terminal_map").fetchall()
        return {r["address"]: (r["label"], r["class"]) for r in rows}

    def label_terminal(self, address: str, label: str, klass: str) -> None:
        self.conn.execute(
            """INSERT INTO terminal_map (address, label, class, hits, updated_at_ms)
               VALUES (?,?,?,0,?)
               ON CONFLICT(address) DO UPDATE SET label=excluded.label,
                   class=excluded.class, updated_at_ms=excluded.updated_at_ms""",
            (address, label, klass, now_ms()),
        )

    # -- smart money -------------------------------------------------------
    def smart_wallets(self, chain: Chain) -> set[str]:
        rows = self.conn.execute(
            "SELECT address FROM wallets WHERE chain = ? AND is_smart = 1", (chain.value,)
        ).fetchall()
        return {r["address"] for r in rows}

    def wallet_record(self, address: str) -> dict | None:
        row = self.conn.execute(
            "SELECT trades, wins, pnl_usd, is_smart FROM wallets WHERE address = ?",
            (address,),
        ).fetchone()
        if row is None:
            return None
        return {
            "trades": int(row["trades"]),
            "wins": int(row["wins"]),
            "pnl_usd": float(row["pnl_usd"]),
            "is_smart": bool(row["is_smart"]),
        }

    def record_buyer_outcome(self, address: str, chain: Chain, pnl_usd: float) -> None:
        """Promote a wallet to smart money only after it shows up on winners.

        One lucky sniper on a single win is not a KOL. Two winning appearances
        with net-positive PnL is the cheapest filter that is not a tweet list.
        """
        row = self.conn.execute(
            "SELECT trades, wins, pnl_usd FROM wallets WHERE address = ?", (address,)
        ).fetchone()
        trades = (int(row["trades"]) if row else 0) + 1
        wins = (int(row["wins"]) if row else 0) + (1 if pnl_usd > 0 else 0)
        pnl = (float(row["pnl_usd"]) if row else 0.0) + pnl_usd
        is_smart = wins >= 2 and pnl > 0 and wins / trades >= 0.5
        self.upsert_wallet(address, chain, trades, wins, pnl, is_smart)

    def upsert_wallet(
        self, address: str, chain: Chain, trades: int, wins: int, pnl_usd: float, is_smart: bool
    ) -> None:
        self.conn.execute(
            """INSERT INTO wallets (address, chain, trades, wins, pnl_usd, is_smart, updated_at_ms)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(address) DO UPDATE SET trades=excluded.trades,
                   wins=excluded.wins, pnl_usd=excluded.pnl_usd,
                   is_smart=excluded.is_smart, updated_at_ms=excluded.updated_at_ms""",
            (address, chain.value, trades, wins, pnl_usd, int(is_smart), now_ms()),
        )

    # -- kv ----------------------------------------------------------------
    def get_kv(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_kv(self, key: str, value: str) -> None:
        self.conn.execute(
            """INSERT INTO kv (key, value) VALUES (?,?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            (key, value),
        )

    def executemany(self, sql: str, rows: Iterable[tuple]) -> None:
        self.conn.executemany(sql, rows)
