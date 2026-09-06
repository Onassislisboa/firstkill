"""Environment + strategy configuration.

No pydantic, no dotenv, no yaml. A .env file is 20 lines of parsing and TOML
reading has been in the standard library since 3.11, so the dependency would
buy nothing but an install step.
"""

from __future__ import annotations

import copy
import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .log import get
from .models import Chain, now_ms

REPO_ROOT = Path(__file__).resolve().parents[2]

# Good enough to make the on-chain features exist while paper trading, and
# rejected outright for live trading by Settings.validate().
PUBLIC_SOLANA_RPC = "https://api.mainnet-beta.solana.com"
log = get("settings")


def load_dotenv(path: Path | None = None, *, override: bool = False) -> None:
    path = path or REPO_ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key) or default)
    except ValueError:
        return default


class Config:
    """Read-only nested config with dotted lookup.

    `cfg["risk.equity_usd"]` beats `cfg["risk"]["equity_usd"]` at every call
    site, and a missing key raising here rather than three frames deeper is
    worth the twelve lines.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def load(cls, path: Path) -> Config:
        with path.open("rb") as fh:
            return cls(tomllib.load(fh))

    def __getitem__(self, dotted: str) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"missing config key: {dotted}")
            node = node[part]
        return node

    def get(self, dotted: str, default: Any = None) -> Any:
        try:
            return self[dotted]
        except KeyError:
            return default

    def overlay(self, nested: dict[str, Any]) -> Config:
        data = copy.deepcopy(self._data)

        def merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
            for key, value in src.items():
                if isinstance(value, dict) and isinstance(dst.get(key), dict):
                    merge(dst[key], value)
                else:
                    dst[key] = value

        merge(data, nested)
        return Config(data)

    def section(self, name: str) -> dict[str, Any]:
        value = self.get(name, {})
        return value if isinstance(value, dict) else {}

    @property
    def raw(self) -> dict[str, Any]:
        return self._data


@dataclass(slots=True)
class Settings:
    mode: str = "paper"
    enabled_chains: list[Chain] = field(default_factory=lambda: [Chain.SOLANA])
    state_dir: Path = field(default_factory=lambda: REPO_ROOT / "state")
    log_level: str = "INFO"

    solana_rpc_url: str = ""
    solana_ws_url: str = ""
    solana_private_key: str = ""
    jupiter_api_key: str = ""
    pumpportal_ws_url: str = ""

    evm_private_key: str = ""
    evm_keys: dict[Chain, str] = field(default_factory=dict)
    rpc_urls: dict[Chain, str] = field(default_factory=dict)
    zeroex_api_key: str = ""
    rh_chain_swap_router: str = ""
    rh_chain_quoter: str = ""
    rh_chain_weth: str = ""

    rh_api_key: str = ""
    rh_private_key: str = ""

    birdeye_api_key: str = ""
    helius_api_key: str = ""
    cope_api_key: str = ""
    twitter_bearer: str = ""
    bubblemaps_api_key: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    @property
    def live(self) -> bool:
        return self.mode == "live"

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        chains: list[Chain] = []
        for token in (_env("ENABLED_CHAINS", "solana")).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                chains.append(Chain(token))
            except ValueError as exc:
                raise ValueError(
                    f"ENABLED_CHAINS contains unknown chain {token!r}; "
                    f"valid values: {[c.value for c in Chain]}"
                ) from exc

        state_dir = Path(_env("STATE_DIR", "state"))
        if not state_dir.is_absolute():
            state_dir = REPO_ROOT / state_dir

        return cls(
            mode=_env("MODE", "paper").lower(),
            enabled_chains=chains or [Chain.SOLANA],
            state_dir=state_dir,
            log_level=_env("LOG_LEVEL", "INFO").upper(),
            solana_rpc_url=_env("SOLANA_RPC_URL"),
            solana_ws_url=_env("SOLANA_WS_URL"),
            solana_private_key=_env("SOLANA_PRIVATE_KEY"),
            jupiter_api_key=_env("JUPITER_API_KEY"),
            pumpportal_ws_url=_env("PUMPPORTAL_WS_URL"),
            evm_private_key=_env("EVM_PRIVATE_KEY"),
            evm_keys={
                Chain.BNB: _env("BNB_PRIVATE_KEY") or _env("EVM_PRIVATE_KEY"),
                Chain.BASE: _env("BASE_PRIVATE_KEY") or _env("EVM_PRIVATE_KEY"),
                Chain.ROBINHOOD_CHAIN: _env("ROBINHOOD_CHAIN_PRIVATE_KEY")
                or _env("EVM_PRIVATE_KEY"),
            },
            rpc_urls={
                Chain.BNB: _env("BNB_RPC_URL"),
                Chain.BASE: _env("BASE_RPC_URL"),
                Chain.ROBINHOOD_CHAIN: _env("ROBINHOOD_CHAIN_RPC_URL"),
            },
            zeroex_api_key=_env("ZEROEX_API_KEY"),
            rh_chain_swap_router=_env("RH_CHAIN_SWAP_ROUTER"),
            rh_chain_quoter=_env("RH_CHAIN_QUOTER_V2"),
            rh_chain_weth=_env("RH_CHAIN_WETH"),
            rh_api_key=_env("RH_API_KEY"),
            rh_private_key=_env("RH_PRIVATE_KEY"),
            birdeye_api_key=_env("BIRDEYE_API_KEY"),
            helius_api_key=_env("HELIUS_API_KEY"),
            cope_api_key=_env("COPE_API_KEY"),
            twitter_bearer=_env("TWITTER_BEARER_TOKEN") or _env("X_BEARER_TOKEN"),
            bubblemaps_api_key=_env("BUBBLEMAPS_API_KEY"),
            telegram_bot_token=_env("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=_env("TELEGRAM_CHAT_ID"),
        )

    def validate(self) -> list[str]:
        """Return blocking problems. Paper mode is permissive on purpose; live
        mode is not, because the failure mode of a half-configured live bot is
        a wallet, not a stack trace."""
        problems: list[str] = []
        if self.mode not in {"paper", "live"}:
            problems.append(f"MODE must be paper or live, got {self.mode!r}")
        if not self.live:
            return problems

        if Chain.SOLANA in self.enabled_chains:
            if not self.solana_rpc_url:
                problems.append("SOLANA_RPC_URL is required for live Solana trading")
            if PUBLIC_SOLANA_RPC in self.solana_rpc_url:
                problems.append(
                    "SOLANA_RPC_URL points at the public endpoint. It will rate-limit "
                    "you out of every trade worth making. Use a paid RPC."
                )
            if not self.solana_private_key:
                problems.append("SOLANA_PRIVATE_KEY is required for live Solana trading")
            if not self.jupiter_api_key:
                problems.append("JUPITER_API_KEY is required (api.jup.ag/swap/v2)")

        evm_chains = [c for c in self.enabled_chains if c in self.rpc_urls]
        missing_evm = [c.value for c in evm_chains if not self.evm_key_for(c)]
        if missing_evm:
            problems.append(f"missing EVM private key for {', '.join(missing_evm)}")
        for chain in evm_chains:
            if not self.rpc_urls.get(chain):
                problems.append(f"missing RPC URL for {chain.value}")
        if Chain.BNB in evm_chains or Chain.BASE in evm_chains:
            if not self.zeroex_api_key:
                problems.append("ZEROEX_API_KEY is required for BNB/Base routing")
        if Chain.ROBINHOOD_CHAIN in evm_chains and not self.rh_chain_swap_router:
            problems.append("RH_CHAIN_SWAP_ROUTER is required for Robinhood Chain")

        if Chain.ROBINHOOD_BROKER in self.enabled_chains:
            if not (self.rh_api_key and self.rh_private_key):
                problems.append("RH_API_KEY and RH_PRIVATE_KEY are required")
        return problems

    def evm_key_for(self, chain: Chain | None = None) -> str:
        if chain is None:
            return self.evm_private_key
        return (self.evm_keys.get(chain) or self.evm_private_key or "").strip()

    def wallet_pubkeys(self) -> dict[str, str]:
        """Public addresses only. Never returns a secret."""
        out: dict[str, str] = {}
        if self.solana_private_key:
            try:
                from solders.keypair import Keypair

                out["solana"] = str(Keypair.from_base58_string(self.solana_private_key).pubkey())
            except Exception:  # noqa: BLE001
                out["solana"] = "set"
        for chain in (Chain.BNB, Chain.BASE, Chain.ROBINHOOD_CHAIN):
            key = self.evm_key_for(chain)
            if not key:
                continue
            try:
                from eth_account import Account

                out[chain.value] = Account.from_key(key).address
            except Exception:  # noqa: BLE001
                out[chain.value] = "set"
        return out


def load_strategy(path: Path | None = None) -> Config:
    return Config.load(path or REPO_ROOT / "config" / "strategy.toml")


AGGRESSIVE_STARTED_KEY = "aggressive_started_ms"
AGGRESSIVE_GRADUATED_KEY = "aggressive_graduated"


def _slot_band(
    equity: float,
    min_usd: float,
    prod_n: int,
    prod_min: float,
    prod_max: float,
    want: int,
) -> tuple[int, float, float]:
    """Same total cap as prod_n * prod_max, more seats only if min size still fits."""
    n = max(prod_n, int(want))
    while n > prod_n:
        max_pct = prod_max * prod_n / n
        min_pct = prod_min * prod_n / n
        if equity > 0:
            min_pct = max(min_pct, min_usd / equity)
        if min_pct <= max_pct + 1e-12:
            return n, min_pct, max_pct
        n -= 1
    return prod_n, prod_min, prod_max


def apply_aggressive_learning(strategy: Config, store: Any) -> Config:
    """Overlay the learning profile. Production scoring/gates in toml stay put."""
    sec = strategy.section("aggressive_learning")
    if not sec.get("enabled"):
        return strategy

    need = int(sec.get("graduate_at_closes", 40))
    closed = int(store.trade_count())
    if closed >= need:
        if store.get_kv(AGGRESSIVE_GRADUATED_KEY) != "1":
            store.set_kv(AGGRESSIVE_GRADUATED_KEY, "1")
            log.info(
                "aggressive_learning graduated",
                extra={"closed": closed, "need": need},
            )
        return strategy

    if not store.get_kv(AGGRESSIVE_STARTED_KEY):
        store.set_kv(AGGRESSIVE_STARTED_KEY, str(now_ms()))
        log.info("aggressive_learning on", extra={"closed": closed, "need": need})

    prod_n = int(strategy.get("risk.max_concurrent_positions", 2))
    equity = float(strategy.get("risk.equity_usd", 0)) + float(store.realized_pnl())
    n, min_pct, max_pct = _slot_band(
        max(0.0, equity),
        float(strategy.get("risk.min_position_usd", 10)),
        prod_n,
        float(strategy.get("risk.min_position_pct", 0.06)),
        float(strategy.get("risk.max_position_pct", 0.16)),
        int(sec.get("max_concurrent_positions", 4)),
    )
    return strategy.overlay(
        {
            "aggressive_learning": {"_active": True},
            "risk": {
                "max_concurrent_positions": n,
                "max_positions_per_chain": n,
                "min_position_pct": min_pct,
                "max_position_pct": max_pct,
            },
        }
    )


def score_floors(strategy: Config, store: Any) -> tuple[float, float]:
    """p/EV used to enter. Aggressive ignores self-tune so learning can actually enter."""
    prod_p = float(strategy.get("scoring.min_probability", 0.56))
    prod_ev = float(strategy.get("scoring.min_expected_value", 0.035))
    if strategy.get("aggressive_learning._active"):
        sec = strategy.section("aggressive_learning")
        return (
            float(sec.get("min_probability", prod_p)),
            float(sec.get("min_expected_value", prod_ev)),
        )
    return (
        store.param("scoring.min_probability", prod_p),
        store.param("scoring.min_expected_value", prod_ev),
    )


def aggressive_closes_since_on(store: Any) -> int:
    started = store.get_kv(AGGRESSIVE_STARTED_KEY)
    if not started:
        return 0
    try:
        since = int(started)
    except ValueError:
        return 0
    row = store.conn.execute(
        "SELECT COUNT(*) AS n FROM trades WHERE closed_at_ms >= ?",
        (since,),
    ).fetchone()
    return int(row["n"])


def load_terminals(path: Path | None = None) -> Config:
    return Config.load(path or REPO_ROOT / "config" / "terminals.toml")


def load_whales(path: Path | None = None) -> list[dict[str, Any]]:
    """Labeled wallets pasted from Moby / Fomo. Empty file is fine."""
    path = path or REPO_ROOT / "config" / "whales.toml"
    if not path.exists():
        return []
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return [dict(row) for row in (data.get("wallet") or []) if row.get("address")]


def whale_addresses(rows: list[dict[str, Any]], *, source: str | None = None) -> set[str]:
    out: set[str] = set()
    for row in rows:
        if source and str(row.get("source", "")).lower() != source:
            continue
        if row.get("chase", True) is False:
            continue
        addr = str(row.get("address") or "").strip()
        if addr:
            out.add(addr)
    return out


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    rows = data if isinstance(data, list) else []
    return [dict(r) for r in rows if str(r.get("address") or "").strip()]


def _save_json_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_kols(state_dir: Path) -> list[dict[str, Any]]:
    return _load_json_rows(state_dir / "kols.json")


def save_kols(state_dir: Path, rows: list[dict[str, Any]]) -> None:
    _save_json_rows(state_dir / "kols.json", rows)


def load_fomo(state_dir: Path) -> list[dict[str, Any]]:
    return _load_json_rows(state_dir / "fomo.json")


def save_fomo(state_dir: Path, rows: list[dict[str, Any]]) -> None:
    _save_json_rows(state_dir / "fomo.json", rows)


def chase_rows(state_dir: Path) -> list[dict[str, Any]]:
    return load_whales() + load_kols(state_dir) + load_fomo(state_dir)


def crowd_addresses(rows: list[dict[str, Any]], kind: str) -> set[str]:
    """kind='fomo' or 'whale'. chase=false wallets are known but not copied."""
    out: set[str] = set()
    for row in rows:
        if row.get("chase", True) is False:
            continue
        addr = str(row.get("address") or "").strip()
        if not addr:
            continue
        source = str(row.get("source") or "").lower()
        klass = str(row.get("class") or "").lower()
        is_fomo = source == "fomo" or klass == "fomo"
        if kind == "fomo" and is_fomo:
            out.add(addr)
        elif kind == "whale" and not is_fomo:
            out.add(addr)
    return out
