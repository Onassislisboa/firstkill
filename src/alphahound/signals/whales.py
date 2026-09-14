"""Whale and Fomo-profile reads.

MobyScreener does not publish an API (checked: docs are the consumer app,
mobyscreener.com/api is an HTML landing page). What it *shows* - who is inside,
what % they hold, whether they are buying or selling - is on-chain and we
already fetch the holders and the recent trades. This module answers those
three questions against a labeled wallet set (paste from Moby) plus size-based
whales.

Fomo is the same story on the execution side (no trading API) and a different
one on the research side: Cope Capital maps Fomo handles to wallets. That is
optional. Without a key, only wallets you labeled `source = "fomo"` count.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

from ..models import Side
from .distribution import Holder
from .flow import Trade


@dataclass(slots=True)
class CrowdRead:
    """One crowd (whales, or Fomo profiles worth chasing) inside a token."""

    inside: int = 0
    hold_pct: float = 0.0
    buy_usd: float = 0.0
    sell_usd: float = 0.0
    wallets: tuple[str, ...] = ()

    @property
    def net_flow(self) -> float:
        denom = self.buy_usd + self.sell_usd
        if denom <= 0:
            return 0.0
        return (self.buy_usd - self.sell_usd) / denom


def _norm(address: str) -> str:
    return address.lower() if address.startswith("0x") else address


def crowd_read(
    holders: list[Holder],
    trades: list[Trade],
    labeled: set[str],
    *,
    size_pct: float = 0.0,
    supply: float = 0.0,
) -> CrowdRead:
    """Holdings and recent flow for a labeled crowd.

    `size_pct` > 0 also treats an unlabeled holder as a whale when they own
    at least that share of circulating supply. Deployer and LP are never the
    crowd we want to chase. `supply` is mint supply when we have it; without
    it, % is vs the holder list (top-20 looks like 100% whales).
    """
    labeled_n = {_norm(a) for a in labeled if a}
    circ = [h for h in holders if not (h.is_lp or h.is_burn or h.is_deployer)]
    listed = sum(h.balance for h in circ)
    total = supply if supply > 0 else listed
    members: dict[str, float] = {}
    if total > 0:
        for h in circ:
            pct = h.balance / total
            if _norm(h.address) in labeled_n or (size_pct > 0 and pct >= size_pct):
                members[_norm(h.address)] = pct
    buy = sell = 0.0
    for t in trades:
        if not t.wallet or _norm(t.wallet) not in members:
            continue
        if t.side is Side.BUY:
            buy += t.size_usd
        else:
            sell += t.size_usd
    return CrowdRead(
        inside=len(members),
        hold_pct=sum(members.values()),
        buy_usd=buy,
        sell_usd=sell,
        wallets=tuple(members),
    )


def who_inside(
    holders: list[Holder],
    trades: list[Trade],
    labeled: dict[str, str],
) -> list[str]:
    """Handles from `labeled` (norm-addr → name) found in holders or recent flow."""
    names: list[str] = []
    seen: set[str] = set()
    circ = sorted(
        (h for h in holders if not (h.is_lp or h.is_burn or h.is_deployer)),
        key=lambda h: -h.balance,
    )
    for h in circ:
        key = _norm(h.address)
        if key in labeled and key not in seen:
            seen.add(key)
            names.append(labeled[key])
    for t in trades:
        if not t.wallet:
            continue
        key = _norm(t.wallet)
        if key in labeled and key not in seen:
            seen.add(key)
            names.append(labeled[key])
    return names


def wallets_in(obj: Any, *, skip: set[str] | None = None) -> set[str]:
    """Pull wallet-shaped strings out of a nested dict (Cope payloads)."""
    skip_n = {a for a in (skip or set()) if a}
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                kl = str(key).lower()
                if kl in {"wallet", "address", "pubkey", "solana_wallet"} and isinstance(
                    value, str
                ):
                    if 32 <= len(value) <= 64 and value not in skip_n:
                        found.add(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)
    return found
