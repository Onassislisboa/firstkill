"""Launch origin: launchpad tokens only, per chain.

Solana is pump.fun. BNB is four.meme. Robinhood Chain is Pons only — not
Uniswap handmade pools, not Pools.trade. The brokerage API is majors.
"""

from __future__ import annotations

from .models import Candidate, Chain
from .settings import Config


def _norm(address: str) -> str:
    return address.lower() if address.startswith("0x") else address


def launchpad_origin(candidate: Candidate, strategy: Config) -> tuple[bool, str]:
    """Whether this token came from a launchpad we trade on its chain."""
    if not bool(strategy.get("launchpads.require_launchpad", True)):
        return True, "launchpad filter off"

    if candidate.chain is Chain.ROBINHOOD_BROKER:
        return False, "robinhood brokerage is majors, not launchpads"

    if candidate.source in {"pumpfun_stream", "pump.fun"}:
        if candidate.chain is Chain.SOLANA:
            return True, "pump.fun stream"
        return False, "pump.fun stream on a non-solana chain"
    if candidate.chain is Chain.ROBINHOOD_CHAIN:
        dex = (candidate.dex_id or "").lower()
        if candidate.source == "hood_stream":
            if dex in {"", "pons"}:
                return True, "pons factory"
            return False, "not a pons launch"
        # Pons V1 seeds a Uniswap V3 pool; Dexscreener labels that `uniswap`.
        # The engine confirms the mint on the Pons factory before it sits.
        if dex in {"pons", "uniswap"}:
            return True, "hood pons venue"
        return False, f"handmade pool on {dex or 'unknown'}"

    cfg = strategy.section(f"launchpads.{candidate.chain.value}")
    if not cfg:
        return False, f"no launchpads configured for {candidate.chain.value}"

    suffixes = tuple(str(s).lower() for s in (cfg.get("mint_suffixes") or []))
    mint = candidate.address.lower()
    for suffix in suffixes:
        if mint.endswith(suffix):
            return True, f"mint suffix .{suffix}"

    dex_ids = {str(d).lower() for d in (cfg.get("dex_ids") or [])}
    dex = (candidate.dex_id or "").lower()
    if dex in dex_ids:
        return True, f"dex {dex}"

    if dex:
        return False, f"handmade pool on {dex}"
    return False, "unknown origin, not a known launchpad"


def known_holder_share(holders, known: set[str]) -> float:
    """Circulating share held by labeled KOLs / learned smart wallets."""
    if not known:
        return 0.0
    circ = [h for h in holders if not (h.is_lp or h.is_burn)]
    total = sum(h.balance for h in circ)
    if total <= 0:
        return 0.0
    known_n = {_norm(a) for a in known}
    held = sum(h.balance for h in circ if _norm(h.address) in known_n)
    return held / total
