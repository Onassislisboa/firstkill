"""Luminos-style distribution read: Bundled / Cabaled / Organic / Unverified.

Public heuristics from https://luminos.capital/guide — not a scrape, not a
price call. Organic means "no manipulation pattern found", never "buy".
Missing data only pushes toward caution, never toward Organic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .models import Features


# Cluster / time-node / co-buy ladder (Luminos: 5% cabal, 20% bundled).
CLUSTER_CABAL = 0.05
CLUSTER_BUNDLED = 0.20
# Ownership cards.
TOP10_CAUTION, TOP10_FLAG = 0.30, 0.50
FRESH_CAUTION, FRESH_FLAG = 0.20, 0.30
# Fresh wallets are expected on a 10-minute pump.fun mint.
FRESH_MIN_AGE_MIN = 180.0
BOT_CABAL = 0.35


@dataclass(slots=True)
class DistRead:
    label: str = "unverified"  # bundled | cabaled | organic | unverified
    fit: int = 0  # 0–100 how strongly it fits the label, not risk
    risk: int = 0  # rug-danger; bundling axis only
    confidence: str = "low"  # high | medium | low
    sources: int = 0
    signals: list[str] = field(default_factory=list)
    hard: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def classify(
    features: Features,
    unknown: set[str] | None = None,
    *,
    age_minutes: float = 0.0,
) -> DistRead:
    unknown = unknown or set()
    f = features

    def known(name: str) -> bool:
        return name not in unknown

    sources = 0
    if known("liquidity_usd"):
        sources += 1
    if known("cluster_pct") or known("top1_pct") or known("top10_pct"):
        sources += 1
    if known("bot_share") or known("axiom_share"):
        sources += 1
    confidence = "high" if sources >= 3 else "medium" if sources == 2 else "low"

    dist_ready = known("cluster_pct") or known("bundle_pct")
    thin_market = known("liquidity_usd") and f.liquidity_usd < 3000
    if thin_market or not dist_ready:
        return DistRead(
            label="unverified",
            fit=0,
            risk=0,
            confidence=confidence,
            sources=sources,
            signals=["data pending" if not dist_ready else "no real market yet"],
        )

    bundled: list[str] = []
    cabal: list[str] = []
    cabal_n = 0
    hard = False

    if known("cluster_pct"):
        if f.cluster_pct > CLUSTER_BUNDLED:
            bundled.append(f"cluster {f.cluster_pct:.0%} linked")
            hard = True
        elif f.cluster_pct >= CLUSTER_CABAL:
            cabal.append(f"cluster {f.cluster_pct:.0%}")
            cabal_n += 1

    if known("bundle_pct"):
        if f.bundle_pct > CLUSTER_BUNDLED:
            bundled.append(f"launch bundle {f.bundle_pct:.0%}")
            hard = True
        elif f.bundle_pct >= CLUSTER_CABAL:
            cabal.append(f"co-buy {f.bundle_pct:.0%}")
            cabal_n += 1

    kol_covers = (
        known("known_holder_pct")
        and known("top1_pct")
        and f.known_holder_pct >= f.top1_pct * 0.6
        and f.top1_pct > 0
    )

    if known("top10_pct") and not kol_covers:
        if f.top10_pct > TOP10_FLAG:
            cabal.append(f"top10 {f.top10_pct:.0%}")
            cabal_n += 2
        elif f.top10_pct >= TOP10_CAUTION:
            cabal.append(f"top10 {f.top10_pct:.0%}")
            cabal_n += 1

    if known("top1_pct") and not kol_covers and f.top1_pct > 0.28:
        cabal.append(f"unknown top1 {f.top1_pct:.0%}")
        cabal_n += 1

    if (
        known("fresh_wallet_pct")
        and age_minutes >= FRESH_MIN_AGE_MIN
        and f.fresh_wallet_pct > FRESH_FLAG
    ):
        bundled.append(f"fresh wallets {f.fresh_wallet_pct:.0%}")

    if known("bot_share") and f.bot_share > BOT_CABAL:
        cabal.append(f"one terminal {f.bot_share:.0%}")
        cabal_n += 1

    if (
        known("unique_buyers_5m")
        and known("net_inflow_usd_5m")
        and f.unique_buyers_5m <= 2
        and f.net_inflow_usd_5m >= 50_000
    ):
        bundled.append(f"{f.unique_buyers_5m:.0f} buyers / ${f.net_inflow_usd_5m:,.0f} inflow")
        hard = True

    # One soft signal is not a classification. A hard trigger is.
    if hard or len(bundled) >= 2:
        fit = min(100, 55 + 15 * len(bundled) + (20 if hard else 0))
        return DistRead(
            label="bundled",
            fit=fit,
            risk=fit,  # only bundling feeds risk
            confidence=confidence,
            sources=sources,
            signals=bundled + cabal,
            hard=hard,
        )
    if cabal_n >= 2:
        fit = min(100, 40 + 12 * len(cabal))
        return DistRead(
            label="cabaled",
            fit=fit,
            risk=0,
            confidence=confidence,
            sources=sources,
            signals=cabal,
        )

    # Would-be Organic with thin holder data stays grey.
    if not (known("cluster_pct") and (known("top10_pct") or known("top1_pct"))):
        return DistRead(
            label="unverified",
            fit=0,
            risk=0,
            confidence=confidence,
            sources=sources,
            signals=["distribution incomplete"],
        )

    spread = 0.0
    if known("cluster_pct"):
        spread = max(spread, f.cluster_pct)
    if known("top10_pct"):
        spread = max(spread, max(0.0, f.top10_pct - 0.20))
    fit = max(0, min(100, int(round(100 - 180 * spread))))
    return DistRead(
        label="organic",
        fit=fit,
        risk=0,
        confidence=confidence,
        sources=sources,
        signals=["no coordinated-distribution pattern"],
    )


def security_cert(
    label: str,
    vetoes: list[str] | None = None,
    mint_revoked: bool | None = None,
) -> str:
    """ok / no / ? — visor badge. Organic passes; bundled/cabaled/live mint fail."""
    if mint_revoked is False:
        return "no"
    if (label or "") in ("bundled", "cabaled"):
        return "no"
    blob = " ".join(vetoes or []).lower()
    if any(w in blob for w in ("bundled", "cabaled", "honeypot", "mint authority", "freeze")):
        return "no"
    if label == "organic":
        return "ok"
    return "?"


def bot_veto(read: DistRead, chain: str) -> str | None:
    """What the bot does with a read. Organic is never a buy signal."""
    if read.label == "bundled":
        return "bundled: " + (read.signals[0] if read.signals else "manufactured supply")
    if read.label == "cabaled":
        return "cabaled: " + (read.signals[0] if read.signals else "insider float")
    if read.label == "unverified" and chain == "solana":
        return "unverified: skip"
    return None
