"""Gates, the scoring model, and expected value.

Three stages, in this order, and the order is the whole design:

1. GATES - absolute vetoes. Unsurvivable risks (rug, honeypot, no exit) are not
   traded off against upside, because a total loss is not a large loss.
2. SCORE - a logistic model over normalized features, giving a win probability.
3. EXPECTED VALUE - probability combined with the realized win/loss
   distribution, minus every cost we can predict.

Stage 3 is the one most bots skip, and skipping it is why they lose money while
their signal looks great: a 60%-win-rate strategy that pays 6% round-trip on a
7% average win is a machine for converting conviction into fees.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import TYPE_CHECKING

from .log import get
from .models import Features, Position, Score, TradeRecord, now_ms
from .verdict import bot_veto, classify
from .portfolio import banked_from_peak
from .settings import Config, score_floors
from .origin import launchpad_origin
from .playbook import gate as pb_gate
from .playbook import section as pb_section
from .store import Store
from .rubric import grade

if TYPE_CHECKING:
    # Import-time only. Keeps the scoring path free of the HTTP stack so the
    # model, gates and expected-value maths stay testable with nothing
    # installed.
    from .signals import Enrichment

log = get("scoring")


# ---------------------------------------------------------------------------
# Normalization. A declared transform per feature rather than a fitted scaler.
#
# A fitted scaler is state that drifts, has to be versioned alongside the
# weights, and silently changes the meaning of every stored feature vector when
# it updates. These transforms are fixed, so a feature vector recorded today
# means the same thing to a model trained next month. Each maps its feature to
# roughly [-1, 1] with 0 at "unremarkable".
# ---------------------------------------------------------------------------


def _clip(x: float, lo: float = -1.5, hi: float = 1.5) -> float:
    return max(lo, min(hi, x))


def _tanh(scale: float):
    return lambda x: math.tanh(x * scale)


def _center(center: float, scale: float):
    return lambda x: _clip((x - center) * scale)


def _log_center(center_log: float, scale: float):
    return lambda x: _clip((math.log10(1.0 + max(0.0, x)) - center_log) / scale)


NORMALIZERS: dict[str, object] = {
    # chart
    "ret_5m": _tanh(3.0),
    "ret_15m": _tanh(1.5),
    "vwap_dev": _tanh(3.0),
    "atr_pct": _tanh(10.0),
    "breakout": lambda x: _clip(x, 0.0, 1.0),
    "volume_z": lambda x: _clip(x / 3.0),
    "body_ratio": lambda x: _clip(x),
    "parabolic": lambda x: _clip(x, 0.0, 1.0),
    # distribution
    "holder_count": _log_center(2.0, 1.5),
    "holder_growth_5m": _tanh(0.1),
    "top10_pct": _center(0.35, 2.8),
    "gini": _center(0.60, 2.5),
    "fresh_wallet_pct": _center(0.40, 2.5),
    "bundle_pct": _center(0.10, 4.0),
    "dev_holding_pct": _center(0.0, 10.0),
    "lp_locked_pct": lambda x: _clip(x, 0.0, 1.0),
    "known_holder_pct": _tanh(3.0),
    "top1_pct": _center(0.12, 5.0),
    # terminal attribution
    "retail_share": _center(0.25, 3.0),
    "retail_share_delta_5m": _tanh(10.0),
    "bot_share": _center(0.15, 3.0),
    "axiom_share": _center(0.15, 3.0),
    "axiom_share_delta_5m": _tanh(12.0),
    "unknown_share": lambda x: _clip(x, 0.0, 1.0),
    # flow
    "unique_buyers_5m": _log_center(1.3, 0.7),
    "buy_sell_ratio": lambda x: math.tanh((x - 1.0) * 1.2),
    "net_inflow_usd_5m": _tanh(1.0 / 20_000.0),
    "avg_buy_size_usd": _tanh(1.0 / 500.0),
    "smart_money_buys": _tanh(0.5),
    "fomo_inside": _tanh(0.35),
    "fomo_net_flow": lambda x: _clip(x, -1.0, 1.0),
    "fomo_supply_pct": lambda x: _clip(x, 0.0, 1.0),
    "whale_hold_pct": lambda x: _clip(x, 0.0, 1.0),
    "whale_net_flow": lambda x: _clip(x, -1.0, 1.0),
    "cluster_pct": lambda x: _clip(x, 0.0, 1.0),
    "twitter_mentions": _log_center(0.7, 0.8),
    "twitter_inst": lambda x: _clip(x, 0.0, 1.0),
    "twitter_fresh": lambda x: _clip(x, 0.0, 1.0),
    "copy_signal": lambda x: _clip(x, 0.0, 1.0),
    "dex_profile": lambda x: _clip(x, 0.0, 1.0),
    "is_vamp": lambda x: _clip(x, 0.0, 1.0),
    "is_beta": lambda x: _clip(x, 0.0, 1.0),
    "is_main": lambda x: _clip(x, 0.0, 1.0),
    "main_ret_5m": _tanh(3.0),
    # liquidity / cost
    "liquidity_usd": _log_center(4.3, 0.8),
    "liq_to_mcap": _tanh(8.0),
    "price_impact": _tanh(15.0),
    "round_trip_cost": _tanh(20.0),
    "token_age_minutes": _tanh(1.0 / 60.0),
}


# ---------------------------------------------------------------------------
# Prior weights.
#
# This is the strategy's thesis written as numbers, and it is what the bot
# trades on until it has enough closed trades to have learned anything. Getting
# the signs right by hand matters far more than any later tuning, because a
# model that starts with the wrong signs has to lose real money to discover it.
#
# The load-bearing pair: retail_share is NEGATIVE and retail_share_delta_5m is
# strongly POSITIVE. High retail share means the crowd already arrived; rising
# retail share means it is arriving now. Being early to the second while the
# first is still low is the entire edge.
# ---------------------------------------------------------------------------

PRIOR_WEIGHTS: dict[str, float] = {
    "ret_5m": 0.55,
    "ret_15m": 0.15,
    "vwap_dev": -0.50,
    "atr_pct": -0.20,
    "breakout": 0.70,
    "volume_z": 0.60,
    "body_ratio": 0.40,
    "parabolic": -1.20,
    "holder_count": 0.30,
    "holder_growth_5m": 0.90,
    "top10_pct": -0.90,
    "gini": -0.45,
    "fresh_wallet_pct": -0.70,
    "bundle_pct": -1.10,
    "dev_holding_pct": -1.00,
    "lp_locked_pct": 0.50,
    # High top-10 is a smell, not a death sentence: known KOLs sitting in
    # that top-10 on a *new* launch is the $TRUMP shape; this weight lets it
    # through. Old coins are rejected by the age gate, not by this.
    "known_holder_pct": 1.30,
    "top1_pct": -0.70,
    "retail_share": -0.80,
    "retail_share_delta_5m": 1.40,
    "bot_share": -1.00,
    "axiom_share": -0.35,
    "axiom_share_delta_5m": 0.85,
    "unknown_share": -0.10,
    "unique_buyers_5m": 0.60,
    "buy_sell_ratio": 0.70,
    "net_inflow_usd_5m": 0.50,
    "avg_buy_size_usd": 0.20,
    "smart_money_buys": 1.60,
    "fomo_inside": 1.10,
    "fomo_net_flow": 1.40,
    "fomo_supply_pct": 2.20,
    "whale_hold_pct": 0.70,
    "whale_net_flow": 1.50,
    "cluster_pct": -1.20,
    "twitter_mentions": 0.90,
    "twitter_inst": 1.80,
    "twitter_fresh": 1.20,
    "copy_signal": 1.00,
    "dex_profile": 1.70,
    "is_vamp": -2.20,
    "is_beta": -0.70,
    "is_main": 0.80,
    "main_ret_5m": 0.90,
    "liquidity_usd": 0.40,
    "liq_to_mcap": 0.30,
    "price_impact": -0.90,
    "round_trip_cost": -1.00,
    "token_age_minutes": -0.30,
}

# Pessimistic on purpose. The unconditional probability that a random new token
# is a profitable trade after costs is well under half, so a model whose bias
# implies otherwise is starting from a lie.
PRIOR_BIAS = -1.20


def normalize(features: Features, unknown: set[str] | None = None) -> dict[str, float]:
    """Normalized feature map, with unmeasured features forced to neutral.

    Zeroing unknowns rather than passing raw defaults is what keeps a missing
    data provider from reading as good news. A token with no holder data should
    score as if holder data were irrelevant, not as if it were perfect.
    """
    unknown = unknown or set()
    raw = features.as_dict()
    out: dict[str, float] = {}
    for name, value in raw.items():
        if name in unknown:
            out[name] = 0.0
            continue
        fn = NORMALIZERS.get(name)
        out[name] = float(fn(float(value))) if callable(fn) else _clip(float(value))
    return out


class Model:
    """Logistic model with named weights.

    Weights are stored by name so adding a feature is backward compatible: an
    unknown name gets its prior, a removed name is ignored. Index-based weight
    vectors turn a one-line feature addition into a silent mis-scoring.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        bias: float = PRIOR_BIAS,
        version: int = 0,
    ) -> None:
        self.weights = dict(PRIOR_WEIGHTS)
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items()})
        self.bias = float(bias)
        self.version = version

    # -- io ---------------------------------------------------------------
    def to_payload(self) -> dict:
        return {"weights": self.weights, "bias": self.bias}

    @classmethod
    def from_payload(cls, payload: dict, version: int = 0) -> Model:
        return cls(payload.get("weights"), float(payload.get("bias", PRIOR_BIAS)), version)

    @classmethod
    def load(cls, store: Store) -> Model:
        active = store.active_weights()
        if active is None:
            return cls()
        version, payload = active
        return cls.from_payload(payload, version)

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_payload(), indent=2), encoding="utf-8")

    # -- inference --------------------------------------------------------
    def logit(self, normalized: dict[str, float]) -> tuple[float, dict[str, float]]:
        contributions = {
            name: self.weights.get(name, 0.0) * value
            for name, value in normalized.items()
            if value != 0.0
        }
        return self.bias + sum(contributions.values()), contributions

    def probability(self, normalized: dict[str, float]) -> tuple[float, dict[str, float]]:
        z, contributions = self.logit(normalized)
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))), contributions


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def _live_x(tw: dict) -> bool:
    h = str(tw.get("official") or "").replace("@", "").strip()
    if len(h) < 2 or h.isdigit():
        return False
    age = tw.get("official_age_min")
    if age is not None and float(age) > 360:
        return False
    return True


@dataclass(slots=True)
class Gate:
    name: str
    # Features/measurements this gate needs. If any is unknown the gate cannot
    # be evaluated, and an unevaluable gate is a veto in live mode.
    requires: tuple[str, ...]
    message: str


def evaluate_gates(
    enr: Enrichment, strategy: Config, store: Store, *, live: bool
) -> tuple[list[str], list[str]]:
    """Return (vetoes, abstained).

    `store.param(...)` is consulted for every threshold so the postmortem loop
    can tighten a gate that keeps costing money without a redeploy - with the
    change recorded in param_history, so a mysteriously silent bot has a paper
    trail.
    """

    chain = enr.candidate.chain

    def p(name: str, default: float) -> float:
        return pb_gate(strategy, chain, name, default, store)

    f = enr.features
    vetoes: list[str] = []
    abstained: list[str] = []
    unknown = enr.unknown

    allowed, origin_reason = launchpad_origin(enr.candidate, strategy)
    if not allowed:
        vetoes.append(f"launchpad: {origin_reason}")

    # Age stays on the visor; it is not a buy veto. The scan list is the
    # selection — paper enters what it already chose.
    if not enr.candidate.created_at_ms:
        abstained.append("age(unmeasured)")

    # Playbook-only floors. Skip when the input was never measured so a missing
    # Twitter key cannot veto every Solana mint.
    min_vol = p("min_volume_5m", 0.0)
    vol_seen = enr.snap is not None or enr.candidate.volume_5m_usd > 0
    if min_vol > 0 and vol_seen and enr.candidate.volume_5m_usd < min_vol:
        vetoes.append(f"volume: {enr.candidate.volume_5m_usd:.0f} < {min_vol:.0f} (5m)")
    min_tw = p("min_twitter_mentions", 0.0)
    tw = getattr(enr, "twitter", None) or {}
    live_x = _live_x(tw)
    if min_tw > 0 and "twitter_mentions" not in unknown and f.twitter_mentions < min_tw:
        vetoes.append(f"twitter: {f.twitter_mentions:.0f} mentions (want {min_tw:.0f})")
    if tw.get("official") and tw.get("official_age_min") is not None:
        off_age = float(tw["official_age_min"])
        if off_age > 360:
            vetoes.append(f"twitter: official quiet {off_age:.0f}m")
    copy_mcap = float(pb_section(strategy, chain).get("copy_max_mcap_usd", 2_000_000))
    copy_min = float(pb_section(strategy, chain).get("copy_min_mcap_usd", 100_000))
    ripped = enr.candidate.ret_5m > 0.20 or (
        "parabolic" not in unknown and f.parabolic >= 0.60
    )
    if copy_min > 0 and enr.candidate.mcap_usd > 0 and enr.candidate.mcap_usd < copy_min:
        vetoes.append(f"mcap: {enr.candidate.mcap_usd:.0f} below {copy_min:.0f} floor")
    if copy_mcap > 0 and enr.candidate.mcap_usd > copy_mcap and ripped:
        vetoes.append("priced: mcap already past copy window")
    max_chase = float(pb_section(strategy, chain).get("max_chase_ret_5m", 0) or 0)
    thesis = live_x or (
        "copy_signal" not in unknown and f.copy_signal >= 1.0
    ) or ("twitter_inst" not in unknown and f.twitter_inst >= 0.5)
    if max_chase > 0 and enr.candidate.ret_5m > max_chase and not thesis:
        vetoes.append("chase: 5m ripped, wait dip")
    max_cluster = p("max_cluster_pct", 0.0)
    probed = enr.chain_probed or enr.mint is not None
    if max_cluster > 0 and "cluster_pct" not in unknown and f.cluster_pct > max_cluster:
        vetoes.append(f"cluster: {f.cluster_pct:.0%} linked supply")
    elif (
        probed
        and ((max_cluster > 0 and "cluster_pct" in unknown) or "top1_pct" in unknown)
        and not enr.candidate.dex_paid
    ):
        # Rugs look fine on mcap until the bubble is measured.
        vetoes.append("rug_filter: distribution unmeasured, skip")
    if (
        chain.value == "solana"
        and bool(pb_section(strategy, chain).get("require_sponsor", False))
        and "whale_n" in (getattr(enr, "crowd", None) or {})
        and not _sponsored(enr)
    ):
        # Cheap prefilter has no whale_n, so it still spends RPC. After crowd
        # is read, a Solana mint with no labeled KOL/whale is a skip — bundles
        # on this chain hide inside fresh wallets.
        vetoes.append("sponsor: no labeled KOL or whale")
    if f.is_vamp >= 1:
        vetoes.append("vamp: clone of an earlier ticker")
    if f.is_beta >= 1 and f.main_ret_5m <= -0.20:
        vetoes.append("beta: main runner is dumping")

    def check(name: str, requires: tuple[str, ...], failed: bool, message: str) -> None:
        missing = [r for r in requires if r in unknown]
        if missing:
            abstained.append(f"{name}(unmeasured:{','.join(missing)})")
            return
        if failed:
            vetoes.append(f"{name}: {message}")

    check(
        "liquidity",
        ("liquidity_usd",),
        f.liquidity_usd < p("min_liquidity_usd", 15000.0),
        f"{f.liquidity_usd:.0f} < {p('min_liquidity_usd', 15000.0):.0f}",
    )
    check(
        "holders",
        ("holder_count",),
        f.holder_count < p("min_holder_count", 60),
        f"{f.holder_count:.0f} holders",
    )
    # Concentration is a score when known wallets sit in the top — $TRUMP's
    # top-10 was well above 55% and it was still a real market. The exception
    # is UNKNOWN wallets stacking circulating supply: that is a rug. LP and
    # burn are already excluded from top1/top10 in distribution.analyze.
    unknown_whale = (
        f.top1_pct > p("max_unknown_top1_pct", 0.50)
        and f.known_holder_pct < f.top1_pct * 0.6
    )
    check(
        "unknown_whale",
        ("top1_pct", "known_holder_pct"),
        unknown_whale,
        f"top holder {f.top1_pct:.0%} is not a known KOL/whale",
    )
    unknown_top10 = (
        f.top10_pct > p("max_top10_pct", 0.50)
        and f.known_holder_pct < f.top10_pct * 0.6
    )
    check(
        "top10",
        ("top10_pct", "known_holder_pct"),
        unknown_top10,
        f"top10 {f.top10_pct:.0%} not explained by known wallets",
    )
    check(
        "dev_holding",
        ("dev_holding_pct",),
        f.dev_holding_pct > p("max_dev_holding_pct", 0.05),
        f"dev holds {f.dev_holding_pct:.1%}",
    )
    check(
        "bundle",
        ("bundle_pct",),
        f.bundle_pct > p("max_bundle_pct", 0.25),
        f"launch bundle {f.bundle_pct:.0%}",
    )
    check(
        "fresh_wallets",
        ("fresh_wallet_pct",),
        f.fresh_wallet_pct > p("max_fresh_wallet_pct", 0.70),
        f"{f.fresh_wallet_pct:.0%} fresh wallets",
    )
    check(
        "sniper_bots",
        ("bot_share",),
        f.bot_share > float(strategy.get("terminals.max_sniper_bot_share", 0.35)),
        f"bot share {f.bot_share:.0%}",
    )
    check(
        "crowd_already_here",
        ("retail_share",),
        f.retail_share > float(strategy.get("terminals.max_retail_terminal_share", 0.45)),
        f"retail terminals already at {f.retail_share:.0%} - too late",
    )
    check(
        "round_trip_cost",
        ("round_trip_cost",),
        f.round_trip_cost > p("max_round_trip_cost_pct", 0.06),
        f"round trip costs {f.round_trip_cost:.1%}",
    )
    # Sellability is not a threshold, it is a fact, and it is the one check
    # whose failure mode is losing the entire position rather than some of it.
    if "sellable" in unknown or not enr.round_trip.ok:
        abstained.append("sellable(unmeasured)")
    elif enr.round_trip.sell_slippage > p("max_sell_slippage_pct", 0.12):
        vetoes.append(
            f"sellable: exit slippage {enr.round_trip.sell_slippage:.1%} - "
            "treating as unsellable"
        )

    if strategy.get("gates.require_mint_authority_revoked", True):
        if enr.mint is None:
            abstained.append("mint_authority(unmeasured)")
        elif enr.mint.mint_authority is not None:
            vetoes.append("mint_authority: still live, supply can be inflated")
    if strategy.get("gates.require_freeze_authority_revoked", True):
        if enr.mint is None:
            abstained.append("freeze_authority(unmeasured)")
        elif enr.mint.freeze_authority is not None:
            vetoes.append("freeze_authority: still live, your account can be frozen")

    allow_unmeasured = bool(strategy.get("gates.allow_unmeasured", False))
    if live and abstained and not allow_unmeasured:
        vetoes.append(
            "unmeasured: " + ", ".join(abstained) + " (set gates.allow_unmeasured "
            "to trade blind on purpose)"
        )
    # Cheap prefilter has mint=None / chain_probed=False. Don't grey-skip
    # every mint before holders are fetched.
    if enr.chain_probed or enr.mint is not None:
        extra = bot_veto(
            classify(f, unknown, age_minutes=enr.candidate.age_minutes),
            chain.value,
        )
        if extra and not (
            extra.startswith("unverified:") and enr.candidate.dex_paid
        ):
            vetoes.append(extra)
    return vetoes, abstained


# Chase/priced = don't buy the rip. Round-trip = don't pay a fat spread.
# Stay on the visor and re-score until the print is actually buyable.
PATIENCE_PREFIXES = ("chase:", "priced:", "round_trip_cost:", "twitter:")


def patience_only(reasons: list[str]) -> bool:
    return bool(reasons) and all(
        any(r.startswith(p) for p in PATIENCE_PREFIXES) for r in reasons
    )


def watch_call(*, vetoed: bool, ok: bool, reasons: list[str]) -> str:
    if ok:
        return "trade"
    if vetoed and not patience_only(reasons):
        return "skip"
    return "wait"


def _sponsored(enr: Enrichment) -> bool:
    crowd = getattr(enr, "crowd", None) or {}
    if crowd.get("kols") or int(crowd.get("whale_n") or 0) >= 1:
        return True
    f = enr.features
    unknown = enr.unknown
    if "copy_signal" not in unknown and f.copy_signal >= 1.0:
        return True
    if "whale_net_flow" not in unknown and f.whale_net_flow > 0:
        return True
    return False


# Age/priced/retail fire because the clock moved, not because the thesis died.
# Unmeasured/rug_filter would sell every Solana bag on a public-RPC 429.
_HOLD_IGNORE = (
    "age:",
    "priced:",
    "crowd_already_here",
    "volume:",
    "holders:",
    "launchpad:",
    "twitter:",
    "rug_filter:",
    "unverified:",
    "unmeasured:",
    "round_trip",
    "sniper_bots",
    "fresh_wallets",
    "liquidity:",
    "rubric:",
    "sponsor:",
)


@dataclass(slots=True)
class HoldReview:
    """Stage 3 result. `cut` is set only when the open bag should go."""

    cut: str | None
    rubric: float
    entry_rubric: float
    strikes: int
    why: str


def _hold_lethal(vetoes: list[str]) -> str | None:
    for v in vetoes:
        if any(v.startswith(p) for p in _HOLD_IGNORE):
            continue
        return v
    return None


def _hold_sponsor(position: Position, enr: Enrichment) -> str | None:
    if "top10_pct" in enr.unknown or "top1_pct" in enr.unknown:
        return None
    still = set((enr.crowd or {}).get("kols") or []) | set(
        (enr.crowd or {}).get("wallets") or []
    )
    if position.entry_sponsors and still.isdisjoint(set(position.entry_sponsors)):
        if enr.features.whale_net_flow <= 0 and enr.features.copy_signal < 1:
            return "sponsor: labeled KOL/whale left"
    return None


def _hold_rubric(position: Position, total: float, strategy: Config) -> str | None:
    min_hold = float(strategy.get("hold.min_rubric", 5.5))
    need = int(strategy.get("hold.strikes", 2))
    if total < min_hold:
        position.hold_strikes += 1
        if position.hold_strikes >= need:
            return f"rubric: hold {total:.1f} < {min_hold:.1f}"
        return None
    position.hold_strikes = 0
    return None


def hold_cut(
    position: Position, enr: Enrichment, score: Score, strategy: Config
) -> HoldReview:
    """Stage 3: lethal immediately, thesis (sponsor/rubric) only after grace."""
    total = float((score.rubric or {}).get("total") or 0.0)
    entry = position.entry_rubric
    lethal_grace_ms = int(float(strategy.get("hold.lethal_grace_seconds", 90)) * 1000)
    thesis_grace_ms = int(float(strategy.get("hold.grace_seconds", 90)) * 1000)
    age_ms = now_ms() - position.opened_at_ms

    def review(cut: str | None, why: str) -> HoldReview:
        return HoldReview(
            cut=cut,
            rubric=total,
            entry_rubric=entry,
            strikes=position.hold_strikes,
            why=why,
        )

    if age_ms < lethal_grace_ms:
        return review(None, "grace")

    lethal = _hold_lethal(score.veto_reasons)
    if lethal:
        return review(lethal, lethal)
    if age_ms < thesis_grace_ms:
        return review(None, "grace")
    sponsor = _hold_sponsor(position, enr)
    if sponsor:
        return review(sponsor, sponsor)
    dropped = _hold_rubric(position, total, strategy)
    line = f"{entry:.1f}→{total:.1f}"
    if dropped:
        return review(dropped, line)
    need = int(strategy.get("hold.strikes", 2))
    if position.hold_strikes:
        line = f"{line} ({position.hold_strikes}/{need})"
    return review(None, line)


# ---------------------------------------------------------------------------
# Expected value
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Payoff:
    """The realized shape of wins and losses, which is what turns a probability
    into a decision. Priors come from the exit ladder; once there is history,
    history wins - your stops do not fill where you set them."""

    avg_win: float
    avg_loss: float
    samples: int
    from_history: bool


def payoff_from_config(strategy: Config) -> Payoff:
    """Prior payoff, before there is any history to learn from.

    Summing the whole ladder - `sum((mult - 1) * frac)` - is the tempting
    one-liner and it is wrong: it prices every winner as if it reached the top
    rung. With a 3.5x top rung that triples the prior win and makes the bot
    enter almost anything on day one, which is the exact moment it knows least.
    So the prior weights the rungs by how often a winner is assumed to reach
    them, and runs each case through the real exit policy.
    """
    ladder = [(float(m), float(f)) for m, f in strategy.get("exits.take_profit_ladder", [[1.35, 1.0]])]
    reach = strategy.get("scoring.prior_winner_ladder_reach", [[1, 0.6], [2, 0.25], [3, 0.15]])
    weights = {int(rung): float(w) for rung, w in reach}
    total = sum(weights.values()) or 1.0

    expected_win = 0.0
    for rung, weight in weights.items():
        # A winner that reached rung N peaked at that rung's multiple.
        peak = ladder[min(rung, len(ladder)) - 1][0] - 1.0
        expected_win += (weight / total) * banked_from_peak(peak, strategy)

    stop = float(strategy.get("exits.stop_loss_pct", 0.28))
    return Payoff(avg_win=expected_win, avg_loss=stop, samples=0, from_history=False)


def payoff_from_history(trades: list[TradeRecord], strategy: Config, min_samples: int = 20) -> Payoff:
    wins = [t.pnl_pct for t in trades if t.pnl_usd > 0]
    losses = [-t.pnl_pct for t in trades if t.pnl_usd <= 0]
    if len(wins) < 5 or len(losses) < 5 or len(trades) < min_samples:
        return payoff_from_config(strategy)
    # Median for losses (robust to the one catastrophic rug), mean for wins
    # (the fat right tail is where the strategy's return actually comes from
    # and medianing it away would understate the edge).
    return Payoff(
        avg_win=fmean(wins),
        avg_loss=max(0.01, median(losses)),
        samples=len(trades),
        from_history=True,
    )


def expected_value(probability: float, payoff: Payoff, round_trip_cost: float) -> float:
    """EV as a fraction of position size, net of the measured round trip."""
    return probability * payoff.avg_win - (1.0 - probability) * payoff.avg_loss - round_trip_cost


class Scorer:
    def __init__(self, model: Model, strategy: Config, store: Store, *, live: bool) -> None:
        self.model = model
        self.strategy = strategy
        self.store = store
        self.live = live
        self._payoff = payoff_from_config(strategy)
        self._payoff_trades = 0

    def refresh_payoff(self) -> Payoff:
        n = self.store.trade_count()
        if n != self._payoff_trades:
            self._payoff = payoff_from_history(self.store.trades(limit=300), self.strategy)
            self._payoff_trades = n
        return self._payoff

    @property
    def payoff(self) -> Payoff:
        return self._payoff

    def prefilter(self, enr: Enrichment) -> list[str]:
        """Vetoes justified by the free data alone.

        Evaluated as if not live on purpose: this pass may only *reject*. In
        live mode an unmeasured gate is a veto, which is right for the real
        decision and wrong here - it would reject every candidate before
        anything got measured, and the bot would never trade at all.
        """
        vetoes, _ = evaluate_gates(enr, self.strategy, self.store, live=False)
        return vetoes

    def score(self, enr: Enrichment) -> Score:
        vetoes, abstained = evaluate_gates(enr, self.strategy, self.store, live=self.live)
        rubric = grade(enr, self.store, self.strategy)
        min_r = float(self.strategy.get("scoring.min_rubric", 7.0))
        if not vetoes and rubric.total < min_r:
            vetoes.append(f"rubric: {rubric.total:.1f} < {min_r:.1f}")
        normalized = normalize(enr.features, enr.unknown)
        probability, contributions = self.model.probability(normalized)
        payoff = self.refresh_payoff()
        cost = enr.features.round_trip_cost or enr.round_trip.total_cost_pct
        ev = expected_value(probability, payoff, cost)

        score = Score(
            probability=probability,
            expected_value=ev,
            contributions=contributions,
            veto_reasons=vetoes,
            dist=classify(
                enr.features, enr.unknown, age_minutes=enr.candidate.age_minutes
            ).as_dict(),
            rubric=rubric.as_visor(),
        )
        if abstained and not vetoes:
            score.contributions.setdefault("_abstained", 0.0)
        return score

    def passes(self, score: Score) -> tuple[bool, str]:
        if score.vetoed:
            return False, score.veto_reasons[0]
        min_p, min_ev = score_floors(self.strategy, self.store)
        if score.probability < min_p:
            return False, f"probability {score.probability:.3f} < {min_p:.3f}"
        if score.expected_value < min_ev:
            return False, f"EV {score.expected_value:+.3f} < {min_ev:.3f}"
        return True, "ok"

    def explain(self, score: Score) -> str:
        drivers = ", ".join(f"{k}{v:+.2f}" for k, v in score.top_drivers(6))
        return f"p={score.probability:.3f} ev={score.expected_value:+.3f} [{drivers}]"
