"""Self-refinement: postmortem, parameter nudges, and weight training.

This is the module the whole project exists for. Three loops, each closing a
different gap between what the bot believed and what happened:

1. POSTMORTEM classifies every loss into a class that implies an action. A
   taxonomy whose buckets do not map to a parameter change is a dashboard, not
   a learning system.

2. NUDGES move one tunable per dominant error class, by a bounded step, with an
   audit row. Bounded because a bot that can move its own thresholds without
   limit will eventually find the setting where it never trades, or the one
   where it always does.

3. TRAINER refits the logistic weights on realized outcomes, with the priors as
   an L2 anchor, a minimum observation count per feature, a holdout, and
   automatic rollback if live PnL degrades. Every one of those guards exists
   because the alternative is a model that rewrites itself from twelve lucky
   trades.

The fourth loop is the one almost nobody builds: FILTER COST. Rejected
candidates are shadow-tracked, so the money the gates cost is measurable and
gates that are too tight get loosened. Without it, a bot only ever learns from
trades it already agreed with, and it converges on silence.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field

from .log import get
from .models import ErrorClass, ExitReason, Features, TradeRecord
from .scoring import PRIOR_BIAS, PRIOR_WEIGHTS, Model, normalize
from .settings import Config
from .store import Store

log = get("learning")


# ---------------------------------------------------------------------------
# 1. Postmortem
# ---------------------------------------------------------------------------


def classify(trade: TradeRecord, strategy: Config) -> ErrorClass:
    """Assign an error class. Order matters: the first matching rule is the one
    whose fix would have mattered most."""
    ladder = strategy.get("exits.take_profit_ladder", [[1.35, 0.34]])
    first_rung_gain = float(ladder[0][0]) - 1.0 if ladder else 0.35
    max_drift = float(strategy.get("execution.max_price_drift_from_signal", 0.05))

    entry_drift = (
        trade.entry_price / trade.signal_price - 1.0
        if trade.signal_price > 0 and trade.entry_price > 0
        else 0.0
    )
    realized = trade.pnl_pct

    if trade.exit_reason is ExitReason.LIQUIDITY_DRAIN and not trade.won:
        return ErrorClass.RUG

    if trade.won:
        # A win that gave back most of its peak is still a lesson. Threshold at
        # 2.5x so ordinary give-back on a volatile asset is not flagged.
        if trade.max_favorable_excursion > max(0.15, realized * 2.5):
            return ErrorClass.EXIT_TOO_FAST
        return ErrorClass.WIN

    if trade.entry_slippage > max(0.10, 3.0 * max_drift):
        return ErrorClass.SLIPPAGE_BLOWOUT

    if entry_drift > max_drift * 1.5:
        return ErrorClass.LATE_ENTRY

    # It was up enough to have banked something and we did not.
    if trade.max_favorable_excursion >= first_rung_gain * 0.6:
        return ErrorClass.EXIT_TOO_SLOW

    f = trade.features
    if f.bundle_pct > 0.20 or f.bot_share > 0.30:
        return ErrorClass.ADVERSE_SELECTION

    return ErrorClass.NO_EDGE


# ---------------------------------------------------------------------------
# 2. Parameter nudges
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Nudge:
    param: str
    # Relative step applied to the current value.
    step: float
    floor: float
    ceiling: float
    rationale: str


# One nudge per class, each with a defensible causal story. If you cannot say
# why the change would have prevented the loss, it does not belong here.
NUDGES: dict[ErrorClass, tuple[Nudge, ...]] = {
    ErrorClass.LATE_ENTRY: (
        Nudge(
            "execution.max_price_drift_from_signal",
            -0.25,
            0.01,
            0.20,
            "fills arriving too far above the signal price; abort sooner",
        ),
    ),
    ErrorClass.SLIPPAGE_BLOWOUT: (
        Nudge(
            "gates.min_liquidity_usd",
            0.30,
            5_000.0,
            2_000_000.0,
            "position size is too large for the pools being traded",
        ),
        Nudge(
            "risk.max_position_pct_of_liquidity",
            -0.25,
            0.001,
            0.05,
            "same cause, attacked from the size side",
        ),
    ),
    ErrorClass.RUG: (
        Nudge(
            "gates.min_liquidity_usd",
            0.40,
            5_000.0,
            2_000_000.0,
            "rugs concentrate in thin pools",
        ),
        Nudge(
            "gates.max_unknown_top1_pct",
            -0.12,
            0.20,
            0.80,
            "an unknown wallet holding most of supply is the precondition for a rug",
        ),
    ),
    ErrorClass.EXIT_TOO_FAST: (
        Nudge(
            "exits.trailing_stop_pct",
            0.20,
            0.08,
            0.60,
            "trailing stop is inside the asset's normal noise",
        ),
    ),
    ErrorClass.EXIT_TOO_SLOW: (
        Nudge(
            "exits.trailing_stop_pct",
            -0.18,
            0.08,
            0.60,
            "gains were reached and given back; tighten the trail",
        ),
    ),
    ErrorClass.ADVERSE_SELECTION: (
        Nudge(
            "gates.max_bundle_pct",
            -0.20,
            0.05,
            0.50,
            "buying into launch bundles; they exit into us",
        ),
    ),
    ErrorClass.NO_EDGE: (
        Nudge(
            "scoring.min_expected_value",
            0.20,
            0.01,
            0.40,
            "signal was clean and the trade still faded; demand more edge",
        ),
    ),
    ErrorClass.EXECUTION_FAIL: (
        Nudge(
            "execution.slippage_bps",
            0.25,
            50.0,
            1500.0,
            "transactions failing to land; pay for inclusion or stop trying",
        ),
    ),
}

# A self-tuned parameter may never drift more than this multiple away from the
# value you wrote in the config. The bot is allowed to adapt, not to redesign
# the strategy behind your back.
MAX_DRIFT_MULTIPLE = 3.0


@dataclass(slots=True)
class PostmortemReport:
    trades: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    pnl_by_class: dict[str, float] = field(default_factory=dict)


def run_postmortem(
    store: Store,
    strategy: Config,
    *,
    window: int = 60,
    min_share: float = 0.22,
    min_count: int = 4,
) -> PostmortemReport:
    """Look at the last `window` closed trades, find the dominant error class,
    and nudge its parameter.

    Only classes that are BOTH frequent enough and expensive enough act. Acting
    on a single bad trade is how a bot talks itself into a corner.
    """
    trades = store.trades(limit=window)
    report = PostmortemReport(trades=len(trades))
    if not trades:
        return report

    counts: Counter[str] = Counter()
    pnl: dict[str, float] = {}
    for trade in trades:
        klass = trade.error_class
        if klass is ErrorClass.WIN and not trade.won:
            klass = classify(trade, strategy)
        counts[klass.value] += 1
        pnl[klass.value] = pnl.get(klass.value, 0.0) + trade.pnl_usd
    report.counts = dict(counts)
    report.pnl_by_class = pnl

    total = len(trades)
    for klass_value, count in counts.most_common():
        if klass_value == ErrorClass.WIN.value:
            continue
        share = count / total
        if share < min_share or count < min_count:
            report.skipped.append(f"{klass_value}: {count}/{total} below action threshold")
            continue
        if pnl.get(klass_value, 0.0) >= 0:
            report.skipped.append(f"{klass_value}: frequent but not costing money")
            continue
        try:
            klass = ErrorClass(klass_value)
        except ValueError:
            continue
        for nudge in NUDGES.get(klass, ()):
            applied = _apply_nudge(store, strategy, nudge, klass_value, share)
            (report.applied if applied else report.skipped).append(
                applied or f"{nudge.param}: already at its bound"
            )
    return report


def _apply_nudge(
    store: Store, strategy: Config, nudge: Nudge, klass: str, share: float
) -> str:
    base = float(strategy.get(nudge.param, 0.0) or 0.0)
    current = store.param(nudge.param, base)
    if current == 0.0 and base == 0.0:
        return ""

    proposed = current * (1.0 + nudge.step)
    proposed = max(nudge.floor, min(nudge.ceiling, proposed))

    if base > 0:
        low, high = base / MAX_DRIFT_MULTIPLE, base * MAX_DRIFT_MULTIPLE
        proposed = max(low, min(high, proposed))

    if abs(proposed - current) / max(abs(current), 1e-9) < 0.01:
        return ""

    reason = f"{klass} at {share:.0%} of recent trades: {nudge.rationale}"
    store.set_param(nudge.param, proposed, reason)
    log.info(
        "parameter nudged",
        extra={"param": nudge.param, "from": current, "to": proposed, "class": klass},
    )
    return f"{nudge.param}: {current:.4g} -> {proposed:.4g} ({klass})"


# ---------------------------------------------------------------------------
# 3. Filter cost - the loop that keeps the bot from converging on silence
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FilterCost:
    gate: str
    rejected: int
    would_have_won: int
    median_counterfactual: float


def filter_cost(store: Store, *, win_threshold: float = 0.20) -> list[FilterCost]:
    """How much each gate cost, measured on shadow-tracked rejections.

    A gate that rejects fifty candidates and forty of them subsequently doubled
    is not protecting you, it is the strategy.
    """
    rows = store.filter_cost_report(limit=1000)
    buckets: dict[str, list[float]] = {}
    for row in rows:
        reason = (row["reason"] or "unknown").split(":")[0].strip() or "unknown"
        buckets.setdefault(reason, []).append(float(row["counterfactual_pct"] or 0.0))

    out: list[FilterCost] = []
    for gate, values in buckets.items():
        values.sort()
        mid = values[len(values) // 2] if values else 0.0
        out.append(
            FilterCost(
                gate=gate,
                rejected=len(values),
                would_have_won=sum(1 for v in values if v >= win_threshold),
                median_counterfactual=mid,
            )
        )
    out.sort(key=lambda c: -c.median_counterfactual)
    return out


def relax_costly_gates(
    store: Store,
    strategy: Config,
    *,
    min_rejected: int = 12,
    min_median_gain: float = 0.25,
) -> list[str]:
    """Loosen gates whose rejections were consistently profitable.

    Deliberately more conservative than the tightening path: a 10% step versus
    the 20-40% steps in NUDGES. Loosening a gate increases the size of the
    losses you can take, so it should happen slower than tightening one.
    """
    relaxable = {
        "liquidity": ("gates.min_liquidity_usd", -0.10),
        "holders": ("gates.min_holder_count", -0.10),
        "unknown_whale": ("gates.max_unknown_top1_pct", 0.10),
        "top10": ("gates.max_top10_pct", 0.10),
        "bundle": ("gates.max_bundle_pct", 0.10),
        "fresh_wallets": ("gates.max_fresh_wallet_pct", 0.10),
        "crowd_already_here": ("terminals.max_retail_terminal_share", 0.10),
        "round_trip_cost": ("gates.max_round_trip_cost_pct", 0.10),
    }
    notes: list[str] = []
    for cost in filter_cost(store):
        target = relaxable.get(cost.gate)
        if target is None:
            continue
        if cost.rejected < min_rejected or cost.median_counterfactual < min_median_gain:
            continue
        param, step = target
        nudge = Nudge(
            param,
            step,
            0.001,
            1_000_000.0,
            f"{cost.would_have_won}/{cost.rejected} rejections would have won "
            f"(median {cost.median_counterfactual:+.0%})",
        )
        applied = _apply_nudge(store, strategy, nudge, f"filter_cost:{cost.gate}", 1.0)
        if applied:
            notes.append(applied)
    return notes


# ---------------------------------------------------------------------------
# 4. Weight training
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrainResult:
    trained: bool = False
    version: int = 0
    promoted: bool = False
    samples: int = 0
    holdout_logloss: float | None = None
    incumbent_logloss: float | None = None
    frozen_features: list[str] = field(default_factory=list)
    note: str = ""


def _logloss(model: Model, samples: list[tuple[dict[str, float], int, float]]) -> float:
    if not samples:
        return float("inf")
    total = 0.0
    weight_sum = 0.0
    for normalized, label, weight in samples:
        p, _ = model.probability(normalized)
        p = min(1.0 - 1e-9, max(1e-9, p))
        total += -weight * (label * math.log(p) + (1 - label) * math.log(1.0 - p))
        weight_sum += weight
    return total / weight_sum if weight_sum else float("inf")


def _samples_from_trades(trades: list[TradeRecord]) -> list[tuple[dict[str, float], int, float]]:
    out = []
    for trade in trades:
        normalized = normalize(trade.features, trade.unknown)
        label = 1 if trade.won else 0
        # Magnitude matters: a +300% winner and a +3% scratch are not equally
        # informative, and unweighted logistic regression treats them as such.
        weight = 1.0 + min(2.0, abs(trade.pnl_pct) / 0.30)
        out.append((normalized, label, weight))
    return out


def train(store: Store, strategy: Config, *, seed: int = 7) -> TrainResult:
    cfg = strategy.section("learning")
    min_trades = int(strategy.get("scoring.min_trades_for_learned_weights", 40))
    trades = store.trades()
    result = TrainResult(samples=len(trades))

    if len(trades) < min_trades:
        result.note = f"{len(trades)}/{min_trades} closed trades; keeping prior weights"
        return result

    samples = _samples_from_trades(trades)
    # Time-ordered holdout. Random splits leak the future into the past on a
    # regime-dependent series and produce a model that looks great and is not.
    split = max(int(len(samples) * 0.8), min_trades // 2)
    train_set, holdout = samples[:split], samples[split:]
    if not holdout:
        result.note = "not enough data for a holdout"
        return result

    min_obs = int(cfg.get("min_observations_per_feature", 25))
    observations: Counter[str] = Counter()
    for normalized, _, _ in train_set:
        for name, value in normalized.items():
            if value != 0.0:
                observations[name] += 1
    frozen = sorted(n for n in PRIOR_WEIGHTS if observations[n] < min_obs)
    result.frozen_features = frozen

    incumbent = Model.load(store)
    challenger = Model(dict(incumbent.weights), incumbent.bias)

    lr = float(cfg.get("learning_rate", 0.04))
    l2 = float(cfg.get("l2_prior_strength", 0.10))
    rng = random.Random(seed)
    order = list(range(len(train_set)))

    for _ in range(60):
        rng.shuffle(order)
        for index in order:
            normalized, label, weight = train_set[index]
            p, _ = challenger.probability(normalized)
            error = (p - label) * weight
            challenger.bias -= lr * error
            for name, value in normalized.items():
                if value == 0.0 or name in frozen:
                    continue
                prior = PRIOR_WEIGHTS.get(name, 0.0)
                current = challenger.weights.get(name, prior)
                gradient = error * value + l2 * (current - prior)
                challenger.weights[name] = current - lr * gradient

    result.holdout_logloss = _logloss(challenger, holdout)
    result.incumbent_logloss = _logloss(incumbent, holdout)
    margin = float(cfg.get("promotion_margin", 0.01))

    version = store.save_weights(
        challenger.to_payload(),
        samples=len(train_set),
        holdout_logloss=result.holdout_logloss,
        note=f"frozen={len(frozen)} incumbent_ll={result.incumbent_logloss:.4f}",
    )
    result.trained = True
    result.version = version

    if result.holdout_logloss + margin < result.incumbent_logloss:
        store.activate_weights(version)
        result.promoted = True
        result.note = (
            f"promoted v{version}: holdout log-loss {result.holdout_logloss:.4f} "
            f"beats {result.incumbent_logloss:.4f}"
        )
    else:
        result.note = (
            f"kept v{incumbent.version}: challenger {result.holdout_logloss:.4f} did not beat "
            f"{result.incumbent_logloss:.4f} by {margin}"
        )
    log.info("training finished", extra={"note": result.note, "frozen": len(frozen)})
    return result


def check_rollback(store: Store, strategy: Config) -> str:
    """Demote the active weights if they are losing money live.

    Holdout log-loss is a proxy; realized PnL is the thing. When the two
    disagree, the money wins.
    """
    active = store.active_weights()
    if active is None:
        return ""
    version, _ = active
    if version <= 1:
        return ""

    window = int(strategy.get("learning.rollback_window_trades", 20))
    n, pnl = store.pnl_by_weights_version(version)
    if n < window:
        return ""
    if pnl >= 0:
        return ""

    previous = version - 1
    while previous >= 1 and store.weights_version_row(previous) is None:
        previous -= 1
    if previous < 1:
        return ""
    prev_n, prev_pnl = store.pnl_by_weights_version(previous)
    if prev_n and prev_pnl <= pnl:
        return ""

    store.activate_weights(previous)
    message = (
        f"rolled back weights v{version} -> v{previous}: {pnl:+.2f} USD over {n} trades"
    )
    log.error("weights rolled back", extra={"note": message})
    store.set_kv("last_rollback", message)
    return message


def export_weights(store: Store, path) -> str:
    active = store.active_weights()
    model = Model.from_payload(active[1], active[0]) if active else Model()
    payload = {"version": model.version, **model.to_payload()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return f"wrote weights v{model.version} to {path}"


def feature_report(store: Store) -> list[tuple[str, float, float, int]]:
    """(feature, active weight, prior weight, observations) for inspection.

    Reading this is how you notice the model has learned something you disagree
    with, which is the only way a human stays in the loop.
    """
    model = Model.load(store)
    observations: Counter[str] = Counter()
    for trade in store.trades():
        for name, value in normalize(trade.features, trade.unknown).items():
            if value != 0.0:
                observations[name] += 1
    names = sorted(PRIOR_WEIGHTS, key=lambda n: -abs(model.weights.get(n, 0.0)))
    return [
        (name, model.weights.get(name, 0.0), PRIOR_WEIGHTS.get(name, 0.0), observations[name])
        for name in names
    ]


def prior_model() -> Model:
    return Model(dict(PRIOR_WEIGHTS), PRIOR_BIAS)


def blank_features() -> Features:
    return Features()
