"""Exit management.

Entries get all the attention and exits keep all the money. On assets that can
go to zero in one block, the exit policy contributes more variance reduction
than any entry filter, and it is also the only part of the strategy that can be
improved without new data - which is why the learning loop tunes it first.

Priority order is not cosmetic. Liquidity drain outranks everything because it
is the only condition where waiting one more tick can mean not exiting at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from .log import get
from .models import ExitReason, Position, now_ms
from .playbook import ladder as pb_ladder
from .playbook import thesis_cut as pb_thesis_cut
from .settings import Config
from .signals.chart import tape_cut_ret, trail_pct_for_regime, volatility_volume_score, vol_regime
from .store import Store

log = get("portfolio")


@dataclass(slots=True)
class ExitOrder:
    # Fraction of the REMAINING position to sell. 1.0 closes it.
    fraction: float
    reason: ExitReason
    note: str = ""


class PositionManager:
    def __init__(self, strategy: Config, store: Store) -> None:
        self.strategy = strategy
        self.store = store

    def _p(self, name: str, default: float) -> float:
        return self.store.param(f"exits.{name}", float(self.strategy.get(f"exits.{name}", default)))

    def _vol_regime(self, position: Position) -> float:
        c = position.candidate
        live = volatility_volume_score(c.volume_5m_usd, c.mcap_usd, c.ret_5m)
        return vol_regime(position.entry_vol_score, live)

    def _ladder(self, position: Position) -> list[tuple[float, float]]:
        return pb_ladder(self.strategy, position.candidate.chain)

    def observe(self, position: Position, price: float, liquidity_usd: float) -> None:
        if price > position.peak_price:
            position.peak_price = price
        if price < position.trough_price:
            position.trough_price = price
        if liquidity_usd > 0 and (
            position.peak_liquidity_usd <= 0 or liquidity_usd > position.peak_liquidity_usd
        ):
            position.peak_liquidity_usd = liquidity_usd

    def evaluate(self, position: Position, price: float, liquidity_usd: float) -> list[ExitOrder]:
        """Exit orders for the current tick, highest priority first.

        Returns at most one full exit; partials can stack when a fast move
        clears several ladder rungs between ticks, which does happen and which
        a single-rung-per-tick implementation would quietly under-sell.
        """
        if price <= 0 or position.tokens_remaining <= 0:
            return []
        self.observe(position, price, liquidity_usd)

        gain = position.gain(price)
        orders: list[ExitOrder] = []

        drain = self._p("liquidity_drawdown_exit", 0.35)
        if (
            position.peak_liquidity_usd > 0
            and liquidity_usd > 0
            and liquidity_usd < position.peak_liquidity_usd * (1.0 - drain)
        ):
            return [
                ExitOrder(
                    1.0,
                    ExitReason.LIQUIDITY_DRAIN,
                    f"liquidity {liquidity_usd:.0f} down from peak {position.peak_liquidity_usd:.0f}",
                )
            ]

        stop = self._p("stop_loss_pct", 0.28)
        if gain <= -stop:
            return [ExitOrder(1.0, ExitReason.STOP_LOSS, f"{gain:.1%} vs stop -{stop:.1%}")]

        age_minutes = (now_ms() - position.opened_at_ms) / 60_000.0
        # ponytail: a 1m wick is not a falsified thesis; wait this long first.
        min_thesis = self._p("thesis_cut_min_minutes", 20.0)

        # Frank: the spike died before we banked initials. Off-peak cut, only
        # while still above the hard stop — otherwise the stop already fired.
        cut = pb_thesis_cut(self.strategy, position.candidate.chain)
        if (
            age_minutes >= min_thesis
            and cut > 0
            and position.ladder_filled == 0
            and position.peak_price > position.entry_price * 1.08
            and price <= position.peak_price * (1.0 - cut)
        ):
            off = 1.0 - price / position.peak_price
            return [
                ExitOrder(
                    1.0,
                    ExitReason.THESIS_CUT,
                    f"{off:.0%} off peak before first take",
                )
            ]

        # -0.12 is the tape *base*. Do not treat a vol-relative widen as a new
        # threshold — 15–20 thesis_cut notes still gate changing this number.
        tape_base = -0.12
        regime = self._vol_regime(position)
        tape = position.candidate.ret_5m
        tape_cut = tape_cut_ret(tape_base, regime)
        if age_minutes >= min_thesis and gain > 0.08 and tape <= tape_cut:
            return [
                ExitOrder(1.0, ExitReason.THESIS_CUT, "5m flipped, selling with tape")
            ]

        ladder = self._ladder(position)
        # `fraction` in the ladder is of the ORIGINAL position, while an
        # ExitOrder is a fraction of what is LEFT. The conversion has to track
        # the rungs queued in this same tick, because a fast move can clear
        # three rungs between ticks: converting each one against the position's
        # unchanged tokens_remaining would sell 0.34 + 0.33*0.66 + ... = 70% of
        # a ladder that is supposed to sell all of it.
        projected = position.tokens_remaining / position.tokens if position.tokens else 0.0
        while position.ladder_filled < len(ladder) and projected > 1e-9:
            multiple, fraction = ladder[position.ladder_filled]
            if gain < multiple - 1.0:
                break
            of_remaining = min(1.0, fraction / projected)
            orders.append(
                ExitOrder(
                    of_remaining,
                    ExitReason.TAKE_PROFIT,
                    f"rung {position.ladder_filled + 1} at {multiple:.2f}x",
                )
            )
            projected -= min(fraction, projected)
            position.ladder_filled += 1
            position.trailing_active = True

        if orders:
            return orders

        if position.trailing_active:
            trail = trail_pct_for_regime(self._p("trailing_stop_pct", 0.22), self._vol_regime(position))
            if position.peak_price > 0 and price <= position.peak_price * (1.0 - trail):
                return [
                    ExitOrder(
                        1.0,
                        ExitReason.TRAILING_STOP,
                        f"{price:.8f} is {1 - price / position.peak_price:.1%} off peak",
                    )
                ]

        time_stop = self._p("time_stop_minutes", 45.0)
        min_gain = self._p("time_stop_min_gain", 0.15)
        if age_minutes >= time_stop and gain < min_gain:
            return [
                ExitOrder(
                    1.0,
                    ExitReason.TIME_STOP,
                    f"{age_minutes:.0f}m old at {gain:+.1%}, capital better spent elsewhere",
                )
            ]
        return []

    def banked_from_peak(self, peak_return: float) -> float:
        return banked_from_peak(peak_return, self.strategy, self.store)

    @staticmethod
    def excursions(position: Position) -> tuple[float, float]:
        """(max favorable, max adverse) as fractions of entry.

        Feeds the postmortem: a trade that reached +180% and closed at +12% is a
        different failure from one that never went anywhere, and only these two
        numbers can tell them apart.
        """
        if position.entry_price <= 0:
            return 0.0, 0.0
        mfe = position.peak_price / position.entry_price - 1.0
        mae = position.trough_price / position.entry_price - 1.0
        return mfe, mae


def banked_from_peak(peak_return: float, strategy: Config, store: Store | None = None) -> float:
    """What the exit policy actually banks on a trade whose peak was
    `peak_return`.

    Shared by the expected-value prior and the backtester so both speak about
    the same exit policy. Keeping two copies of this is how a backtest ends up
    validating a strategy the live bot is not running.
    """

    def p(name: str, default: float) -> float:
        base = float(strategy.get(f"exits.{name}", default))
        return store.param(f"exits.{name}", base) if store else base

    raw = strategy.get("exits.take_profit_ladder", [[1.35, 1.0]])
    ladder = [(float(m), float(f)) for m, f in raw]
    stop = p("stop_loss_pct", 0.28)
    trail = p("trailing_stop_pct", 0.22)

    if peak_return <= -stop:
        return -stop

    banked = 0.0
    unsold = 1.0
    for multiple, fraction in ladder:
        if peak_return >= multiple - 1.0:
            take = min(unsold, fraction)
            banked += take * (multiple - 1.0)
            unsold -= take
    if unsold > 0:
        # Whatever the ladder did not sell exits on the trailing stop, i.e. a
        # haircut off the peak - never at the peak itself.
        exit_return = (1.0 + peak_return) * (1.0 - trail) - 1.0
        banked += unsold * max(-stop, exit_return)
    return banked
