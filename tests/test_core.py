"""The runnable check.

Covers the logic that would lose money silently if it broke: distribution
maths, the sign of the terminal-attribution thesis, gate vetoes, Kelly sizing,
the exit ladder's fraction accounting, and the postmortem taxonomy.

Stdlib only, no fixtures, no plugins:

    python -m unittest discover -s tests -v
    python tests/test_core.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alphahound import backtest, learning  # noqa: E402
from alphahound.models import (  # noqa: E402
    Action,
    Candidate,
    Candle,
    Chain,
    Decision,
    ErrorClass,
    ExitReason,
    Features,
    Position,
    RoundTrip,
    Score,
    Side,
    TradeRecord,
    VenueId,
    legs_for_display,
    now_ms,
)
from alphahound.engine import enrich_due, loop_bug  # noqa: E402
from alphahound.portfolio import PositionManager, banked_from_peak  # noqa: E402
from alphahound.risk import RiskEngine, kelly_fraction, mcap_position_pct  # noqa: E402
from alphahound.scoring import (  # noqa: E402
    Model,
    Payoff,
    evaluate_gates,
    expected_value,
    normalize,
    payoff_from_config,
    watch_call,
)
from alphahound.settings import Config, apply_aggressive_learning, load_strategy, score_floors  # noqa: E402
from alphahound.signals import Enrichment  # noqa: E402
from alphahound.signals import chart, flow  # noqa: E402
from alphahound.signals.distribution import Holder, analyze, gini  # noqa: E402
from alphahound.signals.terminals import (  # noqa: E402
    BuyerTx,
    ShareTracker,
    TerminalRegistry,
    attribute,
    discover_fee_accounts,
)
from alphahound.store import Store, lock_state_dir  # noqa: E402

STRATEGY = load_strategy()


class TestEnrichDue(unittest.TestCase):
    def test_first_look_always_runs(self):
        self.assertTrue(enrich_due(0, 10_000, 15_000))

    def test_skips_until_ttl(self):
        self.assertFalse(enrich_due(1_000, 10_000, 15_000))
        self.assertTrue(enrich_due(1_000, 20_000, 15_000))


class TestLoopBug(unittest.TestCase):
    def test_name_error_is_fatal_timeout_is_not(self):
        self.assertTrue(loop_bug(NameError("mcap_is_dead")))
        self.assertTrue(loop_bug(AttributeError("x")))
        self.assertFalse(loop_bug(RuntimeError("rpc 429")))
        self.assertFalse(loop_bug(TimeoutError()))


def make_store() -> tuple[Store, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory()
    return Store(Path(tmp.name)), tmp


def candles(closes: list[float], volume: float = 1000.0) -> list[Candle]:
    out = []
    for i, close in enumerate(closes):
        prev = closes[i - 1] if i else close
        out.append(
            Candle(
                ts=i * 60_000,
                open=prev,
                high=max(prev, close) * 1.01,
                low=min(prev, close) * 0.99,
                close=close,
                volume=volume,
            )
        )
    return out


class TestDistribution(unittest.TestCase):
    def test_gini_separates_shapes_that_top10_hides(self):
        # Both distributions have 25 holders and 100 units of supply, and a
        # top-10 share that looks similar. Gini is what tells them apart.
        flat = [4.0] * 25
        one_whale = [40.0] + [2.5] * 24
        ten_medium = [4.0] * 10 + [2.5] * 24
        self.assertLess(gini(flat), 0.05)
        self.assertGreater(gini(one_whale), gini(ten_medium) + 0.15)

    def test_lp_and_burn_excluded_from_concentration(self):
        holders = [
            Holder("pool", 800.0, is_lp=True),
            Holder("burn", 100.0, is_burn=True),
            Holder("a", 50.0),
            Holder("b", 30.0),
            Holder("c", 20.0),
        ]
        stats = analyze(holders, now_ms=now_ms())
        self.assertEqual(stats.holder_count, 3)
        # 50 of 100 circulating, not 50 of 1000 total. Counting the pool as a
        # holder makes every healthy token look concentrated.
        self.assertAlmostEqual(stats.top1_pct, 0.5, places=6)
        self.assertAlmostEqual(stats.lp_pct, 0.8, places=6)

    def test_funding_cluster_catches_one_holder_in_twenty_hats(self):
        holders = [Holder(f"w{i}", 5.0, funder="sybil") for i in range(18)] + [
            Holder("real", 10.0, funder="cex")
        ]
        stats = analyze(holders, now_ms=now_ms())
        self.assertGreater(stats.largest_funding_cluster_pct, 0.85)

    def test_token_hops_from_one_sender_are_a_funding_cluster(self):
        from alphahound.signals.distribution import TokenMove, holders_from_moves

        pool = "0x" + "p" * 40
        src = "0x" + "a" * 40
        moves = [
            TokenMove(1, src, f"0x{i:040x}", 10.0) for i in range(1, 9)
        ] + [TokenMove(2, pool, "0x" + "b" * 40, 5.0)]
        holders, launch = holders_from_moves(moves, pools={pool}, first_n=40)
        self.assertEqual(launch, 1)
        stats = analyze(holders, now_ms=now_ms(), launch_slot=launch)
        self.assertGreater(stats.largest_funding_cluster_pct, 0.70)
        self.assertGreater(stats.bundle_pct, 0.70)

    def test_quote_cluster_is_complementary_to_token_hops(self):
        from alphahound.signals.distribution import (
            TokenMove,
            cluster_share_by_root,
            first_pool_buyers,
        )

        pool = "0x" + "p" * 40
        moves = [TokenMove(i, pool, f"0x{i:040x}", 10.0) for i in range(1, 6)]
        buyers = first_pool_buyers(moves, {pool}, 40, {f"0x{i:040x}": 10.0 for i in range(1, 6)})
        self.assertEqual(len(buyers), 5)
        roots = {f"0x{i:040x}": "0x" + "f" * 40 for i in range(1, 5)}
        holders = [Holder(f"0x{i:040x}", 10.0) for i in range(1, 6)]
        # 4 of 5 bags share a quote root → 80%, which the token-hop path would miss.
        self.assertAlmostEqual(cluster_share_by_root(holders, roots, 50.0), 0.8)

    def test_bundle_pct_not_reported_without_a_launch_slot(self):
        holders = [Holder("a", 100.0, acquired_slot=10)]
        stats = analyze(holders, now_ms=now_ms(), launch_slot=0)
        self.assertEqual(stats.bundle_pct, 0.0)
        self.assertTrue(any("launch slot" in note for note in stats.notes))
        from alphahound.signals.distribution import unmeasured_from_stats

        self.assertIn("bundle_pct", unmeasured_from_stats(stats))
        self.assertIn("fresh_wallet_pct", unmeasured_from_stats(stats))

    def test_wallet_ages_present_are_measured(self):
        from alphahound.signals.distribution import unmeasured_from_stats

        t = now_ms()
        stats = analyze(
            [Holder("a", 100.0, first_seen_ms=t, acquired_slot=1)],
            now_ms=t,
            launch_slot=1,
        )
        self.assertEqual(unmeasured_from_stats(stats), set())
        self.assertAlmostEqual(stats.fresh_wallet_pct, 1.0)

    def test_missing_fresh_wallet_age_is_score_neutral_not_a_bonus(self):
        from alphahound.models import Features
        from alphahound.scoring import Model, normalize

        feat = Features(fresh_wallet_pct=0.0, bundle_pct=0.0)
        model = Model()
        _, lied = model.probability(normalize(feat, unknown=set()))
        _, honest = model.probability(
            normalize(feat, unknown={"fresh_wallet_pct", "bundle_pct"})
        )
        self.assertGreater(lied.get("fresh_wallet_pct", 0.0), 0.5)
        self.assertNotIn("fresh_wallet_pct", honest)
        self.assertGreater(lied.get("bundle_pct", 0.0), 0.3)
        self.assertNotIn("bundle_pct", honest)

    def test_evm_aggregate_marks_unmeasured_features_unknown(self):
        from alphahound.signals import Enricher, Enrichment

        c = Candidate(chain=Chain.ROBINHOOD_CHAIN, address="0x" + "ab" * 20, symbol="X")
        enr = Enrichment(candidate=c, features=Features())
        Enricher.__new__(Enricher)._aggregate_only_features(c, enr)
        for name in (
            "holder_growth_5m",
            "axiom_share_delta_5m",
            "unknown_share",
            "smart_money_buys",
            "cluster_pct",
            "lp_locked_pct",
        ):
            self.assertIn(name, enr.unknown)
            self.assertEqual(normalize(Features(), enr.unknown)[name], 0.0)

    def test_evm_lp_lock_skipped_chain_is_unknown_not_unlocked(self):
        from alphahound.signals import Enricher, Enrichment

        e = Enricher.__new__(Enricher)
        e.evm_rpcs = {}
        e._evm_cache = {}
        c = Candidate(chain=Chain.BASE, address="0x" + "ab" * 20, symbol="X")
        enr = Enrichment(candidate=c, features=Features())
        out = asyncio.run(e._evm_lp_lock(c, enr))
        self.assertEqual(out, {})
        self.assertIn("lp_locked_pct", enr.unknown)

    def test_evm_cluster_unmeasured_when_transfer_path_cannot_run(self):
        from alphahound.signals import Enricher, Enrichment

        e = Enricher.__new__(Enricher)
        e.evm_rpcs = {}
        e._evm_cache = {}
        c = Candidate(chain=Chain.ROBINHOOD_CHAIN, address="0x" + "ab" * 20, symbol="X")
        enr = Enrichment(candidate=c, features=Features())
        out = asyncio.run(e._evm_launch_distribution(c, enr))
        self.assertEqual(out, {})
        self.assertIn("cluster_pct", enr.unknown)


class TestChart(unittest.TestCase):
    def test_breakout_ramps_rather_than_flipping(self):
        flat = candles([1.0] * 12)
        self.assertLess(chart.breakout(flat, 10), 0.5)
        broke = candles([1.0] * 11 + [1.20])
        self.assertGreater(chart.breakout(broke, 10), 0.9)

    def test_parabolic_flags_the_chart_that_already_paid_someone_else(self):
        vertical = candles([1.0, 1.4, 1.9, 2.4, 3.0, 3.4])
        self.assertGreater(chart.parabolic(vertical, 5, 1.5), 0.9)
        calm = candles([1.0, 1.01, 1.02, 1.03, 1.02, 1.04])
        self.assertLess(chart.parabolic(calm, 5, 1.5), 0.1)

    def test_candles_built_from_a_trade_stream(self):
        trades = [
            (0, 1.0, 100.0),
            (10_000, 1.2, 50.0),
            (30_000, 0.9, 25.0),
            (61_000, 1.1, 10.0),
        ]
        built = chart.candles_from_trades(trades, 60)
        self.assertEqual(len(built), 2)
        self.assertEqual(built[0].high, 1.2)
        self.assertEqual(built[0].low, 0.9)
        self.assertEqual(built[0].volume, 175.0)

    def test_empty_candles_do_not_explode(self):
        values = chart.extract([], STRATEGY.section("chart"))
        self.assertEqual(values["ret_5m"], 0.0)


class TestFlow(unittest.TestCase):
    def test_buy_sell_ratio_is_usd_weighted_not_count_weighted(self):
        now = now_ms()
        spam = [flow.Trade(now, Side.BUY, 1.0, 2.0, f"w{i}") for i in range(100)]
        spam.append(flow.Trade(now, Side.SELL, 1.0, 5000.0, "whale"))
        # A hundred two-dollar buys must not outvote a five-thousand-dollar sell.
        self.assertLess(flow.buy_sell_ratio(spam), 1.0)

    def test_whale_concentration(self):
        now = now_ms()
        trades = [
            flow.Trade(now, Side.BUY, 1.0, 9000.0, "whale"),
            flow.Trade(now, Side.BUY, 1.0, 1000.0, "retail"),
        ]
        self.assertAlmostEqual(flow.whale_concentration(trades), 0.9, places=6)


class TestTerminalAttribution(unittest.TestCase):
    def registry(self) -> TerminalRegistry:
        return TerminalRegistry(
            [
                {"name": "axiom", "class": "retail", "fee_accounts": ["AXFEE"]},
                {"name": "trojan", "class": "bot", "fee_accounts": ["TJFEE"]},
            ],
            {"jupiter_v6": "JUPPROG"},
        )

    def test_retail_beats_neutral_on_the_same_transaction(self):
        # Every terminal swap also touches Jupiter. Matching greedily would
        # label all of them "jupiter" and destroy the only useful signal.
        label, klass = self.registry().classify(["JUPPROG", "AXFEE", "other"])
        self.assertEqual((label, klass), ("axiom", "retail"))

    def test_attribution_dedups_by_buyer(self):
        txs = [BuyerTx("bot1", ["TJFEE"], 0, 10.0) for _ in range(20)]
        txs.append(BuyerTx("human", ["AXFEE"], 0, 10.0))
        result = attribute(txs, self.registry())
        self.assertEqual(result.total, 2)
        self.assertAlmostEqual(result.class_share("retail"), 0.5, places=6)

    def test_share_tracker_reports_no_wave_from_one_sample(self):
        tracker = ShareTracker()
        tracker.observe("k", 1_000_000, {"retail": 0.4})
        self.assertEqual(tracker.delta("k", "retail", 300_000), 0.0)
        tracker.observe("k", 1_600_000, {"retail": 0.6})
        self.assertAlmostEqual(tracker.delta("k", "retail", 300_000), 0.2, places=6)

    def test_discovery_ranks_breadth_of_buyers_not_transaction_count(self):
        # FEEA is paid by twelve unrelated wallets: the shape of a real fee
        # account. "noise" only shows up in a few of them, and "busybot" pays
        # FEEB sixty times but is a single wallet - which is exactly the thing
        # a transaction-count ranking would put on top.
        txs = [BuyerTx(f"w{i}", ["FEEA"] + (["noise"] if i < 3 else []), 0) for i in range(12)]
        txs += [BuyerTx("busybot", ["FEEB"], 0) for _ in range(60)]
        found = discover_fee_accounts(txs, self.registry(), min_distinct_buyers=8)
        self.assertTrue(found)
        self.assertEqual(found[0].address, "FEEA")
        self.assertNotIn("FEEB", [c.address for c in found])


class TestScoring(unittest.TestCase):
    def test_unknown_features_are_neutral_not_favourable(self):
        features = Features(liquidity_usd=0.0, holder_count=0.0)
        blind = normalize(features, unknown={"liquidity_usd", "holder_count"})
        self.assertEqual(blind["liquidity_usd"], 0.0)
        self.assertEqual(blind["holder_count"], 0.0)
        # Measured-and-terrible must score worse than unmeasured.
        seen = normalize(features)
        self.assertLess(seen["liquidity_usd"], blind["liquidity_usd"])

    def test_the_thesis_level_and_derivative_have_opposite_signs(self):
        model = Model()
        base = dict(
            unique_buyers_5m=60.0,
            buy_sell_ratio=2.0,
            liquidity_usd=60_000.0,
            holder_count=400.0,
        )
        early = Features(retail_share=0.10, retail_share_delta_5m=0.12, **base)
        late = Features(retail_share=0.60, retail_share_delta_5m=0.00, **base)
        p_early, _ = model.probability(normalize(early))
        p_late, _ = model.probability(normalize(late))
        self.assertGreater(
            p_early,
            p_late,
            "arriving before the retail wave must score above arriving after it",
        )

    def test_parabolic_entries_are_penalised(self):
        model = Model()
        calm = Features(breakout=0.9, volume_z=2.0, parabolic=0.0)
        vertical = Features(breakout=0.9, volume_z=2.0, parabolic=1.0)
        self.assertGreater(
            model.probability(normalize(calm))[0], model.probability(normalize(vertical))[0]
        )

    def test_expected_value_subtracts_costs(self):
        payoff = Payoff(avg_win=0.40, avg_loss=0.25, samples=100, from_history=True)
        free = expected_value(0.60, payoff, 0.0)
        costly = expected_value(0.60, payoff, 0.06)
        self.assertAlmostEqual(free - costly, 0.06, places=9)
        # A 60% win rate is not enough when the round trip eats the edge.
        self.assertLess(expected_value(0.60, payoff, 0.16), 0.0)

    def test_prior_payoff_does_not_assume_every_winner_hits_the_top_rung(self):
        strategy = Config(
            {
                "exits": {
                    "take_profit_ladder": [[1.35, 0.34], [2.0, 0.33], [3.5, 0.33]],
                    "stop_loss_pct": 0.28,
                    "trailing_stop_pct": 0.22,
                },
                "scoring": {"prior_winner_ladder_reach": [[1, 0.6], [2, 0.25], [3, 0.15]]},
            }
        )
        prior = payoff_from_config(strategy)
        naive = sum((m - 1.0) * f for m, f in [(1.35, 0.34), (2.0, 0.33), (3.5, 0.33)])

        # The naive whole-ladder sum is what makes a fresh bot fearless.
        self.assertGreater(naive, 1.2)
        self.assertLess(prior.avg_win, naive / 2)
        # ...but not so pessimistic that a genuinely good signal cannot clear
        # the EV floor, or the bot never trades and never learns.
        self.assertGreater(prior.avg_win, banked_from_peak(0.35, strategy))
        floor = float(strategy.get("scoring.min_expected_value", 0.035))
        self.assertGreater(expected_value(0.62, prior, 0.05), floor)
        self.assertLess(expected_value(0.50, prior, 0.05), floor)

    def test_weights_are_stored_by_name_so_new_features_are_safe(self):
        model = Model({"a_brand_new_feature": 0.5})
        self.assertEqual(model.weights["a_brand_new_feature"], 0.5)
        self.assertIn("bundle_pct", model.weights)


class TestGates(unittest.TestCase):
    def setUp(self):
        self.store, self._tmp = make_store()

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def enrichment(self, **overrides) -> Enrichment:
        features = Features(
            liquidity_usd=80_000.0,
            holder_count=500.0,
            top10_pct=0.22,
            top1_pct=0.08,
            cluster_pct=0.02,
            bundle_pct=0.02,
            dev_holding_pct=0.0,
            fresh_wallet_pct=0.10,
            bot_share=0.10,
            retail_share=0.15,
            round_trip_cost=0.02,
            lp_locked_pct=1.0,
        )
        for name, value in overrides.items():
            setattr(features, name, value)
        candidate = Candidate(
            chain=Chain.SOLANA,
            address="mintpump",
            price_usd=1.0,
            dex_id="pumpfun",
            created_at_ms=now_ms() - 60_000,
        )
        return Enrichment(
            candidate=candidate,
            features=features,
            round_trip=RoundTrip(ok=True, sell_slippage=0.03, total_cost_pct=0.02),
            crowd={"whale_n": 1, "kols": ["bagu"], "wallets": ["WHALE1"]},
        )

    def test_clean_candidate_passes(self):
        enr = self.enrichment()
        enr.mint = _FakeMint(None, None)
        vetoes, abstained = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertEqual(vetoes, [], msg=f"unexpected vetoes; abstained={abstained}")

    def test_vamp_is_a_hard_veto(self):
        enr = self.enrichment(is_vamp=1.0)
        enr.mint = _FakeMint(None, None)
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertTrue(any("vamp" in v for v in vetoes))

    def test_beta_is_vetoed_when_the_main_dumps(self):
        enr = self.enrichment(is_beta=1.0, main_ret_5m=-0.25)
        enr.mint = _FakeMint(None, None)
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertTrue(any("beta" in v for v in vetoes))

    def test_unsellable_token_is_vetoed(self):
        enr = self.enrichment()
        enr.mint = _FakeMint(None, None)
        enr.round_trip = RoundTrip(ok=True, sell_slippage=0.60, total_cost_pct=0.02)
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertTrue(any("sellable" in v for v in vetoes))

    def test_watch_call_waits_for_a_dip_not_a_hard_skip(self):
        self.assertEqual(
            watch_call(vetoed=True, ok=False, reasons=["chase: 5m ripped, wait dip"]),
            "wait",
        )
        self.assertEqual(
            watch_call(
                vetoed=True, ok=False, reasons=["round_trip_cost: round trip costs 7.2%"]
            ),
            "wait",
        )
        self.assertEqual(watch_call(vetoed=True, ok=False, reasons=["vamp: clone"]), "skip")
        self.assertEqual(
            watch_call(vetoed=True, ok=False, reasons=["chase: rip", "vamp: clone"]),
            "skip",
        )
        self.assertEqual(
            watch_call(vetoed=True, ok=False, reasons=["mcap: 80000 below 100000 floor"]),
            "wait",
        )
        self.assertEqual(watch_call(vetoed=False, ok=False, reasons=[]), "wait")
        self.assertEqual(watch_call(vetoed=False, ok=True, reasons=[]), "trade")

    def test_volatility_volume_is_a_prior_not_a_gate(self):
        from alphahound.models import Features
        from alphahound.scoring import PRIOR_WEIGHTS, normalize
        from alphahound.signals.chart import volatility_volume_score

        self.assertIn("volatility_volume_score", Features.names())
        self.assertGreater(PRIOR_WEIGHTS["volatility_volume_score"], 0.0)
        self.assertLess(PRIOR_WEIGHTS["volatility_volume_score"], abs(PRIOR_WEIGHTS["parabolic"]))
        self.assertEqual(PRIOR_WEIGHTS["parabolic"], -1.20)
        dip = volatility_volume_score(40_000, 100_000, 0.0)
        ripped = volatility_volume_score(40_000, 100_000, 0.30)
        quiet = volatility_volume_score(5_000, 100_000, 0.0)
        self.assertGreater(dip, quiet)
        self.assertGreater(ripped, dip)
        self.assertLess(dip / ripped, 1.15)
        raw = normalize(Features(volatility_volume_score=0.5), unknown=set())
        missing = normalize(Features(volatility_volume_score=0.5), unknown={"volatility_volume_score"})
        self.assertAlmostEqual(raw["volatility_volume_score"], 0.5)
        self.assertEqual(missing["volatility_volume_score"], 0.0)
        src = Path(__file__).resolve().parents[1] / "src" / "alphahound" / "scoring.py"
        gates = src.read_text(encoding="utf-8")
        gate_fn = gates[gates.index("def evaluate_gates") : gates.index("def patience_only")]
        self.assertNotIn("volatility_volume_score", gate_fn)

    def test_hood_official_x_is_not_required(self):
        enr = self.enrichment(twitter_mentions=5.0)
        enr.candidate.chain = Chain.ROBINHOOD_CHAIN
        enr.candidate.dex_id = "uniswap"
        enr.candidate.mcap_usd = 250_000
        enr.candidate.volume_5m_usd = 12_000
        enr.candidate.ret_5m = 0.0
        enr.mint = _FakeMint(None, None)
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertFalse(any(v.startswith("twitter:") for v in vetoes), vetoes)
        enr.candidate.ret_5m = 0.28
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertTrue(any("chase" in v for v in vetoes), vetoes)
        enr.twitter = {"official": "hotdogcoin", "official_age_min": 18}
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertFalse(any("chase" in v for v in vetoes), vetoes)

    def test_free_data_prefilter_rejects_early_but_never_approves(self):
        from alphahound.scoring import Model, Scorer
        from alphahound.signals import Enricher

        scorer = Scorer(Model(), STRATEGY, self.store, live=True)
        thin = Candidate(
            chain=Chain.SOLANA,
            address="mintpump",
            price_usd=1.0,
            liquidity_usd=900.0,
            dex_id="pumpfun",
        )
        deep = Candidate(
            chain=Chain.SOLANA,
            address="mint2pump",
            price_usd=1.0,
            liquidity_usd=90_000.0,
            dex_id="pumpfun",
        )
        cheap_thin = Enricher.free_enrichment(thin)  # no network touched
        cheap_deep = Enricher.free_enrichment(deep)

        self.assertTrue(
            any("liquidity" in v for v in scorer.prefilter(cheap_thin)),
            "a pool below the floor must die before it costs an RPC call",
        )
        # Passing the prefilter is not approval: everything else is unmeasured,
        # so the real (live) scoring pass still refuses it.
        self.assertEqual(scorer.prefilter(cheap_deep), [])
        self.assertTrue(scorer.score(cheap_deep).vetoed)

    def test_live_mode_vetoes_what_it_cannot_measure(self):
        enr = self.enrichment()
        enr.mint = _FakeMint(None, None)
        enr.unknown.add("holder_count")
        live_vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        paper_vetoes, paper_abstained = evaluate_gates(enr, STRATEGY, self.store, live=False)
        self.assertTrue(any("unmeasured" in v for v in live_vetoes))
        self.assertEqual(paper_vetoes, [])
        self.assertTrue(any("holder_count" in a for a in paper_abstained))

    def test_live_mint_authority_blocks_entry(self):
        enr = self.enrichment()
        enr.mint = _FakeMint("someauthority", None)
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertTrue(any("mint_authority" in v for v in vetoes))

    def test_a_gate_override_takes_effect_without_a_redeploy(self):
        enr = self.enrichment(liquidity_usd=20_000.0)
        enr.mint = _FakeMint(None, None)
        self.assertEqual(evaluate_gates(enr, STRATEGY, self.store, live=True)[0], [])
        self.store.set_param("gates.min_liquidity_usd", 50_000.0, "test")
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertTrue(any("liquidity" in v for v in vetoes))

    def test_handmade_raydium_is_rejected(self):
        from alphahound.origin import launchpad_origin

        handmade = Candidate(
            chain=Chain.SOLANA,
            address="SoRandomHandmade111111111111111111111111111",
            dex_id="raydium",
        )
        ok, reason = launchpad_origin(handmade, STRATEGY)
        self.assertFalse(ok)
        self.assertIn("handmade", reason)

        pump = Candidate(
            chain=Chain.SOLANA,
            address="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxpump",
            dex_id="raydium",
        )
        self.assertTrue(launchpad_origin(pump, STRATEGY)[0])

        bonk = Candidate(
            chain=Chain.SOLANA,
            address="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxbonk",
            dex_id="letsbonk",
        )
        self.assertFalse(launchpad_origin(bonk, STRATEGY)[0])

        trump = Candidate(
            chain=Chain.SOLANA,
            address="6p6xgHyF7AeE6TZkSmFsko444wqoP15icUSqi2jfGiPN",
            dex_id="raydium",
        )
        self.assertFalse(launchpad_origin(trump, STRATEGY)[0])

        uni = Candidate(chain=Chain.ROBINHOOD_CHAIN, address="0xabc", dex_id="uniswap")
        self.assertFalse(launchpad_origin(uni, STRATEGY)[0])
        pons = Candidate(chain=Chain.ROBINHOOD_CHAIN, address="0xabc", dex_id="pons")
        self.assertTrue(launchpad_origin(pons, STRATEGY)[0])
        stream = Candidate(
            chain=Chain.ROBINHOOD_CHAIN, address="0xabc", source="hood_stream", dex_id="pons"
        )
        self.assertTrue(launchpad_origin(stream, STRATEGY)[0])
        handmade_stream = Candidate(
            chain=Chain.ROBINHOOD_CHAIN, address="0xabc", source="hood_stream", dex_id="uniswap"
        )
        self.assertFalse(launchpad_origin(handmade_stream, STRATEGY)[0])
        baby = Candidate(
            chain=Chain.ROBINHOOD_CHAIN,
            address="0xabc",
            source="hood_stream",
            dex_id="pons",
            mcap_usd=12_000,
            volume_5m_usd=100,
            liquidity_usd=4_000,
            created_at_ms=now_ms(),
        )
        enr = Enrichment(candidate=baby, features=Features(liquidity_usd=4_000))
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=False)
        self.assertTrue(
            any(
                v.startswith("mcap:") or v.startswith("liquidity:") or v.startswith("volume:")
                for v in vetoes
            ),
            vetoes,
        )

        broker = Candidate(chain=Chain.ROBINHOOD_BROKER, address="BTC")
        self.assertFalse(launchpad_origin(broker, STRATEGY)[0])

        four = Candidate(chain=Chain.BNB, address="0xdef", dex_id="fourmeme")
        self.assertTrue(launchpad_origin(four, STRATEGY)[0])

        pancake = Candidate(chain=Chain.BNB, address="0xdef", dex_id="pancakeswap")
        self.assertFalse(launchpad_origin(pancake, STRATEGY)[0])

    def test_old_token_stays_eligible(self):
        enr = self.enrichment()
        enr.mint = _FakeMint(None, None)
        enr.candidate.created_at_ms = now_ms() - 4 * 60 * 60 * 1000
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertFalse(any(v.startswith("age:") for v in vetoes), vetoes)

    def test_solana_sponsor_is_optional(self):
        enr = self.enrichment()
        enr.mint = _FakeMint(None, None)
        enr.crowd = {"whale_n": 0, "kols": [], "wallets": []}
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=False)
        self.assertFalse(any(v.startswith("sponsor:") for v in vetoes), vetoes)

    def test_bnb_does_not_require_sponsor(self):
        enr = self.enrichment()
        enr.candidate.chain = Chain.BNB
        enr.candidate.dex_id = "fourmeme"
        enr.candidate.address = "0xdef"
        enr.crowd = {"whale_n": 0, "kols": [], "wallets": []}
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=False)
        self.assertFalse(any(v.startswith("sponsor:") for v in vetoes), vetoes)

    def test_known_kol_excuses_a_concentrated_top10(self):
        # The $TRUMP shape: top holder has 70%, but that holder is a known whale.
        enr = self.enrichment(top10_pct=0.70, top1_pct=0.70, known_holder_pct=0.70)
        enr.mint = _FakeMint(None, None)
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertEqual(vetoes, [], msg=f"KOL-held concentration must not veto: {vetoes}")

        rug = self.enrichment(top10_pct=0.70, top1_pct=0.70, known_holder_pct=0.0)
        rug.mint = _FakeMint(None, None)
        vetoes, _ = evaluate_gates(rug, STRATEGY, self.store, live=True)
        self.assertTrue(any("unknown_whale" in v for v in vetoes))

    def test_unknown_top10_is_a_hard_veto(self):
        enr = self.enrichment(top10_pct=0.62, top1_pct=0.12, known_holder_pct=0.0)
        enr.mint = _FakeMint(None, None)
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertTrue(any(v.startswith("top10:") for v in vetoes), vetoes)

    def test_unmeasured_twitter_is_not_a_live_veto(self):
        enr = self.enrichment()
        enr.mint = _FakeMint(None, None)
        enr.unknown.add("twitter_mentions")
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertEqual(vetoes, [], msg=f"missing twitter key must not halt solana: {vetoes}")

    def test_solana_skips_unmeasured_cluster(self):
        enr = self.enrichment()
        enr.mint = _FakeMint(None, None)
        enr.unknown.add("cluster_pct")
        paper, _ = evaluate_gates(enr, STRATEGY, self.store, live=False)
        self.assertTrue(any("rug_filter" in v for v in paper), paper)
        cheap = self.enrichment()
        cheap.mint = None
        cheap.unknown.add("cluster_pct")
        self.assertFalse(
            any("rug_filter" in v for v in evaluate_gates(cheap, STRATEGY, self.store, live=False)[0])
        )
        enr.candidate.chain = Chain.ROBINHOOD_CHAIN
        enr.candidate.dex_id = "pons"
        enr.candidate.address = "0xabc"
        hood, _ = evaluate_gates(enr, STRATEGY, self.store, live=False)
        self.assertTrue(any("rug_filter" in v for v in hood), hood)

    def test_paid_listing_skips_unmeasured_cluster(self):
        enr = self.enrichment()
        enr.mint = _FakeMint(None, None)
        enr.unknown.add("cluster_pct")
        enr.candidate.dex_paid = True
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=False)
        self.assertFalse(any("rug_filter" in v for v in vetoes), vetoes)
        self.assertFalse(any(v.startswith("unverified:") for v in vetoes), vetoes)

    def test_measured_cluster_vetoes(self):
        enr = self.enrichment(cluster_pct=0.70)
        enr.mint = _FakeMint(None, None)
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertTrue(any("cluster:" in v for v in vetoes), vetoes)

    def test_measured_bundle_is_a_hard_veto(self):
        enr = self.enrichment(bundle_pct=0.45)
        enr.mint = _FakeMint(None, None)
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=False)
        self.assertTrue(any(v.startswith("bundle:") for v in vetoes), vetoes)

    def test_hood_free_lp_nft_is_a_hard_veto(self):
        enr = self.enrichment(lp_locked_pct=0.0)
        enr.mint = _FakeMint(None, None)
        enr.candidate.chain = Chain.ROBINHOOD_CHAIN
        enr.candidate.dex_id = "uniswap"
        enr.candidate.address = "0xabc"
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=False)
        self.assertTrue(any(v.startswith("lp_unlocked:") for v in vetoes), vetoes)
        locked = self.enrichment(lp_locked_pct=1.0)
        locked.mint = _FakeMint(None, None)
        locked.candidate.chain = Chain.ROBINHOOD_CHAIN
        locked.candidate.dex_id = "uniswap"
        locked.candidate.address = "0xabc"
        vetoes, _ = evaluate_gates(locked, STRATEGY, self.store, live=False)
        self.assertFalse(any(v.startswith("lp_unlocked:") for v in vetoes), vetoes)

    def test_solana_does_not_use_the_lp_nft_gate(self):
        enr = self.enrichment(lp_locked_pct=0.0)
        enr.mint = _FakeMint(None, None)
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=False)
        self.assertFalse(any(v.startswith("lp_unlocked:") for v in vetoes), vetoes)

    def test_robinhood_age_is_not_an_entry_veto(self):
        enr = self.enrichment()
        enr.mint = _FakeMint(None, None)
        enr.candidate.chain = Chain.ROBINHOOD_CHAIN
        enr.candidate.dex_id = "pons"
        enr.candidate.address = "0xabc"
        enr.candidate.created_at_ms = now_ms() - 90 * 60_000
        enr.features.twitter_mentions = 20
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertFalse(any(v.startswith("age:") for v in vetoes), vetoes)

    def test_robinhood_silence_on_twitter_vetoes_when_measured(self):
        enr = self.enrichment()
        enr.mint = _FakeMint(None, None)
        enr.candidate.chain = Chain.ROBINHOOD_CHAIN
        enr.candidate.dex_id = "pons"
        enr.candidate.address = "0xabc"
        enr.features.twitter_mentions = 0
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertTrue(any("twitter:" in v for v in vetoes), vetoes)

    def test_priced_in_past_copy_window_vetoes(self):
        enr = self.enrichment()
        enr.mint = _FakeMint(None, None)
        enr.candidate.mcap_usd = 3_000_000
        enr.candidate.ret_5m = 0.35
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertTrue(any(v.startswith("priced:") for v in vetoes), vetoes)

    def test_early_rip_inside_copy_window_is_not_priced(self):
        enr = self.enrichment(parabolic=0.80)
        enr.mint = _FakeMint(None, None)
        enr.candidate.mcap_usd = 150_000
        enr.candidate.ret_5m = 0.40
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertFalse(any(v.startswith("priced:") for v in vetoes), vetoes)

    def test_official_twitter_quiet_vetoes(self):
        enr = self.enrichment()
        enr.mint = _FakeMint(None, None)
        enr.twitter = {"official": "oddcto", "official_age_min": 500, "posts": []}
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertTrue(any("official quiet" in v for v in vetoes), vetoes)


class _FakeMint:
    def __init__(self, mint_authority, freeze_authority):
        self.mint_authority = mint_authority
        self.freeze_authority = freeze_authority


class TestRisk(unittest.TestCase):
    def setUp(self):
        self.store, self._tmp = make_store()
        self.risk = RiskEngine(STRATEGY, self.store)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_kelly_refuses_a_negative_edge(self):
        payoff = Payoff(avg_win=0.30, avg_loss=0.30, samples=50, from_history=True)
        self.assertEqual(kelly_fraction(0.40, payoff), 0.0)
        self.assertGreater(kelly_fraction(0.70, payoff), 0.0)

    def test_thin_pool_binds_before_the_equity_cap(self):
        score = Score(probability=0.90, expected_value=0.5)
        payoff = Payoff(avg_win=1.0, avg_loss=0.25, samples=50, from_history=True)
        thin = Candidate(
            chain=Chain.SOLANA,
            address="a",
            liquidity_usd=2_000.0,
            price_usd=1.0,
            mcap_usd=2_000_000.0,
        )
        sizing = self.risk.size(thin, score, payoff, [])
        self.assertTrue(sizing.allowed)
        self.assertEqual(sizing.binding_constraint, "pool_liquidity")
        self.assertLessEqual(sizing.size_usd, 2_000.0 * 0.008 + 1e-9)

    def test_mcap_scales_size_with_the_bankroll(self):
        self.assertAlmostEqual(
            mcap_position_pct(100_000, 100_000, 2_000_000, 0.06, 0.16), 0.06
        )
        self.assertAlmostEqual(
            mcap_position_pct(2_000_000, 100_000, 2_000_000, 0.06, 0.16), 0.16
        )
        score = Score(probability=0.90, expected_value=0.5)
        payoff = Payoff(avg_win=1.0, avg_loss=0.25, samples=50, from_history=True)
        lo = Candidate(
            chain=Chain.SOLANA,
            address="lo",
            liquidity_usd=10**7,
            price_usd=1.0,
            mcap_usd=100_000.0,
        )
        hi = Candidate(
            chain=Chain.SOLANA,
            address="hi",
            liquidity_usd=10**7,
            price_usd=1.0,
            mcap_usd=2_000_000.0,
        )
        a = self.risk.size(lo, score, payoff, [])
        b = self.risk.size(hi, score, payoff, [])
        self.assertTrue(a.allowed and b.allowed)
        self.assertEqual(a.binding_constraint, "mcap_size")
        self.assertEqual(b.binding_constraint, "mcap_size")
        self.assertAlmostEqual(a.size_usd, self.risk.equity() * 0.06, places=4)
        self.assertAlmostEqual(b.size_usd, self.risk.equity() * 0.16, places=4)
        self.assertGreater(b.size_usd, a.size_usd)

    def test_kill_switch_stops_sizing(self):
        self.risk.engage_kill_switch("test")
        sizing = self.risk.size(
            Candidate(chain=Chain.SOLANA, address="a", liquidity_usd=10**6, price_usd=1.0),
            Score(probability=0.99, expected_value=1.0),
            Payoff(1.0, 0.2, 50, True),
            [],
        )
        self.assertFalse(sizing.allowed)
        self.assertIn("kill switch", sizing.reason)

    def test_equity_shrinks_with_realized_losses(self):
        before = self.risk.equity()
        self.store.record_trade(_trade(pnl=-200.0))
        self.assertAlmostEqual(self.risk.equity(), before - 200.0, places=6)

    def test_already_holding_blocks_a_second_order(self):
        score = Score(probability=0.90, expected_value=0.5)
        payoff = Payoff(avg_win=1.0, avg_loss=0.25, samples=50, from_history=True)
        c = Candidate(chain=Chain.SOLANA, address="mint", liquidity_usd=10**6, price_usd=1.0)
        held = Position(
            candidate=c, venue=VenueId.PAPER, entry_price=1.0, size_usd=20.0, tokens=20.0
        )
        sizing = self.risk.size(c, score, payoff, [held])
        self.assertFalse(sizing.allowed)
        self.assertIn("already", sizing.reason)

    def test_reentry_cooldown_after_a_close(self):
        self.store.record_trade(_trade(pnl=1.0))
        score = Score(probability=0.90, expected_value=0.5)
        payoff = Payoff(avg_win=1.0, avg_loss=0.25, samples=50, from_history=True)
        c = Candidate(chain=Chain.SOLANA, address="mint", liquidity_usd=10**6, price_usd=1.0)
        sizing = self.risk.size(c, score, payoff, [])
        self.assertFalse(sizing.allowed)
        self.assertIn("recently", sizing.reason)

    def test_streak_does_not_halt_when_cooldown_is_off(self):
        from alphahound.risk import COOLDOWN_UNTIL_KEY

        self.store.record_trade(_trade(pnl=-1.0))
        self.risk.note_trade_closed(won=False)
        self.assertEqual(self.store.get_kv(COOLDOWN_UNTIL_KEY, "0"), "0")
        self.store.set_kv(COOLDOWN_UNTIL_KEY, str(10**15))
        halted, reason = self.risk.halted()
        self.assertFalse(halted, reason)

    def test_daily_entry_cap(self):
        self.store.set_param("risk.max_entries_per_day", 5, "test")
        score = Score(probability=0.90, expected_value=0.5)
        payoff = Payoff(avg_win=1.0, avg_loss=0.25, samples=50, from_history=True)
        for i in range(5):
            self.store.record_decision(
                Decision(
                    candidate=Candidate(
                        chain=Chain.SOLANA, address=f"m{i}", liquidity_usd=10**6, price_usd=1.0
                    ),
                    features=Features(),
                    score=score,
                    action=Action.ENTER,
                    size_usd=10.0,
                )
            )
        sizing = self.risk.size(
            Candidate(chain=Chain.SOLANA, address="fresh", liquidity_usd=10**6, price_usd=1.0),
            score,
            payoff,
            [],
        )
        self.assertFalse(sizing.allowed)
        self.assertIn("daily cap", sizing.reason)


class TestAggressiveLearning(unittest.TestCase):
    def setUp(self):
        self.store, self._tmp = make_store()

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_overlay_more_slots_same_total_cap(self):
        cfg = apply_aggressive_learning(STRATEGY, self.store)
        self.assertTrue(cfg.get("aggressive_learning._active"))
        self.assertEqual(int(cfg.get("risk.max_concurrent_positions")), 4)
        self.assertAlmostEqual(float(cfg.get("risk.max_position_pct")), 0.08, places=4)
        self.assertGreaterEqual(float(cfg.get("risk.min_position_pct")), 0.05 - 1e-9)
        self.store.set_param("scoring.min_expected_value", 0.048, "test")
        p, ev = score_floors(cfg, self.store)
        self.assertAlmostEqual(p, 0.38, places=4)
        self.assertAlmostEqual(ev, 0.03, places=4)

    def test_graduates_back_to_production(self):
        for _ in range(40):
            self.store.record_trade(_trade(pnl=-1.0))
        cfg = apply_aggressive_learning(STRATEGY, self.store)
        self.assertFalse(cfg.get("aggressive_learning._active"))
        self.assertEqual(int(cfg.get("risk.max_concurrent_positions")), 2)
        self.assertEqual(self.store.get_kv("aggressive_graduated"), "1")

    def test_tiny_bankroll_keeps_two_slots(self):
        thin = STRATEGY.overlay({"risk": {"equity_usd": 40.0}})
        cfg = apply_aggressive_learning(thin, self.store)
        self.assertEqual(int(cfg.get("risk.max_concurrent_positions")), 2)


class TestExits(unittest.TestCase):
    def setUp(self):
        self.store, self._tmp = make_store()
        self.manager = PositionManager(STRATEGY, self.store)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def position(self) -> Position:
        candidate = Candidate(
            chain=Chain.SOLANA, address="a", price_usd=1.0, liquidity_usd=100_000.0
        )
        return Position(
            candidate=candidate,
            venue=VenueId.PAPER,
            entry_price=1.0,
            size_usd=100.0,
            tokens=100.0,
        )

    def test_ladder_fractions_are_of_the_original_position(self):
        position = self.position()
        # A jump straight past every rung must sell the ladder (60%), not one
        # rung. The remaining 40% is the runner and waits for the trail.
        orders = self.manager.evaluate(position, 5.0, 100_000.0)
        self.assertEqual(len(orders), 3)
        remaining = 1.0
        sold = 0.0
        for order in orders:
            sold += remaining * order.fraction
            remaining -= remaining * order.fraction
        self.assertAlmostEqual(sold, 0.60, places=6)

    def test_liquidity_drain_outranks_take_profit(self):
        position = self.position()
        self.manager.observe(position, 1.0, 100_000.0)
        orders = self.manager.evaluate(position, 4.0, 40_000.0)
        self.assertEqual(len(orders), 1)
        self.assertIs(orders[0].reason, ExitReason.LIQUIDITY_DRAIN)
        self.assertEqual(orders[0].fraction, 1.0)

    def test_stale_peak_liq_after_restore_is_not_a_drain(self):
        position = self.position()
        self.manager.observe(position, 1.0, 162_000.0)
        position.peak_liquidity_usd = 0.0
        orders = self.manager.evaluate(position, 0.80, 91_000.0)
        self.assertFalse(any(o.reason is ExitReason.LIQUIDITY_DRAIN for o in orders))

    def test_stop_loss(self):
        orders = self.manager.evaluate(self.position(), 0.50, 100_000.0)
        self.assertIs(orders[0].reason, ExitReason.STOP_LOSS)

    def test_trailing_stop_only_arms_after_a_rung_fills(self):
        position = self.position()
        self.manager.observe(position, 1.30, 100_000.0)
        # Up 30%, then back to break-even: rung one never filled, so the trail
        # is not armed and the hard stop has not been hit.
        self.assertEqual(self.manager.evaluate(position, 1.0, 100_000.0), [])

        armed = self.position()
        self.manager.evaluate(armed, 1.50, 100_000.0)
        self.assertTrue(armed.trailing_active)
        orders = self.manager.evaluate(armed, 1.50 * 0.70, 100_000.0)
        self.assertIs(orders[0].reason, ExitReason.TRAILING_STOP)

    def test_time_stop_releases_dead_capital(self):
        position = self.position()
        position.opened_at_ms = now_ms() - 250 * 60_000
        orders = self.manager.evaluate(position, 1.02, 100_000.0)
        self.assertIs(orders[0].reason, ExitReason.TIME_STOP)

    def test_fresh_hold_does_not_thesis_cut(self):
        position = self.position()
        self.manager.observe(position, 1.30, 100_000.0)
        self.assertEqual(self.manager.evaluate(position, 0.75, 100_000.0), [])
        position = self.position()
        position.opened_at_ms = now_ms() - 26 * 60_000
        self.manager.observe(position, 1.30, 100_000.0)
        # 42% off peak, still above the 35% hard stop from entry.
        orders = self.manager.evaluate(position, 0.75, 100_000.0)
        self.assertEqual(len(orders), 1)
        self.assertIs(orders[0].reason, ExitReason.THESIS_CUT)

    def test_tape_flip_sells_with_the_5m(self):
        position = self.position()
        position.opened_at_ms = now_ms() - 26 * 60_000
        position.candidate.ret_5m = -0.15
        orders = self.manager.evaluate(position, 1.12, 100_000.0)
        self.assertEqual(len(orders), 1)
        self.assertIs(orders[0].reason, ExitReason.THESIS_CUT)
        self.assertIn("tape", orders[0].note)

    def test_hot_tape_waits_past_twelve_percent(self):
        position = self.position()
        position.opened_at_ms = now_ms() - 26 * 60_000
        position.candidate.mcap_usd = 100_000
        position.candidate.volume_5m_usd = 80_000
        position.entry_vol_score = 1.0
        position.candidate.ret_5m = -0.15
        self.assertEqual(self.manager.evaluate(position, 1.12, 100_000.0), [])
        position.candidate.ret_5m = -0.20
        orders = self.manager.evaluate(position, 1.12, 100_000.0)
        self.assertEqual(len(orders), 1)
        self.assertIs(orders[0].reason, ExitReason.THESIS_CUT)

    def test_hot_trail_uses_token_regime_not_a_new_base(self):
        armed = self.position()
        armed.candidate.mcap_usd = 100_000
        armed.candidate.volume_5m_usd = 80_000
        armed.entry_vol_score = 1.0
        self.manager.evaluate(armed, 1.50, 100_000.0)
        self.assertTrue(armed.trailing_active)
        # 25% off 1.50 is inside a hot 22% * 1.5 trail, outside the raw 22%.
        self.assertEqual(self.manager.evaluate(armed, 1.50 * 0.75, 100_000.0), [])
        orders = self.manager.evaluate(armed, 1.50 * 0.65, 100_000.0)
        self.assertIs(orders[0].reason, ExitReason.TRAILING_STOP)

    def test_excursions(self):
        position = self.position()
        self.manager.observe(position, 3.0, 100_000.0)
        self.manager.observe(position, 0.6, 100_000.0)
        mfe, mae = PositionManager.excursions(position)
        self.assertAlmostEqual(mfe, 2.0, places=6)
        self.assertAlmostEqual(mae, -0.4, places=6)


def _trade(
    pnl: float = -50.0,
    *,
    exit_reason: ExitReason = ExitReason.STOP_LOSS,
    mfe: float = 0.0,
    entry_price: float = 1.0,
    signal_price: float = 1.0,
    slippage: float = 0.0,
    features: Features | None = None,
    error_class: ErrorClass = ErrorClass.NO_EDGE,
) -> TradeRecord:
    return TradeRecord(
        key="solana:mint",
        chain=Chain.SOLANA,
        venue=VenueId.PAPER,
        opened_at_ms=now_ms() - 60_000,
        closed_at_ms=now_ms(),
        entry_price=entry_price,
        exit_price=entry_price * (1.0 + pnl / 100.0),
        signal_price=signal_price,
        size_usd=100.0,
        pnl_usd=pnl,
        fees_usd=1.0,
        exit_reason=exit_reason,
        error_class=error_class,
        features=features or Features(),
        weights_version=0,
        max_favorable_excursion=mfe,
        entry_slippage=slippage,
    )


class TestPostmortem(unittest.TestCase):
    def test_liquidity_drain_is_a_rug_when_it_costs_money(self):
        trade = _trade(pnl=-20.0, exit_reason=ExitReason.LIQUIDITY_DRAIN)
        self.assertIs(learning.classify(trade, STRATEGY), ErrorClass.RUG)

    def test_winning_liquidity_drain_is_not_a_rug(self):
        trade = _trade(pnl=+65.0, exit_reason=ExitReason.LIQUIDITY_DRAIN, mfe=0.70)
        self.assertIs(learning.classify(trade, STRATEGY), ErrorClass.WIN)
        gave_back = _trade(pnl=+10.0, exit_reason=ExitReason.LIQUIDITY_DRAIN, mfe=2.5)
        self.assertIs(learning.classify(gave_back, STRATEGY), ErrorClass.EXIT_TOO_FAST)

    def test_late_entry_detected_from_the_signal_price_gap(self):
        trade = _trade(entry_price=1.20, signal_price=1.0)
        self.assertIs(learning.classify(trade, STRATEGY), ErrorClass.LATE_ENTRY)

    def test_slippage_blowout_outranks_late_entry(self):
        trade = _trade(entry_price=1.20, signal_price=1.0, slippage=0.35)
        self.assertIs(learning.classify(trade, STRATEGY), ErrorClass.SLIPPAGE_BLOWOUT)

    def test_gave_back_a_reachable_gain(self):
        trade = _trade(mfe=0.30)
        self.assertIs(learning.classify(trade, STRATEGY), ErrorClass.EXIT_TOO_SLOW)

    def test_adverse_selection_from_launch_bundles(self):
        trade = _trade(features=Features(bundle_pct=0.40))
        self.assertIs(learning.classify(trade, STRATEGY), ErrorClass.ADVERSE_SELECTION)

    def test_honest_bucket_when_nothing_else_explains_it(self):
        self.assertIs(learning.classify(_trade(), STRATEGY), ErrorClass.NO_EDGE)

    def test_a_win_that_gave_back_its_peak_is_still_flagged(self):
        trade = _trade(pnl=10.0, exit_reason=ExitReason.TRAILING_STOP, mfe=2.5)
        self.assertIs(learning.classify(trade, STRATEGY), ErrorClass.EXIT_TOO_FAST)


class TestSelfTuning(unittest.TestCase):
    def setUp(self):
        self.store, self._tmp = make_store()

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_nudges_are_bounded_by_the_config_value(self):
        base = float(STRATEGY["gates.min_liquidity_usd"])
        nudge = learning.NUDGES[ErrorClass.RUG][0]
        for _ in range(40):
            learning._apply_nudge(self.store, STRATEGY, nudge, "rug", 1.0)
        final = self.store.param(nudge.param, base)
        self.assertLessEqual(
            final,
            base * learning.MAX_DRIFT_MULTIPLE + 1e-6,
            "the bot must not be able to redesign the strategy behind your back",
        )

    def test_postmortem_ignores_a_class_that_is_not_costing_money(self):
        for _ in range(10):
            self.store.record_trade(
                _trade(pnl=+40.0, exit_reason=ExitReason.LIQUIDITY_DRAIN, error_class=ErrorClass.RUG)
            )
        report = learning.run_postmortem(self.store, STRATEGY)
        self.assertTrue(any("not costing money" in note for note in report.skipped))
        self.assertEqual(self.store.all_params(), {})

    def test_postmortem_acts_on_a_dominant_expensive_class(self):
        for _ in range(12):
            self.store.record_trade(
                _trade(pnl=-30.0, exit_reason=ExitReason.LIQUIDITY_DRAIN, error_class=ErrorClass.RUG)
            )
        learning.run_postmortem(self.store, STRATEGY)
        self.assertGreater(
            self.store.param("gates.min_liquidity_usd", 0.0),
            float(STRATEGY["gates.min_liquidity_usd"]),
        )

    def test_training_refuses_to_learn_from_too_little_data(self):
        for _ in range(5):
            self.store.record_trade(_trade())
        result = learning.train(self.store, STRATEGY)
        self.assertFalse(result.trained)
        self.assertIn("keeping prior weights", result.note)

    def test_filter_cost_surfaces_expensive_gates(self):
        from alphahound.models import Action, Decision

        for i in range(15):
            decision = Decision(
                candidate=Candidate(chain=Chain.SOLANA, address=f"m{i}", price_usd=1.0),
                features=Features(),
                score=Score(probability=0.4, expected_value=0.0),
                action=Action.REJECT_GATE,
                reason="liquidity: 9000 < 15000",
            )
            decision_id = self.store.record_decision(decision)
            self.store.resolve_shadow(decision_id, 0.80)
        costs = learning.filter_cost(self.store)
        self.assertTrue(costs)
        self.assertEqual(costs[0].gate, "liquidity")
        self.assertEqual(costs[0].would_have_won, 15)

        notes = learning.relax_costly_gates(self.store, STRATEGY)
        self.assertTrue(notes, "a gate whose rejections all won must be loosened")
        self.assertLess(
            self.store.param("gates.min_liquidity_usd", 0.0),
            float(STRATEGY["gates.min_liquidity_usd"]),
        )


class TestReviewHistory(unittest.TestCase):
    def setUp(self):
        self.store, self._tmp = make_store()

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_review_splits_fumble_from_correct_reject_and_entries(self):
        from alphahound.models import Score

        miss = Decision(
            candidate=Candidate(chain=Chain.SOLANA, address="dead", price_usd=1.0, symbol="DEAD"),
            features=Features(),
            score=Score(probability=0.3, expected_value=0.0, contributions={"retail_share": -0.5}),
            action=Action.REJECT_GATE,
            reason="liquidity: thin",
        )
        win_rej = Decision(
            candidate=Candidate(chain=Chain.SOLANA, address="run", price_usd=1.0, symbol="RUN"),
            features=Features(),
            score=Score(probability=0.4, expected_value=0.01, contributions={"copy_signal": 0.9}),
            action=Action.REJECT_SCORE,
            reason="score: p 0.40 < 0.42",
        )
        self.store.resolve_shadow(self.store.record_decision(miss), 0.05)
        self.store.resolve_shadow(self.store.record_decision(win_rej), 0.80)
        self.store.record_trade(_trade(pnl=12.0, error_class=ErrorClass.WIN))
        self.store.record_trade(_trade(pnl=-8.0, error_class=ErrorClass.NO_EDGE))

        all_rows = {r["outcome"] for r in self.store.review_history()["rows"]}
        self.assertEqual(
            all_rows,
            {"rejeicao_correta", "fumble", "entrada_correta", "entrada_errada"},
        )
        fumbles = self.store.review_history(outcome="fumble")["rows"]
        self.assertEqual(len(fumbles), 1)
        self.assertEqual(fumbles[0]["symbol"], "RUN")
        self.assertEqual(fumbles[0]["contrib"][0][0], "copy_signal")

    def test_review_keeps_first_sight_mcap(self):
        from alphahound.models import Score

        first = Decision(
            candidate=Candidate(
                chain=Chain.SOLANA, address="keep", price_usd=1.0, symbol="KEEP", mcap_usd=55_000
            ),
            features=Features(),
            score=Score(probability=0.2, expected_value=0.0),
            action=Action.REJECT_GATE,
            reason="mcap: 55000 below 100000 floor",
        )
        later = Decision(
            candidate=Candidate(
                chain=Chain.SOLANA, address="keep", price_usd=1.0, symbol="KEEP", mcap_usd=180_000
            ),
            features=Features(),
            score=Score(probability=0.2, expected_value=0.0),
            action=Action.REJECT_GATE,
            reason="cluster: 44% linked supply",
            ts_ms=first.ts_ms + 200_000,
        )
        self.store.resolve_shadow(self.store.record_decision(first), 0.01)
        self.store.resolve_shadow(self.store.record_decision(later), 0.01)
        rows = [r for r in self.store.review_history()["rows"] if r["symbol"] == "KEEP"]
        self.assertTrue(rows)
        self.assertEqual(rows[0]["mcap_first"], 55000)
        self.assertEqual(rows[-1]["mcap_first"], 55000)


class TestExitCopy(unittest.TestCase):
    def test_exit_and_gate_copy_match_known_codes(self):
        from alphahound.models import describe_code, describe_exit

        self.assertIn("stop duro", describe_exit("stop_loss"))
        self.assertIn("funding", describe_code("cluster: 44% linked supply"))
        self.assertIn("esfriou", describe_code("no_edge"))



class TestBacktest(unittest.TestCase):
    def test_simulated_exit_never_credits_perfect_timing(self):
        peak = 3.0
        banked = backtest.simulate_exit(peak, STRATEGY)
        self.assertLess(banked, peak)
        self.assertGreater(banked, 0.5)

    def test_a_position_that_only_fell_takes_the_stop(self):
        self.assertAlmostEqual(
            backtest.simulate_exit(-0.90, STRATEGY),
            -float(STRATEGY["exits.stop_loss_pct"]),
            places=9,
        )


try:  # httpx is the one non-stdlib dependency; the rest of this file needs none.
    from alphahound.providers import Dexscreener
except ImportError:  # pragma: no cover
    Dexscreener = None


@unittest.skipUnless(Dexscreener is not None, "httpx not installed")
class TestProviderCache(unittest.TestCase):
    def test_one_token_asked_for_repeatedly_costs_one_request(self):
        class FakeHttp:
            def __init__(self):
                self.calls = []

            def limit(self, *_args, **_kw):
                pass

            async def get(self, url, **_kw):
                self.calls.append(url)
                return {
                    "pairs": [
                        {
                            "chainId": "solana",
                            "baseToken": {"address": "MINT", "symbol": "X"},
                            "priceUsd": "1.0",
                            "liquidity": {"usd": 50_000},
                        }
                    ]
                }

        http = FakeHttp()
        dex = Dexscreener(http, cache_seconds=60.0)

        async def scenario():
            # The enricher refresh, then a buy quote, then the matching sell
            # quote - three callers, one tick, one token.
            for _ in range(3):
                snaps = await dex.token_pairs(["MINT"])
                self.assertEqual(len(snaps), 1)
                self.assertEqual(snaps[0].price_usd, 1.0)
            await dex.token_pairs(["MINT", "OTHER"])

        asyncio.run(scenario())
        self.assertEqual(len(http.calls), 2, "a cache hit must not hit the network")
        # The second call asks only for what it is missing.
        self.assertIn("other", http.calls[1].lower())
        self.assertNotIn("mint", http.calls[1].rsplit("/", 1)[-1].lower())

    def test_token_pairs_cache_ignores_address_case(self):
        class FakeHttp:
            def __init__(self):
                self.calls = 0

            def limit(self, *_args, **_kw):
                pass

            async def get(self, url, **_kw):
                self.calls += 1
                return {
                    "pairs": [
                        {
                            "chainId": "solana",
                            "baseToken": {"address": "0xAbC", "symbol": "X"},
                            "priceUsd": "1.0",
                            "liquidity": {"usd": 50_000},
                            "marketCap": 80_000,
                        }
                    ]
                }

        http = FakeHttp()
        dex = Dexscreener(http, cache_seconds=60.0)

        async def scenario():
            a = await dex.token_pairs(["0xabc"])
            b = await dex.token_pairs(["0xAbC"])
            self.assertEqual(len(a), 1)
            self.assertEqual(len(b), 1)
            self.assertEqual(a[0].mcap_usd, 80_000)

        asyncio.run(scenario())
        self.assertEqual(http.calls, 1)

    def test_token_pairs_keeps_last_quote_on_fetch_fail(self):
        class FakeHttp:
            def __init__(self):
                self.n = 0

            def limit(self, *_args, **_kw):
                pass

            async def get(self, url, **_kw):
                from alphahound.net import HttpError

                self.n += 1
                if self.n == 1:
                    return {
                        "pairs": [
                            {
                                "chainId": "solana",
                                "baseToken": {"address": "mint", "symbol": "X"},
                                "priceUsd": "1.0",
                                "liquidity": {"usd": 50_000},
                                "marketCap": 90_000,
                            }
                        ]
                    }
                raise HttpError(429, "slow down", url)

        http = FakeHttp()
        dex = Dexscreener(http, cache_seconds=0.0)

        async def scenario():
            first = await dex.token_pairs(["mint"])
            second = await dex.token_pairs(["mint"])
            self.assertEqual(first[0].mcap_usd, 90_000)
            self.assertEqual(second[0].mcap_usd, 90_000)

        asyncio.run(scenario())

    def test_stamp_uses_pair_birth_not_visor_arrival(self):
        from alphahound.providers import PairSnapshot, pair_created_ms

        self.assertEqual(pair_created_ms(1_700_000_000), 1_700_000_000_000)
        self.assertEqual(pair_created_ms(1_700_000_000_000), 1_700_000_000_000)
        birth = now_ms() - 90 * 60_000
        seen = now_ms() - 60_000
        snap = PairSnapshot(
            chain=Chain.ROBINHOOD_CHAIN,
            pair_address="0xpool",
            token_address="0xtoken",
            symbol="AD",
            name="AD",
            price_usd=1.0,
            liquidity_usd=1.0,
            mcap_usd=1.0,
            volume_m5=1.0,
            volume_h1=1.0,
            buys_m5=1,
            sells_m5=1,
            price_change_m5=0.0,
            price_change_h1=0.0,
            created_at_ms=birth,
        )
        c = Candidate(
            chain=Chain.ROBINHOOD_CHAIN,
            address="0xtoken",
            created_at_ms=seen,
            source="hood_stream",
        )
        snap.stamp(c)
        self.assertEqual(c.created_at_ms, birth)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.store, self._tmp = make_store()

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_same_reject_bumps_ticks_one_shadow(self):
        cand = Candidate(chain=Chain.SOLANA, address="mint", price_usd=1.0)

        def rej(reason: str = "liquidity: thin") -> Decision:
            return Decision(
                candidate=cand,
                features=Features(),
                score=Score(probability=0.3, expected_value=0.0),
                action=Action.REJECT_GATE,
                reason=reason,
            )

        first = self.store.record_decision(rej())
        again = self.store.record_decision(rej())
        self.assertEqual(first, again)
        row = self.store.conn.execute(
            "SELECT COUNT(*) AS n, ticks FROM decisions WHERE id = ?", (first,)
        ).fetchone()
        self.assertEqual(row["n"], 1)
        self.assertEqual(row["ticks"], 2)
        self.assertEqual(
            self.store.conn.execute("SELECT COUNT(*) FROM shadow").fetchone()[0], 1
        )
        other = self.store.record_decision(rej("cluster: 30%"))
        self.assertNotEqual(other, first)
        same_p = self.store.record_decision(
            Decision(
                candidate=cand,
                features=Features(),
                score=Score(probability=0.3, expected_value=0.0),
                action=Action.REJECT_SCORE,
                reason="probability 0.389 < 0.420",
            )
        )
        same_p2 = self.store.record_decision(
            Decision(
                candidate=cand,
                features=Features(),
                score=Score(probability=0.31, expected_value=0.0),
                action=Action.REJECT_SCORE,
                reason="probability 0.396 < 0.420",
            )
        )
        self.assertEqual(same_p, same_p2)
        self.assertEqual(
            self.store.conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0], 3
        )
        self.assertEqual(
            self.store.conn.execute("SELECT COUNT(*) FROM shadow").fetchone()[0], 3
        )

    def test_reject_after_gap_without_open_shadow_is_a_new_episode(self):
        cand = Candidate(chain=Chain.SOLANA, address="mint", price_usd=1.0)
        d = Decision(
            candidate=cand,
            features=Features(),
            score=Score(probability=0.3, expected_value=0.0),
            action=Action.REJECT_GATE,
            reason="liquidity: thin",
        )
        first = self.store.record_decision(d)
        self.store.resolve_shadow(first, 0.1)
        self.store.conn.execute(
            "UPDATE decisions SET last_ts_ms = ? WHERE id = ?",
            (now_ms() - Store.EPISODE_GAP_MS - 1, first),
        )
        second = self.store.record_decision(
            Decision(
                candidate=cand,
                features=Features(),
                score=Score(probability=0.3, expected_value=0.0),
                action=Action.REJECT_GATE,
                reason="liquidity: thin",
            )
        )
        self.assertNotEqual(second, first)
        self.assertEqual(
            self.store.conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0], 2
        )

    def test_trade_round_trip_preserves_features(self):
        features = Features(retail_share=0.33, bundle_pct=0.11, axiom_share=0.22)
        self.store.record_trade(_trade(features=features))
        loaded = self.store.trades()[0]
        self.assertAlmostEqual(loaded.features.retail_share, 0.33, places=9)
        self.assertAlmostEqual(loaded.features.axiom_share, 0.22, places=9)

    def test_trade_round_trip_preserves_mcap(self):
        trade = _trade()
        trade.symbol = "ODD"
        trade.mcap_entry_usd = 80_000
        trade.mcap_exit_usd = 120_000
        self.store.record_trade(trade)
        loaded = self.store.trades()[0]
        self.assertEqual(loaded.symbol, "ODD")
        self.assertAlmostEqual(loaded.mcap_entry_usd, 80_000)
        self.assertAlmostEqual(loaded.mcap_exit_usd, 120_000)

    def test_trade_round_trip_preserves_exit_note(self):
        trade = _trade()
        trade.notes = "5m flipped, selling with tape"
        trade.exit_legs = [
            {
                "ts_ms": trade.closed_at_ms,
                "reason": "thesis_cut",
                "note": "5m flipped, selling with tape",
            }
        ]
        self.store.record_trade(trade)
        loaded = self.store.trades()[0]
        self.assertEqual(loaded.notes, "5m flipped, selling with tape")
        self.assertEqual(loaded.exit_legs[0]["note"], "5m flipped, selling with tape")

    def test_exit_legs_keep_vol5m_at_the_sell_tick(self):
        trade = _trade()
        trade.exit_legs = [
            {
                "ts_ms": trade.closed_at_ms,
                "reason": "trailing_stop",
                "vol5m": 12345.6,
                "holder_growth_5m": 2.5,
            }
        ]
        self.store.record_trade(trade)
        shown = legs_for_display(self.store.trades()[0])
        self.assertAlmostEqual(shown[0]["vol5m"], 12345.6)
        self.assertAlmostEqual(shown[0]["holder_growth_5m"], 2.5)

    def test_unmeasured_features_survive_a_round_trip_to_the_learner(self):
        # A 0.0 that was never measured must not train the model as if it were
        # observed: for most normalizers 0.0 is a real, non-neutral value.
        trade = TradeRecord(
            key="solana:mint",
            chain=Chain.SOLANA,
            venue=VenueId.PAPER,
            opened_at_ms=now_ms(),
            closed_at_ms=now_ms(),
            entry_price=1.0,
            exit_price=0.8,
            size_usd=50.0,
            pnl_usd=-10.0,
            fees_usd=1.0,
            exit_reason=ExitReason.STOP_LOSS,
            error_class=ErrorClass.NO_EDGE,
            features=Features(top10_pct=0.0, liquidity_usd=30_000.0),
            weights_version=1,
            unknown={"top10_pct"},
        )
        self.store.record_trade(trade)
        loaded = self.store.trades()[0]
        self.assertEqual(loaded.unknown, {"top10_pct"})
        self.assertEqual(normalize(loaded.features, loaded.unknown)["top10_pct"], 0.0)
        self.assertNotEqual(normalize(loaded.features)["top10_pct"], 0.0)

    def test_a_second_engine_cannot_share_a_state_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state"
            held = lock_state_dir(state)
            try:
                with self.assertRaises(RuntimeError):
                    lock_state_dir(state)
            finally:
                held.close()
            # Released, so a restart works without manual cleanup.
            lock_state_dir(state).close()

    def test_a_wallet_becomes_smart_after_two_wins(self):
        self.store.record_buyer_outcome("KOL1", Chain.SOLANA, +40.0)
        self.assertNotIn("KOL1", self.store.smart_wallets(Chain.SOLANA))
        self.store.record_buyer_outcome("KOL1", Chain.SOLANA, +20.0)
        self.assertIn("KOL1", self.store.smart_wallets(Chain.SOLANA))
        self.store.record_buyer_outcome("SNIPER", Chain.SOLANA, -50.0)
        self.store.record_buyer_outcome("SNIPER", Chain.SOLANA, -50.0)
        self.assertNotIn("SNIPER", self.store.smart_wallets(Chain.SOLANA))

    def test_consecutive_losses(self):
        self.store.record_trade(_trade(pnl=-10.0))
        self.store.record_trade(_trade(pnl=-10.0))
        self.assertEqual(self.store.consecutive_losses(), 2)
        self.store.record_trade(_trade(pnl=+10.0))
        self.assertEqual(self.store.consecutive_losses(), 0)

    def test_param_changes_leave_an_audit_trail(self):
        self.store.set_param("exits.stop_loss_pct", 0.20, "because")
        self.store.set_param("exits.stop_loss_pct", 0.24, "changed mind")
        history = self.store.param_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["reason"], "changed mind")
        self.assertAlmostEqual(history[0]["old_value"], 0.20, places=9)

    def test_weights_versioning_and_activation(self):
        v1 = self.store.save_weights({"weights": {"a": 1.0}, "bias": 0.0}, 100, 0.60, activate=True)
        v2 = self.store.save_weights({"weights": {"a": 2.0}, "bias": 0.0}, 120, 0.55)
        self.assertEqual(self.store.active_weights()[0], v1)
        self.store.activate_weights(v2)
        self.assertEqual(self.store.active_weights()[0], v2)


class TestCrowd(unittest.TestCase):
    def test_whale_hold_pct_and_buy_vs_sell(self):
        from alphahound.signals.distribution import Holder
        from alphahound.signals.whales import crowd_read

        holders = [
            Holder(address="W1", balance=50),
            Holder(address="W2", balance=40),
            Holder(address="R", balance=10),
        ]
        labeled = crowd_read(holders, [], {"W1"})
        self.assertEqual(labeled.inside, 1)
        self.assertAlmostEqual(labeled.hold_pct, 0.50)

        buying = crowd_read(
            holders,
            [
                flow.Trade(ts_ms=1, side=Side.BUY, price=1, size_usd=800, wallet="W1"),
                flow.Trade(ts_ms=1, side=Side.SELL, price=1, size_usd=200, wallet="W1"),
            ],
            {"W1"},
        )
        self.assertGreater(buying.net_flow, 0)

        selling = crowd_read(
            holders,
            [flow.Trade(ts_ms=1, side=Side.SELL, price=1, size_usd=500, wallet="W1")],
            {"W1"},
        )
        self.assertLess(selling.net_flow, 0)

        sized = crowd_read(holders, [], set(), size_pct=0.15)
        self.assertEqual(sized.inside, 2)
        self.assertAlmostEqual(sized.hold_pct, 0.90)

    def test_fomo_supply_pct_is_a_score_prior_not_a_gate(self):
        from alphahound.models import Features
        from alphahound.scoring import PRIOR_WEIGHTS, normalize

        self.assertIn("fomo_supply_pct", Features.names())
        self.assertGreater(PRIOR_WEIGHTS["fomo_supply_pct"], PRIOR_WEIGHTS["fomo_inside"])
        raw = normalize(Features(fomo_supply_pct=0.25), unknown=set())
        missing = normalize(Features(fomo_supply_pct=0.25), unknown={"fomo_supply_pct"})
        self.assertAlmostEqual(raw["fomo_supply_pct"], 0.25)
        self.assertEqual(missing["fomo_supply_pct"], 0.0)
        src = Path(__file__).resolve().parents[1] / "src" / "alphahound" / "scoring.py"
        gates = src.read_text(encoding="utf-8")
        # evaluate_gates lives in this file; the new feature must not be named there.
        gate_fn = gates[gates.index("def evaluate_gates") : gates.index("def patience_only")]
        self.assertNotIn("fomo_supply_pct", gate_fn)

    def test_who_inside_names_labeled_kols(self):
        from alphahound.signals.distribution import Holder
        from alphahound.signals.whales import who_inside

        holders = [
            Holder(address="K1", balance=30),
            Holder(address="R", balance=70),
        ]
        names = who_inside(
            holders,
            [flow.Trade(ts_ms=1, side=Side.BUY, price=1, size_usd=10, wallet="K2")],
            {"K1": "bagu_2", "K2": "yodacalls"},
        )
        self.assertEqual(names, ["bagu_2", "yodacalls"])

    def test_chase_false_is_not_copied(self):
        from alphahound.settings import crowd_addresses

        rows = [
            {"address": "FOMO1", "source": "fomo", "chase": False},
            {"address": "FOMO2", "source": "fomo"},
            {"address": "WHALE1", "source": "moby", "class": "whale"},
        ]
        self.assertEqual(crowd_addresses(rows, "fomo"), {"FOMO2"})
        self.assertEqual(crowd_addresses(rows, "whale"), {"WHALE1"})

    def test_cope_wallet_walk_skips_the_mint(self):
        from alphahound.signals.whales import wallets_in

        payload = {
            "positions": [
                {"handle": "alpha", "wallet": "So11111111111111111111111111111111111111112"},
                {"address": "MintMintMintMintMintMintMintMintMintMint111"},
            ]
        }
        found = wallets_in(payload, skip={"MintMintMintMintMintMintMintMintMintMint111"})
        self.assertEqual(found, {"So11111111111111111111111111111111111111112"})


class TestFees(unittest.TestCase):
    def test_thin_book_asks_more_slip_than_a_busy_one(self):
        from alphahound.fees import plan_for

        thin = Candidate(
            chain=Chain.SOLANA, address="a", volume_5m_usd=3_000, liquidity_usd=20_000
        )
        busy = Candidate(
            chain=Chain.SOLANA, address="b", volume_5m_usd=80_000, liquidity_usd=200_000
        )
        self.assertGreater(
            plan_for(thin, STRATEGY, 0).slippage_bps,
            plan_for(busy, STRATEGY, 0).slippage_bps,
        )

    def test_retries_bump_then_hard_cap(self):
        from alphahound.fees import plan_for

        c = Candidate(chain=Chain.SOLANA, address="a", volume_5m_usd=15_000)
        p0 = plan_for(c, STRATEGY, 0)
        p1 = plan_for(c, STRATEGY, 1)
        p2 = plan_for(c, STRATEGY, 2)
        p9 = plan_for(c, STRATEGY, 9)
        self.assertGreater(p1.slippage_bps, p0.slippage_bps)
        self.assertGreater(p2.slippage_bps, p1.slippage_bps)
        self.assertEqual(p9.slippage_bps, p2.slippage_bps)
        self.assertEqual(p9.priority_lamports, p2.priority_lamports)
        self.assertLessEqual(p2.slippage_bps, int(STRATEGY.get("execution.max_slippage_bps", 350)))
        self.assertLessEqual(
            p2.priority_lamports, int(STRATEGY.get("execution.max_priority_lamports", 350_000))
        )


class TestPlaybook(unittest.TestCase):
    def test_copy_signal_only_on_young_small_launches(self):
        from alphahound.playbook import copy_signal

        buying = dict(
            smart_buys=3.0,
            fomo_inside=0.0,
            fomo_net_flow=0.0,
            whale_net_flow=0.0,
            strategy=STRATEGY,
            chain=Chain.SOLANA,
        )
        self.assertEqual(copy_signal(age_minutes=8, mcap_usd=150_000, **buying), 1.0)
        self.assertEqual(copy_signal(age_minutes=120, mcap_usd=150_000, **buying), 0.0)
        self.assertEqual(copy_signal(age_minutes=8, mcap_usd=5_000_000, **buying), 0.0)
        self.assertEqual(copy_signal(age_minutes=8, mcap_usd=51_000_000, **buying), 0.0)
        self.assertEqual(copy_signal(age_minutes=8, mcap_usd=80_000, **buying), 0.0)
        idle = dict(buying, smart_buys=0.0)
        self.assertEqual(copy_signal(age_minutes=8, mcap_usd=150_000, **idle), 0.0)

    def test_evm_key_falls_back_then_overrides(self):
        from alphahound.settings import Settings

        shared = Settings(evm_private_key="0xaaa")
        self.assertEqual(shared.evm_key_for(Chain.BNB), "0xaaa")
        split = Settings(
            evm_private_key="0xaaa",
            evm_keys={Chain.BNB: "0xbbb", Chain.ROBINHOOD_CHAIN: "0xccc"},
        )
        self.assertEqual(split.evm_key_for(Chain.BNB), "0xbbb")
        self.assertEqual(split.evm_key_for(Chain.ROBINHOOD_CHAIN), "0xccc")
        self.assertEqual(split.evm_key_for(Chain.BASE), "0xaaa")

    def test_solana_window_is_shorter_than_robinhood(self):
        from alphahound.playbook import max_age_minutes

        self.assertLess(
            max_age_minutes(STRATEGY, Chain.SOLANA),
            max_age_minutes(STRATEGY, Chain.ROBINHOOD_CHAIN),
        )

    def test_kols_round_trip(self):
        from alphahound.settings import load_kols, save_kols

        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name)
        save_kols(path, [{"address": "Abc", "handle": "@x", "class": "kol"}])
        rows = load_kols(path)
        self.assertEqual(rows[0]["address"], "Abc")
        tmp.cleanup()

    def test_fomo_round_trip_and_chase(self):
        from alphahound.settings import chase_rows, crowd_addresses, load_fomo, save_fomo

        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name)
        save_fomo(
            path,
            [
                {
                    "address": "FomoWallet111111111111111111111111111111111",
                    "name": "alpha",
                    "handle": "alpha",
                    "class": "fomo",
                    "source": "fomo",
                    "chase": True,
                }
            ],
        )
        rows = load_fomo(path)
        self.assertEqual(rows[0]["name"], "alpha")
        self.assertEqual(
            crowd_addresses(rows, "fomo"),
            {"FomoWallet111111111111111111111111111111111"},
        )
        self.assertIn(rows[0]["address"], {r["address"] for r in chase_rows(path)})
        tmp.cleanup()

    def test_inspect_keeps_old_token(self):
        from unittest.mock import MagicMock

        from alphahound.discovery import Discovery, DiscoveryStats

        d = Discovery.__new__(Discovery)
        d.settings = MagicMock()
        d.settings.enabled_chains = {Chain.SOLANA}
        d.strategy = STRATEGY
        d.stats = DiscoveryStats()
        d._seen = {}
        d._seen_addr = {}
        old = Candidate(
            chain=Chain.SOLANA,
            address="CTPoyCwkjMvoJwU4xvZZqoD8tiYk6yDchySiN5gGpump",
            created_at_ms=now_ms() - 3 * 3600 * 1000,
            source="inspect",
        )
        self.assertTrue(d._accept(old))
        stale = Candidate(
            chain=Chain.SOLANA,
            address="OtherMint111111111111111111111111111111111",
            created_at_ms=now_ms() - 3 * 3600 * 1000,
            source="dexscreener",
        )
        self.assertFalse(d._accept(stale))
        d._seen_addr["mint"] = now_ms()
        self.assertFalse(d._pair_unknown("MINT"))
        self.assertTrue(d._pair_unknown("fresh"))

    def test_hood_factory_log_emits_the_non_weth_token(self):
        from alphahound.discovery import (
            PONS_FACTORY,
            TOPIC_POOL_CREATED,
            TOPIC_TOKEN_LAUNCHED,
            UNI_V3_FACTORY,
            candidates_from_hood_log,
        )

        weth = "0x0Bd7D308f8E1639FAb988df18A8011f41EAcAD97"
        meme = "0xa3602804e096cb73bd8344afc1ff3f3390b899c5"
        pad = lambda a: "0x" + a[2:].lower().rjust(64, "0")
        found = candidates_from_hood_log(
            {
                "address": UNI_V3_FACTORY,
                "topics": [TOPIC_POOL_CREATED, pad(weth), pad(meme), "0x" + "0" * 62 + "0bb8"],
                "data": "0x" + "0" * 64 + "0" * 24 + "b" * 40,
            },
            weth=weth,
        )
        self.assertEqual(len(found), 0)

        pons = candidates_from_hood_log(
            {
                "address": PONS_FACTORY,
                "topics": [TOPIC_TOKEN_LAUNCHED, pad(meme), pad("0x" + "1" * 40), pad(UNI_V3_FACTORY)],
                "data": "0x",
            },
            weth=weth,
        )
        self.assertEqual(pons[0].address.lower(), meme)
        self.assertEqual(pons[0].dex_id, "pons")

    def test_hood_stream_is_observe_not_a_buy_bypass(self):
        from alphahound.engine import early_observe_floors, observe_early

        self.assertTrue(observe_early("hood_stream"))
        self.assertFalse(observe_early("dexscreener_profiles"))
        baby = Candidate(
            chain=Chain.ROBINHOOD_CHAIN,
            address="0x" + "ab" * 20,
            source="hood_stream",
            mcap_usd=0.0,
            volume_5m_usd=0.0,
            liquidity_usd=0.0,
        )
        floors = early_observe_floors(baby, STRATEGY)
        self.assertTrue(any(v.startswith("mcap:") for v in floors), floors)
        self.assertTrue(any(v.startswith("volume:") for v in floors), floors)
        self.assertTrue(any(v.startswith("liquidity:") for v in floors), floors)
        ripe = Candidate(
            chain=Chain.ROBINHOOD_CHAIN,
            address="0x" + "cd" * 20,
            source="hood_stream",
            mcap_usd=150_000,
            volume_5m_usd=8_000,
            liquidity_usd=20_000,
        )
        self.assertEqual(early_observe_floors(ripe, STRATEGY), [])

    def test_buy_floors_and_chase_still_get_a_visor_card(self):
        from alphahound.engine import Engine
        from alphahound.scoring import hide_from_visor, wait_on_visor

        self.assertTrue(wait_on_visor(["chase: 5m ripped, wait dip"]))
        self.assertTrue(wait_on_visor(["mcap: 80000 below 100000 floor"]))
        self.assertTrue(wait_on_visor(["volume: 2000 < 5000 (5m)", "chase: 5m ripped, wait dip"]))
        self.assertFalse(wait_on_visor(["lp_unlocked: 100% da liquidez livre"]))
        self.assertFalse(wait_on_visor(["cluster: 37% linked supply"]))
        self.assertFalse(hide_from_visor(["chase: 5m ripped, wait dip"]))
        self.assertFalse(hide_from_visor(["mcap: 90000 below 100000 floor"]))
        self.assertTrue(hide_from_visor(["lp_unlocked: 100% da liquidez livre"]))
        self.assertTrue(hide_from_visor(["volume: 819 < 5000 (5m)", "cluster: 37% linked supply"]))
        self.assertFalse(
            hide_from_visor(["chase: 5m ripped, wait dip", "volume: 2000 < 5000 (5m)"])
        )
        Engine._assert_loop_helpers(Engine)

    def test_public_hood_rpc_has_no_jsonrpc_websocket(self):
        from alphahound.discovery import _hood_jsonrpc_ws

        self.assertIsNone(_hood_jsonrpc_ws("https://rpc.mainnet.chain.robinhood.com"))
        self.assertTrue(
            (_hood_jsonrpc_ws("https://robinhood-mainnet.g.alchemy.com/v2/x") or "").startswith(
                "wss://"
            )
        )

    def test_hood_factory_log_uses_block_timestamp_when_present(self):
        from alphahound.discovery import PONS_FACTORY, TOPIC_TOKEN_LAUNCHED, candidates_from_hood_log

        weth = "0x0Bd7D308f8E1639FAb988df18A8011f41EAcAD97"
        meme = "0xa3602804e096cb73bd8344afc1ff3f3390b899c5"
        pad = lambda a: "0x" + a[2:].lower().rjust(64, "0")
        found = candidates_from_hood_log(
            {
                "address": PONS_FACTORY,
                "topics": [TOPIC_TOKEN_LAUNCHED, pad(meme)],
                "data": "0x",
                "blockTimestamp": hex(1_700_000_000),
            },
            weth=weth,
        )
        self.assertEqual(found[0].created_at_ms, 1_700_000_000_000)

    def test_erc20_string_decode(self):
        from alphahound.execution.evm import decode_erc20_string

        # dynamic ABI string "PEPE"
        raw = "0x" + "0" * 62 + "20" + "0" * 62 + "04" + "50455045" + "0" * 56
        self.assertEqual(decode_erc20_string(raw), "PEPE")
        self.assertEqual(decode_erc20_string("0x" + "57455448".ljust(64, "0")), "WETH")
        self.assertEqual(decode_erc20_string("0x"), "")

    def test_pnl_curve_sums_per_chain(self):
        from alphahound.preview import pnl_curves

        t0 = now_ms()
        trades = [
            TradeRecord(
                key="solana:a",
                chain=Chain.SOLANA,
                venue=VenueId.PAPER,
                opened_at_ms=t0,
                closed_at_ms=t0 + 1,
                entry_price=1,
                exit_price=2,
                size_usd=10,
                pnl_usd=4,
                fees_usd=0,
                exit_reason=ExitReason.TAKE_PROFIT,
                error_class=ErrorClass.WIN,
                features=Features(),
                weights_version=0,
            ),
            TradeRecord(
                key="bnb:b",
                chain=Chain.BNB,
                venue=VenueId.PAPER,
                opened_at_ms=t0,
                closed_at_ms=t0 + 2,
                entry_price=1,
                exit_price=0.5,
                size_usd=10,
                pnl_usd=-3,
                fees_usd=0,
                exit_reason=ExitReason.STOP_LOSS,
                error_class=ErrorClass.RUG,
                features=Features(),
                weights_version=0,
            ),
        ]
        chart = pnl_curves(trades)
        self.assertEqual(chart["total"], 1)
        self.assertEqual(chart["by_chain"]["solana"], 4)
        self.assertEqual(chart["by_chain"]["bnb"], -3)
        self.assertEqual(chart["series"]["total"][-1]["y"], 1        )

    def test_pnl_windows_marks_partial_history(self):
        from alphahound.preview import pnl_windows
        from alphahound.store import Store

        root = Path(tempfile.mkdtemp())
        store = Store(root)
        now = now_ms()
        day = 86_400_000
        store.record_trade(
            TradeRecord(
                key="solana:a",
                chain=Chain.SOLANA,
                venue=VenueId.PAPER,
                opened_at_ms=now - 3 * day,
                closed_at_ms=now - 3 * day,
                entry_price=1,
                exit_price=2,
                size_usd=10,
                pnl_usd=5,
                fees_usd=0,
                exit_reason=ExitReason.TAKE_PROFIT,
                error_class=ErrorClass.WIN,
                features=Features(),
                weights_version=0,
            )
        )
        w = pnl_windows(store, now=now)
        by = {x["label"]: x for x in w["windows"]}
        self.assertEqual(by["hoje"]["pnl_usd"], 0)
        self.assertEqual(by["7d"]["pnl_usd"], 5)
        self.assertTrue(by["30d"]["partial"])
        self.assertEqual(by["30d"]["history_days"], w["history_days"])
        self.assertLess(w["history_days"], 30)


class TestVerdict(unittest.TestCase):
    def test_cluster_over_20_is_bundled_hard(self):
        from alphahound.verdict import bot_veto, classify

        read = classify(Features(cluster_pct=0.40, liquidity_usd=20_000))
        self.assertEqual(read.label, "bundled")
        self.assertGreaterEqual(read.risk, 50)
        self.assertIsNotNone(bot_veto(read, "solana"))
        self.assertIsNotNone(bot_veto(read, "robinhood_chain"))

    def test_security_cert_organic_ok_bundled_no(self):
        from alphahound.verdict import security_cert

        self.assertEqual(security_cert("organic"), "ok")
        self.assertEqual(security_cert("bundled"), "no")
        self.assertEqual(security_cert("unverified"), "?")
        self.assertEqual(security_cert("organic", mint_revoked=False), "no")

    def test_one_soft_signal_is_not_a_verdict(self):
        from alphahound.verdict import classify

        read = classify(Features(cluster_pct=0.12, liquidity_usd=20_000, top10_pct=0.22))
        self.assertEqual(read.label, "organic")

    def test_cabaled_needs_convergence(self):
        from alphahound.verdict import bot_veto, classify

        read = classify(
            Features(cluster_pct=0.12, top10_pct=0.42, liquidity_usd=20_000)
        )
        self.assertEqual(read.label, "cabaled")
        self.assertEqual(read.risk, 0)
        self.assertIsNotNone(bot_veto(read, "solana"))
        self.assertIsNotNone(bot_veto(read, "robinhood_chain"))

    def test_organic_is_not_a_buy(self):
        from alphahound.verdict import bot_veto, classify

        read = classify(
            Features(cluster_pct=0.02, top10_pct=0.18, top1_pct=0.05, liquidity_usd=40_000)
        )
        self.assertEqual(read.label, "organic")
        self.assertIsNone(bot_veto(read, "solana"))

    def test_missing_cluster_is_unverified_not_organic(self):
        from alphahound.verdict import bot_veto, classify

        read = classify(
            Features(liquidity_usd=20_000, top10_pct=0.18),
            unknown={"cluster_pct", "bundle_pct"},
        )
        self.assertEqual(read.label, "unverified")
        self.assertIsNotNone(bot_veto(read, "solana"))
        self.assertIsNone(bot_veto(read, "robinhood_chain"))

    def test_fresh_wallets_on_a_10min_coin_are_ignored(self):
        from alphahound.verdict import classify

        read = classify(
            Features(cluster_pct=0.02, top10_pct=0.18, fresh_wallet_pct=0.95, liquidity_usd=20_000),
            age_minutes=10,
        )
        self.assertEqual(read.label, "organic")


class TestPack(unittest.TestCase):
    def test_later_same_ticker_is_vamp(self):
        from alphahound.signals.pack import apply_tags

        now = now_ms()
        orig = Candidate(
            chain=Chain.SOLANA,
            address="orig",
            symbol="PEPE",
            created_at_ms=now - 20 * 60_000,
            mcap_usd=80_000,
            volume_5m_usd=20_000,
            ret_5m=0.40,
        )
        clone = Candidate(
            chain=Chain.SOLANA,
            address="clone",
            symbol="PEPE",
            created_at_ms=now - 3 * 60_000,
            mcap_usd=12_000,
            volume_5m_usd=4_000,
        )
        apply_tags([orig, clone])
        self.assertEqual(orig.pack_role, "main")
        self.assertEqual(clone.pack_role, "vamp")

    def test_name_family_beta_vanishes_when_main_dumps(self):
        from alphahound.signals.pack import apply_tags, dump_beta_keys

        now = now_ms()
        main = Candidate(
            chain=Chain.SOLANA,
            address="main",
            symbol="WIF",
            name="dogwifhat",
            created_at_ms=now - 15 * 60_000,
            mcap_usd=200_000,
            volume_5m_usd=50_000,
            ret_5m=-0.35,
        )
        beta = Candidate(
            chain=Chain.SOLANA,
            address="beta",
            symbol="HAT",
            name="another dogwifhat",
            created_at_ms=now - 4 * 60_000,
            mcap_usd=20_000,
            volume_5m_usd=8_000,
        )
        tags = apply_tags([main, beta])
        self.assertEqual(main.pack_role, "main")
        self.assertEqual(beta.pack_role, "beta")
        self.assertIn(beta.key, dump_beta_keys(tags))
        self.assertNotIn(main.key, dump_beta_keys(tags))


class TestTwitterSocials(unittest.TestCase):
    def test_pair_socials_reads_dexscreener_handle(self):
        from alphahound.providers import inst_weight, pair_socials, parse_pair, utility_hint

        handle, blurb = pair_socials(
            {
                "info": {
                    "description": "just a meme coin",
                    "socials": [{"url": "https://x.com/oddcto"}],
                }
            }
        )
        self.assertEqual(handle, "oddcto")
        self.assertEqual(utility_hint(blurb), "meme")
        from alphahound.providers import twitter_handle

        self.assertEqual(twitter_handle("https://x.com/oddcto/status/2094044487358271615"), "oddcto")
        self.assertEqual(twitter_handle("https://x.com/i/communities/123"), "")
        self.assertEqual(twitter_handle("2094044487358271615"), "")
        self.assertEqual(twitter_handle("@Jobscoinpons"), "Jobscoinpons")
        snap = parse_pair(
            {
                "chainId": "solana",
                "pairAddress": "p",
                "baseToken": {"address": "mint", "symbol": "ODD", "name": "odd"},
                "info": {"socials": [{"url": "https://twitter.com/oddcto"}]},
            }
        )
        self.assertIsNotNone(snap)
        self.assertEqual(snap.twitter, "oddcto")
        self.assertEqual(inst_weight(200_000, "business"), 0.85)
        self.assertEqual(inst_weight(500_000, "business"), 1.0)
        self.assertEqual(inst_weight(800, ""), 0.0)

    def test_paid_dex_with_photo_and_aligned_profile(self):
        from alphahound.models import Candidate
        from alphahound.providers import pair_dex_flags, parse_pair

        pair = {
            "chainId": "solana",
            "pairAddress": "p",
            "baseToken": {"address": "mint", "symbol": "ODD", "name": "odd coin"},
            "boosts": {"active": 4},
            "info": {
                "imageUrl": "https://cdn.example/odd.png",
                "description": "ODD is a meme coin",
                "socials": [{"url": "https://x.com/oddcto"}],
                "websites": [{"url": "https://odd.xyz"}],
            },
        }
        paid, photo, aligned = pair_dex_flags(pair)
        self.assertTrue(paid)
        self.assertTrue(photo)
        self.assertTrue(aligned)
        snap = parse_pair(pair)
        c = snap.to_candidate("dexscreener_boosts")
        self.assertTrue(c.dex_paid)
        self.assertGreaterEqual(c.dex_profile, 0.99)
        rug = pair_dex_flags({"info": {}, "boosts": {"active": 0}, "baseToken": {"symbol": "X"}})
        self.assertEqual(rug, (False, False, False))
        blank = Candidate(chain=c.chain, address="x")
        self.assertEqual(blank.dex_profile, 0.0)
        from alphahound.providers import orders_mark_paid

        self.assertTrue(orders_mark_paid([{"type": "tokenProfile", "status": "approved"}]))
        self.assertTrue(orders_mark_paid({"orders": [{"status": "approved"}], "boosts": []}))
        self.assertFalse(orders_mark_paid({"orders": [], "boosts": []}))
        self.assertFalse(orders_mark_paid([{"type": "tokenBoost", "status": "pending"}]))
        self.assertFalse(orders_mark_paid(None))


class TestRubric(unittest.TestCase):
    def setUp(self):
        self.store, self._tmp = make_store()

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_wash_and_higher_lows(self):
        from alphahound.signals.chart import higher_lows
        from alphahound.signals.flow import Trade, wash_ratio

        w = "same"
        buys = [Trade(ts_ms=i, side=Side.BUY, price=1.0, size_usd=10.0, wallet=w) for i in range(8)]
        self.assertGreater(wash_ratio(buys), 0.7)
        rising = candles([1.0, 1.02, 1.05, 1.08, 1.12])
        self.assertGreaterEqual(higher_lows(rising), 0.6)

    def test_dump_wallet_cuts_crowd_score(self):
        from alphahound.rubric import grade
        from alphahound.signals import Enrichment

        self.store.upsert_wallet("dumper", Chain.SOLANA, trades=4, wins=0, pnl_usd=-40.0, is_smart=False)
        f = Features(top10_pct=0.22, copy_signal=1.0, smart_money_buys=2.0, fomo_net_flow=0.4)
        enr = Enrichment(
            candidate=Candidate(chain=Chain.SOLANA, address="mintpump", dex_id="pumpfun"),
            features=f,
            crowd={"wallets": ["dumper"], "kols": ["x"]},
        )
        dump = grade(enr, self.store)
        clean = Enrichment(
            candidate=enr.candidate,
            features=f,
            crowd={"wallets": ["fresh"], "kols": ["x"]},
        )
        self.assertLess(dump.crowd, grade(clean, self.store).crowd)

    def test_strong_token_clears_seven(self):
        from alphahound.rubric import grade

        f = Features(
            top10_pct=0.22,
            gini=0.35,
            cluster_pct=0.04,
            fresh_wallet_pct=0.15,
            copy_signal=1.0,
            smart_money_buys=2.0,
            fomo_net_flow=0.5,
            whale_net_flow=0.3,
            unique_buyers_5m=40,
            buy_sell_ratio=1.8,
            bot_share=0.05,
            body_ratio=0.6,
            parabolic=0.1,
            twitter_inst=0.7,
            twitter_fresh=1.0,
        )
        c = Candidate(
            chain=Chain.SOLANA,
            address="mintpump",
            dex_id="pumpfun",
            dex_paid=True,
            dex_photo=True,
            dex_aligned=True,
        )
        c.created_at_ms = now_ms() - 60_000
        enr = Enrichment(
            candidate=c,
            features=f,
            crowd={"wallets": ["smart1"], "kols": ["bagu"]},
            candles=candles([1.0, 1.01, 1.03, 1.04, 1.06]),
            twitter={"utility": "claims utility", "official": "oddcto"},
        )
        r = grade(enr, self.store)
        self.assertGreaterEqual(r.total, 7.0, r.as_visor())

    def test_unmeasured_grade_stays_neutral_five_inside_the_model(self):
        from alphahound.models import Score
        from alphahound.rubric import grade
        from alphahound.signals import Enricher

        cheap = Enricher.free_enrichment(
            Candidate(chain=Chain.ROBINHOOD_CHAIN, address="0xabc")
        )
        r = grade(cheap, self.store)
        self.assertGreater(r.total, 3.5)
        self.assertLess(r.total, 6.5)
        self.assertEqual(Score(probability=0.0, expected_value=0.0).rubric, {})

    def test_scorer_vetoes_below_rubric_floor(self):
        from alphahound.scoring import Scorer
        from alphahound.signals import Enrichment

        scorer = Scorer(Model(), STRATEGY, self.store, live=False)
        enr = Enrichment(
            candidate=Candidate(
                chain=Chain.SOLANA,
                address="mintpump",
                dex_id="pumpfun",
                price_usd=1.0,
                liquidity_usd=80_000,
                created_at_ms=now_ms() - 60_000,
            ),
            features=Features(
                liquidity_usd=80_000,
                holder_count=500,
                top10_pct=0.40,
                cluster_pct=0.20,
                gini=0.75,
                bot_share=0.30,
                parabolic=0.95,
                body_ratio=-0.4,
                unique_buyers_5m=2,
                buy_sell_ratio=0.4,
            ),
            round_trip=RoundTrip(ok=True, sell_slippage=0.03, total_cost_pct=0.02),
            crowd={"whale_n": 1, "kols": ["bagu"]},
        )
        score = scorer.score(enr)
        self.assertTrue(any(v.startswith("rubric:") for v in score.veto_reasons), score.veto_reasons)


class TestHold(unittest.TestCase):
    def _pos(self, **kw) -> Position:
        c = Candidate(
            chain=Chain.SOLANA,
            address="mintpump",
            dex_id="pumpfun",
            price_usd=1.0,
            liquidity_usd=50_000,
        )
        args = dict(
            candidate=c,
            venue=VenueId.PAPER,
            entry_price=1.0,
            size_usd=20,
            tokens=20,
            opened_at_ms=now_ms() - 26 * 60_000,
            entry_sponsors=["bagu"],
        )
        args.update(kw)
        return Position(**args)

    def test_age_veto_does_not_cut(self):
        from alphahound.scoring import hold_cut

        pos = self._pos()
        score = Score(
            probability=0.7, expected_value=0.1, veto_reasons=["age: 25m old"], rubric={"total": 7.2}
        )
        enr = Enrichment(
            candidate=pos.candidate,
            features=Features(copy_signal=1.0),
            crowd={"whale_n": 1, "kols": ["bagu"], "wallets": []},
        )
        self.assertIsNone(hold_cut(pos, enr, score, STRATEGY).cut)

    def test_live_mint_cuts(self):
        from alphahound.scoring import hold_cut

        pos = self._pos()
        score = Score(
            probability=0.7,
            expected_value=0.1,
            veto_reasons=["mint_authority: still live, supply can be inflated"],
            rubric={"total": 7.2},
        )
        enr = Enrichment(candidate=pos.candidate, features=Features())
        self.assertIn("mint_authority", hold_cut(pos, enr, score, STRATEGY).cut or "")

    def test_grace_blocks_cut(self):
        from alphahound.scoring import hold_cut

        pos = self._pos(opened_at_ms=now_ms())
        score = Score(
            probability=0.7,
            expected_value=0.1,
            veto_reasons=["mint_authority: still live"],
            rubric={"total": 7.2},
        )
        enr = Enrichment(candidate=pos.candidate, features=Features())
        self.assertIsNone(hold_cut(pos, enr, score, STRATEGY).cut)

    def test_rubric_needs_four_strikes(self):
        from alphahound.scoring import hold_cut

        pos = self._pos()
        score = Score(
            probability=0.7, expected_value=0.1, veto_reasons=["rubric: 4.0 < 7.0"], rubric={"total": 4.0}
        )
        enr = Enrichment(
            candidate=pos.candidate,
            features=Features(copy_signal=1.0),
            crowd={"whale_n": 1, "kols": ["bagu"], "wallets": []},
        )
        for _ in range(3):
            self.assertIsNone(hold_cut(pos, enr, score, STRATEGY).cut)
        self.assertIn("rubric: hold", hold_cut(pos, enr, score, STRATEGY).cut or "")

    def test_sponsor_left_cuts(self):
        from alphahound.scoring import hold_cut

        pos = self._pos()
        score = Score(probability=0.7, expected_value=0.1, veto_reasons=[], rubric={"total": 7.0})
        enr = Enrichment(
            candidate=pos.candidate,
            features=Features(whale_net_flow=0.0, copy_signal=0.0),
            crowd={"whale_n": 0, "kols": [], "wallets": []},
        )
        self.assertIn("sponsor:", hold_cut(pos, enr, score, STRATEGY).cut or "")

    def test_parabolic_after_entry_does_not_cut(self):
        from alphahound.scoring import hold_cut

        pos = self._pos()
        score = Score(probability=0.7, expected_value=0.1, veto_reasons=[], rubric={"total": 7.2})
        enr = Enrichment(
            candidate=pos.candidate,
            features=Features(parabolic=0.8, copy_signal=1.0),
            crowd={"whale_n": 1, "kols": ["bagu"], "wallets": []},
        )
        self.assertIsNone(hold_cut(pos, enr, score, STRATEGY).cut)


class TestDeadMcap(unittest.TestCase):
    def test_sub_40k_is_dead_zero_is_not(self):
        from alphahound.engine import mcap_is_dead

        self.assertTrue(mcap_is_dead(39_999, 40_000))
        self.assertFalse(mcap_is_dead(40_000, 40_000))
        self.assertFalse(mcap_is_dead(0, 40_000))

    def test_scan_floor_drops_unquoted_and_sub_50k(self):
        from alphahound.engine import below_scan_mcap, drop_for_scan_mcap

        self.assertTrue(below_scan_mcap(0, 50_000))
        self.assertTrue(below_scan_mcap(3_000, 50_000))
        self.assertFalse(below_scan_mcap(50_000, 50_000))
        self.assertFalse(below_scan_mcap(80_000, 50_000))
        baby = Candidate(
            chain=Chain.ROBINHOOD_CHAIN,
            address="0xabc",
            source="hood_stream",
            mcap_usd=0.0,
        )
        self.assertFalse(drop_for_scan_mcap(baby, 50_000))
        baby.mcap_usd = 3_000
        self.assertTrue(drop_for_scan_mcap(baby, 50_000))
        ds = Candidate(
            chain=Chain.ROBINHOOD_CHAIN,
            address="0xdef",
            source="dexscreener_profiles",
            mcap_usd=0.0,
        )
        self.assertTrue(drop_for_scan_mcap(ds, 50_000))

    def test_unpaid_dex_stays_off_the_visor(self):
        from alphahound.engine import on_scan_visor, unpaid_for_scan

        raw = Candidate(
            chain=Chain.ROBINHOOD_CHAIN,
            address="0xabc",
            source="hood_stream",
            mcap_usd=80_000,
        )
        self.assertTrue(unpaid_for_scan(raw))
        self.assertFalse(on_scan_visor(raw, 50_000))
        raw.dex_paid = True
        self.assertFalse(unpaid_for_scan(raw))
        self.assertTrue(on_scan_visor(raw, 50_000))
        inspect = Candidate(
            chain=Chain.ROBINHOOD_CHAIN,
            address="0xdef",
            source="inspect",
            mcap_usd=80_000,
        )
        self.assertFalse(unpaid_for_scan(inspect))
        self.assertTrue(on_scan_visor(inspect, 50_000))

    def test_skip_call_is_not_a_visor_card(self):
        from alphahound.engine import visor_card

        c = Candidate(
            chain=Chain.ROBINHOOD_CHAIN,
            address="0xabc",
            source="hood_stream",
            mcap_usd=200_000,
            dex_paid=True,
        )
        reads = {c.key: {"call": "skip", "why": "cluster: 40% linked supply"}}
        self.assertFalse(visor_card(c, reads, 50_000))
        reads[c.key]["call"] = "wait"
        self.assertTrue(visor_card(c, reads, 50_000))

    def test_absorb_watch_keeps_quoted_mcap_on_blank_reemit(self):
        from alphahound.engine import absorb_watch

        live = Candidate(
            chain=Chain.ROBINHOOD_CHAIN,
            address="0xabc",
            source="hood_stream",
            mcap_usd=1485629,
            volume_5m_usd=12000,
            last_scored_ms=9,
            dex_paid=True,
        )
        blank = Candidate(
            chain=Chain.ROBINHOOD_CHAIN,
            address="0xabc",
            source="hood_stream",
            mcap_usd=0,
        )
        absorb_watch(live, blank)
        self.assertEqual(live.mcap_usd, 1485629)
        self.assertEqual(live.volume_5m_usd, 12000)
        self.assertEqual(live.last_scored_ms, 9)
        self.assertTrue(live.dex_paid)
        live.dex_id = "pons"
        blank.dex_id = "uniswap"
        blank.mcap_usd = 1
        absorb_watch(live, blank)
        self.assertEqual(live.dex_id, "pons")

    def test_live_floors_replace_stale_mcap_why(self):
        from alphahound.engine import best_setup_p, stamp_live_floors

        stale = {
            "call": "wait",
            "why": "mcap: 80000 below 100000 floor",
            "vetoes": ["mcap: 80000 below 100000 floor", "cluster: 37% linked supply"],
            "p": 0.0,
        }
        live = stamp_live_floors(stale, ["mcap: 92000 below 100000 floor"])
        self.assertEqual(live["why"], "mcap: 92000 below 100000 floor")
        self.assertTrue(any(v.startswith("mcap: 92000") for v in live["vetoes"]))
        self.assertTrue(any(v.startswith("cluster:") for v in live["vetoes"]))
        recovered = stamp_live_floors(live, [])
        self.assertTrue(recovered["why"].startswith("cluster:"))
        self.assertEqual(
            best_setup_p(
                {
                    "a": {"call": "skip", "p": 0.9},
                    "b": {"call": "trade", "p": 0.48},
                },
                ["a", "b"],
            ),
            0.48,
        )


class TestLpLock(unittest.TestCase):
    def test_burn_and_locker_are_not_free(self):
        from alphahound.signals.lp_lock import nft_ids_from_npm_logs, owner_kind, unlocked_fraction

        lockers = {"0x1111111111111111111111111111111111111111"}
        self.assertEqual(owner_kind("0x000000000000000000000000000000000000dead", lockers), "burn")
        self.assertEqual(owner_kind("0x1111111111111111111111111111111111111111", lockers), "locker")
        self.assertEqual(owner_kind("0x2222222222222222222222222222222222222222", lockers), "free")
        self.assertEqual(unlocked_fraction([]), 1.0)
        self.assertAlmostEqual(unlocked_fraction([(100, "free")]), 1.0)
        self.assertAlmostEqual(unlocked_fraction([(100, "burn")]), 0.0)
        self.assertAlmostEqual(unlocked_fraction([(40, "locker"), (60, "free")]), 0.6)
        ids = nft_ids_from_npm_logs(
            [
                {
                    "address": "0x73991a25c818bf1f1128deaab1492d45638de0d3",
                    "topics": [
                        "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
                        "0x" + "0" * 64,
                        "0x" + "1" * 64,
                        "0x" + "a".rjust(64, "0"),
                    ],
                }
            ],
            "0x73991a25c818bf1f1128deaab1492d45638de0d3",
        )
        self.assertEqual(ids, [10])
        from alphahound.signals.lp_lock import _token_ids_from_v4_modify

        salts = _token_ids_from_v4_modify(
            [{"data": "0x" + ("0" * 192) + f"{7:064x}"}]
        )
        self.assertEqual(salts, [7])


class TestManualSellQueue(unittest.TestCase):
    def test_match_and_refuse_closed(self):
        from alphahound.engine import sell_queued_for
        from alphahound.preview import enqueue_manual_sell

        key = "robinhood_chain:0xabc"
        queued = {key}
        self.assertTrue(sell_queued_for(queued, key, "0xabc"))
        self.assertTrue(sell_queued_for({"0xABC"}, key, "0xabc"))
        self.assertFalse(sell_queued_for({"other:0xdef"}, key, "0xabc"))

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            (state / "positions.json").write_text("[]", encoding="utf-8")
            miss = enqueue_manual_sell(state, key)
            self.assertFalse(miss["ok"])
            (state / "positions.json").write_text(
                json.dumps(
                    [
                        {
                            "candidate": {
                                "chain": "robinhood_chain",
                                "address": "0xabc",
                                "symbol": "PUNCH",
                            }
                        }
                    ]
                ),
                encoding="utf-8",
            )
            hit = enqueue_manual_sell(state, "0xABC")
            self.assertTrue(hit["ok"])
            self.assertEqual(hit["key"], key)
            self.assertEqual(hit["symbol"], "PUNCH")
            again = enqueue_manual_sell(state, key)
            self.assertTrue(again["ok"])
            self.assertEqual(json.loads((state / "sell.json").read_text(encoding="utf-8")), [key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
