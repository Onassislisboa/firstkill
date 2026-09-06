"""Holder distribution analysis.

The purpose of this module is to answer one question: if this goes up, who is
standing behind me with something to sell? Every metric here is a different
angle on that.

All functions are pure so they can be unit-tested and replayed. Fetching the
holder set is somebody else's problem (`signals.solana`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

ZERO_ADDR = "0x" + "0" * 40
TRANSFER_TOPIC0 = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"


@dataclass(slots=True)
class TokenMove:
    block: int
    frm: str
    to: str
    amount: float


@dataclass(slots=True)
class Holder:
    address: str
    balance: float
    # Unix ms of the wallet's first observed activity. 0 = unknown.
    first_seen_ms: int = 0
    # Slot/block in which this wallet acquired the token. 0 = unknown.
    acquired_slot: int = 0
    # Wallet that funded this one, when we can see it one hop back.
    funder: str = ""
    is_lp: bool = False
    is_burn: bool = False
    is_deployer: bool = False


@dataclass(slots=True)
class HolderStats:
    holder_count: int = 0
    top1_pct: float = 0.0
    top10_pct: float = 0.0
    gini: float = 0.0
    dev_holding_pct: float = 0.0
    lp_pct: float = 0.0
    burned_pct: float = 0.0
    fresh_wallet_pct: float = 0.0
    bundle_pct: float = 0.0
    largest_funding_cluster_pct: float = 0.0
    notes: list[str] = field(default_factory=list)


def _topic_addr(topic: str) -> str:
    t = (topic or "").lower().replace("0x", "")
    return ("0x" + t[-40:]) if len(t) >= 40 else ""


def log_topic_addr(address: str) -> str:
    return "0x" + address.lower().replace("0x", "").rjust(64, "0")


def decode_transfer_log(raw: dict) -> TokenMove | None:
    topics = raw.get("topics") or []
    if len(topics) < 3 or str(topics[0]).lower() != TRANSFER_TOPIC0:
        return None
    data = raw.get("data") or "0x0"
    try:
        amount = float(int(data, 16))
        block = int(raw.get("blockNumber") or "0x0", 16)
    except ValueError:
        return None
    frm, to = _topic_addr(str(topics[1])), _topic_addr(str(topics[2]))
    if not frm or not to:
        return None
    return TokenMove(block=block, frm=frm, to=to, amount=amount)


def holders_from_moves(
    moves: list[TokenMove],
    *,
    pools: set[str] | None = None,
    deployer: str = "",
    first_n: int = 40,
) -> tuple[list[Holder], int]:
    """Holders + launch block from ERC-20 Transfer logs.

    `funder` is the first non-pool sender into the wallet (token hop). Same-block
    snipers still show up via acquired_slot even when they bought from the pool.
    """
    if not moves:
        return [], 0
    ordered = sorted(moves, key=lambda m: m.block)
    launch = ordered[0].block
    pools = {p.lower() for p in (pools or set()) if p}
    deployer = deployer.lower()
    bal: dict[str, float] = {}
    first_block: dict[str, int] = {}
    funder: dict[str, str] = {}
    early_n = 0
    for m in ordered:
        to, frm = m.to.lower(), m.frm.lower()
        if to not in pools and to != ZERO_ADDR:
            bal[to] = bal.get(to, 0.0) + m.amount
            if to not in first_block:
                first_block[to] = m.block
                if frm not in pools:
                    funder[to] = frm if early_n < first_n else funder.get(to, "")
                    if early_n < first_n:
                        early_n += 1
        if frm not in pools and frm != ZERO_ADDR:
            bal[frm] = bal.get(frm, 0.0) - m.amount
    holders = []
    for addr, amount in bal.items():
        if amount <= 0 or addr in pools:
            continue
        holders.append(
            Holder(
                address=addr,
                balance=amount,
                acquired_slot=first_block.get(addr, 0),
                funder=funder.get(addr, ""),
                is_lp=False,
                is_deployer=bool(deployer) and addr == deployer,
            )
        )
    return holders, launch


def first_pool_buyers(
    moves: list[TokenMove],
    pools: set[str],
    n: int,
    balances: dict[str, float] | None = None,
) -> list[str]:
    """Pool buyers to probe for quote-token funding (cap n).

    Time-ordered first inbound from the pool, then the heaviest remaining bags
    if `balances` is passed — cluster % is a supply share, so dust snipers
    first would miss the GI pattern.
    """
    if n <= 0 or not moves:
        return []
    pools = {p.lower() for p in pools if p}
    ordered = sorted(moves, key=lambda m: m.block)
    seen: set[str] = set()
    pool_first: list[str] = []
    for m in ordered:
        to = m.to.lower()
        if to in seen or to in pools or to == ZERO_ADDR:
            continue
        seen.add(to)
        if m.frm.lower() in pools:
            pool_first.append(to)
    if balances:
        pool_first.sort(key=lambda a: -float(balances.get(a, 0.0)))
    return pool_first[:n]


def cluster_share_by_root(
    holders: list[Holder],
    roots: dict[str, str],
    circ_total: float,
) -> float:
    """Supply share of the largest group sharing a funding root (quote hops)."""
    if circ_total <= 0 or not roots:
        return 0.0
    by_root: dict[str, float] = {}
    for h in holders:
        root = roots.get(h.address.lower())
        if not root:
            continue
        by_root[root] = by_root.get(root, 0.0) + h.balance
    if not by_root:
        return 0.0
    return max(by_root.values()) / circ_total


def gini(values: list[float]) -> float:
    """Gini coefficient of the balance distribution.

    Top-10 share hides the shape: 10 wallets at 4% each and one at 40% give the
    same headline number and completely different risk. Gini separates them.
    """
    vals = sorted(v for v in values if v > 0)
    n = len(vals)
    if n < 2:
        return 0.0
    total = sum(vals)
    if total <= 0:
        return 0.0
    cumulative = 0.0
    for i, v in enumerate(vals, start=1):
        cumulative += i * v
    return (2.0 * cumulative) / (n * total) - (n + 1.0) / n


def analyze(
    holders: list[Holder],
    *,
    now_ms: int,
    launch_slot: int = 0,
    fresh_wallet_max_age_hours: float = 24.0,
    bundle_slot_window: int = 3,
) -> HolderStats:
    """Turn a holder set into the distribution slice of the feature vector.

    LP and burn addresses are excluded from concentration, because counting the
    pool as a whale makes every healthy token look like a rug and every real
    whale look normal - the exact inversion of what you want.
    """
    stats = HolderStats()
    if not holders:
        stats.notes.append("no holder data")
        return stats

    circulating = [h for h in holders if not (h.is_lp or h.is_burn)]
    supply_total = sum(h.balance for h in holders)
    circ_total = sum(h.balance for h in circulating)

    if supply_total > 0:
        stats.lp_pct = sum(h.balance for h in holders if h.is_lp) / supply_total
        stats.burned_pct = sum(h.balance for h in holders if h.is_burn) / supply_total

    stats.holder_count = len(circulating)
    if not circulating or circ_total <= 0:
        stats.notes.append("no circulating holders")
        return stats

    balances = sorted((h.balance for h in circulating), reverse=True)
    stats.top1_pct = balances[0] / circ_total
    stats.top10_pct = sum(balances[:10]) / circ_total
    stats.gini = gini(balances)
    stats.dev_holding_pct = (
        sum(h.balance for h in circulating if h.is_deployer) / circ_total
    )

    fresh_cutoff = now_ms - int(fresh_wallet_max_age_hours * 3_600_000)
    known_age = [h for h in circulating if h.first_seen_ms > 0]
    if known_age:
        fresh = [h for h in known_age if h.first_seen_ms >= fresh_cutoff]
        stats.fresh_wallet_pct = len(fresh) / len(known_age)
    else:
        stats.notes.append("wallet ages unavailable")

    if launch_slot > 0:
        bundled = [
            h
            for h in circulating
            if h.acquired_slot and h.acquired_slot <= launch_slot + bundle_slot_window
        ]
        stats.bundle_pct = sum(h.balance for h in bundled) / circ_total
    else:
        stats.notes.append("launch slot unknown, bundle_pct not computed")

    stats.largest_funding_cluster_pct = funding_cluster_share(circulating, circ_total)
    return stats


def unmeasured_from_stats(stats: HolderStats) -> set[str]:
    """Keys analyze() left at 0 because the input was missing, not because 0 is true.

    Callers must keep these in Enrichment.unknown so normalize() zeros them
    instead of treating the default as a clean launch / old wallets.
    """
    out: set[str] = set()
    for note in stats.notes:
        if "wallet ages unavailable" in note:
            out.add("fresh_wallet_pct")
        if "launch slot unknown" in note:
            out.add("bundle_pct")
    return out


def funding_cluster_share(holders: list[Holder], circ_total: float) -> float:
    """Largest share of supply held by wallets funded from the same source.

    Twenty wallets funded by one address are one holder wearing twenty hats.
    This is the cheapest sybil detector available: one hop back on the funding
    graph, no clustering heuristics, no ML.
    """
    if circ_total <= 0:
        return 0.0
    by_funder: dict[str, float] = {}
    for h in holders:
        if not h.funder:
            continue
        by_funder[h.funder] = by_funder.get(h.funder, 0.0) + h.balance
    if not by_funder:
        return 0.0
    return max(by_funder.values()) / circ_total


def holder_growth(history: list[tuple[int, int]], window_ms: int) -> float:
    """Holders added per minute over the window, from (ts_ms, count) samples.

    The level of holder count is nearly worthless - it is trivially inflated.
    The slope is what a real wave looks like, and it is much more expensive to
    fake convincingly for more than a minute.
    """
    if len(history) < 2:
        return 0.0
    latest_ts, latest_n = history[-1]
    cutoff = latest_ts - window_ms
    base = next((s for s in reversed(history[:-1]) if s[0] <= cutoff), history[0])
    dt_min = (latest_ts - base[0]) / 60_000.0
    if dt_min <= 0:
        return 0.0
    return (latest_n - base[1]) / dt_min


def sellable_supply_overhang(holders: list[Holder], entry_price: float) -> float:
    """Fraction of circulating supply held by wallets with a cost basis far
    below the current price - i.e. the people who are already in profit and
    will be selling into your entry.

    Returns 0.0 when cost bases are unknown, which is honest rather than
    optimistic; a fabricated number here would flow straight into sizing.
    """
    if entry_price <= 0 or not holders:
        return 0.0
    bundled = [h for h in holders if h.acquired_slot and not (h.is_lp or h.is_burn)]
    circ_total = sum(h.balance for h in holders if not (h.is_lp or h.is_burn))
    if circ_total <= 0 or not bundled:
        return 0.0
    return sum(h.balance for h in bundled) / circ_total
