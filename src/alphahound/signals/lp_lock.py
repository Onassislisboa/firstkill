"""LP lock probe: who can pull the pool's liquidity.

Hood Dexscreener pairs are Uniswap V4 poolIds (32-byte); the factory stream
may still emit a V3 pool (20-byte). BNB is usually V2 (the pair is the LP token).

ponytail: this is not a proof of safety. It does not catch (1) a locker the
team owns, (2) a lock that expires in minutes, (3) liquidity sitting
out-of-range while the NFT still exists. Upgrade: remaining lock time from
known locker ABIs + in-range check vs pool tick.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ..models import Chain, now_ms
from .distribution import TRANSFER_TOPIC0, ZERO_ADDR

# Uniswap V3 NonfungiblePositionManager on Robinhood Chain (4663).
HOOD_NFPM = "0x73991a25c818bf1f1128deaab1492d45638de0d3"
HOOD_V4_POOL_MANAGER = "0x8366a39cc670b4001a1121b8f6a443a643e40951"
HOOD_V4_POSM = "0x58daec3116aae6d93017baaea7749052e8a04fa7"

# IncreaseLiquidity(uint256 indexed tokenId, uint128, uint256, uint256)
TOPIC_INCREASE_LIQ = "0x3067048beee31b25b2f1681f88dac838c8bba36af25bfb2b7cf7473a5847e35f"
# Pool Mint(...) — used to bound receipts to this pool's txs.
TOPIC_POOL_MINT = "0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde"
# V4 PoolManager.ModifyLiquidity(PoolId indexed, address indexed, ...)
TOPIC_V4_MODIFY = "0xf208f4912782fd25c7f114ca3723a2d5dd6f3bcc3ac8db5af63baa85f711d5ec"

SEL_OWNER_OF = "0x6352211e"
SEL_POSITIONS = "0x99fbab88"
SEL_TOKEN0 = "0x0dfe1681"
SEL_TOKEN1 = "0xd21220a7"
SEL_V4_POS_LIQ = "0x1efeed33"

DEAD = {
    ZERO_ADDR,
    "0x000000000000000000000000000000000000dead",
    "0x000000000000000000000000000000000000deaD".lower(),
    "0xdead000000000000000000000000000000000000",
}

# Common BNB lockers. Hood has no canonical set yet — toml fills that gap.
BNB_LOCKERS = (
    "0xe2fe530c047f2d85298b07d9333c05737f1435fb",  # Team Finance
    "0x663a5c229c09b049e36dcc11a9b0d4a8eb9db214",  # Unicrypt
    "0x407993575c91ce7643a4d4ccafb9d779f1b6bc0c",  # PinkLock
)


@dataclass(slots=True)
class LpLockProbe:
    locked_pct: float
    unlocked_pct: float
    nfts: int
    rpc: int
    note: str
    ok: bool


def _addr_word(raw: str, word: int) -> str:
    h = (raw or "").removeprefix("0x")
    start = word * 64
    chunk = h[start : start + 64]
    if len(chunk) < 64:
        return ""
    return ("0x" + chunk[-40:]).lower()


def decode_positions(raw: str) -> tuple[str, str, int]:
    """token0, token1, liquidity from NPM.positions(tokenId)."""
    h = (raw or "").removeprefix("0x")
    if len(h) < 512:
        return "", "", 0
    token0 = ("0x" + h[128 + 24 : 192]).lower()
    token1 = ("0x" + h[192 + 24 : 256]).lower()
    liq = int(h[448:512], 16)
    return token0, token1, liq


def nft_ids_from_npm_logs(logs: list, npm: str) -> list[int]:
    want = npm.lower()
    ids: list[int] = []
    seen: set[int] = set()
    for item in logs:
        if str(item.get("address") or "").lower() != want:
            continue
        topics = [str(t).lower() for t in (item.get("topics") or [])]
        if not topics:
            continue
        tid = None
        if topics[0] == TRANSFER_TOPIC0 and len(topics) >= 4:
            tid = int(topics[3], 16)
        elif topics[0] == TOPIC_INCREASE_LIQ and len(topics) >= 2:
            tid = int(topics[1], 16)
        if tid is None or tid in seen:
            continue
        seen.add(tid)
        ids.append(tid)
    return ids


def _token_ids_from_v4_modify(logs: list) -> list[int]:
    """PositionManager uses tokenId as ModifyLiquidity salt."""
    ids: list[int] = []
    seen: set[int] = set()
    for item in logs:
        data = str(item.get("data") or "").replace("0x", "")
        if len(data) < 256:
            continue
        tid = int(data[192:256], 16)
        if tid <= 0 or tid in seen:
            continue
        seen.add(tid)
        ids.append(tid)
    return ids


def owner_kind(owner: str, lockers: set[str]) -> str:
    a = (owner or "").lower()
    if not a or a in DEAD:
        return "burn"
    if a in lockers:
        return "locker"
    return "free"


def unlocked_fraction(rows: list[tuple[int, str]]) -> float:
    """rows = (liquidity, kind). Free share of NFT liquidity; 1.0 if none."""
    total = sum(max(0, liq) for liq, _ in rows)
    if total <= 0:
        return 1.0
    free = sum(liq for liq, kind in rows if kind == "free")
    return free / total


def _pad_uint(value: int) -> str:
    return f"{value:064x}"


def _lockers(chain: Chain, extra: list[str]) -> set[str]:
    out = {a.lower() for a in extra if a}
    if chain is Chain.BNB:
        out.update(BNB_LOCKERS)
    return out


async def probe_lp_lock(
    rpc,
    *,
    chain: Chain,
    pool: str,
    created_at_ms: int,
    extra_lockers: list[str] | None = None,
) -> LpLockProbe:
    if not pool or not str(pool).startswith("0x"):
        return LpLockProbe(0.0, 1.0, 0, 0, "lp lock skipped: no pool", False)
    lockers = _lockers(chain, extra_lockers or [])
    if chain is Chain.BNB:
        return await _probe_v2(rpc, pool.lower(), lockers)
    if chain is Chain.ROBINHOOD_CHAIN:
        raw = pool.lower().replace("0x", "")
        if len(raw) == 64:
            return await _probe_v4(rpc, "0x" + raw, created_at_ms, lockers)
        v2 = await _probe_v2(rpc, pool.lower(), lockers)
        if v2.ok:
            return v2
        return await _probe_v3(rpc, pool.lower(), created_at_ms, lockers)
    return LpLockProbe(0.0, 1.0, 0, 0, "lp lock skipped: chain", False)


async def _probe_v2(rpc, pool: str, lockers: set[str]) -> LpLockProbe:
    n = 0

    async def call(to: str, data: str) -> str:
        nonlocal n
        n += 1
        return await rpc.eth_call(to, data)

    try:
        raw = await call(pool, "0x18160ddd")  # totalSupply
        supply = int(raw, 16) if raw and raw != "0x" else 0
    except Exception as exc:  # noqa: BLE001
        return LpLockProbe(0.0, 1.0, 0, n, f"lp lock v2 failed: {exc}", False)
    if supply <= 0:
        return LpLockProbe(0.0, 1.0, 0, n, "lp lock v2: no LP supply", False)
    held = 0
    for addr in sorted(DEAD | lockers):
        try:
            raw = await call(pool, "0x70a08231" + addr.lower().replace("0x", "").rjust(64, "0"))
            held += int(raw, 16) if raw and raw != "0x" else 0
        except Exception:  # noqa: BLE001
            continue
    locked = min(1.0, held / supply)
    free = max(0.0, 1.0 - locked)
    return LpLockProbe(
        locked, free, 1, n, f"lp lock v2: {free:.0%} free (supply {supply})", True
    )


async def _from_block(rpc, created_at_ms: int) -> tuple[int, int, int]:
    n = 1
    latest = int(await rpc.call("eth_blockNumber", []), 16)
    if not created_at_ms:
        age_s = 7 * 86400.0
    else:
        age_s = (now_ms() - created_at_ms) / 1000.0
        if age_s < 0:
            age_s = 7 * 86400.0
    age_s = max(120.0, age_s)
    # ponytail: Hood blocks are sub-second; 8/s overshoot beats missing the mint.
    start = max(0, latest - int(age_s * 8) - 2_000)
    return start, latest, n


async def _get_logs(rpc, filt: dict) -> tuple[list, int]:
    try:
        data = await rpc.get_logs(filt)
        return (data if isinstance(data, list) else []), 1
    except Exception:  # noqa: BLE001
        to_b = int(filt.get("toBlock") or "0x0", 16)
        frm = max(0, to_b - 4_000)
        filt = {**filt, "fromBlock": hex(frm)}
        data = await rpc.get_logs(filt)
        return (data if isinstance(data, list) else []), 2


async def _probe_v3(rpc, pool: str, created_at_ms: int, lockers: set[str]) -> LpLockProbe:
    n = 0
    try:
        _start, latest, nb = await _from_block(rpc, created_at_ms)
        n += nb
        t0_raw = await rpc.eth_call(pool, SEL_TOKEN0)
        t1_raw = await rpc.eth_call(pool, SEL_TOKEN1)
        n += 2
    except Exception as exc:  # noqa: BLE001
        return LpLockProbe(0.0, 1.0, 0, n, f"lp lock v3 pool failed: {exc}", False)
    token0, token1 = _addr_word(str(t0_raw), 0), _addr_word(str(t1_raw), 0)
    if not token0 or not token1:
        return LpLockProbe(0.0, 1.0, 0, n, "lp lock v3: not a V3 pool", False)

    npm = HOOD_NFPM
    ids: list[int] = []
    mints, n_m = await _get_logs(
        rpc,
        {
            "fromBlock": hex(0),
            "toBlock": hex(latest),
            "address": pool,
            "topics": [TOPIC_POOL_MINT],
        },
    )
    n += n_m
    hashes = list(dict.fromkeys(str(x.get("transactionHash") or "") for x in mints if x.get("transactionHash")))
    for txh in hashes[:16]:
        try:
            receipt = await rpc.receipt(txh)
            n += 1
        except Exception:  # noqa: BLE001
            continue
        logs = (receipt or {}).get("logs") or []
        ids.extend(nft_ids_from_npm_logs(logs, npm))
    ids = list(dict.fromkeys(ids))
    if not ids:
        return LpLockProbe(0.0, 1.0, 0, n, "lp lock v3: no NFT logs in window", False)

    sem = asyncio.Semaphore(8)
    rows: list[tuple[int, str]] = []

    async def one(tid: int) -> tuple[int, str] | None:
        async with sem:
            try:
                pos_raw = await rpc.eth_call(npm, SEL_POSITIONS + _pad_uint(tid))
                own_raw = await rpc.eth_call(npm, SEL_OWNER_OF + _pad_uint(tid))
            except Exception:  # noqa: BLE001
                return None
        t0, t1, liq = decode_positions(str(pos_raw or ""))
        if {t0, t1} != {token0, token1} or liq <= 0:
            return None
        owner = _addr_word(str(own_raw or ""), 0)
        return liq, owner_kind(owner, lockers)

    found = await asyncio.gather(*(one(tid) for tid in ids[:48]))
    n += min(len(ids), 48) * 2
    for item in found:
        if item:
            rows.append(item)

    free = unlocked_fraction(rows)
    locked = max(0.0, 1.0 - free)
    nfts = len(rows)
    kinds = {}
    for _, kind in rows:
        kinds[kind] = kinds.get(kind, 0) + 1
    note = (
        f"lp lock v3: {free:.0%} free nfts={nfts} "
        f"({', '.join(f'{k}={v}' for k, v in sorted(kinds.items())) or 'none'}) rpc={n}"
    )
    return LpLockProbe(locked, free, nfts, n, note, True)


async def _probe_v4(rpc, pool_id: str, created_at_ms: int, lockers: set[str]) -> LpLockProbe:
    """V4: ModifyLiquidity logs on PoolManager, filtered by this poolId."""
    n = 0
    try:
        _start, latest, nb = await _from_block(rpc, created_at_ms)
        n += nb
    except Exception as exc:  # noqa: BLE001
        return LpLockProbe(0.0, 1.0, 0, n, f"lp lock v4 head failed: {exc}", False)
    topic_id = "0x" + pool_id.replace("0x", "").rjust(64, "0")
    posm_topic = "0x" + HOOD_V4_POSM.replace("0x", "").rjust(64, "0")
    mods, n_m = await _get_logs(
        rpc,
        {
            "fromBlock": hex(0),
            "toBlock": hex(latest),
            "address": HOOD_V4_POOL_MANAGER,
            "topics": [TOPIC_V4_MODIFY, topic_id, posm_topic],
        },
    )
    n += n_m
    if not mods:
        mods, n_m2 = await _get_logs(
            rpc,
            {
                "fromBlock": hex(0),
                "toBlock": hex(latest),
                "address": HOOD_V4_POOL_MANAGER,
                "topics": [TOPIC_V4_MODIFY, topic_id],
            },
        )
        n += n_m2
    ids = _token_ids_from_v4_modify(mods)
    if not ids:
        hashes = list(
            dict.fromkeys(str(x.get("transactionHash") or "") for x in mods if x.get("transactionHash"))
        )
        for txh in hashes[:16]:
            try:
                receipt = await rpc.receipt(txh)
                n += 1
            except Exception:  # noqa: BLE001
                continue
            ids.extend(nft_ids_from_npm_logs((receipt or {}).get("logs") or [], HOOD_V4_POSM))
        ids = list(dict.fromkeys(ids))
    if not ids:
        return LpLockProbe(0.0, 1.0, 0, n, "lp lock v4: no position NFTs in window", False)

    sem = asyncio.Semaphore(8)
    rows: list[tuple[int, str]] = []

    async def one(tid: int) -> tuple[int, str] | None:
        async with sem:
            try:
                liq_raw = await rpc.eth_call(HOOD_V4_POSM, SEL_V4_POS_LIQ + _pad_uint(tid))
                own_raw = await rpc.eth_call(HOOD_V4_POSM, SEL_OWNER_OF + _pad_uint(tid))
            except Exception:  # noqa: BLE001
                return None
        liq = int(str(liq_raw or "0x0"), 16) if str(liq_raw or "") not in ("", "0x") else 0
        if liq <= 0:
            return None
        owner = _addr_word(str(own_raw or ""), 0)
        return liq, owner_kind(owner, lockers)

    found = await asyncio.gather(*(one(tid) for tid in ids[:48]))
    n += min(len(ids), 48) * 2
    for item in found:
        if item:
            rows.append(item)

    free = unlocked_fraction(rows)
    locked = max(0.0, 1.0 - free)
    nfts = len(rows)
    kinds: dict[str, int] = {}
    for _, kind in rows:
        kinds[kind] = kinds.get(kind, 0) + 1
    note = (
        f"lp lock v4: {free:.0%} free nfts={nfts} "
        f"({', '.join(f'{k}={v}' for k, v in sorted(kinds.items())) or 'none'}) rpc={n}"
    )
    return LpLockProbe(locked, free, nfts, n, note, True)
