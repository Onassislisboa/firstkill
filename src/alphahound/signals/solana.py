"""Solana chain reader.

Everything the signal layer needs that only the chain can answer: mint
authorities, the holder set, and the transaction stream that buyer attribution
is computed from.

Cost discipline matters here more than anywhere else in the codebase. Enriching
one candidate fully is roughly 1 + 1 + 2 + N_holders + N_txs RPC calls, and at
forty candidates a tick that is how you burn a paid RPC plan in an afternoon.
Every method takes an explicit cap, and the engine spends its budget on
candidates that already passed the cheap filters.
"""

from __future__ import annotations

import asyncio
import base64
import struct
from dataclasses import dataclass
from urllib.parse import urlsplit

from ..log import get
from ..models import Side
from ..net import Http, gather_ok
from ..settings import PUBLIC_SOLANA_RPC
from .distribution import Holder
from .flow import Trade
from .terminals import BuyerTx

log = get("solana")

TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
INCINERATOR = "1nc1nerator11111111111111111111111111111111"
SYSTEM_PROGRAM = "11111111111111111111111111111111"
WSOL = "So11111111111111111111111111111111111111112"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def b58encode(raw: bytes) -> str:
    """Base58 in twelve lines, so `base58` does not become a hard dependency
    of the signal layer."""
    n = int.from_bytes(raw, "big")
    out = ""
    while n > 0:
        n, rem = divmod(n, 58)
        out = _B58_ALPHABET[rem] + out
    leading_zeros = len(raw) - len(raw.lstrip(b"\x00"))
    return "1" * leading_zeros + (out or "1")


@dataclass(slots=True)
class MintInfo:
    mint: str
    supply: float
    decimals: int
    mint_authority: str | None
    freeze_authority: str | None
    program: str

    @property
    def is_token_2022(self) -> bool:
        return self.program == TOKEN_2022_PROGRAM

    @property
    def authorities_revoked(self) -> bool:
        return self.mint_authority is None and self.freeze_authority is None


class RpcError(RuntimeError):
    pass


class SolanaReader:
    def __init__(
        self,
        http: Http,
        rpc_url: str,
        *,
        commitment: str = "confirmed",
        rate_per_sec: float = 25.0,
    ) -> None:
        self.http = http
        self.rpc_url = rpc_url
        self.commitment = commitment
        self._id = 0
        # Without an explicit budget this host inherits the generous default and
        # a single candidate's transaction fan-out will trip any rate limit,
        # after which every retry backs off and the tick stops finishing. The
        # public endpoint tolerates far less than this; a paid one, far more.
        if PUBLIC_SOLANA_RPC in rpc_url:
            rate_per_sec = 5.0
        host = urlsplit(rpc_url).netloc
        if host:
            http.limit(host, rate_per_sec=rate_per_sec, burst=max(2, int(rate_per_sec)))
        # ponytail: first-seen never gets younger. 0 = not fresh (≥200 sigs).
        self._first_seen: dict[str, int] = {}

    async def _rpc(self, method: str, params: list) -> object:
        self._id += 1
        payload = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}
        data = await self.http.post(self.rpc_url, json_body=payload)
        if not isinstance(data, dict):
            raise RpcError(f"{method}: unexpected response type {type(data)}")
        if "error" in data:
            raise RpcError(f"{method}: {data['error']}")
        return data.get("result")

    async def get_slot(self) -> int:
        result = await self._rpc("getSlot", [{"commitment": self.commitment}])
        return int(result or 0)

    # -- mint --------------------------------------------------------------
    async def mint_info(self, mint: str) -> MintInfo | None:
        """Parse the SPL Mint account directly.

        Going through the raw account rather than an indexer API is the point:
        mint and freeze authority are the two facts that decide whether a token
        can be inflated or your wallet frozen, and they are not worth trusting
        to a third party's cache.
        """
        result = await self._rpc(
            "getAccountInfo",
            [mint, {"encoding": "base64", "commitment": self.commitment}],
        )
        value = (result or {}).get("value") if isinstance(result, dict) else None
        if not value:
            return None
        raw = base64.b64decode(value["data"][0])
        if len(raw) < 82:
            return None

        # SPL Mint layout: COption<Pubkey> authority (4+32), u64 supply,
        # u8 decimals, u8 is_initialized, COption<Pubkey> freeze (4+32).
        mint_auth_flag = struct.unpack_from("<I", raw, 0)[0]
        mint_auth = b58encode(raw[4:36]) if mint_auth_flag == 1 else None
        supply_raw = struct.unpack_from("<Q", raw, 36)[0]
        decimals = raw[44]
        freeze_flag = struct.unpack_from("<I", raw, 46)[0]
        freeze_auth = b58encode(raw[50:82]) if freeze_flag == 1 else None

        return MintInfo(
            mint=mint,
            supply=supply_raw / (10**decimals) if decimals else float(supply_raw),
            decimals=decimals,
            mint_authority=mint_auth,
            freeze_authority=freeze_auth,
            program=value.get("owner", TOKEN_PROGRAM),
        )

    # -- holders -----------------------------------------------------------
    async def largest_holders(
        self,
        mint: str,
        *,
        limit: int = 20,
        pool_addresses: set[str] | None = None,
        deployer: str = "",
        resolve_ages: int = 0,
    ) -> list[Holder]:
        """Top token accounts, resolved to their owner wallets.

        getTokenLargestAccounts caps at 20, which is enough for concentration
        metrics and nowhere near enough for a holder count. The holder count
        comes from an indexer (see `dexscreener`/Birdeye) because doing it on
        chain means getProgramAccounts over every token account in existence.
        """
        result = await self._rpc(
            "getTokenLargestAccounts", [mint, {"commitment": self.commitment}]
        )
        entries = (result or {}).get("value") or [] if isinstance(result, dict) else []
        entries = entries[:limit]
        if not entries:
            return []

        addresses = [e["address"] for e in entries]
        owners = await self._token_account_owners(addresses)
        pool_addresses = pool_addresses or set()

        holders: list[Holder] = []
        for entry in entries:
            token_account = entry["address"]
            owner = owners.get(token_account, token_account)
            amount = float(entry.get("uiAmount") or 0.0)
            holders.append(
                Holder(
                    address=owner,
                    balance=amount,
                    is_lp=owner in pool_addresses or token_account in pool_addresses,
                    is_burn=owner in (INCINERATOR, SYSTEM_PROGRAM),
                    is_deployer=bool(deployer) and owner == deployer,
                )
            )

        if resolve_ages > 0:
            await self._fill_wallet_ages(holders[:resolve_ages])
        return holders

    async def _token_account_owners(self, token_accounts: list[str]) -> dict[str, str]:
        if not token_accounts:
            return {}
        result = await self._rpc(
            "getMultipleAccounts",
            [token_accounts, {"encoding": "base64", "commitment": self.commitment}],
        )
        values = (result or {}).get("value") or [] if isinstance(result, dict) else []
        owners: dict[str, str] = {}
        for address, value in zip(token_accounts, values):
            if not value:
                continue
            raw = base64.b64decode(value["data"][0])
            # Token account layout: mint(32) owner(32) amount(8) ...
            if len(raw) >= 64:
                owners[address] = b58encode(raw[32:64])
        return owners

    async def _fill_wallet_ages(self, holders: list[Holder]) -> None:
        """Stamp holders with their first observed activity.

        ponytail: approximated from the oldest of the most recent 200
        signatures. If a wallet has more than 200 lifetime transactions we
        report "not fresh", which is the right answer for the feature this
        feeds even though the timestamp is a floor rather than the truth.
        Upgrade path: paginate `before` until exhaustion, or use an indexer's
        first-transaction endpoint.
        """

        ages = getattr(self, "_first_seen", None)
        if ages is None:
            self._first_seen = {}
            ages = self._first_seen

        async def one(holder: Holder) -> None:
            cached = ages.get(holder.address)
            if cached is not None:
                if cached:
                    holder.first_seen_ms = cached
                return
            try:
                result = await self._rpc(
                    "getSignaturesForAddress",
                    [holder.address, {"limit": 200, "commitment": self.commitment}],
                )
            except (RpcError, Exception):  # noqa: BLE001 - one holder, not the tick
                return
            sigs = result if isinstance(result, list) else []
            if not sigs or len(sigs) >= 200:
                ages[holder.address] = 0
                return
            oldest = sigs[-1].get("blockTime")
            if oldest:
                holder.first_seen_ms = int(oldest) * 1000
                ages[holder.address] = holder.first_seen_ms

        await gather_ok(*(one(h) for h in holders))

    # -- transactions ------------------------------------------------------
    async def recent_activity(
        self,
        address: str,
        mint: str,
        price_usd: float,
        *,
        max_txs: int = 120,
        concurrency: int = 8,
    ) -> tuple[list[Trade], list[BuyerTx]]:
        """Fetch and decode recent swaps touching `address`.

        Returns both flow trades and attribution records from the same fetch,
        because they need the same transactions and fetching twice would double
        the most expensive call in the system.
        """
        result = await self._rpc(
            "getSignaturesForAddress",
            [address, {"limit": max_txs, "commitment": self.commitment}],
        )
        sigs = [s["signature"] for s in (result or []) if not s.get("err")]
        if not sigs:
            return [], []

        sem = asyncio.Semaphore(concurrency)

        async def fetch(sig: str) -> object:
            async with sem:
                return await self._rpc(
                    "getTransaction",
                    [
                        sig,
                        {
                            "encoding": "jsonParsed",
                            "maxSupportedTransactionVersion": 0,
                            "commitment": self.commitment,
                        },
                    ],
                )

        raw_txs = await gather_ok(*(fetch(s) for s in sigs))

        trades: list[Trade] = []
        buys: list[BuyerTx] = []
        for tx in raw_txs:
            decoded = decode_swap(tx, mint, price_usd)
            if decoded is None:
                continue
            trade, buyer_tx = decoded
            trades.append(trade)
            if buyer_tx is not None:
                buys.append(buyer_tx)
        trades.sort(key=lambda t: t.ts_ms)
        return trades, buys

    async def launch_slot(self, mint: str) -> int:
        """Slot of the mint's oldest visible transaction, used as the reference
        for bundle detection. Returns 0 when it cannot be determined, and the
        distribution analyzer then declines to report bundle_pct rather than
        reporting a wrong one."""
        try:
            result = await self._rpc(
                "getSignaturesForAddress",
                [mint, {"limit": 1000, "commitment": self.commitment}],
            )
        except RpcError:
            return 0
        sigs = result if isinstance(result, list) else []
        if not sigs or len(sigs) >= 1000:
            return 0
        return int(sigs[-1].get("slot") or 0)


def decode_swap(
    tx: object, mint: str, price_usd: float
) -> tuple[Trade, BuyerTx | None] | None:
    """Turn a jsonParsed transaction into a trade in `mint`.

    Direction comes from the pre/post token balance delta of the signer, which
    is the only method that does not depend on recognising the DEX program. New
    routers appear constantly; balance deltas do not change.
    """
    if not isinstance(tx, dict):
        return None
    meta = tx.get("meta") or {}
    message = (tx.get("transaction") or {}).get("message") or {}
    block_time = tx.get("blockTime")
    if not block_time:
        return None

    keys_raw = message.get("accountKeys") or []
    account_keys = [
        k.get("pubkey", "") if isinstance(k, dict) else str(k) for k in keys_raw
    ]
    # Address lookup tables hide accounts from accountKeys; the loaded ones are
    # reported separately and terminal fee accounts frequently live there.
    loaded = meta.get("loadedAddresses") or {}
    account_keys += list(loaded.get("writable") or []) + list(loaded.get("readonly") or [])
    if not account_keys:
        return None

    signer = next(
        (
            k.get("pubkey")
            for k in keys_raw
            if isinstance(k, dict) and k.get("signer") and k.get("pubkey")
        ),
        account_keys[0],
    )

    def owner_delta(owner: str) -> float:
        pre = sum(
            float(b["uiTokenAmount"].get("uiAmount") or 0.0)
            for b in meta.get("preTokenBalances") or []
            if b.get("mint") == mint and b.get("owner") == owner
        )
        post = sum(
            float(b["uiTokenAmount"].get("uiAmount") or 0.0)
            for b in meta.get("postTokenBalances") or []
            if b.get("mint") == mint and b.get("owner") == owner
        )
        return post - pre

    delta = owner_delta(signer)
    if abs(delta) < 1e-12:
        return None

    side = Side.BUY if delta > 0 else Side.SELL
    size_usd = abs(delta) * price_usd
    ts_ms = int(block_time) * 1000

    trade = Trade(ts_ms=ts_ms, side=side, price=price_usd, size_usd=size_usd, wallet=signer)
    buyer_tx = (
        BuyerTx(buyer=signer, account_keys=account_keys, ts_ms=ts_ms, size_usd=size_usd)
        if side is Side.BUY
        else None
    )
    return trade, buyer_tx
