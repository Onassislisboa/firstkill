"""EVM venues: 0x Swap v2 for BNB/Base, Uniswap V3 for Robinhood Chain.

Two adapters because the chain coverage genuinely differs. 0x covers BNB (56)
and Base (8453) and handles routing, allowances and calldata for us. Robinhood
Chain (4663, an Arbitrum Orbit L2 with ETH gas) is not on 0x, so it talks to
Uniswap V3's QuoterV2 and SwapRouter02 directly.

The 0x path needs no ABI encoder: the four ERC-20 selectors it uses are
constants, and 0x returns ready calldata. Only the Uniswap path pulls in
`web3`, and only when Robinhood Chain is enabled.
"""

from __future__ import annotations

from typing import Any

from ..log import get
from ..models import EVM_CHAIN_IDS, Candidate, Chain, Fill, Quote, Side, VenueId
from ..net import Http, HttpError
from ..providers import Dexscreener
from ..settings import Config, Settings
from . import ExecutionError
from ..fees import FeePlan, slip_bps

log = get("evm")

ZEROEX_URL = "https://api.0x.org/swap/allowance-holder/quote"
ZEROEX_CHAINS = (Chain.BNB, Chain.BASE)
# 0x's sentinel for the chain's native asset. Selling native means no ERC-20
# approval on the buy leg, which removes an entire transaction and its failure
# modes from the hot path.
NATIVE_SENTINEL = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"

# Standard ERC-20 selectors. Fixed by the spec, so hardcoding them is safe and
# keeps a keccak implementation out of the dependency tree.
SEL_DECIMALS = "0x313ce567"
SEL_BALANCE_OF = "0x70a08231"
SEL_TOTAL_SUPPLY = "0x18160ddd"
SEL_APPROVE = "0x095ea7b3"
SEL_ALLOWANCE = "0xdd62ed3e"
MAX_UINT256 = (1 << 256) - 1

# ETH is priced off the canonical Base WETH pair, which is deep and keyless.
# Robinhood Chain's own WETH has no Dexscreener coverage yet.
WRAPPED_ETH_FOR_PRICING = "0x4200000000000000000000000000000000000006"


def _pad_address(address: str) -> str:
    return address.lower().replace("0x", "").rjust(64, "0")


def _pad_uint(value: int) -> str:
    return f"{value:064x}"


class EvmRpc:
    """Minimal JSON-RPC client. Deliberately not web3: signing comes from
    eth_account and everything else here is four methods."""

    def __init__(self, http: Http, url: str) -> None:
        self.http = http
        self.url = url
        self._id = 0

    async def call(self, method: str, params: list) -> Any:
        self._id += 1
        data = await self.http.post(
            self.url,
            json_body={"jsonrpc": "2.0", "id": self._id, "method": method, "params": params},
        )
        if not isinstance(data, dict):
            raise ExecutionError(f"{method}: bad response {type(data)}")
        if "error" in data:
            raise ExecutionError(f"{method}: {data['error']}")
        return data.get("result")

    async def get_logs(self, filt: dict[str, Any]) -> list:
        data = await self.call("eth_getLogs", [filt])
        return data if isinstance(data, list) else []

    async def eth_call(self, to: str, data: str) -> str:
        return await self.call("eth_call", [{"to": to, "data": data}, "latest"])

    async def decimals(self, token: str) -> int:
        raw = await self.eth_call(token, SEL_DECIMALS)
        return int(raw, 16) if raw and raw != "0x" else 18

    async def balance_of(self, token: str, owner: str) -> int:
        raw = await self.eth_call(token, SEL_BALANCE_OF + _pad_address(owner))
        return int(raw, 16) if raw and raw != "0x" else 0

    async def total_supply(self, token: str) -> int:
        raw = await self.eth_call(token, SEL_TOTAL_SUPPLY)
        return int(raw, 16) if raw and raw != "0x" else 0

    async def nonce(self, address: str) -> int:
        return int(await self.call("eth_getTransactionCount", [address, "pending"]), 16)

    async def gas_price(self) -> int:
        return int(await self.call("eth_gasPrice", []), 16)

    async def send_raw(self, raw_tx: str) -> str:
        return await self.call("eth_sendRawTransaction", [raw_tx])

    async def receipt(self, tx_hash: str) -> dict | None:
        return await self.call("eth_getTransactionReceipt", [tx_hash])


class _EvmSignerMixin:
    settings: Settings

    def _account(self, chain: Chain | None = None):
        key = self.settings.evm_key_for(chain)
        if not key:
            raise ExecutionError("EVM private key is not set for this chain")
        cache: dict = getattr(self, "_accts", None) or {}
        if not hasattr(self, "_accts"):
            self._accts = cache
        slot = chain.value if chain else "_"
        if slot in cache:
            return cache[slot]
        try:
            from eth_account import Account
        except ImportError as exc:  # pragma: no cover
            raise ExecutionError("pip install 'alphahound[evm]' to trade on EVM chains") from exc
        acct = Account.from_key(key)
        cache[slot] = acct
        return acct

    async def _sign_and_send(self, rpc: EvmRpc, tx: dict[str, Any], chain: Chain | None = None) -> str:
        acct = self._account(chain)
        tx.setdefault("from", acct.address)
        tx.setdefault("nonce", await rpc.nonce(acct.address))
        tx.setdefault("gasPrice", await rpc.gas_price())
        tx.setdefault("chainId", tx.pop("_chain_id", None) or int(await rpc.call("eth_chainId", []), 16))
        signed = acct.sign_transaction(tx)
        raw = signed.raw_transaction if hasattr(signed, "raw_transaction") else signed.rawTransaction
        return await rpc.send_raw(raw.hex() if isinstance(raw, bytes) else raw)

    async def _await_receipt(self, rpc: EvmRpc, tx_hash: str, timeout_s: float) -> dict:
        import asyncio

        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            receipt = await rpc.receipt(tx_hash)
            if receipt:
                if int(receipt.get("status", "0x0"), 16) != 1:
                    raise ExecutionError(f"transaction reverted: {tx_hash}")
                return receipt
            await asyncio.sleep(1.0)
        raise ExecutionError(f"timed out waiting for {tx_hash}")


class ZeroExVenue(_EvmSignerMixin):
    """0x Swap API v2, allowance-holder flow. BNB Chain and Base."""

    id = VenueId.ZEROEX

    def __init__(
        self, http: Http, settings: Settings, strategy: Config, dexscreener: Dexscreener
    ) -> None:
        self.http = http
        self.settings = settings
        self.dex = dexscreener
        self.slippage_bps = int(strategy.get("execution.slippage_bps", 250))
        self.confirm_timeout = float(strategy.get("execution.confirm_timeout_seconds", 45))
        self.headers = {"0x-api-key": settings.zeroex_api_key, "0x-version": "v2"}
        self._rpc: dict[Chain, EvmRpc] = {}
        self._decimals: dict[str, int] = {}

    def supports(self, chain: Chain) -> bool:
        return chain in ZEROEX_CHAINS and bool(self.settings.zeroex_api_key)

    def rpc(self, chain: Chain) -> EvmRpc:
        if chain not in self._rpc:
            url = self.settings.rpc_urls.get(chain, "")
            if not url:
                raise ExecutionError(f"no RPC URL for {chain.value}")
            self._rpc[chain] = EvmRpc(self.http, url)
        return self._rpc[chain]

    async def _token_decimals(self, chain: Chain, token: str) -> int:
        key = f"{chain.value}:{token}"
        if key not in self._decimals:
            self._decimals[key] = await self.rpc(chain).decimals(token)
        return self._decimals[key]

    async def _native_price(self, chain: Chain) -> float:
        from . import WRAPPED_NATIVE

        address = WRAPPED_NATIVE.get(chain, "")
        snaps = await self.dex.token_pairs([address]) if address else []
        price = max((s.price_usd for s in snaps), default=0.0)
        if price <= 0:
            raise ExecutionError(f"could not price {chain.value} gas token")
        return price

    async def _fetch_quote(self, chain: Chain, params: dict[str, Any]) -> dict[str, Any]:
        try:
            data = await self.http.get(ZEROEX_URL, params=params, headers=self.headers)
        except HttpError as exc:
            raise ExecutionError(f"0x quote failed ({exc.status}): {exc.body[:200]}") from exc
        if not isinstance(data, dict):
            raise ExecutionError("0x returned a non-object response")
        if not data.get("liquidityAvailable", True):
            raise ExecutionError("0x reports no liquidity for this pair")
        return data

    async def quote(
        self, candidate: Candidate, side: Side, amount: float, fees: FeePlan | None = None
    ) -> Quote:
        chain = candidate.chain
        chain_id = EVM_CHAIN_IDS[chain]
        taker = self._account(chain).address if self.settings.evm_key_for(chain) else None
        native_price = await self._native_price(chain)
        token_decimals = await self._token_decimals(chain, candidate.address)
        slip = slip_bps(fees, self.slippage_bps)

        if side is Side.BUY:
            sell_amount = int((amount / native_price) * 10**18)
            params: dict[str, Any] = {
                "chainId": chain_id,
                "sellToken": NATIVE_SENTINEL,
                "buyToken": candidate.address,
                "sellAmount": sell_amount,
                "slippageBps": slip,
            }
        else:
            params = {
                "chainId": chain_id,
                "sellToken": candidate.address,
                "buyToken": NATIVE_SENTINEL,
                "sellAmount": int(amount * 10**token_decimals),
                "slippageBps": slip,
            }
        if taker:
            params["taker"] = taker

        data = await self._fetch_quote(chain, params)
        data["_params"] = params

        if side is Side.BUY:
            tokens = float(int(data.get("buyAmount") or 0)) / 10**token_decimals
            price = amount / tokens if tokens > 0 else 0.0
            impact = self._impact_from_mid(price, candidate.price_usd)
            return Quote(
                venue=self.id,
                in_amount=amount,
                out_amount=tokens,
                price=price,
                price_impact=impact,
                raw=data,
            )

        native_out = float(int(data.get("buyAmount") or 0)) / 10**18
        usd_out = native_out * native_price
        price = usd_out / amount if amount > 0 else 0.0
        return Quote(
            venue=self.id,
            in_amount=amount,
            out_amount=usd_out,
            price=price,
            price_impact=self._impact_from_mid(candidate.price_usd, price),
            raw=data,
        )

    @staticmethod
    def _impact_from_mid(executed: float, mid: float) -> float:
        """0x v2 does not return a price impact field, so it is derived from the
        gap to the indexed mid price. That gap is what actually costs money,
        which makes it the more useful number anyway."""
        if mid <= 0 or executed <= 0:
            return 0.0
        return max(0.0, executed / mid - 1.0)

    async def execute(
        self, candidate: Candidate, side: Side, amount: float, quote: Quote
    ) -> Fill:
        chain = candidate.chain
        rpc = self.rpc(chain)
        acct = self._account(chain)
        data = quote.raw
        if "transaction" not in data:
            params = dict(data.get("_params") or {})
            params["taker"] = acct.address
            data = await self._fetch_quote(chain, params)

        await self._ensure_allowance(rpc, chain, data, candidate)

        tx_spec = data.get("transaction") or {}
        tx: dict[str, Any] = {
            "to": tx_spec["to"],
            "data": tx_spec["data"],
            "value": int(tx_spec.get("value") or 0),
            "gas": int(tx_spec.get("gas") or 500_000),
            "_chain_id": EVM_CHAIN_IDS[chain],
        }
        if tx_spec.get("gasPrice"):
            tx["gasPrice"] = int(tx_spec["gasPrice"])

        tx_hash = await self._sign_and_send(rpc, tx, chain)
        await self._await_receipt(rpc, tx_hash, self.confirm_timeout)

        token_decimals = await self._token_decimals(chain, candidate.address)
        native_price = await self._native_price(chain)
        if side is Side.BUY:
            filled = float(int(data.get("buyAmount") or 0)) / 10**token_decimals
        else:
            filled = float(int(data.get("buyAmount") or 0)) / 10**18 * native_price

        slippage = 1.0 - filled / quote.out_amount if quote.out_amount > 0 else 0.0
        fee_native = float(int(data.get("totalNetworkFee") or 0)) / 10**18
        return Fill(
            venue=self.id,
            side=side,
            amount_in=amount,
            amount_out=filled,
            price=amount / filled if side is Side.BUY and filled else (filled / amount if amount else 0.0),
            fee_usd=fee_native * native_price,
            tx_id=tx_hash,
            slippage_vs_quote=slippage,
        )

    async def _ensure_allowance(
        self, rpc: EvmRpc, chain: Chain, data: dict[str, Any], candidate: Candidate
    ) -> None:
        """0x reports a missing allowance in `issues.allowance`; the token only
        needs approving on the sell leg, because buys sell the native asset."""
        issue = (data.get("issues") or {}).get("allowance")
        if not issue:
            return
        spender = issue.get("spender")
        if not spender:
            return
        log.info("approving token", extra={"token": candidate.address, "spender": spender})
        approve_tx = {
            "to": candidate.address,
            "data": SEL_APPROVE + _pad_address(spender) + _pad_uint(MAX_UINT256),
            "value": 0,
            "gas": 120_000,
            "_chain_id": EVM_CHAIN_IDS[chain],
        }
        tx_hash = await self._sign_and_send(rpc, approve_tx, chain)
        await self._await_receipt(rpc, tx_hash, self.confirm_timeout)


# ---------------------------------------------------------------------------
# Robinhood Chain (4663) - Uniswap V3 directly.
# ---------------------------------------------------------------------------

QUOTER_V2_ABI = [
    {
        "name": "quoteExactInputSingle",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {
                "name": "params",
                "type": "tuple",
                "components": [
                    {"name": "tokenIn", "type": "address"},
                    {"name": "tokenOut", "type": "address"},
                    {"name": "amountIn", "type": "uint256"},
                    {"name": "fee", "type": "uint24"},
                    {"name": "sqrtPriceLimitX96", "type": "uint160"},
                ],
            }
        ],
        "outputs": [
            {"name": "amountOut", "type": "uint256"},
            {"name": "sqrtPriceX96After", "type": "uint160"},
            {"name": "initializedTicksCrossed", "type": "uint32"},
            {"name": "gasEstimate", "type": "uint256"},
        ],
    }
]

SWAP_ROUTER_02_ABI = [
    {
        "name": "exactInputSingle",
        "type": "function",
        "stateMutability": "payable",
        "inputs": [
            {
                "name": "params",
                "type": "tuple",
                "components": [
                    {"name": "tokenIn", "type": "address"},
                    {"name": "tokenOut", "type": "address"},
                    {"name": "fee", "type": "uint24"},
                    {"name": "recipient", "type": "address"},
                    {"name": "amountIn", "type": "uint256"},
                    {"name": "amountOutMinimum", "type": "uint256"},
                    {"name": "sqrtPriceLimitX96", "type": "uint160"},
                ],
            }
        ],
        "outputs": [{"name": "amountOut", "type": "uint256"}],
    }
]

FEE_TIERS = (100, 500, 3000, 10_000)


class UniswapV3Venue(_EvmSignerMixin):
    """Robinhood Chain. Requires the `evm` extra for ABI encoding.

    The wallet must hold WETH (not raw ETH) because this uses a plain
    exactInputSingle rather than the router's multicall+wrap path - one fewer
    moving part in exchange for one manual wrap.
    """

    id = VenueId.UNISWAP_V3

    def __init__(
        self, http: Http, settings: Settings, strategy: Config, dexscreener: Dexscreener
    ) -> None:
        self.http = http
        self.settings = settings
        self.dex = dexscreener
        self.slippage_bps = int(strategy.get("execution.slippage_bps", 250))
        self.confirm_timeout = float(strategy.get("execution.confirm_timeout_seconds", 45))
        self.weth = settings.rh_chain_weth
        self.router_address = settings.rh_chain_swap_router
        self.quoter_address = settings.rh_chain_quoter
        self._rpc: EvmRpc | None = None
        self._decimals: dict[str, int] = {}
        self._best_fee: dict[str, int] = {}

    def supports(self, chain: Chain) -> bool:
        return chain is Chain.ROBINHOOD_CHAIN and bool(self.router_address and self.quoter_address)

    @property
    def rpc(self) -> EvmRpc:
        if self._rpc is None:
            url = self.settings.rpc_urls.get(Chain.ROBINHOOD_CHAIN, "")
            if not url:
                raise ExecutionError("ROBINHOOD_CHAIN_RPC_URL is not set")
            self._rpc = EvmRpc(self.http, url)
        return self._rpc

    def _codec(self):
        try:
            from web3 import Web3
        except ImportError as exc:  # pragma: no cover
            raise ExecutionError(
                "pip install 'alphahound[evm]' to trade on Robinhood Chain"
            ) from exc
        return Web3()

    async def _token_decimals(self, token: str) -> int:
        if token not in self._decimals:
            self._decimals[token] = await self.rpc.decimals(token)
        return self._decimals[token]

    async def _eth_price(self) -> float:
        snaps = await self.dex.token_pairs([WRAPPED_ETH_FOR_PRICING])
        price = max((s.price_usd for s in snaps), default=0.0)
        if price <= 0:
            raise ExecutionError("could not price ETH")
        return price

    async def _quote_amount_out(self, token_in: str, token_out: str, amount_in: int) -> tuple[int, int]:
        """Best (amountOut, fee) across the standard fee tiers.

        Trying every tier costs four eth_calls and avoids the classic failure
        of quoting 0.3% on a token whose only real pool is 1%.
        """
        w3 = self._codec()
        quoter = w3.eth.contract(address=w3.to_checksum_address(self.quoter_address), abi=QUOTER_V2_ABI)
        best = (0, 0)
        cache_key = f"{token_in}->{token_out}"
        tiers = (self._best_fee[cache_key],) + FEE_TIERS if cache_key in self._best_fee else FEE_TIERS
        for fee in dict.fromkeys(tiers):
            call = quoter.functions.quoteExactInputSingle(
                (
                    w3.to_checksum_address(token_in),
                    w3.to_checksum_address(token_out),
                    amount_in,
                    fee,
                    0,
                )
            )
            data = call._encode_transaction_data()
            try:
                raw = await self.rpc.eth_call(self.quoter_address, data)
            except ExecutionError:
                continue
            if not raw or raw == "0x":
                continue
            amount_out = int(raw[2:66], 16)
            if amount_out > best[0]:
                best = (amount_out, fee)
        if best[0] <= 0:
            raise ExecutionError("no Uniswap V3 pool with liquidity for this pair")
        self._best_fee[cache_key] = best[1]
        return best

    async def quote(
        self, candidate: Candidate, side: Side, amount: float, fees: FeePlan | None = None
    ) -> Quote:
        eth_price = await self._eth_price()
        token_decimals = await self._token_decimals(candidate.address)
        slip = slip_bps(fees, self.slippage_bps)

        if side is Side.BUY:
            amount_in = int((amount / eth_price) * 10**18)
            out_raw, fee = await self._quote_amount_out(self.weth, candidate.address, amount_in)
            tokens = out_raw / 10**token_decimals
            price = amount / tokens if tokens else 0.0
            return Quote(
                venue=self.id,
                in_amount=amount,
                out_amount=tokens,
                price=price,
                price_impact=ZeroExVenue._impact_from_mid(price, candidate.price_usd),
                raw={
                    "fee": fee,
                    "amount_in": amount_in,
                    "amount_out": out_raw,
                    "side": "buy",
                    "slippage_bps": slip,
                },
            )

        amount_in = int(amount * 10**token_decimals)
        out_raw, fee = await self._quote_amount_out(candidate.address, self.weth, amount_in)
        usd_out = out_raw / 10**18 * eth_price
        price = usd_out / amount if amount else 0.0
        return Quote(
            venue=self.id,
            in_amount=amount,
            out_amount=usd_out,
            price=price,
            price_impact=ZeroExVenue._impact_from_mid(candidate.price_usd, price),
            raw={
                "fee": fee,
                "amount_in": amount_in,
                "amount_out": out_raw,
                "side": "sell",
                "slippage_bps": slip,
            },
        )

    async def execute(
        self, candidate: Candidate, side: Side, amount: float, quote: Quote
    ) -> Fill:
        w3 = self._codec()
        acct = self._account(Chain.ROBINHOOD_CHAIN)
        token_in = self.weth if side is Side.BUY else candidate.address
        token_out = candidate.address if side is Side.BUY else self.weth
        amount_in = int(quote.raw["amount_in"])
        slip = int(quote.raw.get("slippage_bps") or self.slippage_bps)
        min_out = int(quote.raw["amount_out"] * (1.0 - slip / 10_000.0))

        await self._approve(token_in, self.router_address)

        router = w3.eth.contract(
            address=w3.to_checksum_address(self.router_address), abi=SWAP_ROUTER_02_ABI
        )
        call = router.functions.exactInputSingle(
            (
                w3.to_checksum_address(token_in),
                w3.to_checksum_address(token_out),
                int(quote.raw["fee"]),
                acct.address,
                amount_in,
                min_out,
                0,
            )
        )
        tx_hash = await self._sign_and_send(
            self.rpc,
            {
                "to": self.router_address,
                "data": call._encode_transaction_data(),
                "value": 0,
                "gas": 400_000,
                "_chain_id": EVM_CHAIN_IDS[Chain.ROBINHOOD_CHAIN],
            },
            Chain.ROBINHOOD_CHAIN,
        )
        await self._await_receipt(self.rpc, tx_hash, self.confirm_timeout)

        eth_price = await self._eth_price()
        token_decimals = await self._token_decimals(candidate.address)
        if side is Side.BUY:
            filled = quote.raw["amount_out"] / 10**token_decimals
        else:
            filled = quote.raw["amount_out"] / 10**18 * eth_price
        return Fill(
            venue=self.id,
            side=side,
            amount_in=amount,
            amount_out=filled,
            price=amount / filled if side is Side.BUY and filled else (filled / amount if amount else 0.0),
            fee_usd=0.0,
            tx_id=tx_hash,
            slippage_vs_quote=0.0,
        )

    async def _approve(self, token: str, spender: str) -> None:
        acct = self._account(Chain.ROBINHOOD_CHAIN)
        current = await self.rpc.eth_call(
            token, SEL_ALLOWANCE + _pad_address(acct.address) + _pad_address(spender)
        )
        if current and current != "0x" and int(current, 16) > 0:
            return
        tx_hash = await self._sign_and_send(
            self.rpc,
            {
                "to": token,
                "data": SEL_APPROVE + _pad_address(spender) + _pad_uint(MAX_UINT256),
                "value": 0,
                "gas": 120_000,
                "_chain_id": EVM_CHAIN_IDS[Chain.ROBINHOOD_CHAIN],
            },
            Chain.ROBINHOOD_CHAIN,
        )
        await self._await_receipt(self.rpc, tx_hash, self.confirm_timeout)
