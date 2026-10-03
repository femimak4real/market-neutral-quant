"""R2.6-R2.8 portfolio simulation, daily mark-to-market and statistics.

Implementation conventions (fixed before any Stage B result existed):
- Ranking ties are broken alphabetically by symbol (deterministic).
- With fewer than 2 * n_per_side eligible symbols the target is flat.
- Held positions on the same side are not resized (R2.6 step 5), so
  realized long/short notional can drift between rebalances; target entry
  notionals are always equal and the drift is reported.
- Funding counts for settlements strictly after the entry fill and strictly
  before the exit fill.
- A position whose symbol delists is closed at the close of its last traded
  bar. Segment ends close everything at the close of the bar ending at the
  segment end.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from research.phase05_funding_gate import economics as E
from research.phase05_funding_gate.common import DAY_MS, HOUR_MS


@dataclass
class SymbolMarket:
    symbol: str
    bar_open: np.ndarray
    open: np.ndarray
    high: np.ndarray
    close: np.ndarray
    window_end: int
    f_time: np.ndarray
    f_cum: np.ndarray            # len(f_time)+1 prefix sums of rate*price
    slippage_bps: float
    m_time: np.ndarray = field(default_factory=lambda: np.array([], np.int64))
    m_oiv: np.ndarray = field(default_factory=lambda: np.array([]))

    def bar_index(self, t: int) -> int:
        i = int(np.searchsorted(self.bar_open, t))
        return i if i < len(self.bar_open) and self.bar_open[i] == t else -1

    def tradable_bar(self, t: int) -> int:
        """Bar opening exactly at t that ends no later than window_end."""
        i = self.bar_index(t)
        return i if i >= 0 and t + HOUR_MS <= self.window_end else -1

    def first_bar_at_or_after(self, t: int) -> int:
        i = int(np.searchsorted(self.bar_open, t))
        if i < len(self.bar_open) and self.bar_open[i] + HOUR_MS <= self.window_end:
            return i
        return -1

    def last_bar_ending_by(self, t: int) -> int:
        i = int(np.searchsorted(self.bar_open + HOUR_MS, min(t, self.window_end),
                                side="right")) - 1
        return i

    def funding_sum(self, a: int, b: int) -> float:
        """Sum of rate*price for settlements with a < s < b."""
        i0 = int(np.searchsorted(self.f_time, a, side="right"))
        i1 = int(np.searchsorted(self.f_time, b, side="left"))
        return float(self.f_cum[i1] - self.f_cum[i0]) if i1 > i0 else 0.0


def make_funding_cum(times: np.ndarray, rates: np.ndarray, prices: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(rates * prices)])


@dataclass
class SimParams:
    n_per_side: int
    gross_notional: float
    fee: float
    leverage: float


@dataclass
class SimResult:
    lots: list[dict]
    eligible: np.ndarray
    full: np.ndarray
    entry_notional: np.ndarray
    exit_notional: np.ndarray
    net_exposure: np.ndarray      # long MV - short MV right after each rebalance

    @property
    def net_pnl(self) -> float:
        return float(sum(l["net"] for l in self.lots))


def fill_matrix(market: list[SymbolMarket], times: np.ndarray) -> np.ndarray:
    out = np.full((len(times), len(market)), -1, dtype=int)
    for j, m in enumerate(market):
        for k, t in enumerate(times):
            out[k, j] = m.tradable_bar(int(t) + HOUR_MS)
    return out


def _close(lot: dict, m: SymbolMarket, p: SimParams, exit_time: int, exit_price: float,
           reason: str, collect: bool) -> dict:
    qty, side = lot["qty"], lot["side"]
    notional = qty * exit_price
    lot.update(exit_time=exit_time, exit_price=exit_price, exit_reason=reason)
    lot["price_pnl"] = side * qty * (exit_price - lot["entry_price"])
    lot["funding"] = -side * qty * m.funding_sum(lot["entry_time"], exit_time)
    lot["exit_fee"] = notional * p.fee
    lot["exit_slip"] = notional * m.slippage_bps / 10_000
    lot["exit_notional"] = notional
    lot["net"] = (lot["price_pnl"] + lot["funding"] - lot["entry_fee"] - lot["exit_fee"]
                  - lot["entry_slip"] - lot["exit_slip"])
    if collect and side < 0:
        a = lot["entry_bar"]
        b = int(np.searchsorted(m.bar_open, exit_time))
        hi = float(m.high[a:max(b, a + 1)].max())
        mc = E.margin_consumption(lot["entry_notional"], p.leverage, qty,
                                  lot["entry_price"], hi)
        lot["max_high"], lot["consumption"] = hi, mc["consumption"]
        lot["adverse_move"] = mc["adverse_move_fraction"]
    return lot


def simulate(market: list[SymbolMarket], times: np.ndarray, scores: np.ndarray,
             fills: np.ndarray, seg_end: int, p: SimParams,
             quadrants: np.ndarray | None = None, collect: bool = True,
             rng: np.random.Generator | None = None) -> SimResult:
    n_t, n_s = scores.shape
    n = p.n_per_side
    per_name = p.gross_notional / (2 * n)
    open_lots: dict[int, dict] = {}
    lots: list[dict] = []
    eligible = np.zeros(n_t, int)
    full = np.zeros(n_t, bool)
    entry_n = np.zeros(n_t)
    exit_n = np.zeros(n_t)
    net_exp = np.zeros(n_t)

    for k in range(n_t):
        T = int(times[k])
        fill = T + HOUR_MS
        # Delisted symbols: close at the close of the last traded bar.
        for j in [j for j, l in open_lots.items() if market[j].window_end <= fill]:
            m = market[j]
            i = m.last_bar_ending_by(m.window_end)
            lots.append(_close(open_lots.pop(j), m, p, int(m.bar_open[i]) + HOUR_MS,
                               float(m.close[i]), "delisted", collect))

        mask = ~np.isnan(scores[k]) & (fills[k] >= 0)
        elig = np.flatnonzero(mask)
        eligible[k] = elig.size
        target: dict[int, int] = {}
        if elig.size >= 2 * n:
            sc = scores[k, elig].copy()
            if rng is not None:
                sc = rng.permutation(sc)          # shuffle scores only
            order = np.lexsort((elig, -sc))       # score desc, then symbol order
            ranked = elig[order]
            target.update({int(j): 1 for j in ranked[:n]})
            target.update({int(j): -1 for j in ranked[-n:]})
        full[k] = len(target) == 2 * n

        for j in [j for j, l in open_lots.items() if target.get(j) != l["side"]]:
            m = market[j]
            i = fills[k, j] if fills[k, j] >= 0 else m.first_bar_at_or_after(fill)
            if i >= 0:
                px, t_exit, why = float(m.open[i]), int(m.bar_open[i]), "rebalance"
            else:
                i = m.last_bar_ending_by(m.window_end)
                px, t_exit, why = float(m.close[i]), int(m.bar_open[i]) + HOUR_MS, "no bar"
            lot = _close(open_lots.pop(j), m, p, t_exit, px, why, collect)
            exit_n[k] += lot["exit_notional"]
            lots.append(lot)

        for j, side in target.items():
            if j in open_lots:
                continue
            m = market[j]
            i = fills[k, j]
            px = float(m.open[i])
            qty = per_name / px
            open_lots[j] = {
                "symbol": m.symbol, "sym_index": j, "side": side, "qty": qty,
                "entry_k": k, "entry_time": int(m.bar_open[i]), "entry_bar": int(i),
                "entry_price": px, "entry_notional": per_name,
                "entry_fee": per_name * p.fee,
                "entry_slip": per_name * m.slippage_bps / 10_000,
                "quadrant": int(quadrants[k, j]) if quadrants is not None else -1,
            }
            entry_n[k] += per_name

        if collect:
            exp = 0.0
            for j, l in open_lots.items():
                m = market[j]
                i = m.bar_index(fill)
                px = float(m.open[i]) if i >= 0 else l["entry_price"]
                exp += l["side"] * l["qty"] * px
            net_exp[k] = exp

    for j in list(open_lots):
        m = market[j]
        i = m.last_bar_ending_by(seg_end)
        lots.append(_close(open_lots.pop(j), m, p, int(m.bar_open[i]) + HOUR_MS,
                           float(m.close[i]), "segment end", collect))
    return SimResult(lots, eligible, full, entry_n, exit_n, net_exp)


def equity_curve(lots: list[dict], market: list[SymbolMarket],
                 boundaries: np.ndarray) -> np.ndarray:
    """Cumulative net PnL at each UTC day boundary (R2.7 daily mark-to-market)."""
    eq = np.zeros(len(boundaries))
    closes = {}
    for lot in lots:
        j = lot["sym_index"]
        m = market[j]
        if j not in closes:
            ends = m.bar_open + HOUR_MS
            idx = np.searchsorted(ends, boundaries, side="right") - 1
            closes[j] = np.where(idx >= 0, m.close[np.maximum(idx, 0)], np.nan)
        side, qty = lot["side"], lot["qty"]
        entered = boundaries >= lot["entry_time"]
        exited = boundaries >= lot["exit_time"]
        mark = np.where(exited, lot["exit_price"], closes[j])
        price = np.where(entered, side * qty * (mark - lot["entry_price"]), 0.0)
        i0 = int(np.searchsorted(m.f_time, lot["entry_time"], side="right"))
        i1 = np.searchsorted(m.f_time, np.minimum(boundaries + 1, lot["exit_time"]),
                             side="left")
        i1 = np.maximum(i1, i0)
        fund = np.where(entered, -side * qty * (m.f_cum[i1] - m.f_cum[i0]), 0.0)
        costs = (np.where(entered, lot["entry_fee"] + lot["entry_slip"], 0.0)
                 + np.where(exited, lot["exit_fee"] + lot["exit_slip"], 0.0))
        eq += np.nan_to_num(price) + fund - costs
    return eq


def day_boundaries(seg_start: int, seg_end: int) -> np.ndarray:
    return np.arange(seg_start, seg_end + 1, DAY_MS, dtype=np.int64)


def beta_stats(strategy_ret: np.ndarray, btc_ret: np.ndarray) -> tuple[float | None, float | None]:
    ok = ~np.isnan(strategy_ret) & ~np.isnan(btc_ret)
    y, x = strategy_ret[ok], btc_ret[ok]
    if len(x) < 3 or np.var(x, ddof=1) == 0:
        return None, None
    beta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    resid = y - (y.mean() + beta * (x - x.mean()))
    se = float(np.sqrt((resid @ resid) / (len(x) - 2) / ((x - x.mean()) @ (x - x.mean()))))
    return beta, se


def breadth_stats(eligible: np.ndarray, n_per_side: int) -> dict:
    if eligible.size == 0:
        return {"median": None, "min": None, "p5": None, "thin_rebalances": 0}
    return {
        "median": float(np.median(eligible)),
        "min": int(eligible.min()),
        "p5": float(np.percentile(eligible, 5)),
        "thin_rebalances": int(((eligible >= 2 * n_per_side)
                                & (eligible <= 2 * n_per_side + 1)).sum()),
    }
