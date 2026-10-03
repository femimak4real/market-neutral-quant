# Economic Validation Gate — Phase 0.5

(Replaces all previous Phase 0.5 text in SPEC.md.)

## 0.5.0 Go / No-Go Rule

PHASE 0.5 IS A GO/NO-GO RESEARCH GATE.

- If the result is **FAIL**: STOP. Do not implement production
  market-data, execution, or trading infrastructure.
- If the result is **INCONCLUSIVE**: STOP. Do not implement production
  infrastructure. Report exactly what additional data, observations, or
  model corrections are required to reach a conclusive result.
- Only **PASS** permits Phase 1.

The agent MUST NOT proceed on the grounds that results were "close",
"promising", or "likely to improve with more data".

## 0.5.1 Scope and Isolation

- Standalone research code in `research/phase05_funding_gate/`.
- Public REST endpoints only. MUST NOT require API keys.
- ZERO imports from the existing Binance Features Bot or any execution code.
- ZERO order-placement capability.
- Output: trades CSV + markdown report + PASS / FAIL / INCONCLUSIVE.

Directory layout:

```
research/phase05_funding_gate/
├── download_data.py
├── backtest.py
├── economics.py
├── report.py
└── tests/
config/phase05.yaml
```

## 0.5.2 Data

For each symbol in the universe:

- Perp funding history: `GET /fapi/v1/fundingRate` (paginate).
  Retain `fundingTime`, `fundingRate`, and `markPrice` per record.
- Perp klines: `GET /fapi/v1/klines`, interval 1h.
- Spot klines: `GET /api/v3/klines`, interval 1h.

Period: `start_date..end_date` from config; minimum 12 months.

Use actual funding settlement timestamps. Do NOT assume 8h intervals;
some symbols use other intervals or have changed interval over time.

If more than 2% of required bars or funding records are missing for a
symbol, exclude it and report why.

## 0.5.3 Timing (No Look-Ahead)

- Decisions are made only at funding settlement times t.
- A decision at t may use only funding rates settled at or before t.
- Entry/exit fills occur at the open of the first 1h bar starting after t
  (spot and perp at the same bar).
- A position receives/pays funding at settlement s only if it was open
  before s.

## 0.5.4 Entry / Exit Rules

Entry (symbol flat) when ALL hold:

- mean(last K settled funding rates) >= entry_threshold
- last settled funding rate > 0

Exit (position open) when:

- last N settled funding rates are all < exit_threshold, OR
- end of data (close with full exit costs; flag as forced exit)

One position per symbol. Fixed notional per symbol. No compounding.
No position-sizing logic.

## 0.5.5 Trade PnL (USDT)

```
qty = notional_per_symbol / spot_entry_price
```

### Funding PnL

For v0.5, funding PnL is estimated as:

```
funding_pnl = sum over settlements s while open of:
              funding_rate_s * qty * funding_price_s
```

where `funding_rate_s` is the signed settled rate (short perp: positive
rate = income, negative rate = cost; the sign is handled by the formula).

Notional-price convention (`funding_price_s`):

`markPrice` from the funding-history record is authoritative for the
funding-price calculation; the backtest MUST NOT substitute the 1h candle
price when `markPrice` is available.

1. Use the `markPrice` returned by `/fapi/v1/fundingRate` for that
   settlement whenever it is present and non-empty. This reconstructs the
   exchange's funding amount for a linear USDⓈ-M contract.
2. Only if `markPrice` is missing or empty, fall back to the perp 1h kline
   open at or immediately after s, and count that settlement as a fallback.
3. The implementation MUST NOT silently assume a kline price equals the
   exchange's funding calculation price.
4. The report MUST state the convention used and the number/percentage of
   settlements that used the fallback, per symbol.

### Other components

```
basis_pnl = qty * ((spot_exit - spot_entry) - (perp_exit - perp_entry))

fees      = qty * (spot_entry * spot_fee + perp_entry * perp_fee
                 + spot_exit  * spot_fee + perp_exit  * perp_fee)

slippage  = 4 fills * slippage_bps[symbol] / 10000 * notional

net_pnl   = funding_pnl + basis_pnl - fees - slippage
```

## 0.5.6 Margin-Consumption Diagnostic (Not a Liquidation Model)

This is a conservative screening diagnostic. It does NOT reproduce
Binance's margin-ratio, maintenance-margin, or liquidation engine, and its
output MUST NOT be described as a liquidation count.

For each trade:

```
futures_margin   = notional_per_symbol / max_leverage
max_adverse_price = highest 1h perp HIGH from entry bar to exit bar
                    (inclusive)
adverse_pnl      = qty * max(0, max_adverse_price - perp_entry)
consumption      = adverse_pnl / futures_margin
remaining_futures_equity = futures_margin - adverse_pnl
```

Flag the trade if:

```
consumption >= diagnostic_margin_consumption_limit
```

Meaning of the limit: the fraction of the initial futures allocation
consumed by the worst unrealized loss on the short perp leg. For example,
0.70 means the short's worst loss reached 70% of its allocated margin.
At max_leverage = 2.0, that corresponds to a perp rise of 35% above entry
(0.70 / 2.0).

Conservative assumptions (deliberate):

- No credit for spot-leg gains (separate wallets, per Section 15).
- No credit for funding received while open.
- No top-ups or transfers.
- The intra-bar high is treated as reached.

Because the real exchange also applies maintenance margin, real
liquidation would occur BEFORE consumption reaches 1.0. A trade with
consumption below the limit is therefore not proven safe.

Report per trade: consumption and the equivalent price move. Report per
symbol: the maximum consumption and the number of flagged trades. Also
report the equivalent price-move threshold implied by the config
(limit / max_leverage).

Flagged trades still count in PnL but MUST be listed individually. Any
trade with consumption >= 1.0 MUST be highlighted as "allocation fully
consumed".

`diagnostic_margin_consumption_limit` is a Phase 0.5 research parameter
only. It is NOT Section 15's `margin_kill_ratio`, which refers to the
exchange's margin ratio. Later phases MUST NOT reuse it as such.

## 0.5.7 Capital Allocation and Return

```
capital_allocation_per_symbol = notional + notional / max_leverage
```

This is a conservative research capital-allocation assumption: full
notional for the spot leg plus a futures margin allocation. It is NOT the
exchange's required margin, and later phases MUST NOT treat it as Binance's
actual initial or maintenance margin requirement.

```
OOS annualized return = OOS net_pnl / total capital_allocation
                        (across included symbols),
                        annualized by OOS calendar duration
```

## 0.5.8 Threshold Selection (Anti-Overfitting)

1. Before any backtest run, write a snapshot of the full config and its
   hash into the report. This includes hurdle_annual_return, all gate
   criteria, and the grid. These values MUST NOT change after OOS results
   are viewed. Any rerun with changed values MUST be reported as a new,
   separate run, with the earlier run's result retained.
2. Split data chronologically: first `is_fraction` = in-sample, remainder =
   out-of-sample (OOS).
3. Evaluate entry_threshold over entry_threshold_grid on IN-SAMPLE data
   only, with ONE threshold for the whole universe (not per symbol).
4. Select the threshold maximizing in-sample universe net PnL.
5. Lock it. Run OOS once. Do NOT re-tune after viewing OOS.

## 0.5.9 Gate Criteria

All criteria are evaluated on OOS data with the locked threshold.

**INCONCLUSIVE** if:

- total OOS trades < min_oos_trades, OR
- fewer than 6 symbols survive data checks, OR
- more than 10% of OOS settlements used the funding-price fallback.

**PASS** only if ALL hold:

1. Universe OOS net PnL > 0.
2. At least min_profitable_symbols symbols have OOS net PnL > 0, with
   at least 3 OOS trades each.
3. Universe OOS net PnL > 0 with fees AND slippage multiplied by
   cost_stress_multiplier.
4. Both neighboring grid thresholds (one step each side) also produce
   universe OOS net PnL > 0.
5. OOS annualized return >= hurdle_annual_return.
6. Max drawdown of the combined OOS equity curve <= max_drawdown_pct of
   total capital allocation.

Otherwise **FAIL**.

Report every criterion as PASS/FAIL with the number behind it.

## 0.5.10 Required Report Contents

Per symbol and universe, for in-sample and OOS separately:

- trade count
- gross funding, basis PnL
- entry fees, exit fees, slippage
- net PnL
- average and median holding period (in funding periods and hours)
- win rate
- max drawdown
- annualized return
- margin-consumption flagged trades and maximum consumption
- forced exits
- funding-price fallback count

Also include:

- config snapshot and hash (from 0.5.8)
- breakeven funding rate per period
  = round-trip cost fraction / median holding periods
- percentage of settlements with rate above breakeven
- full in-sample grid results table
- stressed-cost results
- final verdict: PASS / FAIL / INCONCLUSIVE, and the go/no-go action
  required by 0.5.0
- known limitations (0.5.11)

## 0.5.11 Known Limitations (Must Appear in Report)

- The 1h bar open is not an executable price; slippage_bps approximates
  spread and impact.
- No intra-bar execution, latency, or partial-fill modelling.
- The margin-consumption check is a conservative diagnostic; liquidation
  and maintenance margin are not simulated, and real liquidation would
  occur before consumption reaches 1.0.
- The fee tier is assumed constant over the period.
- Funding PnL uses the convention in 0.5.5; fallback settlements are
  approximations.

## 0.5.12 Config (`config/phase05.yaml`)

```yaml
phase05:
  start_date: "2024-10-01"
  end_date: "2026-09-30"
  symbols: [BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT,
            DOGEUSDT, ADAUSDT, AVAXUSDT, LINKUSDT]
  notional_per_symbol_usdt: 1000
  max_leverage: 2.0
  # Fraction of futures allocation consumed by worst adverse move (0.5.6).
  # NOT Binance margin ratio; NOT Section 15 margin_kill_ratio.
  diagnostic_margin_consumption_limit: 0.70
  fees:
    spot_taker: null   # REQUIRED: your actual rate (fraction)
    perp_taker: null   # REQUIRED: your actual rate (fraction)
  slippage_bps:
    default: 5
    BTCUSDT: 2
    ETHUSDT: 2
  entry_lookback_periods: 3        # K
  entry_threshold_grid: [0.00005, 0.0001, 0.00015,
                         0.0002, 0.0003, 0.0005]
  exit_threshold: 0.0
  exit_periods: 2                  # N
  is_fraction: 0.6
  cost_stress_multiplier: 1.5
  min_oos_trades: 20
  min_profitable_symbols: 3
  # Opportunity-cost parameter: set to your realistic alternative yield
  # BEFORE running. Never adjust after viewing OOS results.
  hurdle_annual_return: 0.05
  max_drawdown_pct: 0.05
```

The script MUST refuse to run if `fees.spot_taker` or `fees.perp_taker`
is null.

## 0.5.13 Required Tests (`research/phase05_funding_gate/tests/`)

- funding PnL sign: positive rate on short = income, negative = cost
- funding price uses markPrice when present, fallback only when empty,
  and the fallback is counted
- four-fill fee calculation
- basis PnL on a hand-computed example
- no look-ahead: a decision at t cannot see the funding rate settled after t
- funding is accrued only for settlements while the position is open
- chronological split: no OOS data used in threshold selection
- script refuses to run with null fees
- gate logic: each PASS/FAIL/INCONCLUSIVE branch on synthetic inputs
- margin diagnostic: notional 1000, leverage 2, entry 100, max high 135
  -> consumption 0.70 -> flagged; max high 134 -> 0.68 -> not flagged;
  price falling only -> consumption 0
