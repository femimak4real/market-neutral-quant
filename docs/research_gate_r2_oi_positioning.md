# Research Gate R2: OI-Conditioned Cross-Sectional Reversal

(Research Gate R1 = Phase 0.5 funding arbitrage, closed by Decision 001.)

Revision 2: incorporates review clarifications (proxy wording, universe
construction, rebalance timing, turnover reporting, beta definition,
permutation procedure, neighbour definition, coverage window, breadth
reporting, anti-leakage). Economic thresholds unchanged.

Revision 2.1 (before any Stage B code or results): defines delisting from
traded volume. The archive keeps publishing frozen zero-volume kline files
after a contract stops trading (observed for REEFUSDT and FTMUSDT in Stage A
run_20261001T131223Z_28317331). Thresholds unchanged.

R2 is a research experiment, not the beginning of a trading bot. A PASS
authorizes a written proposal only.

## R2.0 Go / No-Go Rule

R2 IS A GO/NO-GO RESEARCH GATE, under the same rules as Phase 0.5:

- **FAIL**: STOP. Do not implement production infrastructure.
- **INCONCLUSIVE**: STOP. Report exactly what data or observations are missing.
- **PASS**: permits a written proposal for the next phase only. It does not
  permit building execution.

R2 has two sequential stages. Stage A must pass before Stage B runs.

| Stage | Purpose | Output |
|---|---|---|
| A: Data feasibility | Prove the required data exists at the required granularity | FEASIBLE / NOT FEASIBLE |
| B: Economic gate | Test the hypothesis out-of-sample | PASS / FAIL / INCONCLUSIVE |

If Stage A is NOT FEASIBLE, R2 stops and this spec is revised. Stage B
MUST NOT run on partial or substituted data.

## R2.1 Hypothesis

Short-term crypto price moves partly reverse, and the reversal is stronger
when the move came with a large change in open interest (a positioning
shock):

| OI change | Price move | Positioning-flow proxy | Expected next-period relative return |
|---|---|---|---|
| Up | Up | Crowding-long proxy | Underperform |
| Up | Down | Crowding-short proxy | Outperform (squeeze risk) |
| Down | Down | Flush proxy | Outperform (rebound) |
| Down | Up | Covering proxy | Underperform |

These labels are **positioning-flow proxies, not identification of actual
trader-side activity.** OI and price combinations do not uniquely identify
whether longs or shorts opened or closed positions. For example, OI up with
price up can arise from both sides opening while aggressive buyers move the
price. Reports MUST use the proxy labels above.

All four rows reduce to one signal: **reversal of the price move, weighted
by the size of the OI shock.**

The hypothesis is only worth pursuing if OI **adds** value over plain
reversal. R2 therefore tests the OI signal against a pre-registered control
(plain reversal) with identical construction, costs, and data.

### Liquidation data disclosure

Binance no longer publishes historical liquidation snapshots, and the live
liquidation stream is a per-symbol snapshot (lower bound). R2 does NOT use
actual liquidation data. The "OI down + price down" row is the **flush proxy**.
Reports MUST NOT describe it as liquidations and MUST NOT claim
liquidation-based results.

## R2.2 Universe (point-in-time, pre-registered)

- **Historical symbol master:** the candidate set is every USDⓈ-M USDT
  perpetual active on `start_date`, including contracts later delisted. It is
  built from the data.binance.vision futures/um kline archive listing, which
  retains delisted contracts. It MUST NOT be derived from today's
  `exchangeInfo`, which omits them.
- Eligibility: listed at least `min_listing_days` before `start_date`.
- Exclude: stablecoin-base contracts (e.g. USDCUSDT) and index or basket
  contracts.
- **Pre-start volume:** for each candidate, the 30 calendar days immediately
  preceding `start_date` are read from cached historical 1h klines (archive
  files; the REST endpoint may reject delisted symbols). A candidate lacking
  complete pre-start volume data is excluded with reason
  `INSUFFICIENT_PRESTART_VOLUME_DATA`. It is **never** substituted using
  today's ranking or any later information.
- Selection: top `universe_size` eligible candidates by summed 30-day perp
  quote volume.
- The universe is fixed for the whole period. A symbol delisted mid-period
  is closed at its last **traded** bar and removed. This is reported, never
  back-filled.
- **Delisting date** = the end of the last 1h bar with non-zero traded
  volume, when every later bar to the end of the data has zero volume for at
  least `delisting_zero_volume_days` (7), or when archive files end with
  clean 404s, whichever is earlier. Zero-volume placeholder bars, and any
  metrics or funding records after that point, are not tradable data.
- A candidate with zero traded volume in the pre-start window is
  `NOT_ACTIVE_AT_START`.
- The selected list and the volumes behind it are written to the run's
  pre-registration snapshot before any backtest.

## R2.3 Data

| Data | Source | Granularity |
|---|---|---|
| Open interest (`sum_open_interest`, `sum_open_interest_value`) | `data.binance.vision` futures/um daily `metrics` files | Required: 5-minute rows |
| Perp OHLCV incl. quote volume | data.binance.vision futures/um klines archive (required for delisted contracts); `GET /fapi/v1/klines` for live contracts | 1h |
| Funding rate + markPrice | `GET /fapi/v1/fundingRate` | Per settlement |
| BTCUSDT perp 1h klines | as above | For the beta diagnostic |

The Phase 0.5 downloader and caching are reused. The metrics downloader is
new and MUST cache per file, retry, and record missing days.

## R2.4 Stage A: Data Feasibility

Coverage is measured per symbol over its **required observation window**:

```
required_start = start_date - (zscore_window_days
                               + ceil(max(lookback_grid_hours) / 24)
                               + 1)          # 1-day point-in-time buffer
required_end   = min(end_date, delisting_date)
```

The window never extends before a contract could exist. Within it:

- metrics files present for >= 98% of calendar days
- median rows per present day >= 280 (5-minute cadence = 288)
- `sum_open_interest` non-null and > 0 in >= 99% of rows
- 1h perp klines missing <= 2%
- funding records missing <= 2% (Phase 0.5 method)

Symbols failing any check are excluded and reported.

**NOT FEASIBLE** if fewer than `min_feasible_symbols` remain, or if metrics
cadence is coarser than 5 minutes for most symbols. In that case, STOP.
Stage A writes its own report and does not touch Stage B outputs.

## R2.5 Signal Construction (point-in-time)

Decisions occur at each rebalance time T (daily, 00:00 UTC).

For each symbol i and lookback L (hours):

```
oi_T      = last metrics sum_open_interest_value with create_time <= T - 5m
oi_T-L    = same rule at T - L
d_oi_i    = ln(oi_T / oi_T-L)
ret_i     = ln(perp_close at bar ending <= T  /  perp_close at bar ending <= T - L)
```

The 5-minute guard is deliberate: a metrics row stamped T is not assumed to
be knowable at T.

Time-series normalization (per symbol, trailing window, no look-ahead):

```
z_oi_i  = (d_oi_i - mean(d_oi_i over prior W days)) / std(same)
z_ret_i = (ret_i  - mean(ret_i  over prior W days)) / std(same)
```

computed from values at prior rebalance times only, with W = `zscore_window_days`.
A symbol needs a full W-day history before it can receive a position.

Signals:

```
OI signal (test):    S_i = -z_ret_i * |z_oi_i|
Control (baseline):  C_i = -z_ret_i
```

Both use the same L, the same normalization, and the same portfolio rules.

**Anti-leakage rule.** No information derived from the future target holding
period may enter score construction, universe eligibility, normalization,
parameter selection, or position sizing. Funding rates may be used only for
PnL after a position is open. They MUST NOT influence the OI score, the
control score, or universe selection.

## R2.6 Portfolio Construction

At each rebalance T:

- Rank eligible symbols by score (S for the test, C for the control).
- Long the top `n_per_side`, short the bottom `n_per_side`.
- Equal notional per name: `gross_notional / (2 * n_per_side)`.
  Dollar-neutral by construction.
- Trade only the differences between old and new target positions.

Rebalance sequence at each T:

1. Calculate scores using only information available at T (R2.5).
2. Determine the target portfolio.
3. Execute the differences at the open of the first 1h bar starting after T.
4. A new or changed position becomes active at that execution price.
   Funding settlements before that fill are excluded for the new exposure.
5. Unchanged positions remain open with no trade, no fee, no slippage.
6. At the next rebalance T', repeat. Positions removed or reduced are
   closed at the open of the first 1h bar starting after T'. Funding accrues
   on them until that fill.

Not in R2: beta weighting, volatility weighting, optimization, or stop-losses.

## R2.7 PnL and Costs

Per position, between consecutive rebalances:

```
price_pnl   = qty * (exit_price - entry_price) * side      (side +1 long, -1 short)
funding_pnl = sum over settlements s while held of:
              -side * funding_rate_s * qty * markPrice_s
              (longs pay positive funding; shorts receive it)
fees        = traded_notional * perp_taker_fee     (each fill, entry and exit)
slippage    = traded_notional * slippage_bps / 10000
net         = price_pnl + funding_pnl - fees - slippage
```

- markPrice precedence and fallback follow Phase 0.5 rule 0.5.5 exactly.
- Fees and slippage apply to traded notional only. An unchanged position
  incurs no fee and no slippage.
- Turnover is reported as entry turnover, exit turnover, gross turnover
  (entry + exit), and total fees, each per rebalance and in total.

Daily mark-to-market (for drawdown, beta, and monthly tables): at each UTC
day boundary, open positions are marked at the close of the 1h bar ending at
00:00 UTC, with funding and costs counted on the day they occur.

Capital allocation (research assumption, NOT exchange margin):

```
capital_allocation = gross_notional / max_leverage
```

## R2.8 Short-Squeeze Diagnostic (not a liquidation model)

For each short position, compute the maximum adverse move (highest 1h high
vs entry) while held, and the resulting consumption of that name's margin
allocation (`name_notional / max_leverage`), using the Phase 0.5.6 formula.
Report the worst 10 events, and flag consumption >= `diagnostic_margin_consumption_limit`.

## R2.9 Parameter Selection (anti-overfitting)

- The only selectable parameter is L, from `lookback_grid_hours`.
- Config snapshot, universe, and code version are hashed and written before
  any backtest.
- Chronological split at `is_fraction`.
- Select the single L maximizing **in-sample net PnL of the OI signal**.
  Lock it. Run OOS once, for both the OI signal and the control at the
  locked L.
- No other parameter may be changed after OOS results are viewed. Reruns are
  separate runs, and earlier runs are retained.

## R2.10 Stage B Gate Criteria (OOS, locked L)

**INCONCLUSIVE** if:

- Stage A was not run, or was NOT FEASIBLE
- fewer than `min_oos_rebalances` OOS rebalances had full positions
- fewer than `min_feasible_symbols` symbols remained eligible OOS

**PASS** only if ALL hold:

| # | Criterion |
|---|---|
| 1 | OI-signal OOS net PnL > 0 |
| 2 | OI-signal OOS net PnL > control OOS net PnL (OI adds value) |
| 3 | OI-signal OOS net PnL > 0 with fees and slippage x `cost_stress_multiplier` |
| 4 | Both grid neighbours of the locked L: OI-signal OOS net PnL > 0 |
| 5 | OOS annualized return on capital allocation >= `hurdle_annual_return` |
| 6 | OOS max drawdown <= `max_drawdown_pct` of capital allocation |
| 7 | abs(OOS BTC beta) <= `max_abs_beta` |
| 8 | Permutation test (definition below) |

Otherwise **FAIL**. Every criterion is reported with its number.

**Criterion 4 neighbours** are the grid entries immediately adjacent to the
locked L, by position in `lookback_grid_hours`. They are NOT the
best-performing alternatives. Example: grid [4, 8, 12, 24, 48, 72], locked
L = 24, so neighbours are 12 and 48. If the locked L is the first or last
grid entry, a neighbour is missing and criterion 4 is FAIL.

**Criterion 7 beta** is computed OOS only, for the locked strategy:

```
strategy_daily_return = daily_net_pnl / capital_allocation
BTC_daily_return      = BTC_close_d / BTC_close_(d-1) - 1
beta = Cov(strategy_daily_return, BTC_daily_return) / Var(BTC_daily_return)
```

Both series use the same UTC calendar days (close = the 1h bar ending at
00:00 UTC), and BTC prices come from BTCUSDT perp 1h klines.

**Criterion 8 permutation procedure.** At every OOS rebalance, randomly
shuffle the OI **scores** across that rebalance's eligible symbols. Shuffle
scores only, never returns or PnL. Everything else is preserved: the same
eligible universe, number of longs and shorts, target notionals, rebalance
times, prices, funding, costs, and turnover mechanics. Compute the complete
OOS net PnL. Repeat `n_permutations` times with `permutation_seed`.

```
criterion 8 passes if actual_OOS_net_PnL > quantile(permuted_OOS_net_PnLs,
                                                     permutation_quantile)
```

Criterion 2 is the core of the hypothesis. If the OI signal does not beat
plain reversal, OI is not adding information, and R2 fails even if it is
profitable.

## R2.11 Report Contents

Stage A report:

- per-symbol file coverage, rows per day, null rate, kline and funding gaps
- included and excluded symbols with reasons; FEASIBLE / NOT FEASIBLE

Stage B report (for in-sample and OOS separately, and for both the OI signal
and the control):

- net PnL split into price, funding, fees, slippage
- entry turnover, exit turnover, gross turnover, total fees (R2.7)
- cross-sectional breadth: eligible symbols per rebalance, reported as
  median, minimum, 5th percentile, and the count of rebalances with only 8-9
  eligible symbols (report only; not a gate criterion)
- annualized return, max drawdown, daily-return Sharpe (descriptive only)
- BTC beta and its standard error
- per-month net PnL table
- long-side vs short-side PnL contribution
- quadrant attribution: PnL by the four positioning-flow proxy rows in R2.1
- in-sample L grid table; neighbour results; cost stress
- permutation distribution (quantiles) and the actual result's percentile
- short-squeeze diagnostic (R2.8)
- universe list, delistings, config snapshot, hash, code version
- verdict and go/no-go action (R2.0)
- known limitations (R2.13)

## R2.12 Config (`config/r2_oi.yaml`)

```yaml
r2_oi:
  start_date: "2024-10-01"
  end_date: "2026-09-30"
  universe_size: 20
  min_listing_days: 90
  min_feasible_symbols: 15
  exclude_bases: [USDC, FDUSD, TUSD, USDP, DAI]
  zscore_window_days: 30
  lookback_grid_hours: [4, 8, 12, 24, 48, 72]
  rebalance_hour_utc: 0
  n_per_side: 4
  gross_notional_usdt: 8000
  max_leverage: 2.0
  diagnostic_margin_consumption_limit: 0.70
  fees:
    perp_taker: 0.0005           # Regular user, no BNB discount
  slippage_bps:
    default: 5
    BTCUSDT: 2
    ETHUSDT: 2
  is_fraction: 0.6
  cost_stress_multiplier: 1.5
  min_oos_rebalances: 200
  # Opportunity-cost parameter; set BEFORE running, never after OOS.
  hurdle_annual_return: 0.05
  max_drawdown_pct: 0.10
  max_abs_beta: 0.20
  n_permutations: 1000
  permutation_quantile: 0.95
  permutation_seed: 20261001
  data_dir: data/r2_oi
  reports_dir: reports/r2_oi
```

All values are research parameters to be confirmed before the first run.
Fees MUST be non-null, as in Phase 0.5.

## R2.13 Known Limitations (must appear in report)

- No liquidation data is used. The flush proxy (OI down, price down) does
  not identify liquidations.
- OI/price quadrants are positioning-flow proxies; they do not identify
  which side opened or closed positions.
- The universe is fixed at `start_date`; contracts listed later are excluded.
- 1h bar opens are not executable prices; slippage_bps approximates spread
  and impact. Thin names may be worse than modelled.
- Metrics timestamps are treated conservatively (5-minute guard). Binance's
  exact publication timing is not documented here.
- Funding is charged at actual settlements; the funding-rate regime is the
  one Decision 001 found compressed after December 2025.
- The 2024-10 to 2026-09 period overlaps the R1 study. R1 examined funding
  rates, not OI-signal returns, so R2's OOS signal results are unseen.
- Daily rebalancing at a fixed hour; no intraday risk management.

## R2.14 Reuse from Phase 0.5

Reuse without modification where possible: `common.py` (config validation,
hashing, code version), Phase 0.5 kline and funding download and caching,
markPrice convention, `economics.max_drawdown`, `annualized_return`,
`margin_consumption`, `meets_threshold`, run index, previous-run tracking,
and report table helpers.

New modules go under `research/r2_oi/`. They MUST NOT modify Phase 0.5
outputs or `reports/phase05/`.

## R2.15 Required Tests

- point-in-time: a decision at T never reads metrics with create_time > T - 5m
  or klines ending after T
- z-score uses only prior rebalance values
- S and C signs: price up + OI up gives a negative S (short candidate)
- quadrant mapping matches the R2.1 table on synthetic inputs
- dollar neutrality: long notional equals short notional at every rebalance
- turnover costs: an unchanged portfolio incurs zero fees
- funding sign: a long pays positive funding, a short receives it
- delisting: position closed at last traded bar and symbol removed;
  zero-volume placeholder bars end the window; a short zero-volume run does not
- universe built from point-in-time data and includes a delisted candidate
- permutation test reproducible with a fixed seed
- beta calculation on a synthetic known-beta series
- Stage B refuses to run unless Stage A is FEASIBLE
- universe: a candidate missing pre-start volume is excluded as
  INSUFFICIENT_PRESTART_VOLUME_DATA and never replaced by current data
- coverage window starts at required_start, not before listing
- timing: funding settled between T and the fill is not charged or credited
  to new exposure; removed positions accrue funding until their exit fill
- turnover: entry, exit, and gross turnover match a hand example
- criterion 4 neighbours are grid-adjacent; locked L at either grid edge
  gives FAIL
- beta: the formula returns a known beta on a synthetic series; UTC day
  boundaries align
- permutation: shuffles scores only (prices, funding, and costs identical
  across permutations); the same seed gives the same distribution
- anti-leakage: changing funding data does not change any score or the
  universe
- breadth statistics computed correctly on a synthetic eligibility series
- gate branches: each PASS / FAIL / INCONCLUSIVE path, including grid edge
- script refuses to run with null fees
