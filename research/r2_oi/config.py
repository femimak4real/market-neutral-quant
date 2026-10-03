"""R2 configuration loading and validation. No network, no execution."""
from __future__ import annotations

import math
from pathlib import Path

import yaml

from research.phase05_funding_gate import common as C

# r2-stage-a-0.1: original Stage A (run_20261001T131223Z_28317331).
# r2-stage-a-0.2: delisting also inferred from a trailing zero-volume run
#                 (archive placeholder bars after trading ends).
R2_CODE_VERSION = "r2-stage-a-0.2"
DEFAULT_CONFIG_PATH = C.REPO_ROOT / "config" / "r2_oi.yaml"

REQUIRED_KEYS = (
    "start_date", "end_date", "universe_size", "min_listing_days",
    "min_feasible_symbols", "exclude_bases", "zscore_window_days",
    "lookback_grid_hours", "rebalance_hour_utc", "n_per_side",
    "gross_notional_usdt", "max_leverage", "diagnostic_margin_consumption_limit",
    "fees", "slippage_bps", "is_fraction", "cost_stress_multiplier",
    "min_oos_rebalances", "hurdle_annual_return", "max_drawdown_pct",
    "max_abs_beta", "n_permutations", "permutation_quantile",
    "permutation_seed", "stage_a",
)
STAGE_A_KEYS = (
    "exclude_symbols", "prestart_min_bar_coverage", "min_metrics_day_coverage",
    "min_median_rows_per_day", "max_oi_null_or_zero_ratio",
    "max_metrics_interval_seconds", "max_kline_missing_ratio",
    "max_funding_missing_ratio", "gap_report_minutes", "download_workers",
    "delisting_zero_volume_days",
)


def validate(cfg: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise C.ConfigError(f"Missing r2_oi keys: {', '.join(missing)}")
    fee = (cfg.get("fees") or {}).get("perp_taker")
    if fee is None:
        raise C.ConfigError("fees.perp_taker is null. Set your actual taker fee "
                            "as a fraction (0.05% = 0.0005) before running.")
    if not isinstance(fee, (int, float)) or isinstance(fee, bool) or not 0 <= fee < 0.01:
        raise C.ConfigError(f"fees.perp_taker={fee!r} is not a plausible fee fraction")
    try:
        start, end = C.period_bounds(cfg)
    except ValueError as exc:
        raise C.ConfigError(f"Invalid start_date/end_date: {exc}") from exc
    if end <= start:
        raise C.ConfigError("end_date must be after start_date")
    grid = cfg["lookback_grid_hours"]
    if (not isinstance(grid, list) or len(grid) < 3
            or any(not isinstance(g, int) or g <= 0 for g in grid)
            or any(b <= a for a, b in zip(grid, grid[1:]))):
        raise C.ConfigError("lookback_grid_hours must be >= 3 increasing positive integers")
    for key in ("universe_size", "min_listing_days", "min_feasible_symbols",
                "zscore_window_days", "n_per_side"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise C.ConfigError(f"{key} must be a positive integer")
    if "default" not in (cfg.get("slippage_bps") or {}):
        raise C.ConfigError("slippage_bps must include 'default'")
    sa = cfg["stage_a"] or {}
    missing = [k for k in STAGE_A_KEYS if k not in sa]
    if missing:
        raise C.ConfigError(f"Missing stage_a keys: {', '.join(missing)}")


def load(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.exists():
        raise C.ConfigError(f"Config file not found: {p}")
    with open(p, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or "r2_oi" not in raw:
        raise C.ConfigError("Config must contain a top-level 'r2_oi' section")
    cfg = raw["r2_oi"]
    validate(cfg)
    return cfg


def required_start_ms(cfg: dict) -> int:
    """R2.4: start_date - (zscore_window_days + ceil(max L / 24) + 1) days."""
    start, _ = C.period_bounds(cfg)
    days = (cfg["zscore_window_days"]
            + math.ceil(max(cfg["lookback_grid_hours"]) / 24) + 1)
    return start - days * C.DAY_MS
