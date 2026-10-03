"""Shared configuration, paths and constants for the Phase 0.5 funding gate.

This module has no network access and no execution capability.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "phase05.yaml"

# 0.5.0: original release (run 9513951a).
# 0.5.1: float-tolerant threshold comparisons (economics.meets_threshold).
CODE_VERSION = "0.5.1"

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
YEAR_MS = int(365.25 * DAY_MS)

REQUIRED_KEYS = (
    "start_date",
    "end_date",
    "symbols",
    "notional_per_symbol_usdt",
    "max_leverage",
    "diagnostic_margin_consumption_limit",
    "fees",
    "slippage_bps",
    "entry_lookback_periods",
    "entry_threshold_grid",
    "exit_threshold",
    "exit_periods",
    "is_fraction",
    "cost_stress_multiplier",
    "min_oos_trades",
    "min_profitable_symbols",
    "hurdle_annual_return",
    "max_drawdown_pct",
)

GO_NO_GO_ACTION = {
    "PASS": "PASS permits Phase 1 (market-data architecture).",
    "FAIL": (
        "STOP. Do not implement production market-data, execution, "
        "or trading infrastructure."
    ),
    "INCONCLUSIVE": (
        "STOP. Do not implement production infrastructure. Resolve the "
        "inconclusive reasons listed in this report first."
    ),
}


class ConfigError(ValueError):
    """Raised when the Phase 0.5 configuration is missing or invalid."""


def date_to_ms(value: Any) -> int:
    """Convert a YYYY-MM-DD date (string or date) to UTC epoch milliseconds."""
    dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def period_bounds(cfg: dict) -> tuple[int, int]:
    """Return [start_ms, end_ms) where end_date is inclusive."""
    return date_to_ms(cfg["start_date"]), date_to_ms(cfg["end_date"]) + DAY_MS


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_config(cfg: dict) -> None:
    """Validate config. Refuses to run with null fees (spec 0.5.12)."""
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ConfigError(f"Missing config keys: {', '.join(missing)}")

    fees = cfg.get("fees") or {}
    for key in ("spot_taker", "perp_taker"):
        value = fees.get(key)
        if value is None:
            raise ConfigError(
                f"fees.{key} is null. Set your actual taker fee as a fraction "
                f"(0.1% = 0.001) in config/phase05.yaml before running."
            )
        if not _is_number(value) or value < 0 or value >= 0.01:
            raise ConfigError(
                f"fees.{key}={value!r} is not a plausible fee fraction "
                f"(expected 0 <= fee < 0.01; 0.1% = 0.001)."
            )

    symbols = cfg["symbols"]
    if not isinstance(symbols, list) or not symbols:
        raise ConfigError("symbols must be a non-empty list")

    try:
        start_ms, end_ms = period_bounds(cfg)
    except ValueError as exc:
        raise ConfigError(f"Invalid start_date/end_date: {exc}") from exc
    if end_ms - start_ms < 365 * DAY_MS:
        raise ConfigError("Backtest period must be at least 12 months.")

    grid = cfg["entry_threshold_grid"]
    if (
        not isinstance(grid, list)
        or len(grid) < 3
        or not all(_is_number(g) for g in grid)
        or any(b <= a for a, b in zip(grid, grid[1:]))
    ):
        raise ConfigError(
            "entry_threshold_grid must be a strictly increasing list of at "
            "least 3 numbers."
        )

    for key in ("entry_lookback_periods", "exit_periods", "min_oos_trades",
                "min_profitable_symbols"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ConfigError(f"{key} must be a positive integer")

    if not 0 < cfg["is_fraction"] < 1:
        raise ConfigError("is_fraction must be between 0 and 1")
    if not cfg["max_leverage"] > 0:
        raise ConfigError("max_leverage must be > 0")
    if not cfg["notional_per_symbol_usdt"] > 0:
        raise ConfigError("notional_per_symbol_usdt must be > 0")
    if not isinstance(cfg["slippage_bps"], dict) or "default" not in cfg["slippage_bps"]:
        raise ConfigError("slippage_bps must be a mapping with a 'default' entry")


def load_config(path: str | Path | None = None) -> dict:
    """Load and validate config/phase05.yaml (or a given path)."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or "phase05" not in raw:
        raise ConfigError("Config must contain a top-level 'phase05' section")
    cfg = raw["phase05"]
    validate_config(cfg)
    return cfg


def slippage_bps_for(cfg: dict, symbol: str) -> float:
    table = cfg["slippage_bps"]
    return float(table.get(symbol, table["default"]))


def resolve_dir(cfg: dict, key: str, default: str) -> Path:
    value = Path(cfg.get(key, default))
    return value if value.is_absolute() else REPO_ROOT / value


def config_hash(cfg: dict) -> str:
    canonical = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ms_to_iso(ms: int | float | None) -> str:
    if ms is None:
        return "n/a"
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
