"""R2.5 point-in-time features and signals. No funding data is used here.

Implementation conventions (fixed before any Stage B result existed):
- An observation older than OBS_MAX_STALENESS_MS at its query time is
  treated as missing (data gap), never forward-filled further.
- z-scores use the prior W rebalance values (sample std, ddof=1); the
  current value is never part of its own mean or std.
- Test and control use the same eligibility: both z_ret and z_oi valid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.phase05_funding_gate.common import HOUR_MS

METRICS_GUARD_MS = 5 * 60_000          # R2.5: create_time <= T - 5m
OBS_MAX_STALENESS_MS = HOUR_MS

QUADRANTS = ("Crowding-long proxy", "Crowding-short proxy", "Flush proxy",
             "Covering proxy", "Neutral")


def value_at(obs_times: np.ndarray, values: np.ndarray, query: np.ndarray,
             max_age_ms: int = OBS_MAX_STALENESS_MS) -> np.ndarray:
    """Last value with obs_time <= query, NaN if none or older than max_age."""
    out = np.full(len(query), np.nan)
    if len(obs_times) == 0:
        return out
    idx = np.searchsorted(obs_times, query, side="right") - 1
    ok = idx >= 0
    j = idx[ok]
    fresh = (query[ok] - obs_times[j]) <= max_age_ms
    out[ok] = np.where(fresh, values[j], np.nan)
    return out


def oi_at(m_time: np.ndarray, m_oiv: np.ndarray, t: np.ndarray) -> np.ndarray:
    return value_at(m_time, m_oiv, t - METRICS_GUARD_MS)


def close_at(bar_open: np.ndarray, close: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Close of the last 1h bar ending at or before t."""
    return value_at(bar_open + HOUR_MS, close, t)


def raw_features(m_time, m_oiv, bar_open, close, times: np.ndarray,
                 lookback_hours: int) -> tuple[np.ndarray, np.ndarray]:
    L = lookback_hours * HOUR_MS
    oi_t, oi_l = oi_at(m_time, m_oiv, times), oi_at(m_time, m_oiv, times - L)
    c_t, c_l = close_at(bar_open, close, times), close_at(bar_open, close, times - L)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_oi = np.where((oi_t > 0) & (oi_l > 0), np.log(oi_t / oi_l), np.nan)
        ret = np.where((c_t > 0) & (c_l > 0), np.log(c_t / c_l), np.nan)
    return d_oi, ret


def zscore_prior(x: np.ndarray, window: int) -> np.ndarray:
    """z of x[k] against x[k-window .. k-1]; NaN unless all prior values exist."""
    s = pd.Series(x, dtype=float)
    prior = s.shift(1).rolling(window, min_periods=window)
    mean, std = prior.mean(), prior.std(ddof=1)
    z = (s - mean) / std
    z[(std <= 0) | std.isna() | s.isna()] = np.nan
    return z.to_numpy()


def signals(d_oi: np.ndarray, ret: np.ndarray, window: int) -> dict[str, np.ndarray]:
    z_oi, z_ret = zscore_prior(d_oi, window), zscore_prior(ret, window)
    valid = ~np.isnan(z_oi) & ~np.isnan(z_ret)
    s = np.where(valid, -z_ret * np.abs(z_oi), np.nan)
    c = np.where(valid, -z_ret, np.nan)
    return {"S": s, "C": c, "z_oi": z_oi, "z_ret": z_ret}


def quadrant_codes(d_oi: np.ndarray, ret: np.ndarray) -> np.ndarray:
    """Index into QUADRANTS per R2.1 (positioning-flow proxies)."""
    q = np.full(len(d_oi), 4, dtype=int)
    q[(d_oi > 0) & (ret > 0)] = 0
    q[(d_oi > 0) & (ret < 0)] = 1
    q[(d_oi < 0) & (ret < 0)] = 2
    q[(d_oi < 0) & (ret > 0)] = 3
    return q
