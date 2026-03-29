"""
analysis.py
===========
Module 8 of 11 — Results Analysis.

What this module does:
  Takes the raw simulation output (portfolio values across all paths and months)
  and computes the summary statistics needed for the charts and console output:
    - Percentile bands at every monthly time step
    - Probability of ruin before life_expectancy
    - Median age of ruin (for paths that did ruin)
    - Key milestone values (portfolio at retirement, etc.)

Python concepts introduced:
  - np.percentile with axis=0 (compute across rows, one result per column)
  - np.isnan / np.nanmedian for arrays with NaN values
  - Dictionary comprehensions { key: value for ... }
  - pandas DataFrame for structured tabular output and CSV saving
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# Percentile levels — focused on downside risk (1st, 2.5th, 5th) plus median and upside (75th)
# Also include 25th percentile for interquartile range analysis
PERCENTILES = [1, 2.5, 5, 25, 50, 75]


def compute_percentile_bands(
    portfolio_values: np.ndarray,
) -> dict:
    """
    Compute percentile bands at each monthly time step.

    portfolio_values has shape (n_sims, n_months).
    We compute along axis=0 — across all simulations for each month.

    Returns a dictionary mapping each percentile level to an array of
    portfolio values, one per month.

    Example: result[50] is the median portfolio value at each month.
    """
    # Dictionary comprehension: { percentile: array_of_values_over_time }
    return {
        pct: np.percentile(portfolio_values, pct, axis=0)
        for pct in PERCENTILES
    }


def compute_ruin_probability(ruin_flags: np.ndarray) -> float:
    """
    Calculate the probability of ruin as a percentage.

    ruin_flags is a boolean array — True where the simulation ruined.
    .mean() on a boolean array gives the proportion of True values.
    """
    return float(ruin_flags.mean() * 100.0)


def compute_median_ruin_age(ruin_ages: np.ndarray) -> Optional[float]:
    """
    Calculate the median age at which ruin occurs, for ruined paths only.

    ruin_ages contains NaN for simulations that did not ruin.
    np.nanmedian ignores NaN values when computing the median.

    Returns None if no simulations ruined.
    """
    ruined = ruin_ages[~np.isnan(ruin_ages)]
    if len(ruined) == 0:
        return None
    return float(np.nanmedian(ruined))


def split_by_phase(
    portfolio_values: np.ndarray,
    ages:             np.ndarray,
    retirement_age:   float,
) -> tuple:
    """
    Split the full results array into accumulation and drawdown phases.

    Returns four arrays:
      accum_values : (n_sims, accum_months)
      accum_ages   : (accum_months,)
      draw_values  : (n_sims, draw_months)
      draw_ages    : (draw_months,)
    """
    # Find the index of the first month in the drawdown phase
    split_idx = int(np.searchsorted(ages, retirement_age))

    accum_values = portfolio_values[:, :split_idx]
    accum_ages   = ages[:split_idx]
    draw_values  = portfolio_values[:, split_idx:]
    draw_ages    = ages[split_idx:]

    return accum_values, accum_ages, draw_values, draw_ages


def compute_milestone_values(
    percentile_bands: dict,
    ages:             np.ndarray,
    retirement_age:   float,
) -> dict:
    """
    Extract portfolio values at key ages (retirement and end of projection).

    Returns a dictionary of { percentile: value_at_retirement }.
    """
    ret_idx = int(np.searchsorted(ages, retirement_age))
    ret_idx = min(ret_idx, len(ages) - 1)

    return {
        pct: float(percentile_bands[pct][ret_idx])
        for pct in PERCENTILES
    }


def analyse(
    portfolio_values: np.ndarray,
    ruin_flags:       np.ndarray,
    ruin_ages:        np.ndarray,
    ages:             np.ndarray,
    config:           dict,
    gia_accum:        np.ndarray = None,
    isa_accum:        np.ndarray = None,
    dc_accum:         np.ndarray = None,
) -> dict:
    """
    Run all analysis calculations and return a results dictionary.

    This is the main entry point called by main.py.

    Returns
    -------
    dict with keys:
      percentile_bands         — { pct: array } for all months
      accum_bands, draw_bands  — same, split by phase
      accum_ages, draw_ages    — age arrays for each phase
      ruin_probability         — float (%)
      median_ruin_age          — float or None
      values_at_retirement     — { pct: £ value }
    """
    retirement_age = float(config["user"]["retirement_age"])

    # Full percentile bands (all months)
    percentile_bands = compute_percentile_bands(portfolio_values)

    # Split into phases
    accum_values, accum_ages, draw_values, draw_ages = split_by_phase(
        portfolio_values, ages, retirement_age
    )
    accum_bands = compute_percentile_bands(accum_values)
    draw_bands  = compute_percentile_bands(draw_values)

    # Ruin statistics
    ruin_prob      = compute_ruin_probability(ruin_flags)
    median_ruin    = compute_median_ruin_age(ruin_ages)

    # Milestone values
    values_at_ret  = compute_milestone_values(percentile_bands, ages, retirement_age)

    results = {
        "percentile_bands":       percentile_bands,
        "accum_bands":            accum_bands,
        "draw_bands":             draw_bands,
        "accum_ages":             accum_ages,
        "draw_ages":              draw_ages,
        "ruin_probability":       ruin_prob,
        "median_ruin_age":        median_ruin,
        "values_at_retirement":   values_at_ret,
        "all_ages":               ages,
    }

    # Per-wrapper median series for the stacked accumulation chart
    if gia_accum is not None and isa_accum is not None and dc_accum is not None:
        results["accum_gia_median"] = np.percentile(gia_accum, 50, axis=0)
        results["accum_isa_median"] = np.percentile(isa_accum, 50, axis=0)
        results["accum_dc_median"]  = np.percentile(dc_accum,  50, axis=0)

    return results


def save_summary_csv(results: dict, config: dict, output_path: str = "outputs/summary_stats.csv") -> None:
    """
    Save a summary of percentile values at each month to a CSV file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    ages  = results["all_ages"]
    bands = results["percentile_bands"]

    rows = []
    for i, age in enumerate(ages):
        row = {"age": round(float(age), 2)}
        for pct in PERCENTILES:
            row[f"p{pct}"] = round(float(bands[pct][i]), 2)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Summary CSV saved to: {output_path}")
