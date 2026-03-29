"""
data_loader.py
==============
Module 1 of 11 — run this first to verify your data files are loading correctly.

What this module does:
  1. Loads VWRP daily price data → converts to monthly nominal returns
  2. Loads ONS CPIH inflation data → converts to monthly inflation rates
  3. Aligns both series on their overlapping date range
  4. Computes real (inflation-adjusted) returns: (1 + nominal) / (1 + cpih) - 1
  5. Runs the Merton Jump-Diffusion decomposition:
       a. Iterative sigma-clipping to identify crash ("jump") months
       b. Computes clean diffusion parameters (mu_d, sigma_d) from calm months only
  6. Prints a calibration report so you can see exactly what the model is using

Python concepts introduced here:
  - Functions with type hints (def foo(x: float) -> dict:)
  - pd.read_csv, pd.to_datetime, .resample(), .pct_change()
  - Reading files line by line with open()
  - Sets (for tracking flagged months efficiently)
  - f-strings for formatted output
  - Returning a dictionary from a function
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maps the 3-letter month abbreviations in the CPIH file to 2-digit month numbers
MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}


# ---------------------------------------------------------------------------
# Step 1: Load VWRP daily prices → monthly returns
# ---------------------------------------------------------------------------

def load_vwrp_monthly_returns(filepath: str) -> pd.Series:
    """
    Load VWRP daily price data and return monthly nominal returns.

    The CSV has one row per trading day with columns:
      Date, Open, High, Low, Close, Volume

    Date format: "Friday, March 27, 2026"  (day name, month name, day, year)

    We only need the Close price. We resample to month-end (last trading day
    of each month) and compute the percentage change month-over-month.

    Returns
    -------
    pd.Series
        Monthly nominal returns as decimals (e.g. 0.034 means +3.4%).
        Indexed by month-end dates.
    """
    # thousands="," strips commas from the Volume column (e.g. "342,287" → 342287)
    df = pd.read_csv(filepath, thousands=",")

    # Parse the date column — note: format must exactly match the file
    # "%A" = full day name, "%B" = full month name, "%d" = day, "%Y" = 4-digit year
    df["Date"] = pd.to_datetime(df["Date"], format="%A, %B %d, %Y")

    # The file is stored newest-first, so we sort ascending
    df = df.sort_values("Date").set_index("Date")

    # Resample to month-end: take the LAST available closing price in each month
    # "ME" stands for Month End frequency
    monthly_close = df["Close"].resample("ME").last()

    # pct_change() computes (price_t / price_{t-1}) - 1 for each row
    # dropna() removes the first row (which has no previous value to compare to)
    monthly_returns = monthly_close.pct_change().dropna()

    return monthly_returns


# ---------------------------------------------------------------------------
# Step 2: Load ONS CPIH data → monthly inflation rates
# ---------------------------------------------------------------------------

def load_cpih_monthly_rates(filepath: str) -> pd.Series:
    """
    Load ONS CPIH data and return monthly inflation rates as decimals.

    The file has a tricky structure:
      - Lines 1-8: metadata headers (Title, CDID, Source, etc.) — must be skipped
      - Then three sections of data, all mixed together:
          Annual    rows: "1989","5.7"
          Quarterly rows: "1989 Q1","5.8"
          Monthly   rows: "1989 JAN","5.7"  ← we want ONLY these

    We identify monthly rows by checking that the date field has exactly
    2 tokens and the second token is a 3-letter month abbreviation.

    CPIH values are annual rates in PERCENT (e.g. 5.7 = 5.7% per year).
    We convert them to monthly decimal rates: (1 + annual_rate)^(1/12) - 1

    Returns
    -------
    pd.Series
        Monthly inflation rates as decimals. Indexed by month-end dates.
    """
    rows = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Each data line looks like: "1989 JAN","5.7"
            # We split on the separator between the two quoted values
            parts = line.split('","')
            if len(parts) != 2:
                continue  # Skip metadata lines and any malformed rows

            date_str  = parts[0].strip('"').strip()
            value_str = parts[1].strip('"').strip()

            # Check if this is a monthly row:
            # Must have exactly 2 tokens (e.g. ["1989", "JAN"])
            # and the second token must be a month abbreviation
            tokens = date_str.split()
            if len(tokens) != 2 or tokens[1] not in MONTH_MAP:
                continue

            year      = tokens[0]
            month_num = MONTH_MAP[tokens[1]]

            try:
                # Build a date string like "1989-01-01" and shift to month-end
                # MonthEnd(0) moves a date to the last day of its month
                date = (
                    pd.to_datetime(f"{year}-{month_num}-01")
                    + pd.offsets.MonthEnd(0)
                )

                # Convert from percent to decimal (5.7 → 0.057)
                annual_rate = float(value_str) / 100.0

                # Convert annual rate to monthly rate
                # Maths: if prices rise R% per year, they rise by (1+R)^(1/12)-1 per month
                monthly_rate = (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0

                rows.append({"date": date, "monthly_cpih": monthly_rate})

            except (ValueError, TypeError):
                continue  # Skip any rows with non-numeric values

    # Build a Series from the collected rows
    cpih_series = (
        pd.DataFrame(rows)
        .set_index("date")
        ["monthly_cpih"]
        .sort_index()
    )

    return cpih_series


# ---------------------------------------------------------------------------
# Step 3: Compute real (inflation-adjusted) returns
# ---------------------------------------------------------------------------

def compute_real_returns(
    nominal_returns: pd.Series,
    monthly_cpih: pd.Series,
) -> pd.Series:
    """
    Convert nominal monthly returns to real (inflation-adjusted) returns.

    Formula: real_return = (1 + nominal) / (1 + cpih) - 1

    We align the two series on their overlapping dates. If CPIH data is
    missing for any month (e.g. published with a lag), we forward-fill
    using the most recent available CPIH value.

    Returns
    -------
    pd.Series
        Real monthly returns as decimals. Indexed by month-end dates.
    """
    # Combine into one DataFrame — pandas automatically aligns on dates
    combined = pd.DataFrame({
        "nominal": nominal_returns,
        "cpih":    monthly_cpih,
    })

    # Forward-fill any missing CPIH values (handles publication lags)
    combined["cpih"] = combined["cpih"].ffill()

    # Drop rows where either series has no data at all
    combined = combined.dropna()

    # Apply the real return formula element-wise
    combined["real"] = (1.0 + combined["nominal"]) / (1.0 + combined["cpih"]) - 1.0

    return combined["real"]


# ---------------------------------------------------------------------------
# Step 4: Identify jump (crash) months via iterative sigma-clipping
# ---------------------------------------------------------------------------

def identify_jump_months(
    real_returns: pd.Series,
    k: float = 2.5,
    max_iterations: int = 10,
) -> tuple:
    """
    Separate the return series into "diffusion" months and "jump" months.

    This implements the iterative sigma-clipping algorithm described in CLAUDE.md.
    Only DOWNWARD jumps are flagged — we model crash risk, not positive surprises.

    Parameters
    ----------
    real_returns : pd.Series
        The full real return series.
    k : float
        Threshold multiplier. A month is flagged as a jump if its return is
        more than k standard deviations below the mean. Default: 2.5.
    max_iterations : int
        Safety cap to prevent infinite loops. In practice converges in 2-3 steps.

    Returns
    -------
    diffusion_returns : pd.Series  — the calm months (used to estimate mu_d, sigma_d)
    jump_returns      : pd.Series  — the crash months
    """
    # We track flagged dates as a Python set — sets are efficient for membership checks
    # and make it easy to detect when the set stops changing (convergence)
    flagged_dates = set()

    for iteration in range(max_iterations):

        # Compute mu and sigma from only the non-flagged months
        non_flagged = real_returns[~real_returns.index.isin(flagged_dates)]
        mu    = non_flagged.mean()
        sigma = non_flagged.std()

        # A month is a jump if it falls more than k*sigma below the mean
        threshold = mu - k * sigma

        # Check the FULL original series against the new threshold
        new_flagged = set(real_returns[real_returns < threshold].index)

        # Convergence check: stop when the flagged set no longer changes
        if new_flagged == flagged_dates:
            break

        flagged_dates = new_flagged

    # Split the original series into two groups
    is_jump       = real_returns.index.isin(flagged_dates)
    diffusion_returns = real_returns[~is_jump]
    jump_returns      = real_returns[is_jump]

    return diffusion_returns, jump_returns


# ---------------------------------------------------------------------------
# Step 5: Print the calibration report
# ---------------------------------------------------------------------------

def print_calibration_report(
    date_range:        str,
    n_total:           int,
    real_returns:      pd.Series,
    diffusion_returns: pd.Series,
    jump_returns:      pd.Series,
    k:                 float,
    lambda_annual:     float,
    jump_mean:         float,
    jump_std:          float,
    jump_clip_lower:   float,
    jump_clip_upper:   float,
) -> None:
    """
    Print a formatted calibration report to the console.

    This is important for transparency — it shows exactly what parameters
    the simulation will use, and flags the short sample size caveat.
    """
    mu_d    = diffusion_returns.mean()
    sigma_d = diffusion_returns.std()
    mu_raw  = real_returns.mean()
    sig_raw = real_returns.std()

    empirical_rate = (len(jump_returns) / n_total) * 12   # annualised
    lambda_monthly = lambda_annual / 12

    # Format the list of identified jump months
    if len(jump_returns) == 0:
        jump_lines = ["   (none identified)"]
    else:
        jump_lines = [
            f"   → {d.strftime('%b %Y')}: {r*100:+.1f}% real return"
            for d, r in jump_returns.items()
        ]

    SEP  = "=" * 66
    SEP2 = "-" * 66

    print(f"""
{SEP}
  JUMP-DIFFUSION MODEL — CALIBRATION REPORT
{SEP}
  DATA
    Overlapping period : {date_range}
    Total months used  : {n_total}

  JUMP IDENTIFICATION  (k = {k}σ threshold, downward only)
    Jump months found  : {len(jump_returns)}""")

    for line in jump_lines:
        print(line)

    print(f"""    Diffusion months   : {len(diffusion_returns)}

{SEP2}
  DIFFUSION COMPONENT  (calibrated from {len(diffusion_returns)} calm months)
    μ_d  monthly  : {mu_d:+.4f}   →  annualised: {mu_d * 12 * 100:.2f}%
    σ_d  monthly  :  {sigma_d:.4f}   →  annualised: {sigma_d * (12**0.5) * 100:.2f}%

  For comparison — raw series (all {n_total} months including jumps):
    μ    monthly  : {mu_raw:+.4f}   →  annualised: {mu_raw * 12 * 100:.2f}%
    σ    monthly  :  {sig_raw:.4f}   →  annualised: {sig_raw * (12**0.5) * 100:.2f}%

{SEP2}
  JUMP COMPONENT
    Empirical rate : {empirical_rate:.1%}/yr  ← based on {len(jump_returns)} event(s) in {n_total} months
    {"⚠  Too few observations — using long-run anchor from config instead." if len(jump_returns) < 5 else "Using empirical rate."}
    λ_annual (cfg) : {lambda_annual:.1%}/yr   →  λ_monthly: {lambda_monthly:.5f}
    μ_J            : {jump_mean:.1%}    (mean single-month crash size)
    σ_J            :  {jump_std:.1%}    (spread of crash sizes)
    Clip range     : [{jump_clip_lower:.0%}, {jump_clip_upper:.0%}]

{SEP}
  ⚠  SHORT SAMPLE WARNING
     Only {n_total} months of VWRP history ({date_range}).
     This period includes the COVID crash (2020) and post-COVID
     inflation spike (2021-23). Estimates may not reflect long-run
     equity returns. The jump-diffusion separation helps mitigate
     this, but treat results as illustrative, not precise forecasts.
{SEP}
""")


# ---------------------------------------------------------------------------
# Main entry point — called by main.py
# ---------------------------------------------------------------------------

def load_data(config: dict) -> dict:
    """
    Main entry point for data loading. Called by main.py at startup.

    Loads both CSV files, computes real returns, runs the jump-diffusion
    decomposition, prints the calibration report, and returns a dictionary
    containing all parameters needed by the simulation.

    Parameters
    ----------
    config : dict
        The full configuration dictionary loaded from config.yaml.

    Returns
    -------
    dict with keys:
        mu_d, sigma_d         — diffusion parameters (use these in simulation)
        mu_raw, sigma_raw     — raw parameters (logged for comparison only)
        lambda_monthly        — monthly jump probability
        jump_mean, jump_std   — jump magnitude parameters
        jump_clip_lower/upper — jump bounds
        jump_months           — list of identified jump month strings
        n_diffusion, n_total  — observation counts
        date_range            — human-readable date range string
        monthly_cpih          — full CPIH series (for inflation-adjusting contributions)
    """
    prices_file = config["data"]["prices_file"]
    cpih_file   = config["data"]["cpih_file"]

    # Load and process both files
    print(f"Loading price data from : {prices_file}")
    nominal_returns = load_vwrp_monthly_returns(prices_file)

    print(f"Loading CPIH data from  : {cpih_file}")
    monthly_cpih = load_cpih_monthly_rates(cpih_file)

    # Align and compute real returns
    real_returns = compute_real_returns(nominal_returns, monthly_cpih)

    n_total    = len(real_returns)
    date_range = (
        f"{real_returns.index.min().strftime('%b %Y')} – "
        f"{real_returns.index.max().strftime('%b %Y')}"
    )

    # Jump-diffusion decomposition
    jd_cfg = config["jump_diffusion"]
    k      = jd_cfg["jump_threshold_sigma"]

    diffusion_returns, jump_returns = identify_jump_months(real_returns, k=k)

    # Jump parameters come from config (not estimated from the short sample)
    lambda_annual   = jd_cfg["lambda_annual"]
    jump_mean       = jd_cfg["jump_mean"]
    jump_std        = jd_cfg["jump_std"]
    jump_clip_lower = jd_cfg["jump_clip_lower"]
    jump_clip_upper = jd_cfg["jump_clip_upper"]

    # Print the full calibration report
    print_calibration_report(
        date_range        = date_range,
        n_total           = n_total,
        real_returns      = real_returns,
        diffusion_returns = diffusion_returns,
        jump_returns      = jump_returns,
        k                 = k,
        lambda_annual     = lambda_annual,
        jump_mean         = jump_mean,
        jump_std          = jump_std,
        jump_clip_lower   = jump_clip_lower,
        jump_clip_upper   = jump_clip_upper,
    )

    return {
        "mu_d":             diffusion_returns.mean(),
        "sigma_d":          diffusion_returns.std(),
        "mu_raw":           real_returns.mean(),
        "sigma_raw":        real_returns.std(),
        "lambda_monthly":   lambda_annual / 12,
        "jump_mean":        jump_mean,
        "jump_std":         jump_std,
        "jump_clip_lower":  jump_clip_lower,
        "jump_clip_upper":  jump_clip_upper,
        "jump_months":      [d.strftime("%Y-%m") for d in jump_returns.index],
        "n_diffusion":      len(diffusion_returns),
        "n_total":          n_total,
        "date_range":       date_range,
        "monthly_cpih":     monthly_cpih,
    }


# ---------------------------------------------------------------------------
# Quick test — run this file directly to verify data loading works
# Usage: python data_loader.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    print("Running data_loader.py as a standalone test...")
    print("This will load your data files and print the calibration report.")
    print()

    # Load config
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("ERROR: config.yaml not found. Make sure you run this from the project root.")
        exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Run the data loader
    result = load_data(config)

    print("Data loading complete. Parameters returned:")
    print(f"  mu_d           : {result['mu_d']:.6f}")
    print(f"  sigma_d        : {result['sigma_d']:.6f}")
    print(f"  lambda_monthly : {result['lambda_monthly']:.6f}")
    print(f"  jump_months    : {result['jump_months']}")
    print(f"  n_total        : {result['n_total']}")
    print()
    print("✓ If you see the calibration report above and these values,")
    print("  your data files are loading correctly. Proceed to validator.py.")
