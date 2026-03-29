"""
main.py
=======
Module 11 of 11 — Entry Point.

What this module does:
  Ties all the other modules together and runs the full pipeline:
    1. Load and validate config.yaml
    2. Load market data and calibrate the jump-diffusion model
    3. Pre-calculate DB pension income (deterministic)
    4. Run the Monte Carlo simulation
    5. Analyse results (percentiles, ruin probability)
    6. Generate charts
    7. Save summary CSV
    8. Print a clean console summary

Usage:
  python main.py

Python concepts introduced (recap):
  - Importing from multiple local modules
  - Timing code execution with time.time()
  - Organising a program with a main() function and if __name__ == "__main__"
  - String formatting for a clean console summary
"""

import time
import yaml
from pathlib import Path

from validator      import validate, print_validation_report
from data_loader    import load_data
from pension_db     import build_db_income_schedule
from tax            import TaxBands
from simulation     import run_simulation
from analysis       import analyse, save_summary_csv
from visualisation  import generate_all_charts


# ---------------------------------------------------------------------------
# Console summary printer
# ---------------------------------------------------------------------------

def print_summary(results: dict, config: dict, db_income_schedule: list, elapsed: float) -> None:
    """Print a clean summary of key results to the console."""
    SEP  = "=" * 62
    SEP2 = "-" * 62

    ruin_prob   = results["ruin_probability"]
    median_ruin = results["median_ruin_age"]
    at_ret      = results["values_at_retirement"]
    ret_age     = config["user"]["retirement_age"]
    life_exp    = config["user"]["life_expectancy"]
    n_sims      = config["simulation"]["n_simulations"]

    # Ruin colour indicator for terminal
    if ruin_prob > 10:
        ruin_flag = "⚠  HIGH"
    elif ruin_prob > 5:
        ruin_flag = "⚠  MODERATE"
    else:
        ruin_flag = "✓  LOW"

    print(f"""
{SEP}
  MONTE CARLO RETIREMENT SIMULATION — RESULTS
{SEP}
  Simulations run   : {n_sims:,}
  Simulation time   : {elapsed:.1f} seconds
{SEP2}
  PORTFOLIO AT RETIREMENT (age {ret_age:.0f}, today's money)
    1st   percentile  : £{at_ret[1]:>12,.0f}
    2.5th percentile  : £{at_ret[2.5]:>12,.0f}
    5th   percentile  : £{at_ret[5]:>12,.0f}
    Median (50th)     : £{at_ret[50]:>12,.0f}
    75th  percentile  : £{at_ret[75]:>12,.0f}
{SEP2}""")

    if db_income_schedule:
        print("  DB PENSION INCOME (guaranteed, from normal pension age)")
        for db in db_income_schedule:
            monthly = db["monthly_income_gross"]
            print(f"    {db['pension_id']:<20}: £{monthly:,.0f}/month  "
                  f"(£{monthly*12:,.0f}/yr)  from age {db['normal_pension_age']:.0f}")
        print(SEP2)

    print(f"""  RUIN PROBABILITY (before age {life_exp:.0f})
    Probability of ruin : {ruin_prob:.1f}%  {ruin_flag}""")

    if median_ruin:
        print(f"    Median age of ruin   : {median_ruin:.0f}")
    else:
        print(f"    No simulations ruined before age {life_exp:.0f}")

    print(f"""
  Note: "Ruin" = capital portfolio reaches £0.
  DB pension income continues beyond capital ruin (if applicable).
{SEP}
  Outputs saved to outputs/ folder:
    accumulation_chart.png
    drawdown_chart.png
    summary_stats.csv
{SEP}
""")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full simulation pipeline."""

    start_time = time.time()

    # ── 1. Load config ─────────────────────────────────────────────────────
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            "config.yaml not found. Make sure you run main.py from the project root."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("\n" + "=" * 62)
    print("  MONTE CARLO RETIREMENT PORTFOLIO SIMULATOR")
    print("=" * 62 + "\n")

    # ── 2. Validate config ─────────────────────────────────────────────────
    print("Step 1/7 — Validating config.yaml...")
    errors, warnings = validate(config)
    print_validation_report(errors, warnings)   # Raises SystemExit on errors

    # ── 3. Load market data + calibrate jump-diffusion ─────────────────────
    print("Step 2/7 — Loading market data and calibrating return model...")
    market_params = load_data(config)

    # ── 4. Pre-calculate DB pension income ─────────────────────────────────
    print("Step 3/7 — Calculating DB pension income...")
    db_income_schedule = build_db_income_schedule(config)
    if not db_income_schedule:
        print("  No DB schemes configured — DC/ISA/GIA only.")
    print()

    # ── 5. Set up tax bands ────────────────────────────────────────────────
    bands = TaxBands.from_config(config)

    # ── 6. Run simulation ──────────────────────────────────────────────────
    print("Step 4/7 — Running Monte Carlo simulation...")
    portfolio_values, ruin_flags, ruin_ages, ages, gia_accum, isa_accum, dc_accum = run_simulation(
        config             = config,
        market_params      = market_params,
        db_income_schedule = db_income_schedule,
        bands              = bands,
    )

    # ── 7. Analyse results ─────────────────────────────────────────────────
    print("\nStep 5/7 — Analysing results...")
    results = analyse(portfolio_values, ruin_flags, ruin_ages, ages, config, gia_accum, isa_accum, dc_accum)

    # ── 8. Save summary CSV ────────────────────────────────────────────────
    print("Step 6/7 — Saving summary CSV...")
    save_summary_csv(results, config)

    # ── 9. Generate charts ─────────────────────────────────────────────────
    print("Step 7/7 — Generating charts...")
    generate_all_charts(results, config, db_income_schedule)

    # ── 10. Print summary ──────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print_summary(results, config, db_income_schedule, elapsed)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
