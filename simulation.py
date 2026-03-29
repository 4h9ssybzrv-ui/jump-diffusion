"""
simulation.py
=============
Module 6 of 11 — Core Monte Carlo Engine.

What this module does:
  Runs n_simulations independent retirement scenarios from current_age to
  life_expectancy. Each simulation path samples market returns from the
  Merton Jump-Diffusion model, applies contributions during accumulation,
  then hands off to drawdown.py during the drawdown phase.

  All random numbers are generated upfront using NumPy (fast), then the
  business logic loops over each simulation path (clear and debuggable).

Python concepts introduced:
  - np.random.default_rng — modern NumPy random number generation
  - Pre-generating large arrays for performance
  - np.where — vectorised conditional (like an if-else, but for arrays)
  - np.clip — cap values within a range
  - Nested loops: simulations × months
  - Progress reporting with print()
"""

import numpy as np
from wrappers import GIAWrapper, ISAWrapper, DCPotWrapper, build_wrappers_from_config
from drawdown import run_monthly_drawdown


# ---------------------------------------------------------------------------
# Random return generation
# ---------------------------------------------------------------------------

def generate_monthly_returns(
    rng:             np.random.Generator,
    n_sims:          int,
    n_months:        int,
    mu_d:            float,
    sigma_d:         float,
    lambda_monthly:  float,
    jump_mean:       float,
    jump_std:        float,
    jump_clip_lower: float,
    jump_clip_upper: float,
) -> np.ndarray:
    """
    Pre-generate all monthly returns for all simulations upfront.

    Why do this upfront rather than inside the loop?
    NumPy can generate millions of random numbers in one call much faster
    than calling it one-by-one inside a Python loop.

    Each month's total return = diffusion component + jump component.

    Returns
    -------
    np.ndarray of shape (n_sims, n_months)
        Total real monthly return for each simulation × month combination.
    """
    # Diffusion component: every month has one
    diffusion = rng.normal(loc=mu_d, scale=sigma_d, size=(n_sims, n_months))

    # Jump indicator: True if a jump occurs in this month
    # rng.random() gives uniform [0,1]; compare to lambda to get Bernoulli
    jump_occurred = rng.random(size=(n_sims, n_months)) < lambda_monthly

    # Jump size: drawn for every month, but only used where jump_occurred is True
    jump_sizes = rng.normal(loc=jump_mean, scale=jump_std, size=(n_sims, n_months))
    jump_sizes = np.clip(jump_sizes, jump_clip_lower, jump_clip_upper)

    # Combine: add jump only where it occurred
    # np.where(condition, value_if_true, value_if_false) — works element-wise
    total_returns = diffusion + np.where(jump_occurred, jump_sizes, 0.0)

    return total_returns


# ---------------------------------------------------------------------------
# Contribution helpers
# ---------------------------------------------------------------------------

def _monthly_contribution_real(
    base_contribution:    float,
    real_growth_rate:     float,
    month_index:          int,
) -> float:
    """
    Calculate the real-terms contribution for a given month.

    Since the simulation works entirely in real (inflation-adjusted) terms,
    we don't need to inflate contributions — CPI adjustment is implicit.
    We only apply any ADDITIONAL real growth on top of inflation.

    Parameters
    ----------
    base_contribution : Monthly contribution in today's money (£)
    real_growth_rate  : Annual real growth rate (e.g. 0.02 = 2% above inflation)
    month_index       : Month number from simulation start (0-indexed)

    Returns
    -------
    float : Contribution amount in real terms for this month (£)
    """
    years_elapsed = month_index / 12.0
    return base_contribution * ((1.0 + real_growth_rate) ** years_elapsed)


def _apply_contributions(
    gia:           GIAWrapper,
    isa:           ISAWrapper,
    dc_list:       list,
    config:        dict,
    month_index:   int,
) -> None:
    """
    Apply monthly contributions to all wrappers, respecting annual limits.

    For DC pots, the £60,000 combined annual allowance is shared across
    all DC schemes — so we pass the combined year-to-date figure to each.
    """
    # GIA contribution
    gia_cfg  = config.get("gia", {})
    gia_base = float(gia_cfg.get("monthly_contribution", 0))
    gia_grow = float(gia_cfg.get("real_contribution_growth", 0))
    if gia_base > 0:
        gia.contribute(_monthly_contribution_real(gia_base, gia_grow, month_index))

    # ISA contribution
    isa_cfg  = config.get("isa", {})
    isa_base = float(isa_cfg.get("monthly_contribution", 0))
    isa_grow = float(isa_cfg.get("real_contribution_growth", 0))
    if isa_base > 0:
        isa.contribute(_monthly_contribution_real(isa_base, isa_grow, month_index))

    # DC pension contributions
    # Must track combined year-to-date across all DC pots for the annual allowance
    combined_dc_ytd = sum(dc.annual_contributions_ytd for dc in dc_list)

    for dc, pension_cfg in zip(dc_list, _get_dc_configs(config)):
        dc_base = float(pension_cfg.get("dc", {}).get("monthly_contribution", 0))
        dc_grow = float(pension_cfg.get("dc", {}).get("real_contribution_growth", 0))
        if dc_base > 0:
            amount = _monthly_contribution_real(dc_base, dc_grow, month_index)
            contributed = dc.contribute(amount, combined_dc_ytd)
            combined_dc_ytd += contributed


def _get_dc_configs(config: dict) -> list:
    """Return pension config entries that have a DC component."""
    return [p for p in config.get("pensions", []) if p.get("type") in ("DC", "hybrid")]


def _apply_lump_sums(
    gia:     GIAWrapper,
    isa:     ISAWrapper,
    dc_list: list,
    config:  dict,
    age:     float,
) -> None:
    """
    Apply any lump sums scheduled for the current age.

    We check within a half-month window to avoid floating-point timing issues.
    """
    lump_sums = config.get("lump_sums") or []
    dc_map    = {p["id"]: dc for p, dc in zip(_get_dc_configs(config), dc_list)}

    for ls in lump_sums:
        ls_age  = float(ls.get("age", -1))
        amount  = float(ls.get("amount", 0))
        wrapper = str(ls.get("wrapper", "")).lower()

        # Apply when we're within half a month of the scheduled age
        if abs(age - ls_age) < (1/24):
            if wrapper == "gia":
                gia.add_lump_sum(amount)
            elif wrapper == "isa":
                isa.add_lump_sum(amount)
            elif wrapper in dc_map:
                combined_ytd = sum(dc.annual_contributions_ytd for dc in dc_list)
                dc_map[wrapper].add_lump_sum(amount, combined_ytd)


def _total_portfolio(gia: GIAWrapper, isa: ISAWrapper, dc_list: list) -> float:
    """Sum of all wrapper balances — the total portfolio value."""
    return gia.balance + isa.balance + sum(dc.balance for dc in dc_list)


def _reset_annual_limits(gia: GIAWrapper, isa: ISAWrapper, dc_list: list) -> None:
    """Reset year-to-date contribution trackers at the start of each new year."""
    gia.reset_annual_contributions()
    isa.reset_annual_contributions()
    for dc in dc_list:
        dc.reset_annual_contributions()


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def run_simulation(
    config:              dict,
    market_params:       dict,
    db_income_schedule:  list,
    bands,                         # TaxBands instance from tax.py
) -> tuple:
    """
    Run the full Monte Carlo simulation.

    Accumulation phase: current_age → retirement_age
      - Apply jump-diffusion returns each month
      - Add contributions (with real growth and annual limits)
      - Apply any lump sums at the right age

    Drawdown phase: retirement_age → life_expectancy
      - Apply jump-diffusion returns
      - Draw from wrappers via drawdown.py
      - Detect ruin

    Parameters
    ----------
    config             : Full config dictionary
    market_params      : Output from data_loader.load_data()
    db_income_schedule : Output from pension_db.build_db_income_schedule()
    bands              : TaxBands instance

    Returns
    -------
    portfolio_values : np.ndarray (n_sims, n_months) — total portfolio each month
    ruin_flags       : np.ndarray bool (n_sims,)     — True if simulation ruined
    ruin_ages        : np.ndarray float (n_sims,)    — age at ruin, NaN if no ruin
    ages             : np.ndarray float (n_months,)  — age at each month index
    """
    # ── Configuration ──────────────────────────────────────────────────────
    sim_cfg        = config["simulation"]
    user_cfg       = config["user"]
    n_sims         = int(sim_cfg["n_simulations"])
    seed           = sim_cfg.get("random_seed")          # None = truly random
    current_age    = float(user_cfg["current_age"])
    retirement_age = float(user_cfg["retirement_age"])
    life_exp       = float(user_cfg["life_expectancy"])

    accum_months = int(round((retirement_age - current_age) * 12))
    total_months = int(round((life_exp       - current_age) * 12))

    # Age at each monthly time step
    ages = current_age + np.arange(total_months) / 12.0

    # ── Random number generation ───────────────────────────────────────────
    print(f"Generating random returns for {n_sims:,} simulations × "
          f"{total_months} months...")

    rng = np.random.default_rng(seed)
    all_returns = generate_monthly_returns(
        rng             = rng,
        n_sims          = n_sims,
        n_months        = total_months,
        mu_d            = market_params["mu_d"],
        sigma_d         = market_params["sigma_d"],
        lambda_monthly  = market_params["lambda_monthly"],
        jump_mean       = market_params["jump_mean"],
        jump_std        = market_params["jump_std"],
        jump_clip_lower = market_params["jump_clip_lower"],
        jump_clip_upper = market_params["jump_clip_upper"],
    )
    print(f"  Generated {all_returns.size:,} return samples. "
          f"Starting simulation loop...\n")

    # ── Results storage ────────────────────────────────────────────────────
    portfolio_values = np.zeros((n_sims, total_months), dtype=np.float64)
    ruin_flags       = np.zeros(n_sims, dtype=bool)
    ruin_ages        = np.full(n_sims, np.nan, dtype=np.float64)

    # Per-wrapper balance tracking during accumulation (for stacked chart)
    gia_values_accum = np.zeros((n_sims, accum_months), dtype=np.float64)
    isa_values_accum = np.zeros((n_sims, accum_months), dtype=np.float64)
    dc_values_accum  = np.zeros((n_sims, accum_months), dtype=np.float64)

    # ── Simulation loop ────────────────────────────────────────────────────
    for sim_idx in range(n_sims):

        # Progress update every 1,000 simulations
        if sim_idx % 1000 == 0:
            pct = sim_idx / n_sims * 100
            print(f"  Simulation {sim_idx+1:>6,} / {n_sims:,}  ({pct:.0f}%)")

        returns_this_sim = all_returns[sim_idx]

        # Initialise wrappers fresh for each simulation path
        gia, isa, dc_list = build_wrappers_from_config(config)
        cumulative_tax_free = 0.0  # Tracks LSA usage for this path

        # ── Accumulation phase ─────────────────────────────────────────────
        for m in range(accum_months):
            age = ages[m]

            # Apply this month's return to all market-exposed wrappers
            r = returns_this_sim[m]
            gia.apply_return(r)
            isa.apply_return(r)
            for dc in dc_list:
                dc.apply_return(r)

            # Reset annual contribution limits at the start of each year
            # m % 12 == 0 catches month 0 (first month) — skip it, nothing to reset yet
            if m > 0 and m % 12 == 0:
                _reset_annual_limits(gia, isa, dc_list)

            # Add monthly contributions
            _apply_contributions(gia, isa, dc_list, config, m)

            # Apply any scheduled lump sums
            _apply_lump_sums(gia, isa, dc_list, config, age)

            # Record total portfolio value and per-wrapper breakdown for this month
            portfolio_values[sim_idx, m] = _total_portfolio(gia, isa, dc_list)
            gia_values_accum[sim_idx, m] = gia.balance
            isa_values_accum[sim_idx, m] = isa.balance
            dc_values_accum[sim_idx, m]  = sum(dc.balance for dc in dc_list)

        # ── Drawdown phase ─────────────────────────────────────────────────
        for m in range(accum_months, total_months):
            age = ages[m]

            # Apply return first (growth before withdrawal — standard convention)
            r = returns_this_sim[m]
            gia.apply_return(r)
            isa.apply_return(r)
            for dc in dc_list:
                dc.apply_return(r)

            # Apply any lump sums during drawdown (e.g. late inheritance)
            _apply_lump_sums(gia, isa, dc_list, config, age)

            # Run monthly drawdown
            ruined, cumulative_tax_free = run_monthly_drawdown(
                gia                    = gia,
                isa                    = isa,
                dc_list                = dc_list,
                db_income_schedule     = db_income_schedule,
                config                 = config,
                current_age            = age,
                bands                  = bands,
                cumulative_tax_free    = cumulative_tax_free,
            )

            if ruined:
                ruin_flags[sim_idx] = True
                ruin_ages[sim_idx]  = age
                # All subsequent months stay at zero (portfolio is gone)
                portfolio_values[sim_idx, m:] = 0.0
                break

            portfolio_values[sim_idx, m] = _total_portfolio(gia, isa, dc_list)

    print(f"\n  Simulation complete. {n_sims:,} paths run.")
    return portfolio_values, ruin_flags, ruin_ages, ages, gia_values_accum, isa_values_accum, dc_values_accum
