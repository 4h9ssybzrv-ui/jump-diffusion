"""
tax.py
======
Module 3 of 11 — UK Income Tax Engine.

What this module does:
  Calculates UK income tax on monthly income using the correct 2024/25 bands.
  Handles the DC pension UFPLS rule: each pension withdrawal is 25% tax-free
  and 75% taxable, subject to the lifetime Lump Sum Allowance (LSA = £268,275).

  The key challenge: DC tax is calculated at the MARGINAL rate on top of any
  DB income already received in the same month. DB income uses up the personal
  allowance and lower bands first, so DC income may be taxed at a higher rate.
  We solve for the required gross withdrawal using binary search (bisection).

  This module is self-contained and can be tested independently.

Python concepts introduced:
  - Dataclasses (@dataclass) — a clean way to group related data
  - The bisection / binary search algorithm
  - Docstrings with parameter and return type documentation
  - Using min() and max() for clamping values
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# TaxBands dataclass — holds all UK tax thresholds
# ---------------------------------------------------------------------------

@dataclass
class TaxBands:
    """
    Holds UK income tax thresholds and rates for one tax year.

    All income thresholds are stored in annual (£/year) terms.
    The calculate_monthly_tax() method converts them to monthly internally.

    The defaults here match 2024/25 England & Wales.
    Load from config.yaml so they can be updated each April without code changes.
    """
    personal_allowance: float = 12570.0   # 0% tax on income below this (annual)
    basic_rate_limit:   float = 50270.0   # 20% on income between PA and here (annual)
    higher_rate_limit:  float = 125140.0  # 40% on income between BRL and here (annual)
    basic_rate:         float = 0.20
    higher_rate:        float = 0.40
    additional_rate:    float = 0.45
    cgt_rate:           float = 0.20      # CGT on GIA gains (simplified flat rate)
    lump_sum_allowance: float = 268275.0  # Lifetime cap on tax-free pension cash

    @classmethod
    def from_config(cls, config: dict) -> "TaxBands":
        """
        Create a TaxBands instance from the config dictionary.
        This is a 'class method' — it creates a new instance from data,
        rather than operating on an existing instance.
        """
        tb = config.get("tax_bands", {})
        return cls(
            personal_allowance = tb.get("personal_allowance", 12570.0),
            basic_rate_limit   = tb.get("basic_rate_limit",   50270.0),
            higher_rate_limit  = tb.get("higher_rate_limit",  125140.0),
            basic_rate         = tb.get("basic_rate",         0.20),
            higher_rate        = tb.get("higher_rate",        0.40),
            additional_rate    = tb.get("additional_rate",    0.45),
            cgt_rate           = tb.get("cgt_rate",           0.20),
            lump_sum_allowance = tb.get("lump_sum_allowance", 268275.0),
        )


# ---------------------------------------------------------------------------
# Core tax calculation
# ---------------------------------------------------------------------------

def calculate_monthly_income_tax(
    monthly_taxable_income: float,
    bands: TaxBands,
) -> float:
    """
    Calculate UK income tax on a given monthly taxable income.

    We convert the annual band thresholds to monthly equivalents
    and apply them in order: personal allowance → basic → higher → additional.

    Parameters
    ----------
    monthly_taxable_income : float
        The total taxable income for this month (£).
        For example: DB gross income + DC taxable portion.
    bands : TaxBands
        Tax band thresholds and rates.

    Returns
    -------
    float
        Monthly income tax owed (£).

    Example
    -------
    If monthly_taxable_income = £3,000:
      Personal allowance: £1,047.50 @ 0%  → £0.00 tax
      Basic rate band:    £1,952.50 @ 20% → £390.50 tax
      Total tax: £390.50
    """
    # Convert annual thresholds to monthly by dividing by 12
    pa  = bands.personal_allowance / 12   # Monthly personal allowance
    brl = bands.basic_rate_limit   / 12   # Monthly basic rate limit
    hrl = bands.higher_rate_limit  / 12   # Monthly higher rate limit

    if monthly_taxable_income <= 0:
        return 0.0

    tax    = 0.0
    income = monthly_taxable_income

    # Step 1: Personal Allowance — no tax on the first £1,047.50/month
    income -= min(income, pa)

    # Step 2: Basic Rate (20%) — on income between PA and BRL
    basic_band = brl - pa
    basic_taxable = min(income, basic_band)
    tax    += basic_taxable * bands.basic_rate
    income -= basic_taxable

    # Step 3: Higher Rate (40%) — on income between BRL and HRL
    higher_band = hrl - brl
    higher_taxable = min(income, higher_band)
    tax    += higher_taxable * bands.higher_rate
    income -= higher_taxable

    # Step 4: Additional Rate (45%) — on income above HRL
    tax += income * bands.additional_rate

    return tax


# ---------------------------------------------------------------------------
# UFPLS: DC pension withdrawal tax calculation
# ---------------------------------------------------------------------------

def calculate_ufpls_tax_free_portion(
    dc_gross_withdrawal: float,
    cumulative_tax_free_taken: float,
    lsa_limit: float,
) -> float:
    """
    Calculate the tax-free portion of a DC UFPLS withdrawal.

    Under UFPLS (Uncrystallised Fund Pension Lump Sum), 25% of each
    withdrawal is tax-free — but only up to the lifetime Lump Sum
    Allowance (LSA = £268,275 across all pension schemes).

    Once the LSA is exhausted, 100% of withdrawals become taxable.

    Parameters
    ----------
    dc_gross_withdrawal       : Gross amount withdrawn from DC pot (£)
    cumulative_tax_free_taken : Total tax-free cash already taken (£, tracked per path)
    lsa_limit                 : Lifetime cap on tax-free cash (£, from config)

    Returns
    -------
    float : Tax-free portion of this withdrawal (£). Always >= 0.
    """
    lsa_remaining   = max(0.0, lsa_limit - cumulative_tax_free_taken)
    potential_free  = 0.25 * dc_gross_withdrawal
    tax_free        = min(potential_free, lsa_remaining)
    return tax_free


def calculate_dc_net_received(
    dc_gross_withdrawal:       float,
    db_gross_income_this_month: float,
    cumulative_tax_free_taken: float,
    bands:                     TaxBands,
) -> tuple:
    """
    Calculate the net cash received from a DC withdrawal, accounting for:
      - The 25% UFPLS tax-free portion (subject to LSA cap)
      - Income tax on the 75% taxable portion at the MARGINAL rate
        (i.e. on top of DB income already received this month)

    Parameters
    ----------
    dc_gross_withdrawal        : Gross amount taken from DC pot (£)
    db_gross_income_this_month : DB pension income received this month (£).
                                 This has already used up some of the tax bands.
    cumulative_tax_free_taken  : Tax-free cash taken so far (tracked per sim path)
    bands                      : TaxBands instance

    Returns
    -------
    net_received    : float — cash in hand after tax (£)
    tax_paid        : float — income tax on the DC portion (£)
    tax_free_used   : float — tax-free cash used from LSA this month (£)
    """
    if dc_gross_withdrawal <= 0:
        return 0.0, 0.0, 0.0

    # Work out the tax-free portion for this withdrawal
    tax_free = calculate_ufpls_tax_free_portion(
        dc_gross_withdrawal, cumulative_tax_free_taken, bands.lump_sum_allowance
    )
    dc_taxable = dc_gross_withdrawal - tax_free

    # Tax on the DC taxable portion:
    # We calculate the MARGINAL tax — i.e. the extra tax caused by adding
    # dc_taxable on top of the DB income already received.
    # This ensures DB income uses up the lower bands first.
    total_taxable       = db_gross_income_this_month + dc_taxable
    tax_on_total        = calculate_monthly_income_tax(total_taxable,       bands)
    tax_on_db_only      = calculate_monthly_income_tax(db_gross_income_this_month, bands)
    tax_on_dc_portion   = tax_on_total - tax_on_db_only

    net_received = dc_gross_withdrawal - tax_on_dc_portion
    return net_received, tax_on_dc_portion, tax_free


def gross_up_dc_for_target_net(
    net_needed:                 float,
    db_gross_income_this_month: float,
    cumulative_tax_free_taken:  float,
    bands:                      TaxBands,
    tolerance:                  float = 0.50,
    max_iterations:             int   = 60,
) -> tuple:
    """
    Find the gross DC withdrawal needed to yield a target net cash amount.

    Because the relationship between gross and net is nonlinear (due to the
    UFPLS tax-free portion interacting with the income tax bands), we can't
    simply divide by (1 - tax_rate). Instead we use the bisection method:
    repeatedly halve a search interval until we find the gross that gives
    the right net.

    Bisection is a classic algorithm — worth understanding:
      1. Start with a range [lo, hi] that definitely contains the answer.
      2. Try the midpoint. Is the result too high or too low?
      3. Discard the half that can't contain the answer.
      4. Repeat until close enough (within tolerance).

    Parameters
    ----------
    net_needed                 : Target net cash from DC (£)
    db_gross_income_this_month : DB income already received this month (£)
    cumulative_tax_free_taken  : Tax-free cash taken so far (£)
    bands                      : TaxBands instance
    tolerance                  : Stop when result is within £0.50 of target
    max_iterations             : Safety cap

    Returns
    -------
    gross_withdrawal : float — gross amount to take from DC pot (£)
    tax_paid         : float — income tax on the DC withdrawal (£)
    tax_free_used    : float — LSA tax-free cash used (£)
    """
    if net_needed <= 0:
        return 0.0, 0.0, 0.0

    # Search bounds:
    # Lower bound: gross must be at least net_needed (0% tax is the best case)
    # Upper bound: 3× net_needed is more than enough (even at 45% tax + UFPLS)
    lo = net_needed
    hi = net_needed * 3.0

    gross = hi  # Default to upper bound if bisection doesn't converge

    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0

        net_from_mid, _, _ = calculate_dc_net_received(
            mid, db_gross_income_this_month, cumulative_tax_free_taken, bands
        )

        if abs(net_from_mid - net_needed) < tolerance:
            gross = mid
            break

        # If net_from_mid is less than needed, we need a larger gross → raise lo
        # If net_from_mid is more than needed, the gross is too high → lower hi
        if net_from_mid < net_needed:
            lo = mid
        else:
            hi = mid

    # Compute final net, tax, and tax-free from the converged gross
    net_received, tax_paid, tax_free_used = calculate_dc_net_received(
        gross, db_gross_income_this_month, cumulative_tax_free_taken, bands
    )

    return gross, tax_paid, tax_free_used


# ---------------------------------------------------------------------------
# GIA tax helper
# ---------------------------------------------------------------------------

def calculate_gia_net_received(
    gross_withdrawal:  float,
    balance:           float,
    cost_basis:        float,
    bands:             TaxBands,
) -> tuple:
    """
    Calculate net cash from a GIA withdrawal.

    CGT applies to the gains portion only. The capital (cost basis) portion
    is returned tax-free. We use a proportional cost basis method:
      gains_proportion = (balance - cost_basis) / balance

    Note: CGT annual exempt amount (£3,000 in 2024/25) is not modelled.
    This is a documented simplification.

    Returns
    -------
    net_received  : float — cash after CGT (£)
    tax_paid      : float — CGT paid (£)
    """
    if gross_withdrawal <= 0 or balance <= 0:
        return 0.0, 0.0

    gains_proportion = max(0.0, (balance - cost_basis) / balance)
    tax_paid         = gross_withdrawal * gains_proportion * bands.cgt_rate
    net_received     = gross_withdrawal - tax_paid

    return net_received, tax_paid


# ---------------------------------------------------------------------------
# Quick test
# Usage: python tax.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml
    from pathlib import Path

    print("Running tax.py as a standalone test...\n")

    config_path = Path("config.yaml")
    if not config_path.exists():
        print("ERROR: config.yaml not found.")
        exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    bands = TaxBands.from_config(config)

    print("=== Test 1: Monthly income tax on various income levels ===")
    test_incomes = [0, 800, 1047.50, 2000, 4189.17, 6000, 10000]
    for income in test_incomes:
        tax = calculate_monthly_income_tax(income, bands)
        print(f"  Monthly income £{income:>8,.2f} → tax £{tax:>7,.2f}  "
              f"(effective rate {tax/income*100:.1f}%)" if income > 0
              else f"  Monthly income £{income:>8,.2f} → tax £{tax:>7,.2f}")

    print()
    print("=== Test 2: DC UFPLS withdrawal ===")
    print("  Scenario: No DB income, £0 LSA used, need £1,500 net from DC")
    gross, tax, free = gross_up_dc_for_target_net(
        net_needed=1500,
        db_gross_income_this_month=0,
        cumulative_tax_free_taken=0,
        bands=bands,
    )
    print(f"  Gross withdrawal needed : £{gross:,.2f}")
    print(f"  Of which tax-free (25%) : £{free:,.2f}")
    print(f"  Income tax paid         : £{tax:,.2f}")
    print(f"  Net received            : £{gross - tax:,.2f}")

    print()
    print("=== Test 3: DC withdrawal on top of DB income ===")
    print("  Scenario: £1,500/month DB income, need £1,000 more net from DC")
    gross, tax, free = gross_up_dc_for_target_net(
        net_needed=1000,
        db_gross_income_this_month=1500,
        cumulative_tax_free_taken=0,
        bands=bands,
    )
    print(f"  DB income this month    : £1,500.00")
    print(f"  Gross DC withdrawal     : £{gross:,.2f}")
    print(f"  Tax-free portion (25%)  : £{free:,.2f}")
    print(f"  Income tax on DC        : £{tax:,.2f}")
    print(f"  Net from DC             : £{gross - tax:,.2f}")
    print()
    print("✓ If the numbers above look reasonable, tax.py is working correctly.")
    print("  Key check: Test 3 tax should be higher than Test 2 (DB uses up lower bands).")
