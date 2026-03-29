"""
drawdown.py
===========
Module 7 of 11 — Monthly Drawdown Orchestration.

What this module does:
  Handles one month of the drawdown phase. Given the current wrapper balances
  and the user's target spending, it:
    1. Calculates how much DB income covers (from all accessible DB schemes)
    2. Draws the remaining shortfall from GIA → ISA → DC in that order
    3. Applies the correct tax to each withdrawal (via tax.py)
    4. Detects and reports ruin

Python concepts introduced:
  - Guard clauses: checking conditions early to simplify the main logic
  - sum() with a generator expression
  - Returning multiple values cleanly
  - Careful floating-point comparisons (using small epsilon, not == 0)
"""

from tax import (
    TaxBands,
    calculate_monthly_income_tax,
    calculate_dc_net_received,
    gross_up_dc_for_target_net,
)
from wrappers import GIAWrapper, ISAWrapper, DCPotWrapper


# A small threshold to avoid floating-point "almost zero" issues
EPSILON = 0.01   # £0.01 — effectively zero for our purposes


def run_monthly_drawdown(
    gia:                 GIAWrapper,
    isa:                 ISAWrapper,
    dc_list:             list,
    db_income_schedule:  list,
    config:              dict,
    current_age:         float,
    bands:               TaxBands,
    cumulative_tax_free: float,
) -> tuple:
    """
    Execute one month of the drawdown phase.

    The drawdown target is the user's desired NET monthly spending (today's money).
    We fill this from sources in order:
      1. DB pension income (net of income tax) — from all accessible schemes
      2. GIA (net of CGT on gains)
      3. ISA (tax-free)
      4. DC pot(s) (net of UFPLS income tax, gated by pension_access_age)

    Ruin is declared if:
      - GIA and ISA are empty AND no DC pot is yet accessible (age < access_age)
      - All sources (GIA + ISA + all DC pots) are empty at any age

    Parameters
    ----------
    gia, isa, dc_list    : Current wrapper states (modified in place)
    db_income_schedule   : List of DB income dicts from pension_db.py
    config               : Full config dict
    current_age          : User's current age this month
    bands                : TaxBands instance from tax.py
    cumulative_tax_free  : Running total of LSA used so far in this sim path (£)

    Returns
    -------
    ruined              : bool  — True if the portfolio is ruined this month
    cumulative_tax_free : float — Updated LSA tracker (may be unchanged)
    """

    # ── Step 0: Determine monthly net spending target ──────────────────────
    dd_cfg = config["drawdown"]

    if dd_cfg["mode"] == "amount":
        # Fixed amount in today's money (real terms) — stays constant each month
        target_net_monthly = float(dd_cfg["annual_amount"]) / 12.0
    else:
        # Percentage of current total portfolio
        total_portfolio = gia.balance + isa.balance + sum(dc.balance for dc in dc_list)
        target_net_monthly = total_portfolio * (float(dd_cfg["percentage"]) / 100.0) / 12.0

        # Apply cap if set (e.g., "don't withdraw more than £50k/year even if 4% suggests more")
        if dd_cfg.get("percentage_cap") is not None:
            cap_monthly = float(dd_cfg["percentage_cap"]) / 12.0
            target_net_monthly = min(target_net_monthly, cap_monthly)

        # Apply floor if set (e.g., "withdraw at least £20k/year even if 4% suggests less")
        if dd_cfg.get("percentage_floor") is not None:
            floor_monthly = float(dd_cfg["percentage_floor"]) / 12.0
            target_net_monthly = max(target_net_monthly, floor_monthly)

    remaining = target_net_monthly  # How much net income we still need to source

    # ── Step 1: DB income ──────────────────────────────────────────────────
    # Sum gross income from all DB schemes whose NPA has been reached
    total_db_gross = sum(
        db["monthly_income_gross"]
        for db in db_income_schedule
        if current_age >= db["normal_pension_age"]
    )

    if total_db_gross > EPSILON:
        # All DB gross income is taxed together as regular income
        db_income_tax  = calculate_monthly_income_tax(total_db_gross, bands)
        total_db_net   = total_db_gross - db_income_tax
        remaining      = max(0.0, remaining - total_db_net)

    # Early exit: DB income alone covers the full target — no capital needed
    if remaining <= EPSILON:
        return False, cumulative_tax_free

    # ── Step 2: GIA withdrawal ─────────────────────────────────────────────
    if gia.balance > EPSILON and remaining > EPSILON:

        # Work out what gross to take in order to receive `remaining` net.
        # Net = gross × (1 - CGT_rate × gains_proportion)
        # Rearranging: gross = net / (1 - CGT_rate × gains_proportion)
        if gia.balance > 0:
            gains_proportion = max(0.0, (gia.balance - gia.cost_basis) / gia.balance)
        else:
            gains_proportion = 0.0

        effective_tax_rate = gains_proportion * bands.cgt_rate
        if effective_tax_rate < 1.0:
            gross_needed = remaining / (1.0 - effective_tax_rate)
        else:
            gross_needed = remaining  # Shouldn't happen, but safe fallback

        # Cap to what's actually available
        gross_needed = min(gross_needed, gia.balance)

        net_received, tax_paid, actual_gross = gia.withdraw(gross_needed, bands.cgt_rate)
        remaining = max(0.0, remaining - net_received)

    # ── Step 3: ISA withdrawal ─────────────────────────────────────────────
    if isa.balance > EPSILON and remaining > EPSILON:
        # ISA is tax-free: 1:1 gross → net
        net_received, actual_gross = isa.withdraw(remaining)
        remaining = max(0.0, remaining - net_received)

    # ── Step 4: DC pot withdrawals (UFPLS, age-gated) ─────────────────────
    for dc in dc_list:
        if remaining <= EPSILON:
            break
        if dc.balance <= EPSILON:
            continue
        if current_age < dc.pension_access_age:
            continue  # Age gate: too young to access this pot

        # Gross up: how much gross DC do we need to receive `remaining` net?
        # This accounts for UFPLS (25% tax-free, 75% taxable) and the
        # marginal tax rate on top of DB income already received.
        gross_needed, _, _ = gross_up_dc_for_target_net(
            net_needed                  = remaining,
            db_gross_income_this_month  = total_db_gross,
            cumulative_tax_free_taken   = cumulative_tax_free,
            bands                       = bands,
        )

        # Cap to what's available in the pot
        gross_needed = min(gross_needed, dc.balance)

        # Execute the withdrawal from the DC pot
        actual_gross, accessible = dc.withdraw(gross_needed, current_age)

        if not accessible:
            continue   # Shouldn't happen here (we checked above), but safe

        # Calculate exact net and tax-free for the actual gross withdrawn
        net_received, dc_tax_paid, tax_free_used = calculate_dc_net_received(
            dc_gross_withdrawal         = actual_gross,
            db_gross_income_this_month  = total_db_gross,
            cumulative_tax_free_taken   = cumulative_tax_free,
            bands                       = bands,
        )

        # Update the lifetime LSA tracker
        cumulative_tax_free += tax_free_used
        remaining = max(0.0, remaining - net_received)

    # ── Ruin detection ─────────────────────────────────────────────────────
    total_remaining = gia.balance + isa.balance + sum(dc.balance for dc in dc_list)

    # Ruin case 1: Everything is gone
    if total_remaining <= EPSILON:
        return True, cumulative_tax_free

    # Ruin case 2: GIA and ISA are empty, DC not yet accessible, can't cover target
    # (User is stuck waiting for pension access age with no liquid assets)
    gia_isa_empty = gia.balance <= EPSILON and isa.balance <= EPSILON
    dc_with_balance_locked = any(
        dc.balance > EPSILON and current_age < dc.pension_access_age
        for dc in dc_list
    )

    if gia_isa_empty and dc_with_balance_locked and remaining > EPSILON:
        return True, cumulative_tax_free

    return False, cumulative_tax_free
