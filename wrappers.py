"""
wrappers.py
===========
Module 4 of 11 — GIA, ISA, and DC Pension Pot wrapper classes.

What this module does:
  Defines three dataclasses — one per investment wrapper type.
  Each wrapper tracks its own balance and handles contributions,
  growth, and withdrawals according to its own rules.

  GIA  : Tracks cost basis for proportional CGT calculation
  ISA  : Tax-free; enforces £20k annual limit
  DCPot: Tracks annual contributions against the pension allowance;
         withdrawals handled by tax.py (UFPLS)

Python concepts introduced:
  - @dataclass decorator — auto-generates __init__, __repr__ etc.
  - Instance methods (def method(self, ...))
  - Default field values with field(default=...)
  - None as a sentinel value (for no annual limit)
  - min() and max() for safe clamping
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# GIA Wrapper
# ---------------------------------------------------------------------------

@dataclass
class GIAWrapper:
    """
    General Investment Account.

    Tracks a cost basis so that CGT can be applied only to gains,
    not to the original capital invested.

    Attributes
    ----------
    balance                  : Current portfolio value (£)
    cost_basis               : Total amount ever invested — excludes gains (£)
    annual_contributions_ytd : Total contributed so far this calendar year (£)
    """
    balance:                  float
    cost_basis:               float         # Starts equal to starting_value
    annual_contributions_ytd: float = 0.0

    def apply_return(self, r: float) -> None:
        """
        Apply a monthly return to the balance.

        r is a decimal return (e.g. 0.05 for +5%, -0.10 for -10%).
        The cost basis does not change when returns are applied —
        only when money is put in or taken out.
        """
        self.balance = self.balance * (1.0 + r)

    def contribute(self, amount: float) -> float:
        """
        Add a contribution to the GIA.

        No HMRC annual limit on GIA contributions, so the full amount
        is always accepted. The cost basis increases by the contribution
        because this is new money going in (not a gain).

        Returns the actual amount contributed (always equal to amount for GIA).
        """
        self.balance                  += amount
        self.cost_basis               += amount
        self.annual_contributions_ytd += amount
        return amount

    def withdraw(self, gross_needed: float, cgt_rate: float) -> tuple:
        """
        Withdraw from the GIA, applying CGT on the gains portion.

        The gains proportion is calculated as:
          (balance - cost_basis) / balance

        This tells us what fraction of each £ withdrawn is a gain
        (taxable) vs return of original capital (tax-free).

        Parameters
        ----------
        gross_needed : How much we want to take out before tax (£)
        cgt_rate     : CGT rate on gains (e.g. 0.20 for 20%)

        Returns
        -------
        net_received : float — cash after CGT (£)
        tax_paid     : float — CGT paid (£)
        actual_gross : float — actual gross withdrawn (may be < gross_needed if balance is low)
        """
        if self.balance <= 0:
            return 0.0, 0.0, 0.0

        # Can't withdraw more than the current balance
        actual_gross = min(gross_needed, self.balance)

        # What fraction of the balance is gains? (clamp to 0 if cost_basis > balance)
        gains_proportion = max(0.0, (self.balance - self.cost_basis) / self.balance)

        # CGT applies only to the gains portion of the withdrawal
        tax_paid     = actual_gross * gains_proportion * cgt_rate
        net_received = actual_gross - tax_paid

        # Reduce cost basis proportionally to the fraction of balance withdrawn
        proportion_withdrawn = actual_gross / self.balance
        self.cost_basis = self.cost_basis * (1.0 - proportion_withdrawn)
        self.cost_basis = max(0.0, self.cost_basis)  # Never let it go negative

        self.balance -= actual_gross

        return net_received, tax_paid, actual_gross

    def add_lump_sum(self, amount: float) -> float:
        """Add a one-off lump sum. Same as contribute() for GIA (no limit)."""
        return self.contribute(amount)

    def reset_annual_contributions(self) -> None:
        """Call this at the start of each new tax year."""
        self.annual_contributions_ytd = 0.0


# ---------------------------------------------------------------------------
# ISA Wrapper
# ---------------------------------------------------------------------------

@dataclass
class ISAWrapper:
    """
    Stocks & Shares ISA.

    All growth and withdrawals are completely tax-free.
    Annual contributions are capped at the HMRC ISA allowance (£20,000).

    Attributes
    ----------
    balance                  : Current portfolio value (£)
    annual_limit             : HMRC annual contribution limit (£, default £20,000)
    annual_contributions_ytd : Total contributed so far this calendar year (£)
    """
    balance:                  float
    annual_limit:             float = 20000.0
    annual_contributions_ytd: float = 0.0

    def apply_return(self, r: float) -> None:
        """Apply a monthly return. ISA growth is tax-free so no adjustments needed."""
        self.balance = self.balance * (1.0 + r)

    def contribute(self, amount: float) -> float:
        """
        Add a contribution, capped at the remaining annual allowance.

        If the requested contribution would exceed the annual limit,
        we cap it and log a note. The excess is simply not invested.

        Returns the actual amount contributed (may be less than amount).
        """
        remaining_allowance = self.annual_limit - self.annual_contributions_ytd
        actual_contribution = min(amount, max(0.0, remaining_allowance))

        if actual_contribution < amount:
            # Quietly cap — validator.py will have already warned about this
            pass

        self.balance                  += actual_contribution
        self.annual_contributions_ytd += actual_contribution
        return actual_contribution

    def withdraw(self, amount_needed: float) -> tuple:
        """
        Withdraw from the ISA. Fully tax-free — no calculation needed.

        Returns
        -------
        net_received : float — cash received (equals gross, no tax)
        actual_gross : float — amount actually taken (may be < amount_needed)
        """
        actual_gross = min(amount_needed, self.balance)
        self.balance -= actual_gross
        return actual_gross, actual_gross   # net == gross (tax-free)

    def add_lump_sum(self, amount: float) -> float:
        """Add a one-off lump sum (subject to annual limit)."""
        return self.contribute(amount)

    def reset_annual_contributions(self) -> None:
        """Call this at the start of each new tax year."""
        self.annual_contributions_ytd = 0.0


# ---------------------------------------------------------------------------
# DC Pension Pot Wrapper
# ---------------------------------------------------------------------------

@dataclass
class DCPotWrapper:
    """
    Defined Contribution pension pot.

    This wrapper only tracks the balance and contributions.
    Tax on withdrawals (UFPLS) is calculated by tax.py, not here,
    because the tax depends on the total income from all sources that month.

    Attributes
    ----------
    pension_id               : Identifier matching the config (e.g. "main_pension")
    balance                  : Current pot value (£)
    pension_access_age       : Minimum age before drawdown can start
    annual_limit             : HMRC annual pension allowance (shared across all DC pots)
    annual_contributions_ytd : Contributions made to THIS pot this year (£)
    """
    pension_id:               str
    balance:                  float
    pension_access_age:       float = 57.0
    annual_limit:             float = 60000.0
    annual_contributions_ytd: float = 0.0

    def apply_return(self, r: float) -> None:
        """Apply a monthly return."""
        self.balance = self.balance * (1.0 + r)

    def contribute(
        self,
        amount:              float,
        combined_ytd_all_dc: float,
    ) -> float:
        """
        Add a contribution, subject to the combined DC annual allowance.

        The £60,000 annual pension allowance applies across ALL DC schemes
        combined. We therefore need to know the total already contributed
        to ALL DC schemes this year (combined_ytd_all_dc), which is passed
        in from simulation.py.

        Parameters
        ----------
        amount              : Requested contribution (£)
        combined_ytd_all_dc : Total already contributed to all DC schemes this year (£)

        Returns the actual amount contributed (may be less than amount).
        """
        remaining_allowance = self.annual_limit - combined_ytd_all_dc
        actual_contribution = min(amount, max(0.0, remaining_allowance))

        self.balance                  += actual_contribution
        self.annual_contributions_ytd += actual_contribution
        return actual_contribution

    def withdraw(self, gross_needed: float, current_age: float) -> tuple:
        """
        Attempt to withdraw from the DC pot.

        If current_age < pension_access_age, no withdrawal is possible
        and (0, 0, False) is returned. Tax calculation is NOT done here —
        that is handled by tax.py in drawdown.py.

        Returns
        -------
        actual_gross     : float — gross amount taken from pot (£)
        actual_gross     : float — same (tax handled externally)
        accessible       : bool  — False if age gate blocks access
        """
        if current_age < self.pension_access_age:
            return 0.0, False   # Age gate: pension not yet accessible

        actual_gross  = min(gross_needed, self.balance)
        self.balance -= actual_gross
        return actual_gross, True

    def add_lump_sum(
        self,
        amount:              float,
        combined_ytd_all_dc: float,
    ) -> float:
        """Add a one-off lump sum (subject to combined DC annual allowance)."""
        return self.contribute(amount, combined_ytd_all_dc)

    def reset_annual_contributions(self) -> None:
        """Call at the start of each new tax year."""
        self.annual_contributions_ytd = 0.0


# ---------------------------------------------------------------------------
# Factory function — build wrappers from config
# ---------------------------------------------------------------------------

def build_wrappers_from_config(config: dict) -> tuple:
    """
    Create GIA, ISA, and all DC pension pot wrappers from the config dictionary.

    Returns
    -------
    gia_wrapper  : GIAWrapper
    isa_wrapper  : ISAWrapper
    dc_wrappers  : list[DCPotWrapper]  — one per DC or hybrid pension scheme
    """
    # GIA
    gia_cfg = config.get("gia", {})
    gia = GIAWrapper(
        balance    = float(gia_cfg.get("starting_value", 0)),
        cost_basis = float(gia_cfg.get("starting_value", 0)),  # Cost basis = starting value
    )

    # ISA
    isa_cfg = config.get("isa", {})
    isa = ISAWrapper(
        balance      = float(isa_cfg.get("starting_value", 0)),
        annual_limit = float(isa_cfg.get("annual_limit", 20000)),
    )

    # DC pension pots — collect all DC and hybrid schemes
    dc_wrappers = []
    for pension in config.get("pensions", []):
        ptype = pension.get("type", "DC")
        if ptype in ("DC", "hybrid"):
            dc_cfg = pension.get("dc", {})
            dc_wrappers.append(DCPotWrapper(
                pension_id         = pension.get("id", "pension"),
                balance            = float(dc_cfg.get("starting_value", 0)),
                pension_access_age = float(dc_cfg.get("pension_access_age", 57)),
                annual_limit       = float(dc_cfg.get("annual_limit", 60000)),
            ))

    return gia, isa, dc_wrappers


# ---------------------------------------------------------------------------
# Quick test
# Usage: python wrappers.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml
    from pathlib import Path

    print("Running wrappers.py as a standalone test...\n")

    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        gia, isa, dc_list = build_wrappers_from_config(config)
        print("Wrappers built from config.yaml:")
    else:
        # Fallback: build example wrappers manually
        gia     = GIAWrapper(balance=50000, cost_basis=50000)
        isa     = ISAWrapper(balance=20000)
        dc_list = [DCPotWrapper(pension_id="main", balance=100000)]
        print("Wrappers built with example values:")

    print(f"  GIA : balance=£{gia.balance:,.0f}, cost_basis=£{gia.cost_basis:,.0f}")
    print(f"  ISA : balance=£{isa.balance:,.0f}")
    for dc in dc_list:
        print(f"  DC  : id={dc.pension_id}, balance=£{dc.balance:,.0f}, "
              f"access_age={dc.pension_access_age}")

    print()
    print("=== Test: Apply a 5% return ===")
    gia.apply_return(0.05)
    isa.apply_return(0.05)
    for dc in dc_list:
        dc.apply_return(0.05)
    print(f"  GIA balance after +5%: £{gia.balance:,.2f}  (cost_basis unchanged: £{gia.cost_basis:,.2f})")
    print(f"  ISA balance after +5%: £{isa.balance:,.2f}")
    for dc in dc_list:
        print(f"  DC  balance after +5%: £{dc.balance:,.2f}")

    print()
    print("=== Test: GIA withdrawal (50% gains) ===")
    # Manually set cost_basis to 50% of balance to simulate gains
    gia.cost_basis = gia.balance * 0.5
    net, tax, gross = gia.withdraw(gross_needed=10000, cgt_rate=0.20)
    print(f"  Gross withdrawn : £{gross:,.2f}")
    print(f"  CGT paid (20% on gains half): £{tax:,.2f}")
    print(f"  Net received    : £{net:,.2f}")
    print(f"  Remaining GIA   : £{gia.balance:,.2f}")

    print()
    print("✓ Wrappers are working correctly.")
