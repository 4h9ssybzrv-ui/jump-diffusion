"""
validator.py
============
Module 2 of 11 — Input validation. Run before the simulation starts.

What this module does:
  Checks config.yaml for errors and problems before any simulation code runs.
  Rather than letting a bad value cause a confusing Python crash deep inside the
  simulation, this catches issues early and explains them in plain English.

  Three levels of feedback:
    ✓  Pass    — everything looks fine
    ⚠  Warning — simulation will run, but you should be aware of something
    ✗  Error   — simulation will NOT run until this is fixed

Python concepts introduced:
  - Lists as accumulators (collecting messages as we go)
  - Functions that return multiple values as a tuple
  - Iterating over a list of dictionaries (the pensions list)
  - f-strings with variable values
  - Early return to stop execution
"""


# ---------------------------------------------------------------------------
# Internal helpers — each checks one category of rules
# ---------------------------------------------------------------------------

def _check_ages(config: dict, errors: list, warnings: list) -> None:
    """Validate age and timeline settings."""
    u = config["user"]
    current_age    = u["current_age"]
    retirement_age = u["retirement_age"]
    life_exp       = u["life_expectancy"]

    if not isinstance(current_age, (int, float)) or current_age < 18:
        errors.append("current_age must be a number >= 18.")

    if not isinstance(retirement_age, (int, float)) or retirement_age <= current_age:
        errors.append(
            f"retirement_age ({retirement_age}) must be greater than "
            f"current_age ({current_age})."
        )

    if not isinstance(life_exp, (int, float)) or life_exp <= retirement_age:
        errors.append(
            f"life_expectancy ({life_exp}) must be greater than "
            f"retirement_age ({retirement_age})."
        )

    if isinstance(life_exp, (int, float)) and life_exp > 120:
        errors.append("life_expectancy above 120 is not supported.")


def _check_drawdown(config: dict, errors: list, warnings: list) -> None:
    """Validate drawdown settings."""
    dd = config["drawdown"]
    mode = dd.get("mode", "amount")

    if mode not in ("amount", "percentage"):
        errors.append(
            f"drawdown.mode must be 'amount' or 'percentage', got '{mode}'."
        )

    if mode == "amount":
        amount = dd.get("annual_amount", 0)
        if not isinstance(amount, (int, float)) or amount <= 0:
            errors.append(
                f"drawdown.annual_amount must be a positive number, got {amount}."
            )

    if mode == "percentage":
        pct = dd.get("percentage", 0)
        if not isinstance(pct, (int, float)) or not (0 < pct <= 100):
            errors.append(
                f"drawdown.percentage must be between 0 and 100, got {pct}."
            )


def _check_wrappers(config: dict, errors: list, warnings: list) -> None:
    """Validate GIA and ISA settings."""
    # GIA
    gia = config.get("gia", {})
    if gia.get("starting_value", 0) < 0:
        errors.append("gia.starting_value cannot be negative.")
    if gia.get("monthly_contribution", 0) < 0:
        errors.append("gia.monthly_contribution cannot be negative.")

    # ISA
    isa = config.get("isa", {})
    if isa.get("starting_value", 0) < 0:
        errors.append("isa.starting_value cannot be negative.")

    monthly_isa_contrib = isa.get("monthly_contribution", 0)
    isa_limit           = isa.get("annual_limit", 20000)

    if monthly_isa_contrib < 0:
        errors.append("isa.monthly_contribution cannot be negative.")

    if isinstance(monthly_isa_contrib, (int, float)) and isinstance(isa_limit, (int, float)):
        if monthly_isa_contrib * 12 > isa_limit:
            warnings.append(
                f"ISA monthly contribution (£{monthly_isa_contrib:,.0f}) × 12 = "
                f"£{monthly_isa_contrib * 12:,.0f}, which exceeds the annual ISA limit "
                f"of £{isa_limit:,.0f}. Contributions will be capped at the limit."
            )


def _check_pensions(config: dict, errors: list, warnings: list) -> None:
    """Validate all pension schemes in the pensions list."""
    pensions = config.get("pensions", [])
    u        = config["user"]
    current_age    = u.get("current_age", 0)
    retirement_age = u.get("retirement_age", 0)
    life_exp       = u.get("life_expectancy", 100)

    if not isinstance(pensions, list) or len(pensions) == 0:
        warnings.append(
            "No pension schemes configured. Only GIA and ISA will be modelled."
        )
        return

    # Track combined DC contributions for the annual allowance check
    total_dc_monthly = 0.0
    pension_ids      = []

    for pension in pensions:
        pid  = pension.get("id", "(unnamed)")
        ptype = pension.get("type", "DC")

        if pid in pension_ids:
            errors.append(f"Duplicate pension id '{pid}'. Each pension must have a unique id.")
        pension_ids.append(pid)

        if ptype not in ("DC", "DB", "hybrid"):
            errors.append(
                f"Pension '{pid}': type must be 'DC', 'DB', or 'hybrid', got '{ptype}'."
            )

        # ── DC checks ──────────────────────────────────────────────────────
        if ptype in ("DC", "hybrid"):
            dc = pension.get("dc", {})

            if dc.get("starting_value", 0) < 0:
                errors.append(f"Pension '{pid}': dc.starting_value cannot be negative.")

            dc_contrib = dc.get("monthly_contribution", 0)
            if dc_contrib < 0:
                errors.append(f"Pension '{pid}': dc.monthly_contribution cannot be negative.")
            total_dc_monthly += dc_contrib

            access_age = dc.get("pension_access_age", 57)

            if isinstance(access_age, (int, float)):
                if access_age > retirement_age:
                    gap = int(access_age - retirement_age)
                    warnings.append(
                        f"Pension '{pid}': DC pot is not accessible until age {access_age}, "
                        f"but retirement is planned at {retirement_age}. "
                        f"GIA and ISA must bridge a {gap}-year gap before pension access. "
                        f"Make sure your GIA/ISA balances are sufficient."
                    )
                if access_age > life_exp:
                    errors.append(
                        f"Pension '{pid}': pension_access_age ({access_age}) exceeds "
                        f"life_expectancy ({life_exp}). This DC pot will never be accessed."
                    )

        # ── DB checks ──────────────────────────────────────────────────────
        if ptype in ("DB", "hybrid"):
            db = pension.get("db", {})
            input_mode = db.get("input_mode", "projected")

            if input_mode not in ("projected", "calculate"):
                errors.append(
                    f"Pension '{pid}': db.input_mode must be 'projected' or 'calculate'."
                )

            npa = db.get("normal_pension_age", 67)

            if isinstance(npa, (int, float)):
                if npa > life_exp:
                    errors.append(
                        f"Pension '{pid}': DB normal_pension_age ({npa}) exceeds "
                        f"life_expectancy ({life_exp}). This DB income will never be received."
                    )
                if npa > retirement_age:
                    gap = int(npa - retirement_age)
                    warnings.append(
                        f"Pension '{pid}': DB income does not start until age {npa}, "
                        f"but retirement is planned at {retirement_age}. "
                        f"GIA/ISA must cover a {gap}-year gap before DB income begins."
                    )

            if input_mode == "projected":
                income = db.get("projected_annual_income", 0)
                if not isinstance(income, (int, float)) or income <= 0:
                    errors.append(
                        f"Pension '{pid}': projected_annual_income must be a positive "
                        f"number when input_mode is 'projected'."
                    )

            if input_mode == "calculate":
                salary  = db.get("current_pensionable_salary", 0)
                yrs_now = db.get("years_of_service_to_date", 0)
                yrs_ret = db.get("years_of_service_at_retirement", 0)

                if not isinstance(salary, (int, float)) or salary <= 0:
                    errors.append(
                        f"Pension '{pid}': current_pensionable_salary must be positive."
                    )
                if not isinstance(yrs_now, (int, float)) or yrs_now < 0:
                    errors.append(
                        f"Pension '{pid}': years_of_service_to_date cannot be negative."
                    )
                if not isinstance(yrs_ret, (int, float)) or yrs_ret < yrs_now:
                    errors.append(
                        f"Pension '{pid}': years_of_service_at_retirement ({yrs_ret}) "
                        f"cannot be less than years_of_service_to_date ({yrs_now})."
                    )

    # Check combined DC contributions against annual allowance
    combined_annual_dc = total_dc_monthly * 12
    dc_annual_limit    = 60000  # HMRC annual pension allowance
    if combined_annual_dc > dc_annual_limit:
        warnings.append(
            f"Combined annual DC pension contributions (£{combined_annual_dc:,.0f}) "
            f"exceed the HMRC annual allowance of £{dc_annual_limit:,.0f}. "
            f"Contributions will be capped at the annual limit."
        )


def _check_lump_sums(config: dict, errors: list, warnings: list) -> None:
    """Validate lump sum injections."""
    lump_sums = config.get("lump_sums", [])
    if not lump_sums:
        return

    u = config["user"]
    current_age    = u.get("current_age", 0)
    retirement_age = u.get("retirement_age", 0)
    life_exp       = u.get("life_expectancy", 100)

    # Build list of valid wrapper names
    valid_wrappers = {"gia", "isa"}
    for p in config.get("pensions", []):
        valid_wrappers.add(p.get("id", ""))

    for i, ls in enumerate(lump_sums):
        label = f"lump_sums[{i}]"

        age    = ls.get("age")
        amount = ls.get("amount")
        wrapper = ls.get("wrapper", "")

        if not isinstance(age, (int, float)) or age < current_age:
            errors.append(
                f"{label}: age ({age}) is in the past (current age: {current_age})."
            )

        if not isinstance(amount, (int, float)) or amount <= 0:
            errors.append(f"{label}: amount must be a positive number.")

        if isinstance(age, (int, float)) and age > life_exp:
            errors.append(
                f"{label}: age ({age}) exceeds life_expectancy ({life_exp})."
            )

        if isinstance(age, (int, float)) and retirement_age < age <= life_exp:
            warnings.append(
                f"{label}: lump sum at age {age} occurs after retirement age "
                f"({retirement_age}). It will be added but no further regular "
                f"contributions will be made after retirement."
            )

        if wrapper.lower() not in valid_wrappers:
            errors.append(
                f"{label}: wrapper '{wrapper}' not recognised. "
                f"Must be one of: {sorted(valid_wrappers)}."
            )


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate(config: dict) -> tuple:
    """
    Run all validation checks on the config dictionary.

    Returns
    -------
    errors   : list[str] — problems that prevent the simulation from running
    warnings : list[str] — things to be aware of (simulation still runs)
    """
    errors   = []
    warnings = []

    _check_ages(config, errors, warnings)
    _check_drawdown(config, errors, warnings)
    _check_wrappers(config, errors, warnings)
    _check_pensions(config, errors, warnings)
    _check_lump_sums(config, errors, warnings)

    return errors, warnings


def print_validation_report(errors: list, warnings: list) -> None:
    """
    Print a formatted validation report.
    Called by main.py before the simulation starts.
    """
    SEP = "=" * 60

    print(f"\n{SEP}")
    print("  INPUT VALIDATION REPORT")
    print(SEP)

    all_good = True

    if not errors and not warnings:
        print("  ✓  All checks passed. No issues found.")
    else:
        if warnings:
            all_good = False
            for w in warnings:
                # Word-wrap long warnings to keep the report readable
                words  = w.split()
                line   = "  ⚠  "
                indent = "     "
                for word in words:
                    if len(line) + len(word) + 1 > 60:
                        print(line)
                        line = indent + word + " "
                    else:
                        line += word + " "
                print(line)

        if errors:
            all_good = False
            for e in errors:
                words  = e.split()
                line   = "  ✗  "
                indent = "     "
                for word in words:
                    if len(line) + len(word) + 1 > 60:
                        print(line)
                        line = indent + word + " "
                    else:
                        line += word + " "
                print(line)

    print(SEP + "\n")

    if errors:
        raise SystemExit(
            f"Simulation stopped: {len(errors)} error(s) found in config.yaml. "
            f"Please fix the issues above and try again."
        )

    if all_good:
        print("Proceeding with simulation...\n")
    else:
        print(f"Proceeding with simulation ({len(warnings)} warning(s) noted)...\n")


# ---------------------------------------------------------------------------
# Quick test
# Usage: python validator.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml
    from pathlib import Path

    print("Running validator.py as a standalone test...")

    config_path = Path("config.yaml")
    if not config_path.exists():
        print("ERROR: config.yaml not found.")
        exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    errors, warnings = validate(config)
    print_validation_report(errors, warnings)

    if not errors:
        print("✓ Validation passed. Your config.yaml looks good.")
