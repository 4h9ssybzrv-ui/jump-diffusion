# V2 Streamlit Implementation Plan

## Overview
Build a Streamlit web app (`app.py`) that wraps the existing Monte Carlo simulation. Users can configure all inputs via sidebar widgets and see charts + stats inline. The app will work locally and be deployable to Streamlit Community Cloud for free.

---

## Key Challenges Identified

### 1. **visualisation.py Returns None** (CRITICAL)
- Current functions save PNG to disk and return `None`
- Streamlit needs `matplotlib.Figure` objects to display charts
- **Solution:** Refactor `visualisation.py` to optionally return figures instead of closing them

### 2. **Config Structure More Complex Than CLAUDE.md**
- CLAUDE.md describes a simple "pension" wrapper, but actual code has:
  - `pensions`: list of schemes (DC/DB/hybrid)
  - `jump_diffusion`: complex calibration config (not simple "black_swan")
  - `tax_bands`: detailed UK tax thresholds
  - `lump_sums`: feature not mentioned in CLAUDE.md
- **Solution:** Build Streamlit UI that mirrors the **actual** config structure, not the simplified CLAUDE.md version

### 3. **validator.py Calls SystemExit on Errors**
- Currently crashes the app if config is invalid
- **Solution:** Wrap validation in try/except, display errors as `st.error()` instead

### 4. **CPU-Bound Simulation Blocks UI**
- `run_simulation()` with 10,000 simulations takes ~10+ seconds
- Will freeze Streamlit UI during computation
- **Solution:** Use `st.cache_data` to cache results and avoid re-running unless inputs change

### 5. **Hardcoded File Paths**
- CSVs are in `data/stock_prices.csv` and `data/cpih.csv`
- CLAUDE.md mentions optional upload, but actual spec says bundled data only
- **Solution:** Use bundled data; simplify by removing upload feature (out of scope per CLAUDE.md)

---

## Implementation Strategy

### Phase 1: Refactor visualisation.py (Minimal Change)

**Goal:** Make chart functions return figures without breaking existing CLI.

**Approach:**
- Modify `generate_accumulation_chart()` and `generate_drawdown_chart()` to return the `matplotlib.Figure` object
- Keep the `plt.savefig()` calls (for CLI backward compatibility)
- Add `return fig` before `plt.close()`
- Existing `main.py` won't break (it ignores the return value)

**Files modified:** `visualisation.py`

---

### Phase 2: Create app.py (Streamlit Entry Point)

**Structure:**

```python
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import simulation modules
from data_loader import load_data
from pension_db import build_db_income_schedule
from tax import TaxBands
from wrappers import build_wrappers_from_config
from simulation import run_simulation
from analysis import analyse
from visualisation import generate_accumulation_chart, generate_drawdown_chart
from validator import validate

# Page config
st.set_page_config(page_title="Monte Carlo Retirement Simulator", layout="wide")

def main():
    st.title("Monte Carlo Retirement Simulator")
    st.write("""
    All values shown are in **real terms (today's money)**, adjusted for inflation.
    This simulation uses UK historical stock market returns and CPIH inflation data.
    """)

    # Build config dict from sidebar widgets
    config = build_config_from_widgets()

    # Validate
    try:
        errors, warnings = validate(config)
        if errors:
            st.error("Configuration errors:\n" + "\n".join(errors))
            return
        if warnings:
            st.warning("Configuration warnings:\n" + "\n".join(warnings))
    except SystemExit:
        # validator.py calls SystemExit on critical errors; catch it
        st.error("Configuration validation failed. Please check your inputs.")
        return

    # Run simulation button
    if st.button("Run Simulation", key="run_sim"):
        run_and_display_simulation(config)

def build_config_from_widgets():
    # Build a config dict from Streamlit sidebar inputs
    # Mirror the actual YAML structure, not CLAUDE.md's simplified version
    pass

def run_and_display_simulation(config):
    try:
        # Load data
        market_params = load_data(config)
        db_income = build_db_income_schedule(config)
        tax_bands = TaxBands.from_config(config)

        # Run simulation (with spinner for feedback)
        n_sims = config["simulation"]["n_simulations"]
        with st.spinner(f"Running {n_sims} Monte Carlo simulations..."):
            portfolio_values, ruin_flags, ruin_ages, ages, gia_accum, isa_accum, dc_accum = run_simulation(
                config, market_params, db_income, tax_bands
            )

        # Analyse results
        analysis = analyse(portfolio_values, ruin_flags, ruin_ages, ages, config, gia_accum, isa_accum, dc_accum)

        # Display charts
        st.subheader("Accumulation Phase")
        fig_accum = generate_accumulation_chart(analysis, config, db_income)
        st.pyplot(fig_accum)

        st.subheader("Drawdown Phase")
        fig_draw = generate_drawdown_chart(analysis, config, db_income)
        st.pyplot(fig_draw)

        # Display summary stats
        st.subheader("Summary Statistics")
        summary_df = build_summary_table(analysis)
        st.dataframe(summary_df, use_container_width=True)

        # Assumptions expander
        with st.expander("See all assumptions"):
            st.write(format_assumptions(config, market_params))

    except Exception as e:
        st.error(f"Simulation failed: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()
```

**Sidebar widget groups:**
1. **Simulation Settings:** n_simulations (slider 100–10,000, **default 1000**), random_seed (default 42)
2. **User Settings:** current_age, retirement_age, life_expectancy, pension_access_age
3. **GIA Settings:** starting_value, monthly_contribution, annual_limit, tax_rate
4. **ISA Settings:** starting_value, monthly_contribution, annual_limit
5. **Pensions Settings:** For each pension scheme in config:
   - Scheme type (DC/DB/hybrid selector)
   - If DC: starting_value, monthly_contribution, annual_limit
   - If DB: scheme name, accrual rate, starting salary, annual salary growth
6. **Drawdown Settings:** mode (radio: amount/percentage), annual_amount, percentage
7. **Tax Settings:** Expander with income tax bands (optional — could pre-fill 2024/25 defaults)
8. **Jump-Diffusion Settings:** lambda_annual, jump_mean, jump_std, thresholds

**Key helper functions:**
- `build_config_from_widgets()` — construct full config dict from all sidebar inputs
- `build_summary_table()` — format analysis results as a Pandas DataFrame for `st.dataframe()`
- `format_assumptions()` — create formatted text of all assumptions used

---

### Phase 3: Create requirements.txt

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
pyyaml>=6.0
```

Remove `seaborn` (unused in existing code).

---

### Phase 4: Testing & Refinement

**Local testing:**
1. Install Streamlit: `pip install streamlit`
2. Run: `streamlit run app.py`
3. Test all sidebar widgets update config correctly
4. Test validation catches invalid inputs
5. Test charts render properly (no blank areas, legends readable)
6. Test performance (how long does 10,000 sims take? Should be <15 seconds)

**Known limitations for now:**
- No scenario comparison (out of scope per CLAUDE.md)
- No CSV export (out of scope)
- No shareable URL encoding of inputs (out of scope)
- No custom pension schemes—only pre-configured templates

---

## Implementation Order (Step-by-Step)

1. **Modify visualisation.py** — Add `return fig` before `plt.close()` in both chart functions
2. **Create app.py skeleton** — Basic page structure, sidebar, button
3. **Implement config widget builder** — Test that sidebar inputs → valid config dict
4. **Hook up simulation** — Make sure config flows through to `run_simulation()` correctly
5. **Render charts** — Call modified visualisation.py, display with `st.pyplot()`
6. **Build summary table** — Format analysis results nicely
7. **Add assumptions expander** — Display all config assumptions
8. **Test end-to-end** — Run locally, tweak UI, verify calculations
9. **Create requirements.txt**
10. **Document for Cloud deployment** — README on how to deploy to Streamlit Community Cloud

---

## Design Decisions

### A. Caching Strategy
- Use `@st.cache_data` on `load_data()` call (market params, inflation don't change often)
- Do NOT cache `run_simulation()` result because users will tweak config frequently
  - Instead, show `st.spinner()` during simulation to indicate it's working
  - Alternative: add a "Save Results" button to manually cache results between runs

### B. Config Complexity
- The **actual** config structure is far more complex than CLAUDE.md describes
- Rather than simplify for Streamlit, we should build the **full** Streamlit UI
- This means users get ALL the advanced options: lump sums, DB pensions, tax bands, etc.
- BUT: Default values should be sensible (2024/25 UK tax bands, simple DC pension)

### C. Error Handling
- Catch validator errors and display as `st.error()` (don't crash app)
- Wrap simulator in try/except to catch numerical errors gracefully
- Display last-known results if simulation fails

### D. File Structure
- Keep `app.py` in the root (same level as `main.py`)
- Keep all data loading and simulation modules unchanged
- Only modify `visualisation.py` (minimal, backward-compatible change)

---

## User Decisions (Confirmed)

✅ **1. Default config:** Show sensible defaults (35-year-old, £50k GIA, etc.)
✅ **2. Pension complexity:** Build FULL UI (support DC, DB, hybrid schemes)
✅ **3. Tax bands visibility:** Hide in expander to reduce sidebar clutter
✅ **4. Simulation speed:** Default to 1,000 simulations (faster feedback), slider allows 100–10,000

---

## Success Criteria

✅ App launches without errors
✅ All sidebar widgets render and are interactive
✅ Clicking "Run Simulation" produces both charts
✅ Summary stats table displays correctly
✅ Assumptions expander shows all key inputs
✅ Color coding in charts matches CLAUDE.md (green/yellow/red for ruin risk)
✅ Closing the app doesn't corrupt any files (all in-memory)
✅ `streamlit run app.py` works from project root
✅ Can push to GitHub and deploy to Streamlit Cloud with no code changes

