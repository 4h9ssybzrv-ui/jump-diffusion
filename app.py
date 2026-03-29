"""
app.py — Streamlit web interface for Monte Carlo Retirement Simulator

Wraps the existing simulation modules (simulation.py, analysis.py, visualisation.py, etc.)
and provides an interactive web UI with sidebar controls.

All values are displayed in real terms (today's money).
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import traceback

# Import simulation modules
from data_loader import load_data
from pension_db import build_db_income_schedule
from tax import TaxBands
from simulation import run_simulation
from analysis import analyse
from visualisation import generate_accumulation_chart, generate_drawdown_chart
from validator import validate


# ============================================================================
# Page Configuration & Session State
# ============================================================================

st.set_page_config(
    page_title="Monte Carlo Retirement Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for persistent results across sidebar interactions
if "sim_results" not in st.session_state:
    st.session_state.sim_results = None
if "last_config" not in st.session_state:
    st.session_state.last_config = None


# ============================================================================
# Cached Data Loader
# ============================================================================

@st.cache_data
def load_market_data():
    """Load CSV data and calibrate jump-diffusion parameters. Cached for speed."""
    # Build a minimal config just for loading data
    base_config = {
        "data": {
            "prices_file": "data/VWRP Dailies - Sheet1.csv",
            "cpih_file": "data/series-280326.csv",
        },
        "jump_diffusion": {
            "jump_threshold_sigma": 2.0,
            "lambda_annual": 0.10,
            "jump_mean": -0.40,
            "jump_std": 0.10,
            "jump_clip_lower": -0.60,
            "jump_clip_upper": -0.05,
        },
    }
    return load_data(base_config)


# ============================================================================
# Config Builder — Sidebar Widgets
# ============================================================================

def build_config_from_sidebar():
    """
    Build full config dict from Streamlit sidebar widgets.
    Exactly mirrors the config.yaml structure expected by simulation.py.
    """

    with st.sidebar:
        st.header("Configuration")

        # ---- Simulation Settings ----
        st.subheader("Simulation Settings")
        n_sims = st.slider(
            "Number of simulations",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="More simulations = more accurate, but slower. 1000 is a good balance.",
        )
        random_seed = st.number_input(
            "Random seed (for reproducibility)",
            value=42,
            min_value=0,
            max_value=999999,
            help="Same seed = same results. Use 0 for random each time.",
        )

        # ---- User / Personal Settings ----
        st.subheader("Personal Details")
        current_age = st.number_input(
            "Current age",
            value=35,
            min_value=18,
            max_value=100,
        )
        retirement_age = st.number_input(
            "Retirement age",
            value=65,
            min_value=current_age + 1,
            max_value=100,
        )
        life_expectancy = st.number_input(
            "Life expectancy (for ruin calculation)",
            value=100,
            min_value=retirement_age + 1,
            max_value=120,
        )

        # ---- GIA Settings ----
        st.subheader("GIA (General Investment Account)")
        gia_starting = st.number_input(
            "GIA starting value (£)",
            value=50000,
            min_value=0,
            step=1000,
        )
        gia_monthly = st.number_input(
            "GIA monthly contribution (£)",
            value=500,
            min_value=0,
            step=100,
        )

        # ---- ISA Settings ----
        st.subheader("ISA (Individual Savings Account)")
        isa_starting = st.number_input(
            "ISA starting value (£)",
            value=20000,
            min_value=0,
            step=1000,
        )
        isa_monthly = st.number_input(
            "ISA monthly contribution (£)",
            value=500,
            min_value=0,
            step=100,
        )

        # ---- DC Pension Settings ----
        st.subheader("Defined Contribution (DC) Pension")
        dc_starting = st.number_input(
            "DC pension starting value (£)",
            value=100000,
            min_value=0,
            step=1000,
        )
        dc_monthly = st.number_input(
            "DC pension monthly contribution (£)",
            value=1000,
            min_value=0,
            step=100,
        )
        dc_access_age = st.number_input(
            "DC pension access age",
            value=57,
            min_value=55,
            max_value=75,
        )

        # ---- DB Pension Toggle ----
        st.subheader("DB Pension (Optional)")
        has_db = st.checkbox("I have a Defined Benefit (DB) pension", value=False)
        db_normal_age = 67  # Default
        db_annual_income = 0.0
        if has_db:
            db_normal_age = st.number_input(
                "DB pension normal retirement age",
                value=67,
                min_value=55,
                max_value=75,
            )
            db_annual_income = st.number_input(
                "DB pension annual income (£, in today's money)",
                value=20000,
                min_value=1,
                step=1000,
            )

        # ---- Drawdown Settings ----
        st.subheader("Drawdown Strategy")
        drawdown_mode = st.radio(
            "Drawdown mode",
            options=["amount", "percentage"],
            help="Amount: fixed £ per year. Percentage: % of total portfolio.",
        )

        if drawdown_mode == "amount":
            annual_amount = st.number_input(
                "Annual drawdown (£, in today's money)",
                value=30000,
                min_value=1,
                step=1000,
            )
            annual_percentage = 4.0  # Safe default for percentage mode if not used
        else:
            annual_percentage = st.number_input(
                "Annual drawdown (% of portfolio)",
                value=4.0,
                min_value=0.1,
                max_value=10.0,
                step=0.1,
            )
            annual_amount = 30000  # Safe default for amount mode if not used

        # ---- Return Assumptions ----
        st.subheader("Return Assumptions")
        st.info("""
        The simulator calculates mean returns from historical UK stock data.
        You can adjust this multiplier to account for different market expectations.

        **Default (0.5x):** Conservative, more realistic long-term forecast.
        **1.0x:** Use historical data as-is (may be optimistic if recent bull market).
        """)
        mu_multiplier = st.slider(
            "Mean return adjustment (multiplier of data)",
            min_value=0.25,
            max_value=1.50,
            value=0.50,
            step=0.05,
            help="0.5x = half the historical average. Good for conservative projections.",
        )

        # ---- Tax Settings (Hardcoded for MVP) ----
        with st.expander("Tax Settings (UK 2024/25 defaults)"):
            st.info("""
            Tax bands are preset to UK 2024/25 rates:
            - Standard rate: 20% on gains in GIA
            - Pension withdrawal: 40% income tax
            - ISA: Tax-free

            Custom tax rates are not supported in this version.
            """)

    # Build the full config dict
    config = {
        "simulation": {
            "n_simulations": n_sims,
            "random_seed": random_seed if random_seed > 0 else None,
        },
        "user": {
            "current_age": current_age,
            "retirement_age": retirement_age,
            "life_expectancy": life_expectancy,
        },
        "tax_bands": {
            # UK 2024/25 defaults (same as config.yaml)
            "pa": 12570,
            "basic_rate_threshold": 50270,
            "higher_rate_threshold": 125140,
            "rates": [0.20, 0.40, 0.45],
        },
        "gia": {
            "starting_value": gia_starting,
            "monthly_contribution": gia_monthly,
            "real_contribution_growth": 0.0,
            "annual_limit": None,  # No limit for GIA
        },
        "isa": {
            "starting_value": isa_starting,
            "monthly_contribution": isa_monthly,
            "real_contribution_growth": 0.0,
            "annual_limit": 20000,  # HMRC limit
        },
        "pensions": [
            {
                "id": "main_pension",
                "type": "DC",
                "dc": {
                    "starting_value": dc_starting,
                    "monthly_contribution": dc_monthly,
                    "annual_limit": 60000,  # HMRC pension allowance
                    "pension_access_age": dc_access_age,
                },
            }
        ],
        "drawdown": {
            "mode": drawdown_mode,
            "annual_amount": annual_amount,
            "percentage": annual_percentage,
        },
        "lump_sums": [],  # Not supported in Streamlit MVP
        "jump_diffusion": {
            # Use defaults from config.yaml
            "jump_threshold_sigma": 2.0,
            "lambda_annual": 0.10,
            "jump_mean": -0.40,
            "jump_std": 0.10,
            "jump_clip_lower": -0.60,
            "jump_clip_upper": -0.05,
        },
        "data": {
            "prices_file": "data/VWRP Dailies - Sheet1.csv",
            "cpih_file": "data/series-280326.csv",
        },
    }

    # Add DB pension if enabled
    if has_db:
        config["pensions"].append({
            "id": "db_pension",
            "type": "DB",
            "db": {
                "normal_pension_age": db_normal_age,
                "projected_annual_income": db_annual_income,
            },
        })

    return config, mu_multiplier


# ============================================================================
# Pipeline & Results Rendering
# ============================================================================

def run_pipeline(config, mu_multiplier=1.0):
    """
    Run the full simulation pipeline: load data, build DB schedule, run simulation,
    analyse results. Returns (results_dict, db_schedule, market_params).

    Parameters
    ----------
    config : dict
        Full configuration dict from build_config_from_sidebar()
    mu_multiplier : float, default 1.0
        Adjustment factor for mean return (mu). If 0.5, uses half the historical average.
    """
    try:
        # Load market data (cached)
        market_params = load_market_data()

        # Adjust mean return based on user's multiplier
        market_params["mu_d"] = market_params["mu_d"] * mu_multiplier
        market_params["mu_d_original"] = load_market_data()["mu_d"]  # Store original for display
        market_params["mu_multiplier"] = mu_multiplier

        # Build DB income schedule
        db_schedule = build_db_income_schedule(config)

        # Build tax bands
        tax_bands = TaxBands.from_config(config)

        # Run simulation
        n_sims = config["simulation"]["n_simulations"]
        with st.spinner(f"Running {n_sims} Monte Carlo simulations... (this may take ~10-30s)"):
            portfolio_values, ruin_flags, ruin_ages, ages, gia_accum, isa_accum, dc_accum = run_simulation(
                config, market_params, db_schedule, tax_bands
            )

        # Analyse results
        results = analyse(
            portfolio_values,
            ruin_flags,
            ruin_ages,
            ages,
            config,
            gia_accum=gia_accum,
            isa_accum=isa_accum,
            dc_accum=dc_accum,
        )

        return results, db_schedule, market_params

    except Exception as e:
        st.error(f"❌ Simulation failed: {str(e)}")
        st.write("**Traceback:**")
        st.code(traceback.format_exc())
        return None, None, None


def render_results(results, config, db_schedule, market_params):
    """
    Display charts, summary stats, and assumptions for completed simulation.
    Generates matplotlib figures and closes them after display to prevent memory leaks.
    """
    try:
        # ---- Charts ----
        st.subheader("📊 Accumulation Phase (Saving Years)")
        fig_accum = generate_accumulation_chart(results, config, db_schedule, return_fig=True)
        st.pyplot(fig_accum, use_container_width=True)
        plt.close(fig_accum)  # Prevent memory leak

        st.subheader("📉 Drawdown Phase (Retirement Years)")
        fig_draw = generate_drawdown_chart(results, config, db_schedule, return_fig=True)
        st.pyplot(fig_draw, use_container_width=True)
        plt.close(fig_draw)  # Prevent memory leak

        # ---- Key Metrics ----
        col1, col2, col3, col4, col5 = st.columns(5)
        retirement_age = config["user"]["retirement_age"]
        idx_retirement = np.argmin(np.abs(results["all_ages"] - retirement_age))

        with col1:
            median_val = results["values_at_retirement"][50]
            st.metric("Median at Retirement", f"£{median_val:,.0f}")

        with col2:
            p5_val = results["values_at_retirement"][5]
            st.metric("5th Percentile", f"£{p5_val:,.0f}")

        with col3:
            p75_val = results["values_at_retirement"][75]
            st.metric("75th Percentile", f"£{p75_val:,.0f}")

        with col4:
            ruin_pct = results["ruin_probability"]
            st.metric("Ruin Probability", f"{ruin_pct:.1f}%")

        with col5:
            # Calculate annual withdrawal amount
            drawdown_config = config["drawdown"]
            if drawdown_config["mode"] == "percentage":
                withdrawal_pct = drawdown_config["percentage"]
                withdrawal_amount = median_val * (withdrawal_pct / 100)
                st.metric(
                    "Annual Withdrawal (Median)",
                    f"£{withdrawal_amount:,.0f}",
                    delta=f"{withdrawal_pct:.1f}% of portfolio"
                )
            else:
                withdrawal_amount = drawdown_config["annual_amount"]
                withdrawal_pct = (withdrawal_amount / median_val) * 100
                st.metric(
                    "Annual Withdrawal (Fixed)",
                    f"£{withdrawal_amount:,.0f}",
                    delta=f"{withdrawal_pct:.1f}% of median"
                )

        # ---- Percentile Table ----
        st.subheader("📋 Portfolio Values by Age (Percentiles)")

        # Sample every 5 years for readability
        all_ages = results["all_ages"]
        bands = results["percentile_bands"]

        sampled_indices = [i for i, age in enumerate(all_ages) if int(age) % 5 == 0]

        table_data = {
            "Age": [int(all_ages[i]) for i in sampled_indices],
            "1st %ile (£)": [f"{bands[1][i]:,.0f}" for i in sampled_indices],
            "5th %ile (£)": [f"{bands[5][i]:,.0f}" for i in sampled_indices],
            "50th %ile (£)": [f"{bands[50][i]:,.0f}" for i in sampled_indices],
            "75th %ile (£)": [f"{bands[75][i]:,.0f}" for i in sampled_indices],
        }
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # ---- DB Pension Income Table (if applicable) ----
        if db_schedule:
            st.subheader("💷 DB Pension Income (Annual)")
            db_data = {
                "Scheme": [d["pension_id"] for d in db_schedule],
                "Normal Pension Age": [int(d["normal_pension_age"]) for d in db_schedule],
                "Annual Income (£)": [f"£{d['monthly_income_gross']*12:,.0f}" for d in db_schedule],
            }
            st.dataframe(pd.DataFrame(db_data), use_container_width=True, hide_index=True)

        # ---- Assumptions Expander ----
        with st.expander("📋 Model Assumptions & Data"):
            col_left, col_right = st.columns(2)

            with col_left:
                st.write("**Simulation Parameters**")
                st.write(f"- Number of simulations: {config['simulation']['n_simulations']:,}")
                st.write(f"- Current age: {config['user']['current_age']}")
                st.write(f"- Retirement age: {config['user']['retirement_age']}")
                st.write(f"- Life expectancy: {config['user']['life_expectancy']}")

                st.write("**Jump-Diffusion Model**")
                jd = config["jump_diffusion"]
                st.write(f"- Annual jump probability: {jd['lambda_annual']*100:.0f}%")
                st.write(f"- Mean jump size: {jd['jump_mean']*100:.0f}%")
                st.write(f"- Jump std dev: {jd['jump_std']*100:.0f}%")
                st.write(f"- Clipping range: {jd['jump_clip_lower']*100:.0f}% to {jd['jump_clip_upper']*100:.0f}%")

                st.write("**Market Data**")
                st.write(f"- Returns calibrated from: UK stock market (VWRP)")
                st.write(f"- Inflation adjusted with: CPIH (UK Consumer Price Index)")
                if market_params:
                    orig_mu = market_params.get("mu_d_original", market_params["mu_d"])
                    mult = market_params.get("mu_multiplier", 1.0)
                    adj_mu = market_params["mu_d"]
                    st.write(f"- Mean monthly return (data): {orig_mu*100:.2f}% → adjusted: {adj_mu*100:.2f}% (×{mult:.2f})")
                    st.write(f"- Volatility (σ): {market_params['sigma_d']*100:.2f}%")

            with col_right:
                st.write("**Portfolio Setup**")
                st.write(f"- GIA starting: £{config['gia']['starting_value']:,.0f} / month: £{config['gia']['monthly_contribution']:,.0f}")
                st.write(f"- ISA starting: £{config['isa']['starting_value']:,.0f} / month: £{config['isa']['monthly_contribution']:,.0f}")

                for pen in config["pensions"]:
                    if pen["type"] == "DC":
                        dc = pen["dc"]
                        st.write(f"- DC starting: £{dc['starting_value']:,.0f} / month: £{dc['monthly_contribution']:,.0f}")
                        st.write(f"  Access age: {dc['pension_access_age']}")
                    elif pen["type"] == "DB":
                        db = pen["db"]
                        st.write(f"- DB pension: £{db['projected_annual_income']:,.0f}/yr @ age {db['normal_pension_age']}")

                st.write("**Drawdown**")
                if config["drawdown"]["mode"] == "amount":
                    annual_amount = config['drawdown']['annual_amount']
                    median_at_ret = results["values_at_retirement"][50]
                    rate = (annual_amount / median_at_ret * 100) if median_at_ret > 0 else 0
                    st.write(f"- Mode: Fixed amount")
                    st.write(f"- Annual withdrawal: £{annual_amount:,.0f}")
                    st.write(f"- Withdrawal rate (median): {rate:.2f}% of portfolio")
                else:
                    pct = config['drawdown']['percentage']
                    median_at_ret = results["values_at_retirement"][50]
                    amount = median_at_ret * (pct / 100)
                    st.write(f"- Mode: Percentage")
                    st.write(f"- Withdrawal rate: {pct:.1f}% of portfolio")
                    st.write(f"- Annual amount (median): £{amount:,.0f}")

                st.write("**Tax Assumptions**")
                st.write("- GIA: 20% CGT on gains only")
                st.write("- ISA: Tax-free")
                st.write("- Pension: 40% income tax on withdrawals")

        st.success("✅ Simulation complete!")

    except Exception as e:
        st.error(f"Error rendering results: {str(e)}")
        st.code(traceback.format_exc())


# ============================================================================
# Main App
# ============================================================================

def main():
    st.title("🎯 Monte Carlo Retirement Simulator")
    st.markdown("""
    All values shown are in **real terms (today's money)**, adjusted for inflation using CPIH.

    This simulator uses UK historical stock market returns and applies a jump-diffusion model
    to stress-test your retirement portfolio across accumulation and drawdown phases.
    """)

    # Build config from sidebar
    config, mu_multiplier = build_config_from_sidebar()

    # Validate config
    validation_errors = []
    validation_warnings = []
    try:
        errors, warnings = validate(config)
        validation_errors = errors or []
        validation_warnings = warnings or []
    except SystemExit:
        # validator.py may call SystemExit on critical errors; catch it
        validation_errors = ["Configuration validation failed. Please check your inputs."]

    # Display validation messages
    if validation_errors:
        st.error("⚠️ **Configuration Errors:**")
        for err in validation_errors:
            st.write(f"- {err}")

    if validation_warnings:
        st.warning("⚠️ **Configuration Warnings:**")
        for warn in validation_warnings:
            st.write(f"- {warn}")

    # Run Simulation Button
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        run_button = st.button(
            "🚀 Run Simulation",
            disabled=len(validation_errors) > 0,
            use_container_width=True,
        )

    with col2:
        if st.button("🔄 Reset Results", use_container_width=True):
            st.session_state.sim_results = None
            st.rerun()

    # Run simulation if button clicked
    if run_button:
        results, db_schedule, market_params = run_pipeline(config, mu_multiplier)
        if results:
            st.session_state.sim_results = (results, config, db_schedule, market_params)

    # Display stored results
    if st.session_state.sim_results:
        results, cfg, db_sched, mkt_params = st.session_state.sim_results
        render_results(results, cfg, db_sched, mkt_params)
    else:
        # Show placeholder
        st.info(
            "👈 Adjust settings in the sidebar, then click **Run Simulation** to see results."
        )


if __name__ == "__main__":
    main()
