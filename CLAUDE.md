# Monte Carlo Retirement Portfolio Simulator — Project Specification

## Project Overview

A Python-based Monte Carlo simulation tool that models the long-term real (inflation-adjusted)
growth and drawdown of a retirement portfolio across three investment wrappers: GIA, ISA, and
Pension. Returns are derived from historical stock price data adjusted for inflation (CPIH).
Black swan shocks are overlaid to stress-test the portfolio.

All portfolio values are expressed in **real terms (today's money)** throughout — accumulation
and drawdown phases alike.

---

## Repository Structure

```
project/
├── main.py                  # Entry point — runs simulation and generates outputs
├── config.yaml              # All user-defined inputs (see spec below)
├── simulation.py            # Core Monte Carlo engine
├── wrappers.py              # GIA / ISA / Pension wrapper logic (tax, limits, contributions)
├── drawdown.py              # Drawdown logic (order, tax, ruin detection)
├── analysis.py              # Percentile calculations and ruin probability
├── visualisation.py         # Chart generation (accumulation + drawdown)
├── data_loader.py           # CSV ingestion and return series preparation
├── data/
│   ├── stock_prices.csv     # Monthly stock price growth rates (user provided)
│   └── cpih.csv             # Annualised CPIH inflation data (user provided)
└── outputs/
    ├── accumulation_chart.png
    ├── drawdown_chart.png
    └── summary_stats.csv
```

---

## Configuration File: `config.yaml`

All user-facing inputs live here. The simulation reads this at runtime.

```yaml
simulation:
  n_simulations: 10000          # Number of Monte Carlo paths
  random_seed: 42               # Set for reproducibility; set to null for random

user:
  current_age: 35               # User's current age
  retirement_age: 65            # Age at which drawdown begins
  pension_access_age: 60        # Minimum age to access pension wrapper
  life_expectancy: 100          # Age used for ruin probability calculation

data:
  stock_prices_file: "data/stock_prices.csv"
  cpih_file: "data/cpih.csv"

wrappers:
  gia:
    starting_value: 50000       # Current GIA portfolio value (£)
    monthly_contribution: 500   # Monthly contribution during accumulation (£, pre-inflation-adj)
    annual_limit: null          # No enforced annual contribution limit for GIA
    tax_rate_on_gains: 0.20     # 20% CGT applied to gains portion only on withdrawal

  isa:
    starting_value: 20000       # Current ISA portfolio value (£)
    monthly_contribution: 500   # Monthly contribution during accumulation (£, pre-inflation-adj)
    annual_limit: 20000         # HMRC annual ISA allowance (£)
    tax_rate_on_gains: 0.0      # Tax-free

  pension:
    starting_value: 100000      # Current pension pot value (£)
    monthly_contribution: 1000  # Monthly contribution (£, pre-inflation-adj)
    annual_limit: 60000         # HMRC annual pension allowance (£)
    tax_rate_on_withdrawal: 0.40  # Income tax on pension withdrawals

drawdown:
  mode: "amount"                # "amount" (fixed £) or "percentage" (% of portfolio)
  annual_amount: 30000          # Used if mode = "amount" (£, in today's money)
  percentage: 4.0               # Used if mode = "percentage" (e.g. 4.0 = 4% of total portfolio)
  order: ["GIA", "ISA", "Pension"]  # Wrapper depletion order — do not change without understanding ruin logic

black_swan:
  annual_probability: 0.10      # 10% chance of a shock event each year (~once per decade)
  shock_mean: -0.40             # Mean shock magnitude (-40%, grounded in 1987/2008/2020 history)
  shock_std: 0.10               # Std of shock (-1 sigma = -30%, +1 sigma = -50%)
  # Shock is sampled from N(shock_mean, shock_std) and clipped to [-0.80, -0.10]
  # Shocks apply equally across all wrappers simultaneously
```

---

## Input Data Format

### `stock_prices.csv`
Monthly stock price growth rates (NOT price levels). Column format:

```
date,monthly_return
1990-01-01,0.0234
1990-02-01,-0.0187
...
```
- `date`: Any parseable date string (monthly frequency)
- `monthly_return`: Decimal return for that month (e.g. 0.02 = 2%)
- Dividends are assumed to be included in the return series (total return index)

### `cpih.csv`
Annualised CPIH inflation rates. Column format:

```
date,annual_cpih
1990-01-01,0.065
1991-01-01,0.053
...
```
- `date`: Year or year-month reference
- `annual_cpih`: Decimal annual inflation rate (e.g. 0.03 = 3%)
- Converted to monthly rate internally: `monthly_cpih = (1 + annual_cpih)^(1/12) - 1`

---

## Module Specifications

### `data_loader.py`

**Responsibilities:**
- Load and validate both CSVs
- Convert annualised CPIH to monthly inflation rate
- Calculate real monthly return series:
  `real_return = (1 + nominal_return) / (1 + monthly_cpih) - 1`
- Compute `mu` (mean) and `sigma` (std dev) of the real monthly return series
- Log mu and sigma to console for transparency

**Key output:**
```python
{
  "mu": float,       # Mean real monthly return
  "sigma": float,    # Std dev of real monthly return
  "monthly_cpih_series": pd.Series  # For inflation-adjusting contributions
}
```

---

### `simulation.py`

**Responsibilities:**
Core Monte Carlo engine. Runs `n_simulations` paths from `current_age` to `life_expectancy`.

**Algorithm — per simulation path:**

1. **Initialise** wrapper balances from config (`starting_value` for GIA, ISA, Pension)
2. **Track cost basis** for GIA separately (to calculate gains on withdrawal)
3. **For each month** during accumulation (`current_age` → `retirement_age`):
   a. Sample real monthly return: `r ~ N(mu, sigma)`
   b. Check for black swan shock (annual probability check each January):
      - If shock occurs: `shock ~ N(shock_mean, shock_std)`, clipped to `[-0.80, -0.10]`
      - Apply shock multiplicatively to all wrapper balances simultaneously
      - Black swan shocks are treated as a nominal event (no additional inflation adjustment)
   c. Apply return to all wrapper balances: `balance *= (1 + r)`
   d. Apply inflation-adjusted contributions to each wrapper:
      - `contribution_t = base_contribution * cumulative_inflation_factor`
      - Enforce annual limits: if `year_to_date_contribution + contribution_t > annual_limit`,
        cap contribution to remaining allowance (excess is simply not invested in that wrapper)
      - Update GIA cost basis by contribution amount added
   e. Since all values are in real terms: returns are real, and contributions are inflation-adjusted
      to maintain real purchasing power

4. **For each month** during drawdown (`retirement_age` → `life_expectancy`):
   a. Apply real return and black swan logic (same as accumulation)
   b. No further contributions during drawdown
   c. Calculate gross withdrawal required (inflation-adjusted `annual_amount / 12` or `%` of total)
   d. Withdraw from wrappers in order: GIA → ISA → Pension (subject to pension_access_age):
      - **GIA withdrawal**: Calculate gain proportion = `(balance - cost_basis) / balance`
        Apply tax: `net_withdrawal = gross * (1 - tax_rate * gain_proportion)` — i.e. tax is
        only on the gains portion. Reduce cost basis proportionally.
      - **ISA withdrawal**: No tax. `net_withdrawal = gross`
      - **Pension withdrawal**: Only accessible if `current_age >= pension_access_age`.
        Apply 40% tax: `net_withdrawal = gross / (1 - 0.40)` — gross up to get required
        pre-tax withdrawal, then deduct tax.
      - **Ruin condition**: If GIA and ISA are both £0 AND current age < pension_access_age,
        mark simulation as **ruined** at this month and stop further drawdown for that path.
        If pension is accessible and all three are depleted, also mark as ruined.
   e. Record total portfolio value (GIA + ISA + Pension) each month for that path

5. **Store** monthly portfolio values per simulation as a `(n_simulations, n_months)` array

---

### `wrappers.py`

**Responsibilities:**
- Wrapper dataclass/class for GIA, ISA, Pension
- Tracks: `balance`, `cost_basis` (GIA only), `annual_contributions_ytd`
- Methods:
  - `apply_return(r)` — apply real monthly return
  - `apply_shock(shock)` — apply black swan shock, update cost basis proportionally for GIA
  - `contribute(amount, year)` — add contribution, enforce annual limit, update cost basis
  - `withdraw(gross_amount)` — apply tax logic, return net received, reduce balance/cost basis

---

### `drawdown.py`

**Responsibilities:**
- Orchestrates drawdown order across wrappers
- Handles pension age gate
- Returns: net cash received, ruin flag, updated wrapper states

---

### `analysis.py`

**Responsibilities:**
- Take `(n_simulations, n_months)` results array
- Compute percentile bands at each time step: `[1, 2.5, 5, 50, 75]`
  - Focus on downside risk (1st, 2.5th, 5th) plus median and 75th upside
- Compute **ruin probability**:
  - Count simulations where portfolio hits £0 before `life_expectancy`
  - Express as a percentage of total simulations
- Compute **median age of ruin** (for ruined simulations only)
- Compute **median per-wrapper balances** during accumulation (GIA / ISA / DC) for stacked chart
- Output summary stats to `outputs/summary_stats.csv`

---

### `visualisation.py`

**Responsibilities:**
Produce two charts, saved to `outputs/`.

**Chart 1 — Accumulation Phase** (`accumulation_chart.png`):
- X-axis: Age (current_age → retirement_age)
- Y-axis: Total real portfolio value (£, today's money)
- **Stacked area chart** showing median wrapper composition:
  - GIA (amber), ISA (indigo), Pension DC (teal) — stacked bottom to top
- Uncertainty band overlay: 1st–75th percentile range (grey, semi-transparent)
- Dashed median total line overlaid on the stack
- Annotate retirement values at 5th, 50th, 75th percentiles
- Title: "Portfolio Growth — Accumulation Phase (Real Terms)"

**Chart 2 — Drawdown Phase** (`drawdown_chart.png`):
- X-axis: Age (retirement_age → life_expectancy)
- Y-axis: Total real portfolio value (£, today's money)
- Asymmetric percentile fan bands: 1–75 (lightest), 2.5–75 (middle), 5–75 (darkest)
- Median (50th) as solid line
- Vertical dashed line at DC pension_access_age (orange) and DB normal_pension_age (purple)
- Annotate ruin probability prominently (top-right text box), colour-coded by severity:
  `"Probability of Ruin: X.X%"`
- Title: "Portfolio Drawdown — Drawdown Phase (Real Terms)"

---

### `main.py`

Entry point. Orchestrates the full pipeline:

```
1. Load config.yaml
2. Load and preprocess data (data_loader.py)
3. Log mu, sigma, and CPIH assumptions to console
4. Run Monte Carlo simulation (simulation.py)
5. Analyse results (analysis.py)
6. Generate charts (visualisation.py)
7. Print summary to console:
   - Median portfolio at retirement
   - 2.5th and 97.5th percentile at retirement
   - Probability of ruin
   - Median age of ruin (if applicable)
```

---

## Key Assumptions & Design Decisions

| Assumption | Value / Rationale |
|---|---|
| All values in real terms | Removes inflation noise; shows purchasing power in today's money |
| Real return = nominal − CPIH | `(1 + nominal) / (1 + monthly_cpih) − 1` |
| Contributions inflation-adjusted | Maintains constant real purchasing power of contributions |
| Black swan probability | 10% per year (~once per decade, consistent with 1987/2008/2020 history) |
| Black swan mean shock | −40% (mean of historical major crashes: −33% to −57%) |
| Black swan shock std | 10% (1σ range: −30% to −50%) |
| Black swan shock clip | Clipped to [−80%, −10%] to prevent extreme outliers |
| GIA tax | 20% on gains portion only (proportional cost basis method) |
| Pension tax | 40% on gross withdrawal (salary-sacrifice contributions, no relief modelled) |
| ISA tax | None |
| Drawdown order | GIA → ISA → Pension (pension gated by pension_access_age) |
| Ruin definition | Portfolio = £0 before age 100 (life_expectancy in config) |
| Dividends | Included in historical return series; not modelled separately |
| State pension | Not modelled |
| Contribution limits | ISA £20k/yr, Pension £60k/yr; excess is silently uncapped for GIA |


---

# V2 — Streamlit Web Application

## Overview

V2 wraps the existing Python simulation in a Streamlit web interface, making it shareable with
friends via a public URL. The core simulation modules (`simulation.py`, `wrappers.py`,
`drawdown.py`, `analysis.py`, `data_loader.py`) are **unchanged**. V2 adds only a new
front-end entry point and minor refactoring to support in-memory config (replacing `config.yaml`).

Hosted for free on **Streamlit Community Cloud** (streamlit.io/cloud). No backend server,
no database, no user accounts.

---

## V2 Repository Structure

```
project/
├── app.py                   # NEW — Streamlit entry point (replaces main.py for web use)
├── main.py                  # Unchanged — still works as CLI entry point
├── config.yaml              # Unchanged — still used by main.py CLI
├── simulation.py            # Unchanged
├── wrappers.py              # Unchanged
├── drawdown.py              # Unchanged
├── analysis.py              # Unchanged
├── visualisation.py         # Minor change — functions return figures instead of saving to disk
├── data_loader.py           # Unchanged
├── data/
│   ├── stock_prices.csv     # Bundled — same file, committed to repo
│   └── cpih.csv             # Bundled — same file, committed to repo
├── requirements.txt         # NEW — lists Python dependencies for Streamlit Cloud deployment
└── outputs/                 # Unchanged — still used by CLI main.py
    ├── accumulation_chart.png
    ├── drawdown_chart.png
    └── summary_stats.csv
```

---

## New File: `app.py`

**Responsibilities:**
- Render the full Streamlit UI
- Collect all user inputs via sidebar widgets
- Build a config dict from those inputs (same structure as `config.yaml`)
- Call `data_loader`, `simulation`, `analysis`, and `visualisation` directly
- Display charts and summary stats inline in the page

**Page layout:**

```
┌─────────────────────────────────────────────────────────┐
│  Sidebar                  │  Main panel                 │
│  ─────────                │  ──────────                 │
│  [Simulation Settings]    │  Title + brief description  │
│  [User Settings]          │                             │
│  [GIA Settings]           │  [Run Simulation] button    │
│  [ISA Settings]           │                             │
│  [Pension Settings]       │  Chart 1: Accumulation      │
│  [Drawdown Settings]      │  Chart 2: Drawdown          │
│  [Black Swan Settings]    │  Summary stats table        │
│                           │  Assumptions expander       │
└─────────────────────────────────────────────────────────┘
```

---

## Sidebar Input Groups

All inputs mirror `config.yaml` exactly. Grouped with `st.sidebar.header()` separators.

### Simulation Settings
| Field | Widget | Default | Notes |
|---|---|---|---|
| Number of simulations | `st.slider` | 1000 | Range: 100–10,000. Default lowered from 10k for speed |
| Random seed | `st.number_input` | 42 | Integer; allows reproducibility |

### User Settings
| Field | Widget | Default |
|---|---|---|
| Current age | `st.number_input` | 35 |
| Retirement age | `st.number_input` | 65 |
| Pension access age | `st.number_input` | 60 |
| Life expectancy | `st.number_input` | 100 |

### GIA Settings
| Field | Widget | Default |
|---|---|---|
| Starting value (£) | `st.number_input` | 50,000 |
| Monthly contribution (£) | `st.number_input` | 500 |
| CGT rate on gains | `st.slider` | 0.20 | Range: 0.0–0.45 |

### ISA Settings
| Field | Widget | Default |
|---|---|---|
| Starting value (£) | `st.number_input` | 20,000 |
| Monthly contribution (£) | `st.number_input` | 500 |
| Annual limit (£) | `st.number_input` | 20,000 |

### Pension Settings
| Field | Widget | Default |
|---|---|---|
| Starting value (£) | `st.number_input` | 100,000 |
| Monthly contribution (£) | `st.number_input` | 1,000 |
| Annual limit (£) | `st.number_input` | 60,000 |
| Tax rate on withdrawal | `st.slider` | 0.40 | Range: 0.0–0.60 |

### Drawdown Settings
| Field | Widget | Default | Notes |
|---|---|---|---|
| Drawdown mode | `st.radio` | "amount" | Options: "amount" / "percentage" |
| Annual amount (£) | `st.number_input` | 30,000 | Shown only if mode = "amount" |
| Percentage (%) | `st.number_input` | 4.0 | Shown only if mode = "percentage" |

### Black Swan Settings
| Field | Widget | Default |
|---|---|---|
| Annual probability | `st.slider` | 0.10 | Range: 0.0–0.50 |
| Mean shock magnitude | `st.slider` | -0.40 | Range: -0.80–-0.10 |
| Shock std dev | `st.slider` | 0.10 | Range: 0.01–0.30 |

---

## Main Panel

### On load (before simulation runs)
- App title: **"Monte Carlo Retirement Simulator"**
- One-paragraph description explaining what the tool does and that all values are in real (today's) money
- A note that data is based on bundled historical UK stock and CPIH inflation data
- Prominent **"Run Simulation"** button (`st.button`)

### After simulation runs
- `st.spinner("Running simulation...")` shown during computation
- **Chart 1** — Accumulation phase (`st.pyplot`)
- **Chart 2** — Drawdown phase (`st.pyplot`)
- **Summary stats table** (`st.dataframe`) with:
  - Median portfolio at retirement
  - 5th percentile portfolio at retirement
  - 75th percentile portfolio at retirement
  - Probability of ruin (%)
  - Median age of ruin (if any simulations ruined)
- **Assumptions expander** (`st.expander("See assumptions")`) — collapsible section listing
  all key assumptions from the assumptions table in this spec, so users understand the model

---

## Change to `visualisation.py`

The only required code change to existing modules. Currently `visualisation.py` saves charts
to disk with `plt.savefig()`. For Streamlit, functions must **return the matplotlib figure
object** instead of (or in addition to) saving to disk.

Proposed approach — add an optional `return_fig` parameter:

```python
def plot_accumulation(data, config, return_fig=False):
    fig, ax = plt.subplots(...)
    # ... existing chart logic ...
    if return_fig:
        return fig
    else:
        fig.savefig("outputs/accumulation_chart.png")
```

This keeps `main.py` CLI behaviour intact while allowing `app.py` to call
`plot_accumulation(..., return_fig=True)` and pass the figure to `st.pyplot(fig)`.

---

## New File: `requirements.txt`

Required for Streamlit Community Cloud to install dependencies:

```
streamlit
pandas
numpy
matplotlib
pyyaml
```

---

## Deployment: Streamlit Community Cloud

Step-by-step for deployment:

1. Push the repository to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click "New app" → select the repo → set entry point to `app.py`
4. Click "Deploy" — Streamlit Cloud installs `requirements.txt` automatically
5. A public URL is generated (e.g. `https://yourname-retirement-sim.streamlit.app`)
6. Share that URL with friends — no login required to use the tool

**Note on "sleeping":** Streamlit Community Cloud free tier puts apps to sleep after
~7 days of inactivity. The first visit after sleep takes ~30 seconds to wake up.
Subsequent visits are instant.

---

## V2 Scope Boundaries

| In scope | Out of scope |
|---|---|
| Streamlit UI wrapping all config fields | User accounts or saved scenarios |
| Bundled historical data (no upload) | CSV upload by users |
| Both charts rendered inline | Scenario comparison |
| Summary stats table | PDF/export of results |
| Assumptions expander | Mobile-optimised layout |
| Free Streamlit Cloud hosting | Custom domain |

---

## Future Enhancements (Out of Scope for v2)

- Shareable URL encoding of inputs (encode config as URL query params)
- Scenario comparison (run two configs side by side)
- CSV download of simulation output
- State pension income toggle
- Pension tax-free lump sum (25% PCLS)
- Mobile layout optimisation
