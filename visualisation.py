"""visualisation.py — Chart generation for accumulation and drawdown phases.

Accumulation chart: stacked area by wrapper (GIA / ISA / DC pension) showing
median composition, with a shaded uncertainty band (1st–75th percentile range).

Drawdown chart: percentile fan chart with ruin probability annotation.

Percentile set: [1, 2.5, 5, 50, 75] — focused on downside risk.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# Colour palette
BLUE       = "#2196F3"   # base blue for drawdown bands
DARK_BLUE  = "#0D47A1"   # median line
GIA_COLOR  = "#FF9800"   # amber  — GIA
ISA_COLOR  = "#5C6BC0"   # indigo — ISA
DC_COLOR   = "#26A69A"   # teal   — DC pension


def _fmt(v):
    """Format a £ value compactly: £1.2M, £450K, £999."""
    if v >= 1e6:
        return f"£{v/1e6:.1f}M"
    if v >= 1e3:
        return f"£{v/1e3:.0f}K"
    return f"£{v:.0f}"


def _style(ax, title, xl, yl):
    """Apply consistent axis styling."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xl, fontsize=11)
    ax.set_ylabel(yl, fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt(x)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0)


def _bands(ax, ages, bands):
    """
    Draw percentile fan bands and median line for the drawdown chart.

    Bands are asymmetric — focused on downside risk:
      1st–75th  (lightest): outer range
      2.5th–75th           : middle band
      5th–75th  (darkest) : inner band
      50th                : median line
    """
    ax.fill_between(ages, bands[1],   bands[75], alpha=0.10, color=BLUE, label="1–75th pct")
    ax.fill_between(ages, bands[2.5], bands[75], alpha=0.16, color=BLUE, label="2.5–75th pct")
    ax.fill_between(ages, bands[5],   bands[75], alpha=0.24, color=BLUE, label="5–75th pct")
    ax.plot(ages, bands[50], color=DARK_BLUE, linewidth=2, label="Median")


def generate_accumulation_chart(
    results,
    config,
    db_schedule,
    output_path="outputs/accumulation_chart.png",
    return_fig: bool = False,
):
    """
    Stacked area chart showing how GIA, ISA, and DC pension grow during
    accumulation, based on median simulation paths.

    A shaded band (1st–75th percentile of total portfolio) overlays the stack
    to communicate outcome uncertainty.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    aa  = results["accum_ages"]
    ab  = results["accum_bands"]
    ra  = float(config["user"]["retirement_age"])
    ns  = int(config["simulation"]["n_simulations"])

    gia_med = results.get("accum_gia_median")
    isa_med = results.get("accum_isa_median")
    dc_med  = results.get("accum_dc_median")

    fig, ax = plt.subplots(figsize=(11, 6))

    if gia_med is not None and isa_med is not None and dc_med is not None:
        # Stacked area: median wrapper composition
        ax.stackplot(
            aa,
            gia_med, isa_med, dc_med,
            labels=["GIA", "ISA", "Pension (DC)"],
            colors=[GIA_COLOR, ISA_COLOR, DC_COLOR],
            alpha=0.80,
        )

        # Uncertainty band: 1st–75th percentile of total portfolio
        ax.fill_between(
            aa, ab[1], ab[75],
            alpha=0.14, color="grey",
            label="1–75th pct range",
        )
        # Median total as a dashed reference line
        ax.plot(
            aa, ab[50],
            color="#1a1a2e", linewidth=1.5, linestyle="--",
            alpha=0.6, label="Median total",
        )
    else:
        # Fallback: plain percentile bands if wrapper data unavailable
        _bands(ax, aa, ab)

    _style(ax, "Portfolio Growth — Accumulation Phase (Real Terms)", "Age", "Portfolio Value (£)")

    # Annotate retirement values at the right edge
    for pct, lbl, off in [(75, "75th", 0.05), (50, "Median", 0.0), (5, "5th", -0.05)]:
        val = float(ab[pct][-1])
        if val > 0:
            ax.annotate(
                f"{lbl}: {_fmt(val)}",
                xy=(ra, val), xytext=(ra - 2, val * (1 + off)),
                fontsize=8, color=DARK_BLUE, ha="right",
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.8),
            )

    # Annotate DB pension income at retirement (if applicable)
    if db_schedule:
        tdb = sum(d["monthly_income_gross"] for d in db_schedule)
        if tdb > 0:
            ax.annotate(
                "DB at ret: " + _fmt(tdb) + "/mo",
                xy=(ra, float(ab[50][-1])),
                xytext=(ra * 0.72, float(ab[75][-1]) * 0.82),
                fontsize=9, color="#388E3C",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", alpha=0.8),
                arrowprops=dict(arrowstyle="->", color="#388E3C", lw=1.2),
            )

    ax.legend(loc="upper left", fontsize=9, framealpha=0.7)
    fig.text(
        0.5, -0.01,
        f"Based on {ns:,} simulations | Jump-Diffusion | Real terms | Stacked = median paths",
        ha="center", fontsize=8, color="grey",
    )
    plt.tight_layout()
    if return_fig:
        # Streamlit mode: return the figure object so st.pyplot(fig) can render it.
        # Caller is responsible for closing the figure after display.
        return fig
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Accumulation chart saved: {output_path}")


def generate_drawdown_chart(
    results,
    config,
    db_schedule,
    output_path="outputs/drawdown_chart.png",
    return_fig: bool = False,
):
    """
    Percentile fan chart for the drawdown phase, with:
      - Vertical lines at DC pension access age(s) and DB normal pension age(s)
      - Ruin probability annotation (colour-coded by severity)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    da  = results["draw_ages"]
    db2 = results["draw_bands"]
    rp  = results["ruin_probability"]
    mr  = results["median_ruin_age"]
    ns  = int(config["simulation"]["n_simulations"])

    fig, ax = plt.subplots(figsize=(11, 6))
    _bands(ax, da, db2)
    _style(ax, "Portfolio Drawdown — Drawdown Phase (Real Terms)", "Age", "Portfolio Value (£)")

    # Vertical lines at DC pension access ages
    seen = set()
    for pen in config.get("pensions", []):
        if pen.get("type") in ("DC", "hybrid"):
            access_age = float(pen.get("dc", {}).get("pension_access_age", 57))
            if access_age not in seen and da[0] <= access_age <= da[-1]:
                ax.axvline(x=access_age, color="#F57C00", linestyle="--", lw=1.4, alpha=0.8)
                ax.text(
                    access_age + 0.3, ax.get_ylim()[1] * 0.95,
                    "DC " + str(int(access_age)), fontsize=8, color="#F57C00", va="top",
                )
                seen.add(access_age)

    # Vertical lines at DB normal pension ages
    for d in db_schedule:
        npa = float(d["normal_pension_age"])
        if da[0] <= npa <= da[-1]:
            ax.axvline(x=npa, color="#7B1FA2", linestyle=":", lw=1.4, alpha=0.8)
            ax.text(
                npa + 0.3, ax.get_ylim()[1] * 0.80,
                "DB " + str(int(npa)), fontsize=8, color="#7B1FA2", va="top",
            )

    # Ruin probability box — colour by severity
    rc  = "#C62828" if rp > 10 else ("#F57C00" if rp > 5 else "#388E3C")
    bc2 = "#FFEBEE" if rp > 10 else ("#FFF3E0" if rp > 5 else "#E8F5E9")
    rt  = f"Probability of Ruin: {rp:.1f}%"
    if mr:
        rt += f" | Median ruin age: {mr:.0f}"
    ax.annotate(
        rt,
        xy=(0.97, 0.95), xycoords="axes fraction",
        fontsize=11, fontweight="bold", color=rc, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=bc2, edgecolor=rc, alpha=0.9),
    )

    ax.legend(loc="upper right", fontsize=9, framealpha=0.7, bbox_to_anchor=(0.97, 0.72))
    fig.text(
        0.5, -0.01,
        f"Based on {ns:,} simulations | Jump-Diffusion | Real terms",
        ha="center", fontsize=8, color="grey",
    )
    plt.tight_layout()
    if return_fig:
        # Streamlit mode: return the figure object so st.pyplot(fig) can render it.
        # Caller is responsible for closing the figure after display.
        return fig
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Drawdown chart saved:     {output_path}")


def generate_all_charts(results, config, db_schedule):
    print("Generating charts...")
    generate_accumulation_chart(results, config, db_schedule)
    generate_drawdown_chart(results, config, db_schedule)
    print("  Done.")
