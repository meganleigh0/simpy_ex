# ==========================================================
# UNIVERSAL EVMS PLOT GENERATOR — Bulletproof Cost-Set Mapping
# ==========================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import re

# --- Cobra file dictionary ---
cobra_files = {
    "Abrams_STS_2022": "data/Cobra-Abrams STS 2022.xlsx",
    "Abrams_STS": "data/Cobra-Abrams STS.xlsx",
    "XM30": "data/Cobra-XM30.xlsx"
}

output_dir = "EVMS_Output"
os.makedirs(output_dir, exist_ok=True)


# ---------------------------------------
# Normalize column names
# ---------------------------------------
def normalize_columns(df):
    df = df.rename(columns={c: c.strip().upper().replace(" ", "").replace("-", "").replace("_", "") for c in df.columns})
    return df


# ---------------------------------------
# Fuzzy matcher for cost-set names
# ---------------------------------------
def fuzzy_find(column_list, patterns):
    """
    column_list: list of pivoted cost-set column names
    patterns: list of acceptable fuzzy matches for BCWS/BCWP/ACWP
    Returns first matching column or None
    """
    for col in column_list:
        clean = col.replace("_", "").replace("-", "").upper()
        for p in patterns:
            if re.search(p, clean):
                return col
    return None


# ---------------------------------------
# Compute EV metrics safely
# ---------------------------------------
def compute_ev_metrics(df):

    if "COSTSET" not in df.columns:
        raise ValueError("No COST-SET column detected. Inspect your file.")

    if "DATE" not in df.columns:
        raise ValueError("No DATE column detected. Inspect your file.")

    # Pivot cost-sets
    pivot = df.pivot_table(index="DATE", columns="COSTSET", values="HOURS", aggfunc="sum").reset_index()
    cost_cols = [c for c in pivot.columns if c != "DATE"]

    # Find cost-set names (fuzzy)
    bcws_col = fuzzy_find(cost_cols, ["BCWS", "BUDGETEDCOSTWORKSCHED", "PMBBCWS", "SCH"])
    bcwp_col = fuzzy_find(cost_cols, ["BCWP", "BUDGETEDCOSTWORKPERF", "PMBBCWP", "PERF"])
    acwp_col = fuzzy_find(cost_cols, ["ACWP", "ACTUALCOSTWORKPERF", "ACTUAL"])

    missing = []
    if bcws_col is None: missing.append("BCWS")
    if bcwp_col is None: missing.append("BCWP")
    if acwp_col is None: missing.append("ACWP")

    if missing:
        raise ValueError(f"Missing required cost sets: {missing}. Columns found: {cost_cols}")

    # Extract series
    BCWS = pivot[bcws_col].replace(0, np.nan)
    BCWP = pivot[bcwp_col].replace(0, np.nan)
    ACWP = pivot[acwp_col].replace(0, np.nan)

    # Create output metrics dataframe
    out = pd.DataFrame({"DATE": pivot["DATE"]})

    out["Monthly CPI"] = BCWP / ACWP
    out["Monthly SPI"] = BCWP / BCWS
    out["Cumulative CPI"] = BCWP.cumsum() / ACWP.cumsum()
    out["Cumulative SPI"] = BCWP.cumsum() / BCWS.cumsum()

    return out


# ---------------------------------------
# Build EVMS plot
# ---------------------------------------
def create_evms_plot(program, evdf):

    dates = evdf["DATE"]

    fig = go.Figure()

    # Background bands
    fig.add_hrect(y0=0.90, y1=0.97, fillcolor="red", opacity=0.25, line_width=0)
    fig.add_hrect(y0=0.97, y1=1.00, fillcolor="yellow", opacity=0.25, line_width=0)
    fig.add_hrect(y0=1.00, y1=1.07, fillcolor="green", opacity=0.25, line_width=0)
    fig.add_hrect(y0=1.07, y1=1.20, fillcolor="lightblue", opacity=0.25, line_width=0)

    # Monthly CPI
    fig.add_trace(go.Scatter(
        x=dates, y=evdf["Monthly CPI"],
        mode="markers",
        marker=dict(symbol="diamond", size=10, color="yellow"),
        name="Monthly CPI"
    ))

    # Monthly SPI
    fig.add_trace(go.Scatter(
        x=dates, y=evdf["Monthly SPI"],
        mode="markers",
        marker=dict(size=9, color="black"),
        name="Monthly SPI"
    ))

    # Cumulative CPI
    fig.add_trace(go.Scatter(
        x=dates, y=evdf["Cumulative CPI"],
        mode="lines",
        line=dict(color="blue", width=4),
        name="Cumulative CPI"
    ))

    # Cumulative SPI
    fig.add_trace(go.Scatter(
        x=dates, y=evdf["Cumulative SPI"],
        mode="lines",
        line=dict(color="darkgrey", width=4),
        name="Cumulative SPI"
    ))

    fig.update_layout(
        title=f"{program} EVMS Indices",
        xaxis_title="Month",
        yaxis_title="EV Index",
        yaxis=dict(range=[0.90, 1.20]),
        template="simple_white",
        height=600,
    )

    return fig


# ---------------------------------------
# Main loop
# ---------------------------------------
for program, path in cobra_files.items():

    print(f"Processing {program} …")

    df = pd.read_excel(path)
    df = normalize_columns(df)

    evdf = compute_ev_metrics(df)
    fig = create_evms_plot(program, evdf)

    fig.write_image(f"{output_dir}/{program}_EVMS.png", scale=3)
    fig.write_html(f"{output_dir}/{program}_EVMS.html")

    print(f"✔ Completed EVMS for {program}")

print("✔ ALL PROGRAMS COMPLETE")