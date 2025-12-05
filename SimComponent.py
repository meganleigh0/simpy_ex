# =====================================================================
# UNIVERSAL EVMS PLOT GENERATOR — AUTO-DETECTS ALL COST-SET FORMATS
# =====================================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re, os

cobra_files = {
    "Abrams_STS_2022": "data/Cobra-Abrams STS 2022.xlsx",
    "Abrams_STS": "data/Cobra-Abrams STS.xlsx",
    "XM30": "data/Cobra-XM30.xlsx"
}

output_dir = "EVMS_Output"
os.makedirs(output_dir, exist_ok=True)


def normalize_columns(df):
    df = df.rename(columns={c: c.strip().upper().replace(" ", "").replace("-", "").replace("_", "") for c in df.columns})
    return df


# ----------------------------------------------------------------------------
# AUTO-MAP COST-SET VALUES TO BCWS / BCWP / ACWP EVEN WHEN NAMES ARE WEIRD
# ----------------------------------------------------------------------------
def map_cost_sets(cost_cols):
    """
    Takes the COSTSET unique column values, automatically detects:
    - BCWS
    - BCWP
    - ACWP
    """

    # Clean text for fuzzy matching
    cleaned = {col: col.replace("_", "").replace("-", "").upper() for col in cost_cols}

    bcws = bcwp = acwp = None

    for orig, clean in cleaned.items():

        # ACWP Detection
        if ("ACWP" in clean) or ("ACTUAL" in clean) or ("ACWPHRS" in clean):
            acwp = orig
        
        # BCWS Detection
        elif ("BCWS" in clean) or ("BUDGET" in clean) or ("PLAN" in clean):
            bcws = orig
        
        # BCWP Detection
        elif ("BCWP" in clean) or ("PROGRESS" in clean) or ("EARNED" in clean):
            bcwp = orig

    return bcws, bcwp, acwp


# ----------------------------------------------------------------------------
# Compute EV metrics
# ----------------------------------------------------------------------------
def compute_ev_metrics(df):

    if "COSTSET" not in df.columns:
        raise ValueError("Missing COSTSET column.")

    pivot = df.pivot_table(index="DATE", columns="COSTSET", values="HOURS", aggfunc="sum").reset_index()
    cost_cols = [c for c in pivot.columns if c != "DATE"]

    bcws_col, bcwp_col, acwp_col = map_cost_sets(cost_cols)

    missing = []
    if bcws_col is None: missing.append("BCWS")
    if bcwp_col is None: missing.append("BCWP")
    if acwp_col is None: missing.append("ACWP")

    if missing:
        raise ValueError(f"Missing required cost sets: {missing}. Found: {cost_cols}")

    BCWS = pivot[bcws_col].replace(0, np.nan)
    BCWP = pivot[bcwp_col].replace(0, np.nan)
    ACWP = pivot[acwp_col].replace(0, np.nan)

    out = pd.DataFrame({"DATE": pivot["DATE"]})
    out["Monthly CPI"] = BCWP / ACWP
    out["Monthly SPI"] = BCWP / BCWS
    out["Cumulative CPI"] = BCWP.cumsum() / ACWP.cumsum()
    out["Cumulative SPI"] = BCWP.cumsum() / BCWS.cumsum()

    return out


# ----------------------------------------------------------------------------
# EVMS Plot Builder
# ----------------------------------------------------------------------------
def create_evms_plot(program, evdf):

    fig = go.Figure()

    # EV performance background zones
    fig.add_hrect(y0=0.90, y1=0.97, fillcolor="red", opacity=0.25, line_width=0)
    fig.add_hrect(y0=0.97, y1=1.00, fillcolor="yellow", opacity=0.25, line_width=0)
    fig.add_hrect(y0=1.00, y1=1.07, fillcolor="green", opacity=0.25, line_width=0)
    fig.add_hrect(y0=1.07, y1=1.20, fillcolor="lightblue", opacity=0.25, line_width=0)

    fig.add_trace(go.Scatter(x=evdf["DATE"], y=evdf["Monthly CPI"],
                             mode="markers",
                             marker=dict(symbol="diamond", size=10, color="yellow"),
                             name="Monthly CPI"))

    fig.add_trace(go.Scatter(x=evdf["DATE"], y=evdf["Monthly SPI"],
                             mode="markers",
                             marker=dict(size=9, color="black"),
                             name="Monthly SPI"))

    fig.add_trace(go.Scatter(x=evdf["DATE"], y=evdf["Cumulative CPI"],
                             mode="lines",
                             line=dict(color="blue", width=4),
                             name="Cumulative CPI"))

    fig.add_trace(go.Scatter(x=evdf["DATE"], y=evdf["Cumulative SPI"],
                             mode="lines",
                             line=dict(color="darkgrey", width=4),
                             name="Cumulative SPI"))

    fig.update_layout(
        title=f"{program} EVMS Trend",
        xaxis_title="Month",
        yaxis_title="EV Index",
        yaxis=dict(range=[0.90, 1.20]),
        template="simple_white",
        height=600
    )

    return fig


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
for program, path in cobra_files.items():

    print(f"Processing {program} …")

    df = pd.read_excel(path)
    df = normalize_columns(df)

    evdf = compute_ev_metrics(df)
    fig = create_evms_plot(program, evdf)

    fig.write_image(f"{output_dir}/{program}_EVMS.png", scale=3)
    fig.write_html(f"{output_dir}/{program}_EVMS.html")

    print(f"✔ Completed EVMS for {program}")

print("ALL PROGRAM EVMS COMPLETE ✓")