# ==========================================================
# UNIVERSAL EVMS PLOT GENERATOR FOR ANY COBRA EXPORT
# Works with Abrams STS 2022, Abrams STS, XM30, SEP, etc.
# ==========================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# ---------------------------------------
# User-configurable paths
# ---------------------------------------
cobra_files = {
    "Abrams_STS_2022": "data/Cobra-Abrams STS 2022.xlsx",
    "Abrams_STS": "data/Cobra-Abrams STS.xlsx",
    "XM30": "data/Cobra-XM30.xlsx"
}

output_dir = "EVMS_Output"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------
# Helper: Auto-standardize cost-set column names
# ---------------------------------------
def normalize_columns(df):
    df = df.rename(columns={c: c.strip().upper().replace(" ", "").replace("-", "") for c in df.columns})
    
    # Map variations → standard labels
    column_map = {
        "COSTSET": "COST_SET",
        "COSTSET": "COST_SET",
        "DATE": "DATE",
        "HOURS": "HOURS",
        "SUBTEAM": "SUB_TEAM",
        "SUBTEAM": "SUB_TEAM"
    }
    
    for old, new in column_map.items():
        for c in df.columns:
            if old in c:
                df = df.rename(columns={c: new})
    
    return df


# ---------------------------------------
# Helper: Compute EV indices
# ---------------------------------------
def compute_ev_metrics(df):
    # Require these fields
    needed = ["COST_SET", "DATE"]
    if not all(col in df.columns for col in needed):
        raise ValueError("Missing COST_SET or DATE column after normalization.")

    # Pivot cost sets into columns
    pivot = df.pivot_table(
        index="DATE",
        columns="COST_SET",
        values="HOURS",   # Cobra exports usually store values in HOURS column
        aggfunc="sum"
    ).reset_index()

    # Standardize expected cost sets
    cols = pivot.columns
    BCWS = pivot[cols[cols.str.contains("BCWS", case=False)][0]]
    BCWP = pivot[cols[cols.str.contains("BCWP", case=False)][0]]
    ACWP = pivot[cols[cols.str.contains("ACWP", case=False)][0]]

    # Build metrics table
    out = pd.DataFrame()
    out["DATE"] = pivot["DATE"]

    out["Monthly CPI"] = BCWP / ACWP
    out["Monthly SPI"] = BCWP / BCWS

    # Cumulative
    out["Cumulative CPI"] = BCWP.cumsum() / ACWP.cumsum()
    out["Cumulative SPI"] = BCWP.cumsum() / BCWS.cumsum()

    return out


# ---------------------------------------
# Helper: Plot EVMS
# ---------------------------------------
def create_evms_plot(program, evdf):
    dates = evdf["DATE"]
    
    fig = go.Figure()

    # -----------------------------------
    # Background colored performance bands
    # -----------------------------------
    fig.add_hrect(y0=0.90, y1=0.97, fillcolor="red", opacity=0.35, line_width=0)
    fig.add_hrect(y0=0.97, y1=1.00, fillcolor="yellow", opacity=0.35, line_width=0)
    fig.add_hrect(y0=1.00, y1=1.07, fillcolor="green", opacity=0.35, line_width=0)
    fig.add_hrect(y0=1.07, y1=1.20, fillcolor="lightblue", opacity=0.35, line_width=0)

    # -----------------------------------
    # Monthly CPI (diamond)
    # -----------------------------------
    fig.add_trace(go.Scatter(
        x=dates, y=evdf["Monthly CPI"],
        mode="markers",
        marker=dict(symbol="diamond", size=10, color="yellow"),
        name="Monthly CPI"
    ))

    # Monthly SPI (dot)
    fig.add_trace(go.Scatter(
        x=dates, y=evdf["Monthly SPI"],
        mode="markers",
        marker=dict(size=10, color="black"),
        name="Monthly SPI"
    ))

    # -----------------------------------
    # Cumulative CPI (blue line)
    # -----------------------------------
    fig.add_trace(go.Scatter(
        x=dates, y=evdf["Cumulative CPI"],
        mode="lines", line=dict(width=4, color="blue"),
        name="Cumulative CPI"
    ))

    # Cumulative SPI (grey line)
    fig.add_trace(go.Scatter(
        x=dates, y=evdf["Cumulative SPI"],
        mode="lines", line=dict(width=4, color="darkgrey"),
        name="Cumulative SPI"
    ))

    fig.update_layout(
        title=f"{program} — EVMS Indices",
        xaxis_title="Month",
        yaxis_title="EV Indices",
        yaxis=dict(range=[0.90, 1.20]),
        template="simple_white",
        height=600,
    )

    return fig


# ---------------------------------------
# MAIN LOOP — Process each COBRA file
# ---------------------------------------

for program, path in cobra_files.items():

    print(f"Processing {program} ...")

    df = pd.read_excel(path)
    df = normalize_columns(df)
    evdf = compute_ev_metrics(df)

    fig = create_evms_plot(program, evdf)

    fig.write_image(f"{output_dir}/{program}_EVMS.png", scale=3)
    fig.write_html(f"{output_dir}/{program}_EVMS.html")

    print(f"✔ Saved EVMS plot for {program}")

print("ALL EVMS PLOTS COMPLETE ✓")