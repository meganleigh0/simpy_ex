import pandas as pd
import numpy as np
import plotly.graph_objects as go
from IPython.display import display

# -------------------------------------------------------------------
# 0) Load Cobra data (uses existing DATA_PATH, SHEET_NAME, ANCHOR)
# -------------------------------------------------------------------
xl = pd.ExcelFile(DATA_PATH)
cobra = xl.parse(SHEET_NAME)

cobra["DATE"] = pd.to_datetime(cobra["DATE"], errors="coerce")
cobra = cobra[cobra["DATE"].notna()].copy()

# -------------------------------------------------------------------
# 1) PROGRAM MANPOWER TABLE  (Program total, labeled "SHC")
# -------------------------------------------------------------------

# 9/80 available hours table from OpPlan (2024–2028)
available_9_80 = {
    2024: [142, 160, 196, 156, 160, 191, 151, 160, 191, 160, 151, 155],
    2025: [124, 160, 200, 160, 160, 191, 152, 160, 191, 191, 160, 173],
    2026: [147, 160, 200, 160, 151, 195, 156, 160, 191, 160, 151, 151],
    2027: [147, 160, 192, 160, 151, 190, 151, 160, 191, 160, 151, 160],
    2028: [151, 160, 200, 160, 160, 191, 151, 160, 191, 160, 142, 160],
}

def get_available_hours(year, month):
    try:
        return available_9_80[year][month - 1]
    except KeyError:
        return np.nan

# Monthly totals for the *whole* program (no SUB_TEAM filter)
month_totals = (
    cobra.groupby([cobra["DATE"].dt.to_period("M"), "COST-SET"])["HOURS"]
    .sum()
    .unstack(fill_value=0.0)
)
# convert PeriodIndex -> Timestamp (end of month)
month_totals.index = month_totals.index.to_timestamp("M")

# ensure required cost-sets exist
for k in ["BCWS", "ACWP", "ETC", "BCWP"]:
    if k not in month_totals.columns:
        month_totals[k] = 0.0

# current month = last month <= ANCHOR that exists in the data
valid_idx = month_totals.index[month_totals.index <= ANCHOR]
if len(valid_idx) == 0:
    # no data before ANCHOR – just make an empty row
    program_manpower_tbl = pd.DataFrame(
        {
            "SUB_TEAM": ["SHC"],
            "Demand": [np.nan],
            "Actual": [np.nan],
            "Next Month BCWS": [np.nan],
            "Next Month ETC": [np.nan],
        }
    )
else:
    cur_period = valid_idx.max()

    # next month = first month > current that exists (if any)
    future_idx = month_totals.index[month_totals.index > cur_period]
    next_period = future_idx.min() if len(future_idx) > 0 else None

    cur = month_totals.loc[cur_period]
    if next_period is not None:
        nxt = month_totals.loc[next_period]
    else:
        # if there is no future month in Cobra, treat as zeros
        nxt = pd.Series({"BCWS": 0.0, "ACWP": 0.0, "ETC": 0.0})

    cur_year, cur_month = cur_period.year, cur_period.month
    cur_avail = get_available_hours(cur_year, cur_month)

    if next_period is not None:
        next_year, next_month = next_period.year, next_period.month
        next_avail = get_available_hours(next_year, next_month)
    else:
        next_avail = np.nan

    # Convert hours -> FTE (headcount)
    demand = cur["BCWS"] / cur_avail if cur_avail and not np.isnan(cur_avail) else np.nan
    actual = cur["ACWP"] / cur_avail if cur_avail and not np.isnan(cur_avail) else np.nan
    next_bcws_fte = (
        nxt["BCWS"] / next_avail if next_avail and not np.isnan(next_avail) else np.nan
    )
    next_etc_fte = (
        nxt["ETC"] / next_avail if next_avail and not np.isnan(next_avail) else np.nan
    )

    program_manpower_tbl = pd.DataFrame(
        {
            "SUB_TEAM": ["SHC"],  # label like the dashboard
            "Demand": [round(demand, 1)],
            "Actual": [round(actual, 1)],
            "Next Month BCWS": [round(next_bcws_fte, 1)],
            "Next Month ETC": [round(next_etc_fte, 1)],
        }
    )

# -------------------------------------------------------------------
# 2) EVMS TABLE (Monthly & Cumulative CPI / SPI) – using same month_totals
# -------------------------------------------------------------------
ev = month_totals[month_totals.index <= ANCHOR].copy().sort_index()

ev["CPI_month"] = np.where(ev["ACWP"] == 0, np.nan, ev["BCWP"] / ev["ACWP"])
ev["SPI_month"] = np.where(ev["BCWS"] == 0, np.nan, ev["BCWP"] / ev["BCWS"])

ev["BCWP_cum"] = ev["BCWP"].cumsum()
ev["ACWP_cum"] = ev["ACWP"].cumsum()
ev["BCWS_cum"] = ev["BCWS"].cumsum()

ev["CPI_cum"] = np.where(ev["ACWP_cum"] == 0, np.nan, ev["BCWP_cum"] / ev["ACWP_cum"])
ev["SPI_cum"] = np.where(ev["BCWS_cum"] == 0, np.nan, ev["BCWP_cum"] / ev["BCWS_cum"])

evms_tbl = ev[["CPI_month", "SPI_month", "CPI_cum", "SPI_cum"]].copy()
evms_tbl["Month"] = evms_tbl.index.to_period("M").strftime("%b-%y")
evms_tbl = evms_tbl.reset_index(drop=True)

# -------------------------------------------------------------------
# 3) EVMS PLOTLY CHART
# -------------------------------------------------------------------
fig = go.Figure()

if not evms_tbl.empty:
    # performance bands
    fig.add_hrect(y0=0.90, y1=0.95, fillcolor="#ff4d4d", opacity=0.3, line_width=0, layer="below")
    fig.add_hrect(y0=0.95, y1=1.00, fillcolor="#ffd633", opacity=0.3, line_width=0, layer="below")
    fig.add_hrect(y0=1.00, y1=1.10, fillcolor="#66cc66", opacity=0.3, line_width=0, layer="below")
    fig.add_hrect(y0=1.10, y1=1.20, fillcolor="#99ccff", opacity=0.3, line_width=0, layer="below")

    x_vals = evms_tbl["Month"]

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=evms_tbl["CPI_month"],
            mode="markers",
            name="Monthly CPI",
            marker=dict(symbol="diamond", size=10, color="#f4b183"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=evms_tbl["SPI_month"],
            mode="markers",
            name="Monthly SPI",
            marker=dict(symbol="circle", size=10, color="#000000"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=evms_tbl["CPI_cum"],
            mode="lines",
            name="Cumulative CPI",
            line=dict(color="#0050b3", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=evms_tbl["SPI_cum"],
            mode="lines",
            name="Cumulative SPI",
            line=dict(color="#7f7f7f", width=3),
        )
    )

    fig.update_layout(
        title="EV Indices",
        xaxis_title="Month",
        yaxis_title="Index",
        yaxis=dict(range=[0.9, 1.2]),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=60, b=80),
    )

# -------------------------------------------------------------------
# 4) SHOW TABLES + CHART
# -------------------------------------------------------------------
print("Program Manpower (SHC):")
display(program_manpower_tbl)

print("\nEVMS Indices Table:")
display(evms_tbl[["Month", "CPI_month", "SPI_month", "CPI_cum", "SPI_cum"]])

if not evms_tbl.empty:
    fig.show()
else:
    print("\nNo EVMS data to plot.")