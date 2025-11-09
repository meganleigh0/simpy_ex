# ----- ONE-CELL STREAMLIT DASHBOARD -----
# Run with: streamlit run this_file.py
# Data requirements: columns -> DATE, CHG#, COST-SET in {ACWP, BCWP, BCWS, ETC}, HOURS

import sys, types, os
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="XM30 EV Dashboard", layout="wide")

# ----------------------------- Helpers -----------------------------
COST_ORDER = ["ACWP","BCWP","BCWS","ETC"]

def coalesce_cost_columns(df, value_col):
    """Pivot CHG# x COST-SET -> columns ACWP, BCWP, BCWS, ETC with zeros for missing."""
    pt = (df.pivot_table(index="CHG#", columns="COST-SET", values=value_col,
                         aggfunc="sum", fill_value=0)
            .reindex(columns=COST_ORDER, fill_value=0))
    pt["TOTAL"] = pt.sum(axis=1, numeric_only=True)
    return pt

def ev_rollup(df):
    """Return totals (single row) ACWP, BCWP, BCWS, ETC and derived EV metrics."""
    s = (df.pivot_table(index=None, columns="COST-SET", values="HOURS",
                        aggfunc="sum", fill_value=0)
           .reindex(COST_ORDER, fill_value=0))
    totals = {k: float(s.get(k, pd.Series([0])).iloc[0] if k in s else 0.0) for k in COST_ORDER}

    ACWP = totals.get("ACWP",0.0)
    BCWP = totals.get("BCWP",0.0)
    BCWS = totals.get("BCWS",0.0)
    ETC  = totals.get("ETC", 0.0)

    # Earned value metrics
    SPI = np.nan if BCWS == 0 else BCWP/BCWS
    CPI = np.nan if ACWP == 0 else BCWP/ACWP
    SV  = BCWP - BCWS
    SVp = np.nan if BCWS == 0 else 100*SV/BCWS
    CV  = BCWP - ACWP
    CVp = np.nan if BCWP == 0 else 100*CV/BCWP

    # BAC/EAC/VAC
    BAC = BCWS                                    # common baseline = cumulative BCWS
    EAC = ACWP + ETC                              # simple EAC model
    VAC = BAC - EAC
    VACp = np.nan if BAC == 0 else 100*VAC/BAC

    # Remaining work & TCPI
    BCWR = BAC - BCWP
    # TCPI using EAC definition above: (BAC-EV)/(EAC-AC)
    denom = (EAC - ACWP)
    TCPI = np.nan if denom == 0 else (BAC - BCWP)/denom

    metrics = dict(ACWP=ACWP, BCWP=BCWP, BCWS=BCWS, ETC=ETC,
                   SPI=SPI, CPI=CPI, SV=SV, SVp=SVp, CV=CV, CVp=CVp,
                   BAC=BAC, EAC=EAC, VAC=VAC, VACp=VACp, BCWR=BCWR, TCPI=TCPI)
    return metrics

def format_money(x): 
    return f"{x:,.2f}"

def format_ratio(x):
    return "—" if pd.isna(x) else f"{x:0.2f}"

def format_pct(x):
    return "—" if pd.isna(x) else f"{x:0.2f}"

def week_start(d):  # Monday-start weeks
    return (d - pd.to_timedelta(d.dt.dayofweek, unit="D"))

# ----------------------------- Data load -----------------------------
if "xm30_cobra_export_weekly_extract" in globals():
    df = globals()["xm30_cobra_export_weekly_extract"].copy()
else:
    st.info("Upload the weekly COBRA export (CSV or Excel) if it’s not already loaded as a DataFrame named `xm30_cobra_export_weekly_extract`.")
    up = st.file_uploader("Upload CSV/Excel", type=["csv","xlsx","xls"])
    if not up:
        st.stop()
    if up.name.lower().endswith(".csv"):
        df = pd.read_csv(up)
    else:
        df = pd.read_excel(up)

# Normalize columns
rename_map = {c.lower(): c for c in df.columns}
# Ensure expected labels exist exactly:
required = ["DATE","CHG#","COST-SET","HOURS"]
# try light normalization if needed
for col in list(df.columns):
    if col.strip().upper() in {"DATE","CHG#","COST-SET","HOURS"} and col != col.strip().upper():
        df.rename(columns={col: col.strip().upper()}, inplace=True)
if not set(required).issubset(df.columns):
    st.error(f"Missing required columns. Found {list(df.columns)}; need {required}")
    st.stop()

# Types
df["DATE"] = pd.to_datetime(df["DATE"])
df["COST-SET"] = df["COST-SET"].astype(str).str.upper()
df["COST-SET"] = df["COST-SET"].replace({"BCW S":"BCWS"})  # light cleanup if needed

# ----------------------------- Controls -----------------------------
st.sidebar.header("Filters")
all_chg = sorted(df["CHG#"].astype(str).unique().tolist())
sel_chg = st.sidebar.multiselect("PLUG / CHG# (optional)", options=all_chg, default=all_chg)
min_d, max_d = df["DATE"].min().date(), df["DATE"].max().date()
as_of = st.sidebar.date_input("As-of date", value=max_d, min_value=min_d, max_value=max_d)
as_of = pd.Timestamp(as_of)

# Period selection
st.sidebar.subheader("Period options")
weeks_back = st.sidebar.number_input("4-Week window length", min_value=1, max_value=12, value=4, step=1)

# Current Status Period = week containing as_of (Mon start)
df["WEEK_START"] = week_start(df["DATE"])
status_week_start = week_start(pd.Series([as_of]))[0]
status_week_end = status_week_start + pd.Timedelta(days=6)

# Apply CHG# filter
df = df[df["CHG#"].astype(str).isin(sel_chg)]

# ----------------------------- Build period slices -----------------------------
# CUM = up to as_of
df_cum = df[df["DATE"] <= as_of]

# 4WK window ending as_of (inclusive)
fourwk_start = as_of - pd.Timedelta(weeks=weeks_back) + pd.Timedelta(days=1)
df_4wk = df[(df["DATE"] >= fourwk_start) & (df["DATE"] <= as_of)]

# Current Status Period = the accounting week containing as_of
df_status = df[(df["WEEK_START"] == status_week_start)]

# ----------------------------- Header -----------------------------
st.title("XM30 Earned Value Dashboard")
st.caption(f"As-of: **{as_of.date()}**, Current Week: **{status_week_start.date()} – {status_week_end.date()}**")
st.write("Columns used: `DATE`, `CHG#`, `COST-SET` ∈ {ACWP, BCWP, BCWS, ETC}, `HOURS`.")

# ----------------------------- Summary blocks -----------------------------
def summary_block(label, dfx):
    m = ev_rollup(dfx)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{label} • ACWP", format_money(m["ACWP"]))
    c2.metric(f"{label} • BCWP", format_money(m["BCWP"]))
    c3.metric(f"{label} • BCWS", format_money(m["BCWS"]))
    c4.metric(f"{label} • ETC",  format_money(m["ETC"]))
    return m

with st.container():
    st.subheader("CUM (to As-of)")
    m_cum = summary_block("CUM", df_cum)

with st.container():
    st.subheader(f"Last {weeks_back} Weeks")
    m_4wk = summary_block("4 Week", df_4wk)

with st.container():
    st.subheader("Current Status Period (week of As-of)")
    m_stat = summary_block("Status", df_status)

# ----------------------------- EV Indices table -----------------------------
def metrics_table(label, m):
    rows = [
        ("SPI",  format_ratio(m["SPI"])),
        ("CPI",  format_ratio(m["CPI"])),
        ("SV",   format_money(m["SV"])),
        ("SV%",  format_pct(m["SVp"])),
        ("CV",   format_money(m["CV"])),
        ("CV%",  format_pct(m["CVp"])),
        ("BAC",  format_money(m["BAC"])),
        ("EAC",  format_money(m["EAC"])),
        ("VAC",  format_money(m["VAC"])),
        ("VAC%", format_pct(m["VACp"])),
        ("BCWR", format_money(m["BCWR"])),
        ("ETC",  format_money(m["ETC"])),
        ("TCPI", format_ratio(m["TCPI"])),
    ]
    dfm = pd.DataFrame(rows, columns=["Metric", label])
    return dfm

left, mid, right = st.columns(3)
with left:
    st.markdown("#### EV Metrics — CUM")
    st.table(metrics_table("CUM", m_cum).set_index("Metric"))
with mid:
    st.markdown("#### EV Metrics — 4 Wk")
    st.table(metrics_table("4 Wk", m_4wk).set_index("Metric"))
with right:
    st.markdown("#### EV Metrics — Current Status")
    st.table(metrics_table("Status", m_stat).set_index("Metric"))

# ----------------------------- Detail pivots by CHG# -----------------------------
st.markdown("---")
st.subheader("Detail by CHG#")

tab1, tab2, tab3 = st.tabs(["CUM", "4 Wk", "Status"])
with tab1:
    st.dataframe(coalesce_cost_columns(df_cum, "HOURS"))
with tab2:
    st.dataframe(coalesce_cost_columns(df_4wk, "HOURS"))
with tab3:
    st.dataframe(coalesce_cost_columns(df_status, "HOURS"))

# ----------------------------- Optional charts -----------------------------
st.markdown("---")
st.subheader("Trends (Weekly Totals)")

# Build weekly totals by COST-SET
weekly = (df[df["DATE"] <= as_of]
          .groupby([ "WEEK_START","COST-SET"], as_index=False)["HOURS"].sum())
weekly = weekly[weekly["COST-SET"].isin(COST_ORDER)]

import plotly.express as px
line = px.line(weekly, x="WEEK_START", y="HOURS", color="COST-SET",
               category_orders={"COST-SET": COST_ORDER},
               labels={"WEEK_START":"Week", "HOURS":"Hours"})
st.plotly_chart(line, use_container_width=True)

# ----------------------------- Export buttons -----------------------------
st.download_button("Download CUM by CHG# (CSV)",
                   coalesce_cost_columns(df_cum, "HOURS").reset_index().to_csv(index=False).encode(),
                   file_name=f"cum_by_plug_{as_of.date()}.csv")

st.download_button("Download 4 Wk by CHG# (CSV)",
                   coalesce_cost_columns(df_4wk, "HOURS").reset_index().to_csv(index=False).encode(),
                   file_name=f"4wk_by_plug_{as_of.date()}.csv")

st.download_button("Download Status Week by CHG# (CSV)",
                   coalesce_cost_columns(df_status, "HOURS").reset_index().to_csv(index=False).encode(),
                   file_name=f"statusweek_by_plug_{as_of.date()}.csv")
# ----- end one cell -----