# ONE-CELL STREAMLIT DASHBOARD (robust loader)
import os, sys, types
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="XM30 EV Dashboard", layout="wide")

# ---------------- Helpers ----------------
COST_ORDER = ["ACWP","BCWP","BCWS","ETC"]

def week_start(d):  # Monday-week; change offset if your cadence is different
    return (d - pd.to_timedelta(d.dt.dayofweek, unit="D"))

def coalesce_cost_columns(df, value_col):
    pt = (df.pivot_table(index="CHG#", columns="COST-SET", values=value_col,
                         aggfunc="sum", fill_value=0)
            .reindex(columns=COST_ORDER, fill_value=0))
    pt["TOTAL"] = pt.sum(axis=1, numeric_only=True)
    return pt

def ev_rollup(df):
    s = (df.pivot_table(index=None, columns="COST-SET", values="HOURS",
                        aggfunc="sum", fill_value=0)
           .reindex(COST_ORDER, fill_value=0))
    # pull floats safely
    getv = lambda k: (float(s[k].iloc[0]) if k in s else 0.0)
    ACWP, BCWP, BCWS, ETC = (getv("ACWP"), getv("BCWP"), getv("BCWS"), getv("ETC"))

    SPI = np.nan if BCWS == 0 else BCWP/BCWS
    CPI = np.nan if ACWP == 0 else BCWP/ACWP
    SV  = BCWP - BCWS
    SVp = np.nan if BCWS == 0 else 100*SV/BCWS
    CV  = BCWP - ACWP
    CVp = np.nan if BCWP == 0 else 100*CV/BCWP

    BAC = BCWS
    EAC = ACWP + ETC
    VAC = BAC - EAC
    VACp = np.nan if BAC == 0 else 100*VAC/BAC

    BCWR = BAC - BCWP
    denom = (EAC - ACWP)
    TCPI = np.nan if denom == 0 else (BAC - BCWP)/denom

    return dict(ACWP=ACWP, BCWP=BCWP, BCWS=BCWS, ETC=ETC,
                SPI=SPI, CPI=CPI, SV=SV, SVp=SVp, CV=CV, CVp=CVp,
                BAC=BAC, EAC=EAC, VAC=VAC, VACp=VACp, BCWR=BCWR, TCPI=TCPI)

fmt_money = lambda x: f"{x:,.2f}"
fmt_ratio = lambda x: "—" if pd.isna(x) else f"{x:0.2f}"
fmt_pct   = lambda x: "—" if pd.isna(x) else f"{x:0.2f}"

def metrics_table(label, m):
    rows = [
        ("SPI",  fmt_ratio(m["SPI"])),
        ("CPI",  fmt_ratio(m["CPI"])),
        ("SV",   fmt_money(m["SV"])),
        ("SV%",  fmt_pct(m["SVp"])),
        ("CV",   fmt_money(m["CV"])),
        ("CV%",  fmt_pct(m["CVp"])),
        ("BAC",  fmt_money(m["BAC"])),
        ("EAC",  fmt_money(m["EAC"])),
        ("VAC",  fmt_money(m["VAC"])),
        ("VAC%", fmt_pct(m["VACp"])),
        ("BCWR", fmt_money(m["BCWR"])),
        ("ETC",  fmt_money(m["ETC"])),
        ("TCPI", fmt_ratio(m["TCPI"]))
    ]
    return pd.DataFrame(rows, columns=["Metric", label]).set_index("Metric")

# -------------- Data load (resilient) --------------
st.sidebar.header("Data")
df = None

# 1) If a DataFrame named exactly below exists (e.g., when launching from a notebook kernel), use it.
if "xm30_cobra_export_weekly_extract" in globals():
    df = globals()["xm30_cobra_export_weekly_extract"].copy()

# 2) Or let the user upload a file:
uploaded = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv","xlsx","xls"])

# 3) Or type a local path (optional, for your Windows path use \\ or raw string)
path_hint = st.sidebar.text_input("…or enter a local file path (CSV/XLSX)")

if df is None:
    if uploaded is not None:
        # safe guard BEFORE using uploaded.name
        name = uploaded.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            # allow sheet choice
            xls = pd.ExcelFile(uploaded)
            sheet = st.sidebar.selectbox("Excel sheet", xls.sheet_names, index=0)
            df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
    elif path_hint:
        if not os.path.exists(path_hint):
            st.error(f"Path not found: {path_hint}")
            st.stop()
        if path_hint.lower().endswith(".csv"):
            df = pd.read_csv(path_hint)
        else:
            xls = pd.ExcelFile(path_hint)
            sheet = st.sidebar.selectbox("Excel sheet", xls.sheet_names, index=0)
            df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
    else:
        st.info("Load a DataFrame named `xm30_cobra_export_weekly_extract`, or upload a file, or enter a path.")
        st.stop()

# Normalize expected columns (accept a few common variants)
colmap = {c.upper().replace(" ", "").replace("_",""): c for c in df.columns}
def pick(*aliases):
    for a in aliases:
        k = a.upper().replace(" ","").replace("_","")
        if k in colmap: return colmap[k]
    return None

col_date   = pick("DATE", "WEEKDATE", "ASOFDATE")
col_chg    = pick("CHG#", "CHG", "PLUG", "CONTROL_ACCOUNT", "CONTROLACCT")
col_cost   = pick("COST-SET", "COST_SET", "COSTSET")
col_hours  = pick("HOURS", "QTY", "VALUE")

need = [col_date, col_chg, col_cost, col_hours]
if any(x is None for x in need):
    st.error(f"Missing required columns. Found: {list(df.columns)}. "
             f"Need date/CHG#/COST-SET/HOURS (common aliases supported).")
    st.stop()

# Canonicalize names
df = df.rename(columns={col_date:"DATE", col_chg:"CHG#", col_cost:"COST-SET", col_hours:"HOURS"})
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df = df.dropna(subset=["DATE"])
df["COST-SET"] = df["COST-SET"].astype(str).str.upper().str.replace(" ", "")
# tidy up one-off typos
df["COST-SET"] = df["COST-SET"].replace({"BCW S":"BCWS", "BCW":"BCWP"})  # extend if needed
df["WEEK_START"] = week_start(df["DATE"])

# ---------------- UI controls ----------------
st.sidebar.header("Filters")
all_chg = sorted(df["CHG#"].astype(str).unique().tolist())
sel_chg = st.sidebar.multiselect("PLUG / CHG#", options=all_chg, default=all_chg)

min_d, max_d = df["DATE"].min().date(), df["DATE"].max().date()
as_of = st.sidebar.date_input("As-of date", value=max_d, min_value=min_d, max_value=max_d)
as_of = pd.Timestamp(as_of)

weeks_back = st.sidebar.number_input("Rolling window (weeks)", min_value=1, max_value=12, value=4, step=1)

status_week_start = week_start(pd.Series([as_of]))[0]
status_week_end   = status_week_start + pd.Timedelta(days=6)

df = df[df["CHG#"].astype(str).isin(sel_chg)]

# ---------------- Period slices ----------------
df_cum   = df[df["DATE"] <= as_of]
fourwk_start = as_of - pd.Timedelta(weeks=weeks_back) + pd.Timedelta(days=1)
df_4wk  = df[(df["DATE"] >= fourwk_start) & (df["DATE"] <= as_of)]
df_stat = df[(df["WEEK_START"] == status_week_start)]

# ---------------- Header ----------------
st.title("XM30 Earned Value Dashboard")
st.caption(f"As-of **{as_of.date()}**  |  Current Status Week: **{status_week_start.date()} – {status_week_end.date()}**")
st.write("Using columns: `DATE`, `CHG#`, `COST-SET` (ACWP/BCWP/BCWS/ETC), `HOURS`.")

# ---------------- Summary cards ----------------
def summary_block(label, data):
    m = ev_rollup(data)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric(f"{label} • ACWP", fmt_money(m["ACWP"]))
    c2.metric(f"{label} • BCWP", fmt_money(m["BCWP"]))
    c3.metric(f"{label} • BCWS", fmt_money(m["BCWS"]))
    c4.metric(f"{label} • ETC",  fmt_money(m["ETC"]))
    return m

st.subheader("CUM through As-of")
m_cum = summary_block("CUM", df_cum)

st.subheader(f"Last {weeks_back} Weeks")
m_4wk = summary_block("4 Week", df_4wk)

st.subheader("Current Status Period (Accounting Week of As-of)")
m_stat = summary_block("Status", df_stat)

# ---------------- Metrics tables ----------------
left, mid, right = st.columns(3)
with left:
    st.markdown("#### EV Metrics — CUM")
    st.table(metrics_table("CUM", m_cum))
with mid:
    st.markdown("#### EV Metrics — 4 Wk")
    st.table(metrics_table("4 Wk", m_4wk))
with right:
    st.markdown("#### EV Metrics — Status")
    st.table(metrics_table("Status", m_stat))

# ---------------- Detail pivots ----------------
st.markdown("---")
st.subheader("Detail by CHG#")
t1, t2, t3 = st.tabs(["CUM", "4 Wk", "Status"])
with t1: st.dataframe(coalesce_cost_columns(df_cum, "HOURS"))
with t2: st.dataframe(coalesce_cost_columns(df_4wk, "HOURS"))
with t3: st.dataframe(coalesce_cost_columns(df_stat, "HOURS"))

# ---------------- Trend chart ----------------
st.markdown("---")
st.subheader("Weekly Totals Trend")
weekly = (df[df["DATE"] <= as_of]
          .groupby(["WEEK_START","COST-SET"], as_index=False)["HOURS"].sum())
weekly = weekly[weekly["COST-SET"].isin(COST_ORDER)]
fig = px.line(weekly, x="WEEK_START", y="HOURS", color="COST-SET",
              category_orders={"COST-SET": COST_ORDER},
              labels={"WEEK_START":"Week", "HOURS":"Hours"})
st.plotly_chart(fig, use_container_width=True)