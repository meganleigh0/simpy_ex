# ====================== ONE-CELL EV DASHBOARD (no Streamlit) ======================
# Works in a Jupyter cell or plain Python. Produces summary tables, pivots, and a chart.
# Requires: pandas, numpy, plotly (for the chart; you can remove the chart if not installed)

import pandas as pd, numpy as np, io, sys, os, datetime as dt

# ------------------- User-configurable defaults -------------------
AS_OF = None                 # e.g., dt.date(2025, 10, 25). If None, uses max DATE in data
WEEK_START_DAY = "MON"       # One of: "MON","TUE","WED","THU","FRI","SAT","SUN"
FOUR_WK_LEN = 4              # rolling window length in weeks for the "4 Week" block
EXPORT_EXCEL = False         # True to export an Excel with all tables at the end
EXPORT_PATH = "ev_dashboard_output.xlsx"

# ------------------- Load data (robust) -------------------
def _load_df():
    # 1) If an in-memory DataFrame exists, use it
    g = globals()
    if "xm30_cobra_export_weekly_extract" in g and isinstance(g["xm30_cobra_export_weekly_extract"], pd.DataFrame):
        return g["xm30_cobra_export_weekly_extract"].copy()

    # 2) Otherwise, open a file picker
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(title="Select CSV or Excel", filetypes=[("Data files","*.csv *.xlsx *.xls")])
        if not path:
            raise RuntimeError("No file selected and no in-memory DataFrame found.")
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        else:
            # If multiple sheets, use the first by default
            xls = pd.ExcelFile(path)
            return pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    except Exception as e:
        raise RuntimeError(f"Could not load data. {e}")

df = _load_df()

# ------------------- Normalize columns -------------------
# We expect DATE, CHG#, COST-SET, HOURS exactly (case-insensitive, allow spaces/underscores)
norm = {c: c.strip().upper().replace(" ", "").replace("_","") for c in df.columns}
df.rename(columns={old: new for old,new in zip(df.columns, [norm[c] for c in df.columns])}, inplace=True)

rename_map = {
    "DATE":"DATE",
    "CHG#":"CHG#",
    "CHG":"CHG#",
    "PLUG":"CHG#",
    "COST-SET":"COST-SET",
    "COSTSET":"COST-SET",
    "HOURS":"HOURS"
}

# Rebuild a second map from normalized keys back to required names
rev = {k.strip().upper().replace(" ","").replace("_",""): v for k,v in rename_map.items()}
fixed_cols = {}
for c in df.columns:
    key = c  # already normalized
    if key in rev:
        fixed_cols[c] = rev[key]
    else:
        fixed_cols[c] = c  # keep as-is

df.rename(columns=fixed_cols, inplace=True)

required = ["DATE","CHG#","COST-SET","HOURS"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

# Types & cleaning
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df = df.dropna(subset=["DATE"]).copy()
df["COST-SET"] = df["COST-SET"].astype(str).str.upper().str.strip()
df["HOURS"] = pd.to_numeric(df["HOURS"], errors="coerce").fillna(0.0)

# ------------------- Period helpers -------------------
DAY_TO_NUM = dict(MON=0,TUE=1,WED=2,THU=3,FRI=4,SAT=5,SUN=6)
if WEEK_START_DAY.upper() not in DAY_TO_NUM:
    raise ValueError("WEEK_START_DAY must be one of MON,TUE,WED,THU,FRI,SAT,SUN")
anchor = DAY_TO_NUM[WEEK_START_DAY.upper()]

def week_start(series_dt):
    # shift so that anchor becomes day 0
    # Example: if anchor=2 (WED), then Wednesday is week start
    dow = series_dt.dt.dayofweek
    offset = (dow - anchor) % 7
    return series_dt - pd.to_timedelta(offset, unit="D")

df["WEEK_START"] = week_start(df["DATE"])

# Determine AS_OF
if AS_OF is None:
    AS_OF = df["DATE"].max().date()
AS_OF = pd.Timestamp(AS_OF)

status_week_start = week_start(pd.Series([AS_OF]))[0]
status_week_end   = status_week_start + pd.Timedelta(days=6)

# Slices
df_cum   = df[df["DATE"] <= AS_OF]
fourwk_start = AS_OF - pd.Timedelta(weeks=FOUR_WK_LEN) + pd.Timedelta(days=1)
df_4wk   = df[(df["DATE"] >= fourwk_start) & (df["DATE"] <= AS_OF)]
df_stat  = df[df["WEEK_START"] == status_week_start]

# ------------------- EV math -------------------
COST_ORDER = ["ACWP","BCWP","BCWS","ETC"]

def coalesce_cost_columns(dfx, value_col="HOURS"):
    pt = (dfx.pivot_table(index="CHG#", columns="COST-SET", values=value_col,
                          aggfunc="sum", fill_value=0)
            .reindex(columns=COST_ORDER, fill_value=0))
    pt["TOTAL"] = pt.sum(axis=1, numeric_only=True)
    return pt

def ev_rollup(dfx):
    s = (dfx.pivot_table(index=None, columns="COST-SET", values="HOURS",
                         aggfunc="sum", fill_value=0)
            .reindex(COST_ORDER, fill_value=0))
    val = {k: float((s[k].iloc[0] if k in s else 0.0)) for k in COST_ORDER}

    ACWP, BCWP, BCWS, ETC = val.get("ACWP",0.0), val.get("BCWP",0.0), val.get("BCWS",0.0), val.get("ETC",0.0)
    SPI = np.nan if BCWS == 0 else BCWP/BCWS
    CPI = np.nan if ACWP == 0 else BCWP/ACWP
    SV  = BCWP - BCWS
    SVp = np.nan if BCWS == 0 else 100*SV/BCWS
    CV  = BCWP - ACWP
    CVp = np.nan if BCWP == 0 else 100*CV/BCWP
    BAC = BCWS
    EAC = ACWP + ETC
    VAC = BAC - EAC
    VACp= np.nan if BAC == 0 else 100*VAC/BAC
    BCWR= BAC - BCWP
    denom = (EAC - ACWP)
    TCPI = np.nan if denom == 0 else (BAC - BCWP)/denom
    return dict(ACWP=ACWP,BCWP=BCWP,BCWS=BCWS,ETC=ETC,SPI=SPI,CPI=CPI,SV=SV,SVp=SVp,CV=CV,CVp=CVp,
                BAC=BAC,EAC=EAC,VAC=VAC,VACp=VACp,BCWR=BCWR,TCPI=TCPI)

def fmt_money(x): return f"{x:,.2f}"
def fmt_ratio(x): return "—" if pd.isna(x) else f"{x:0.2f}"
def fmt_pct(x):   return "—" if pd.isna(x) else f"{x:0.2f}"

def metrics_df(label, m):
    return pd.DataFrame({
        "Metric": ["SPI","CPI","SV","SV%","CV","CV%","BAC","EAC","VAC","VAC%","BCWR","ETC","TCPI"],
        label: [
            fmt_ratio(m["SPI"]), fmt_ratio(m["CPI"]),
            fmt_money(m["SV"]),  fmt_pct(m["SVp"]),
            fmt_money(m["CV"]),  fmt_pct(m["CVp"]),
            fmt_money(m["BAC"]), fmt_money(m["EAC"]),
            fmt_money(m["VAC"]), fmt_pct(m["VACp"]),
            fmt_money(m["BCWR"]),fmt_money(m["ETC"]),
            fmt_ratio(m["TCPI"])
        ]
    })

# ------------------- Compute blocks -------------------
m_cum  = ev_rollup(df_cum)
m_4wk  = ev_rollup(df_4wk)
m_stat = ev_rollup(df_stat)

cum_by_plug  = coalesce_cost_columns(df_cum)
wk_by_plug   = coalesce_cost_columns(df_4wk)
stat_by_plug = coalesce_cost_columns(df_stat)

# ------------------- Display results -------------------
from IPython.display import display, HTML

print(f"XM30 Earned Value Dashboard — As-of: {AS_OF.date()}   (Status week: {status_week_start.date()}–{status_week_end.date()}, start={WEEK_START_DAY})\n")

# Summary blocks
def head_row(title, m):
    row = pd.DataFrame([[
        fmt_money(m["ACWP"]), fmt_money(m["BCWP"]), fmt_money(m["BCWS"]), fmt_money(m["ETC"])
    ]], columns=["ACWP","BCWP","BCWS","ETC"])
    row.index = [title]
    return row

display(pd.concat([
    head_row("CUM", m_cum),
    head_row(f"Last {FOUR_WK_LEN} Weeks", m_4wk),
    head_row("Current Status", m_stat)
]))

# EV Metrics tables
print("\nEV Metrics — CUM")
display(metrics_df("CUM", m_cum).set_index("Metric"))
print("\nEV Metrics — 4 Wk")
display(metrics_df("4 Wk", m_4wk).set_index("Metric"))
print("\nEV Metrics — Current Status")
display(metrics_df("Status", m_stat).set_index("Metric"))

# Detail pivots
print("\nDetail by CHG# — CUM")
display(cum_by_plug)
print("\nDetail by CHG# — 4 Wk")
display(wk_by_plug)
print("\nDetail by CHG# — Current Status")
display(stat_by_plug)

# ------------------- Trend chart (weekly totals) -------------------
try:
    import plotly.express as px
    weekly = (df[df["DATE"] <= AS_OF]
              .groupby(["WEEK_START","COST-SET"], as_index=False)["HOURS"].sum())
    weekly = weekly[weekly["COST-SET"].isin(COST_ORDER)]
    fig = px.line(weekly, x="WEEK_START", y="HOURS", color="COST-SET",
                  category_orders={"COST-SET": COST_ORDER},
                  labels={"WEEK_START":"Week","HOURS":"Hours"})
    fig.update_layout(title=f"Weekly Totals through {AS_OF.date()}")
    fig.show()
except Exception as e:
    print(f"(Skipping chart; plotly not available) {e}")

# ------------------- Optional export -------------------
if EXPORT_EXCEL:
    with pd.ExcelWriter(EXPORT_PATH, engine="xlsxwriter") as xw:
        head = pd.concat([
            head_row("CUM", m_cum),
            head_row(f"Last {FOUR_WK_LEN} Weeks", m_4wk),
            head_row("Current Status", m_stat)
        ])
        head.to_excel(xw, sheet_name="Summary")
        metrics_df("CUM", m_cum).to_excel(xw, sheet_name="Metrics", index=False, startrow=0)
        metrics_df("4 Wk", m_4wk).to_excel(xw, sheet_name="Metrics", index=False, startrow=16)
        metrics_df("Status", m_stat).to_excel(xw, sheet_name="Metrics", index=False, startrow=32)
        cum_by_plug.to_excel(xw, sheet_name="Detail_CUM")
        wk_by_plug.to_excel(xw, sheet_name="Detail_4Wk")
        stat_by_plug.to_excel(xw, sheet_name="Detail_Status")
    print(f"\nExported: {os.path.abspath(EXPORT_PATH)}")
# ==================== end one cell ====================