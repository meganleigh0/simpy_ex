# ================= ONE-CELL EV DASHBOARD (uses in-memory df) =================
# Assumes you already have: xm30_cobra_export_weekly_extract (pandas DataFrame)
# Columns expected (any case/spacing): DATE, CHG# (or CHG/PLUG), COST-SET, HOURS

import pandas as pd, numpy as np
from IPython.display import display

# ----------------------- PARAMETERS -----------------------
AS_OF = None              # e.g., '2025-10-25'. If None, uses max DATE.
WEEK_START_DAY = "MON"    # accounting week start: MON/TUE/WED/THU/FRI/SAT/SUN
FOUR_WK_LEN = 4           # rolling window length for the "4 Week" block
COST_ORDER = ["ACWP","BCWP","BCWS","ETC"]

# ----------------------- GET DATAFRAME -----------------------
df = xm30_cobra_export_weekly_extract.copy()

# ---- Fix common causes of "Grouper ... not 1-dimensional" ----
# Flatten MultiIndex columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ["_".join([str(x) for x in tup if str(x)!='']) for tup in df.columns]

# Drop exact duplicate column names (keep first)
df = df.loc[:, ~df.columns.duplicated(keep="first")]

# Normalize column names
def norm(c): return str(c).strip().upper().replace(" ", "").replace("_","")
norm_map = {c: norm(c) for c in df.columns}
df.rename(columns=norm_map, inplace=True)

# Map variants to required names
alias = {
    "DATE": "DATE",
    "CHG#": "CHG#",
    "CHG": "CHG#",
    "PLUG": "CHG#",
    "COST-SET": "COST-SET",
    "COSTSET": "COST-SET",
    "HOURS": "HOURS"
}
df.rename(columns={c: alias.get(c, c) for c in df.columns}, inplace=True)

# Ensure required columns exist once and are 1-D
required = ["DATE","CHG#","COST-SET","HOURS"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

# Types
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df = df.dropna(subset=["DATE"]).copy()
df["COST-SET"] = df["COST-SET"].astype(str).str.upper().str.strip()
df["HOURS"] = pd.to_numeric(df["HOURS"], errors="coerce").fillna(0.0)

# ----------------------- PERIOD SLICES -----------------------
DAY_TO_NUM = dict(MON=0,TUE=1,WED=2,THU=3,FRI=4,SAT=5,SUN=6)
anchor = DAY_TO_NUM[WEEK_START_DAY.upper()]

def week_start(sdt):
    dow = sdt.dt.dayofweek
    offset = (dow - anchor) % 7
    return sdt - pd.to_timedelta(offset, unit="D")

df["WEEK_START"] = week_start(df["DATE"])
if AS_OF is None:
    AS_OF = df["DATE"].max().normalize()
else:
    AS_OF = pd.Timestamp(AS_OF).normalize()

status_week_start = week_start(pd.Series([AS_OF]))[0]
status_week_end   = status_week_start + pd.Timedelta(days=6)

df_cum  = df[df["DATE"] <= AS_OF]
df_4wk  = df[(df["DATE"] > AS_OF - pd.Timedelta(weeks=FOUR_WK_LEN)) & (df["DATE"] <= AS_OF)]
df_stat = df[df["WEEK_START"] == status_week_start]

# ----------------------- HELPERS -----------------------
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
    v = {k: float(s.get(k, pd.Series([0])).iloc[0] if k in s else 0.0) for k in COST_ORDER}
    ACWP, BCWP, BCWS, ETC = v["ACWP"], v["BCWP"], v["BCWS"], v["ETC"]
    SPI = np.nan if BCWS == 0 else BCWP/BCWS
    CPI = np.nan if ACWP == 0 else BCWP/ACWP
    SV, CV = BCWP - BCWS, BCWP - ACWP
    SVp = np.nan if BCWS == 0 else 100*SV/BCWS
    CVp = np.nan if BCWP == 0 else 100*CV/BCWP
    BAC = BCWS
    EAC = ACWP + ETC
    VAC = BAC - EAC
    VACp= np.nan if BAC == 0 else 100*VAC/BAC
    BCWR= BAC - BCWP
    denom = (EAC - ACWP)
    TCPI = np.nan if denom == 0 else (BAC - BCWP)/denom
    return dict(ACWP=ACWP,BCWP=BCWP,BCWS=BCWS,ETC=ETC,SPI=SPI,CPI=CPI,SV=SV,SVp=SVp,
                CV=CV,CVp=CVp,BAC=BAC,EAC=EAC,VAC=VAC,VACp=VACp,BCWR=BCWR,TCPI=TCPI)

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

# ----------------------- CALC & DISPLAY -----------------------
m_cum  = ev_rollup(df_cum)
m_4wk  = ev_rollup(df_4wk)
m_stat = ev_rollup(df_stat)

def head_row(title, m):
    row = pd.DataFrame([[fmt_money(m["ACWP"]), fmt_money(m["BCWP"]),
                         fmt_money(m["BCWS"]), fmt_money(m["ETC"])]],
                       columns=["ACWP","BCWP","BCWS","ETC"])
    row.index = [title]; return row

print(f"XM30 EV Dashboard — As-of {AS_OF.date()}  |  Status week: {status_week_start.date()}–{status_week_end.date()} (start={WEEK_START_DAY})")
display(pd.concat([head_row("CUM", m_cum),
                   head_row(f"Last {FOUR_WK_LEN} Weeks", m_4wk),
                   head_row("Current Status", m_stat)]))

print("\nEV Metrics — CUM");         display(metrics_df("CUM", m_cum).set_index("Metric"))
print("\nEV Metrics — 4 Wk");        display(metrics_df("4 Wk", m_4wk).set_index("Metric"))
print("\nEV Metrics — Current");     display(metrics_df("Status", m_stat).set_index("Metric"))

print("\nDetail by CHG# — CUM");     display(coalesce_cost_columns(df_cum))
print("\nDetail by CHG# — 4 Wk");    display(coalesce_cost_columns(df_4wk))
print("\nDetail by CHG# — Status");  display(coalesce_cost_columns(df_stat))
# =================================================================