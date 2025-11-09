# ================= ONE-CELL EV DASHBOARD (in-memory df only) =================
# Expects a DataFrame named: xm30_cobra_export_weekly_extract
# Required logical columns (any case/spacing): DATE, CHG# (or CHG/PLUG), COST-SET, HOURS

import pandas as pd, numpy as np
from IPython.display import display

# ----------------------- PARAMETERS -----------------------
AS_OF = None              # e.g., '2025-10-25' or None to use max DATE in data
WEEK_START_DAY = "MON"    # one of: MON/TUE/WED/THU/FRI/SAT/SUN
FOUR_WK_LEN = 4           # rolling window length for the "4 Week" block
COST_ORDER = ["ACWP","BCWP","BCWS","ETC"]

# ----------------------- GET DATAFRAME -----------------------
if "xm30_cobra_export_weekly_extract" not in globals():
    raise RuntimeError("DataFrame 'xm30_cobra_export_weekly_extract' not found in memory.")
df = xm30_cobra_export_weekly_extract.copy()

# ----------------------- MAKE COLUMNS SAFE -----------------------
# 1) Flatten MultiIndex columns (from some Excel imports)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ["_".join([str(x) for x in tup if str(x) != ""]).strip() for tup in df.columns]

# 2) Normalize names (collapse case/space/underscore)
def _norm(c): return str(c).strip().upper().replace(" ", "").replace("_","")
df.rename(columns={c: _norm(c) for c in df.columns}, inplace=True)

# 3) Map common aliases to canonical names
alias = {
    "DATE": "DATE",
    "CHG#": "CHG#",
    "CHG": "CHG#",
    "PLUG": "CHG#",
    "COST-SET": "COST-SET",
    "COSTSET": "COST-SET",
    "HOURS": "HOURS",
}
df.rename(columns={c: alias.get(c, c) for c in df.columns}, inplace=True)

# 4) Coalesce duplicates of key columns introduced by aliasing (CHG & PLUG -> CHG#)
def _coalesce_same_name(frame, name):
    """If multiple columns share 'name', merge left->right (first non-null) and keep a single column."""
    mask = (frame.columns == name)
    if mask.sum() > 1:
        dup_cols = list(frame.columns[mask])
        frame[name] = frame[dup_cols].bfill(axis=1).iloc[:, 0]
        frame.drop(columns=dup_cols[1:], inplace=True)

for key in ["DATE", "CHG#", "COST-SET", "HOURS"]:
    if key in df.columns:
        _coalesce_same_name(df, key)

# 5) Remove any remaining duplicate (non-key) names
df = df.loc[:, ~df.columns.duplicated(keep="first")]

# ----------------------- TYPE CLEANUP -----------------------
required = ["DATE","CHG#","COST-SET","HOURS"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df = df.dropna(subset=["DATE"]).copy()
df["COST-SET"] = df["COST-SET"].astype(str).str.upper().str.strip()
df["HOURS"] = pd.to_numeric(df["HOURS"], errors="coerce").fillna(0.0)

# ----------------------- PERIODS -----------------------
DAY_TO_NUM = dict(MON=0,TUE=1,WED=2,THU=3,FRI=4,SAT=5,SUN=6)
anchor = DAY_TO_NUM[WEEK_START_DAY.upper()]

def _week_start(sdt):
    dow = sdt.dt.dayofweek
    offset = (dow - anchor) % 7
    return sdt - pd.to_timedelta(offset, unit="D")

df["WEEK_START"] = _week_start(df["DATE"])
AS_OF = pd.Timestamp(df["DATE"].max().normalize() if AS_OF is None else AS_OF).normalize()
status_week_start = _week_start(pd.Series([AS_OF]))[0]
status_week_end   = status_week_start + pd.Timedelta(days=6)

df_cum  = df[df["DATE"] <= AS_OF]
df_4wk  = df[(df["DATE"] > AS_OF - pd.Timedelta(weeks=FOUR_WK_LEN)) & (df["DATE"] <= AS_OF)]
df_stat = df[df["WEEK_START"] == status_week_start]

# ----------------------- HELPERS -----------------------
def _pivot_by_plug(dfx, value_col="HOURS"):
    """Rows=CHG#, Cols=COST-SET, Values=sum(HOURS) with a TOTAL column."""
    pt = (dfx.pivot_table(index="CHG#", columns="COST-SET", values=value_col,
                          aggfunc="sum", fill_value=0)
            .reindex(columns=COST_ORDER, fill_value=0))
    pt["TOTAL"] = pt.sum(axis=1, numeric_only=True)
    return pt

def _ev_rollup(dfx):
    s = (dfx.pivot_table(index=None, columns="COST-SET", values="HOURS",
                         aggfunc="sum", fill_value=0)
           .reindex(COST_ORDER, fill_value=0))
    v = {k: float((s[k].iloc[0] if k in s else 0.0)) for k in COST_ORDER}
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
    return dict(ACWP=ACWP,BCWP=BCWP,BCWS=BCWS,ETC=ETC,
                SPI=SPI,CPI=CPI,SV=SV,SVp=SVp,CV=CV,CVp=CVp,
                BAC=BAC,EAC=EAC,VAC=VAC,VACp=VACp,BCWR=BCWR,TCPI=TCPI)

def _fmt_money(x): return f"{x:,.2f}"
def _fmt_ratio(x): return "—" if pd.isna(x) else f"{x:0.2f}"
def _fmt_pct(x):   return "—" if pd.isna(x) else f"{x:0.2f}"

def _metrics_df(label, m):
    return pd.DataFrame({
        "Metric": ["SPI","CPI","SV","SV%","CV","CV%","BAC","EAC","VAC","VAC%","BCWR","ETC","TCPI"],
        label: [
            _fmt_ratio(m["SPI"]), _fmt_ratio(m["CPI"]),
            _fmt_money(m["SV"]),  _fmt_pct(m["SVp"]),
            _fmt_money(m["CV"]),  _fmt_pct(m["CVp"]),
            _fmt_money(m["BAC"]), _fmt_money(m["EAC"]),
            _fmt_money(m["VAC"]), _fmt_pct(m["VACp"]),
            _fmt_money(m["BCWR"]),_fmt_money(m["ETC"]),
            _fmt_ratio(m["TCPI"])
        ]
    })

def _head_row(title, m):
    row = pd.DataFrame([[ _fmt_money(m["ACWP"]), _fmt_money(m["BCWP"]),
                          _fmt_money(m["BCWS"]), _fmt_money(m["ETC"]) ]],
                       columns=["ACWP","BCWP","BCWS","ETC"])
    row.index = [title]
    return row

# ----------------------- CALCULATE -----------------------
m_cum  = _ev_rollup(df_cum)
m_4wk  = _ev_rollup(df_4wk)
m_stat = _ev_rollup(df_stat)

cum_by_plug  = _pivot_by_plug(df_cum)
wk_by_plug   = _pivot_by_plug(df_4wk)
stat_by_plug = _pivot_by_plug(df_stat)

# ----------------------- DISPLAY -----------------------
print(f"XM30 EV Dashboard — As-of {AS_OF.date()} | Status week: {status_week_start.date()}–{status_week_end.date()} (start={WEEK_START_DAY})")
display(pd.concat([_head_row("CUM", m_cum),
                   _head_row(f"Last {FOUR_WK_LEN} Weeks", m_4wk),
                   _head_row("Current Status", m_stat)]))

print("\nEV Metrics — CUM");         display(_metrics_df("CUM", m_cum).set_index("Metric"))
print("\nEV Metrics — 4 Wk");        display(_metrics_df("4 Wk", m_4wk).set_index("Metric"))
print("\nEV Metrics — Current");     display(_metrics_df("Status", m_stat).set_index("Metric"))

print("\nDetail by CHG# — CUM");     display(cum_by_plug)
print("\nDetail by CHG# — 4 Wk");    display(wk_by_plug)
print("\nDetail by CHG# — Status");  display(stat_by_plug)
# =================================================================