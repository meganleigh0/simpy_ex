# ==================== EV SUMMARY TABLES (CUM • 4WK • STATUS) ====================
# Uses in-memory DataFrame: xm30_cobra_export_weekly_extract
# Needs columns (any case/spacing): DATE, COST-SET (ACWP/BCWP/BCWS/ETC), HOURS

import pandas as pd
import numpy as np
from IPython.display import display

# ---------- Settings ----------
AS_OF = None               # e.g., '2025-10-25'; if None, uses max DATE in the data
WEEK_START_DAY = "MON"     # accounting week start: MON/TUE/WED/THU/FRI/SAT/SUN
FOUR_WK_LEN = 4            # length of the rolling window for the "4 Week" table
COST_ORDER = ["ACWP","BCWP","BCWS","ETC"]

# ---------- Get the DataFrame ----------
if "xm30_cobra_export_weekly_extract" not in globals():
    raise RuntimeError("DataFrame 'xm30_cobra_export_weekly_extract' not found in memory.")
df = xm30_cobra_export_weekly_extract.copy()

# ---------- Light, safe column normalization (no uploads, no prompts) ----------
def _norm(c): return str(c).strip().upper().replace(" ", "").replace("_","")
df.rename(columns={c: _norm(c) for c in df.columns}, inplace=True)

# map common aliases -> canonical names (we only need DATE/COST-SET/HOURS for the three tables)
alias = {"DATE":"DATE","COST-SET":"COST-SET","COSTSET":"COST-SET","HOURS":"HOURS"}
df.rename(columns={c: alias.get(c, c) for c in df.columns}, inplace=True)

need = ["DATE","COST-SET","HOURS"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns for summaries: {missing}. Found: {list(df.columns)}")

# Types/cleanup
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df = df.dropna(subset=["DATE"]).copy()
df["COST-SET"] = df["COST-SET"].astype(str).str.upper().str.strip()
df["HOURS"] = pd.to_numeric(df["HOURS"], errors="coerce").fillna(0.0)

# ---------- Period helpers ----------
DAY_TO_NUM = dict(MON=0,TUE=1,WED=2,THU=3,FRI=4,SAT=5,SUN=6)
anchor = DAY_TO_NUM[WEEK_START_DAY.upper()]

def _week_start(s):
    dow = s.dt.dayofweek
    offset = (dow - anchor) % 7
    return s - pd.to_timedelta(offset, unit="D")

df["WEEK_START"] = _week_start(df["DATE"])

AS_OF = pd.Timestamp(df["DATE"].max().normalize() if AS_OF is None else AS_OF).normalize()
status_week_start = _week_start(pd.Series([AS_OF]))[0]
status_week_end   = status_week_start + pd.Timedelta(days=6)
fourwk_start = AS_OF - pd.Timedelta(weeks=FOUR_WK_LEN) + pd.Timedelta(days=1)

# ---------- Core summarizer (no CHG# needed) ----------
def _sum_table(dfx, title):
    s = (dfx.pivot_table(index=None, columns="COST-SET", values="HOURS",
                         aggfunc="sum", fill_value=0)
           .reindex(COST_ORDER, fill_value=0))
    # single-row table with nice labels
    row = pd.DataFrame([ [ float(s.get(k, pd.Series([0])).iloc[0]) for k in COST_ORDER ] ],
                       columns=COST_ORDER, index=[title])
    return row.round(2)

# ---------- Build the three tables ----------
df_cum   = df[df["DATE"] <= AS_OF]
df_4wk   = df[(df["DATE"] >= fourwk_start) & (df["DATE"] <= AS_OF)]
df_stat  = df[df["WEEK_START"] == status_week_start]

tbl_cum  = _sum_table(df_cum,  "CUM")
tbl_4wk  = _sum_table(df_4wk,  f"Last {FOUR_WK_LEN} Weeks")
tbl_stat = _sum_table(df_stat, "Current Status")

# ---------- Display ----------
print(f"As-of: {AS_OF.date()}  |  Status week: {status_week_start.date()}–{status_week_end.date()}  (start={WEEK_START_DAY})\n")
print("CUMULATIVE:")
display(tbl_cum)
print("\n4-WEEK:")
display(tbl_4wk)
print("\nCURRENT STATUS PERIOD:")
display(tbl_stat)
# ============================================================================== 