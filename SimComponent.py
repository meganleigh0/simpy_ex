# ===================== THREE SUMMARY TABLES: CUM • 4WK • STATUS =====================
import pandas as pd
import numpy as np
from IPython.display import display

# ---------- SETTINGS ----------
AS_OF = None               # e.g., '2025-10-15' ; if None, uses max DATE in the data
WEEK_START_DAY = "MON"     # MON/TUE/WED/THU/FRI/SAT/SUN (accounting calendar anchor)
FOUR_WK_LEN = 4            # 4-week window length
COST_ORDER = ["ACWP","BCWP","BCWS","ETC"]

# ---------- USE YOUR IN-MEMORY DF ----------
df = xm30_cobra_export_weekly_extract.copy()

# ---------- CLEAN TYPES (do NOT rename your columns) ----------
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df = df.dropna(subset=["DATE"]).copy()
df["COST-SET"] = df["COST-SET"].astype(str).str.upper().str.strip()
df["HOURS"] = pd.to_numeric(df["HOURS"], errors="coerce").fillna(0.0)

# ---------- PERIOD BOUNDARIES ----------
if AS_OF is None:
    AS_OF = df["DATE"].max().normalize()
else:
    AS_OF = pd.Timestamp(AS_OF).normalize()

DAY_TO_NUM = dict(MON=0,TUE=1,WED=2,THU=3,FRI=4,SAT=5,SUN=6)
anchor = DAY_TO_NUM[WEEK_START_DAY.upper()]

def week_start(s):
    dow = s.dt.dayofweek
    offset = (dow - anchor) % 7
    return s - pd.to_timedelta(offset, unit="D")

df["WEEK_START"] = week_start(df["DATE"])
status_week_start = week_start(pd.Series([AS_OF]))[0]
status_week_end   = status_week_start + pd.Timedelta(days=6)
fourwk_start = AS_OF - pd.Timedelta(weeks=FOUR_WK_LEN) + pd.Timedelta(days=1)

# ---------- 1) CUMULATIVE (to AS_OF) ----------
# your working approach: sort -> cumsum within CHG# & COST-SET
df_sorted = df.sort_values(["CHG#","COST-SET","DATE"]).copy()
df_sorted["HOURS_CUM"] = df_sorted.groupby(["CHG#","COST-SET"], sort=False)["HOURS"].cumsum()

# take max cum value per group up to AS_OF (monotonic, so max == last cumulative)
cum_max = (
    df_sorted[df_sorted["DATE"] <= AS_OF]
    .groupby(["CHG#","COST-SET"], as_index=False)["HOURS_CUM"].max()
)

cum_pivot = (
    cum_max.pivot(index="CHG#", columns="COST-SET", values="HOURS_CUM")
    .reindex(columns=COST_ORDER, fill_value=0)
    .fillna(0)
)

# totals row for the dashboard top block
tbl_cum = cum_pivot.sum(axis=0).to_frame().T
tbl_cum.index = ["CUM"]
tbl_cum = tbl_cum.reindex(columns=COST_ORDER).round(2)

# ---------- 2) LAST 4 WEEKS (rolling window ending AS_OF) ----------
df_4wk = df[(df["DATE"] >= fourwk_start) & (df["DATE"] <= AS_OF)]
tbl_4wk = (
    df_4wk.pivot_table(index=None, columns="COST-SET", values="HOURS",
                       aggfunc="sum", fill_value=0)
    .reindex(columns=COST_ORDER, fill_value=0)
)
tbl_4wk.index = [f"Last {FOUR_WK_LEN} Weeks"]
tbl_4wk = tbl_4wk.round(2)

# ---------- 3) CURRENT STATUS PERIOD (accounting week containing AS_OF) ----------
df_status = df[(df["WEEK_START"] == status_week_start) & (df["DATE"] <= AS_OF)]
tbl_status = (
    df_status.pivot_table(index=None, columns="COST-SET", values="HOURS",
                          aggfunc="sum", fill_value=0)
    .reindex(columns=COST_ORDER, fill_value=0)
)
tbl_status.index = ["Current Status"]
tbl_status = tbl_status.round(2)

# ---------- DISPLAY ----------
print(f"As-of: {AS_OF.date()}  |  Status week: {status_week_start.date()}–{status_week_end.date()} (start={WEEK_START_DAY})\n")

print("CUMULATIVE:")
display(tbl_cum)

print("\n4-WEEK:")
display(tbl_4wk)

print("\nCURRENT STATUS PERIOD:")
display(tbl_status)
# ================================================================================