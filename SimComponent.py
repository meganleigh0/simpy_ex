# ================= THREE TABLES: CUM • 4WK • STATUS (anchored to 2025-10-15) =================
import pandas as pd
from IPython.display import display

# ---------- CONFIG ----------
FILE_START_DATE = pd.Timestamp("2025-10-15")  # <- anchor to your file's first valid date
WEEK_START_DAY  = "MON"                       # accounting week start: MON/TUE/WED/THU/FRI/SAT/SUN
FOUR_WK_LEN     = 4                           # rolling window length (weeks)
COST_ORDER      = ["ACWP","BCWP","BCWS","ETC"]

# Optional: automatically cap AS_OF to "today"
# AUTO_TODAY = pd.Timestamp("today").normalize()

# ---------- DATA ----------
df = xm30_cobra_export_weekly_extract.copy()

# Keep your exact column names; just clean types/values
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df = df.dropna(subset=["DATE"]).copy()
df["COST-SET"] = df["COST-SET"].astype(str).str.upper().str.strip()
df["HOURS"] = pd.to_numeric(df["HOURS"], errors="coerce").fillna(0.0)

# Filter using the file start date
df = df[df["DATE"] >= FILE_START_DATE].copy()

# AS_OF := latest date present *after* the start-date filter
AS_OF = df["DATE"].max().normalize()

# If you want to auto-limit by today's date instead, uncomment the next two lines:
# if AS_OF > AUTO_TODAY: 
#     AS_OF = AUTO_TODAY

# ---------- PERIOD HELPERS ----------
DAY_TO_NUM = {"MON":0,"TUE":1,"WED":2,"THU":3,"FRI":4,"SAT":5,"SUN":6}
anchor = DAY_TO_NUM[WEEK_START_DAY.upper()]

def week_start(s):
    dow = s.dt.dayofweek
    offset = (dow - anchor) % 7
    return s - pd.to_timedelta(offset, unit="D")

df["WEEK_START"] = week_start(df["DATE"])
status_week_start = week_start(pd.Series([AS_OF]))[0]
status_week_end   = status_week_start + pd.Timedelta(days=6)
fourwk_start      = AS_OF - pd.Timedelta(weeks=FOUR_WK_LEN) + pd.Timedelta(days=1)

# ---------- PIVOT HELPER ----------
def pivot_by_plug(dfx):
    """Rows = CHG#, Cols = COST-SET, Values = SUM(HOURS)."""
    if dfx.empty:
        # preserve shape even when empty
        return pd.DataFrame(columns=COST_ORDER, index=[]).astype(float).fillna(0.0)
    pt = (
        dfx.groupby(["CHG#","COST-SET"], as_index=False)["HOURS"].sum()
           .pivot(index="CHG#", columns="COST-SET", values="HOURS")
           .reindex(columns=COST_ORDER, fill_value=0)
           .fillna(0)
           .sort_index()
    )
    pt.loc["TOTAL"] = pt.sum(numeric_only=True, axis=0)
    return pt.round(4)

# ---------- BUILD TABLES ----------
# 1) CUMULATIVE (all data up to AS_OF)
cum_pivot = pivot_by_plug(df[df["DATE"] <= AS_OF])

# 2) 4-WEEK (rolling window ending AS_OF, inclusive)
wk4_mask  = (df["DATE"] >= fourwk_start) & (df["DATE"] <= AS_OF)
wk4_pivot = pivot_by_plug(df[wk4_mask])

# 3) STATUS PERIOD (accounting week containing AS_OF)
stat_pivot = pivot_by_plug(df[df["WEEK_START"] == status_week_start])

# ---------- SHOW ----------
print(f"As-of: {AS_OF.date()}  |  Status week: {status_week_start.date()}–{status_week_end.date()}  (start={WEEK_START_DAY})\n")

print("CUMULATIVE (by CHG#):")
display(cum_pivot)

print("\n4-WEEK (by CHG#):")
display(wk4_pivot)

print("\nCURRENT STATUS PERIOD (by CHG#):")
display(stat_pivot)
# =================================================================================================