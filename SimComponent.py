# ================= THREE PIVOTS (by CHG#): CUM • 4-WEEK • STATUS =================
import pandas as pd
from IPython.display import display

# ---- Config ----
FILE_START_DATE = pd.Timestamp("2025-10-15")   # anchor start date from your file
WEEK_START_DAY  = "MON"                        # MON/TUE/WED/THU/FRI/SAT/SUN
COST_ORDER      = ["ACWP","BCWP","BCWS","ETC"]
FOUR_WK_LEN     = 4

# ---- Data (uses your in-memory DF) ----
df = xm30_cobra_export_weekly_extract.copy()

# Keep your column names; just clean types / labels
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df = df.dropna(subset=["DATE"]).copy()
df["COST-SET"] = (
    df["COST-SET"].astype(str)
    .str.upper()
    .str.replace(r"\s+", "", regex=True)   # remove embedded spaces like "BCW S"
)
df["HOURS"] = pd.to_numeric(df["HOURS"], errors="coerce").fillna(0.0)

# Filter from the file's start date forward
df = df[df["DATE"] >= FILE_START_DATE].copy()

# AS_OF = latest date in file after filter
AS_OF = df["DATE"].max().normalize()
# If you want to cap to today's date instead, uncomment the next two lines:
# TODAY_CAP = pd.Timestamp("today").normalize()
# if AS_OF > TODAY_CAP: AS_OF = TODAY_CAP

# ---- Accounting weeks ----
DAY_TO_NUM = {"MON":0,"TUE":1,"WED":2,"THU":3,"FRI":4,"SAT":5,"SUN":6}
anchor = DAY_TO_NUM[WEEK_START_DAY.upper()]

def week_start(s):
    dow = s.dt.dayofweek
    offset = (dow - anchor) % 7
    return s - pd.to_timedelta(offset, unit="D")

df["WEEK_START"] = week_start(df["DATE"])
status_week_start = week_start(pd.Series([AS_OF]))[0]
status_week_end   = status_week_start + pd.Timedelta(days=6)

# Last FOUR_WK_LEN distinct accounting weeks (ending at AS_OF)
weeks_le_asof = (df.loc[df["DATE"] <= AS_OF, "WEEK_START"]
                   .drop_duplicates()
                   .sort_values())
last_k_weeks = weeks_le_asof.tail(FOUR_WK_LEN).tolist()
fourwk_start = min(last_k_weeks)
fourwk_end   = status_week_end   # align end to end of AS_OF week

# ---- Helper to build the CHG# x COST-SET pivot with SUM(HOURS) ----
def make_pivot(dfx):
    if dfx.empty:
        return pd.DataFrame(columns=COST_ORDER, index=[])
    pt = (
        dfx.groupby(["CHG#","COST-SET"], as_index=False)["HOURS"].sum()
           .pivot(index="CHG#", columns="COST-SET", values="HOURS")
           .reindex(columns=COST_ORDER, fill_value=0)
           .fillna(0)
           .sort_index()
    )
    pt.loc["TOTAL"] = pt.sum(numeric_only=True, axis=0)
    return pt.round(4)

# ---- Build the three tables (all SUM(HOURS)) ----
# 1) CUM: from FILE_START_DATE through AS_OF
cum_pivot = make_pivot(df[df["DATE"] <= AS_OF])

# 2) 4-WEEK: last 4 accounting weeks (sum within that window)
wk4_mask  = (df["WEEK_START"] >= fourwk_start) & (df["DATE"] <= fourwk_end)
wk4_pivot = make_pivot(df[wk4_mask])

# 3) STATUS: week containing AS_OF
stat_mask = (df["WEEK_START"] == status_week_start) & (df["DATE"] <= status_week_end)
stat_pivot = make_pivot(df[stat_mask])

# ---- Show + quick sanity counts ----
print(f"As-of: {AS_OF.date()}  |  Status week: {status_week_start.date()}–{status_week_end.date()}  (week start={WEEK_START_DAY})\n")

print("CUMULATIVE (by CHG#):    rows =", len(cum_pivot.index) - 1)  # minus TOTAL
display(cum_pivot)

print("\n4-WEEK (by CHG#):        rows =", len(wk4_pivot.index) - 1)
display(wk4_pivot)

print("\nSTATUS PERIOD (by CHG#): rows =", len(stat_pivot.index) - 1)
display(stat_pivot)
# ================================================================================