# ==================== EV DASHBOARD TABLES (by CHG#): CUM • 4WK • STATUS ====================
# One cell: builds three pivots that match Excel AND provides "detail" tables per column.
# Expected columns in your in-memory DataFrame: DATE, CHG#, COST-SET, HOURS

import pandas as pd
from IPython.display import display

# ----------------------------- SETTINGS -----------------------------
FILE_START_DATE = pd.Timestamp("2025-09-21")   # <-- set to the start date your Excel uses
WEEK_START_DAY  = "SUN"                        # "SUN" matches your screenshots; change if needed
COST_ORDER      = ["ACWP","BCWP","BCWS","ETC"]
FOUR_WK_LEN     = 4                            # last 4 accounting weeks
DROP_ALL_ZERO_ROWS = True                      # drop CHG# rows with 0 in all four cols (Excel usually does)

# Optional: cap AS_OF to today instead of file max
# TODAY_CAP = pd.Timestamp("today").normalize()

# ----------------------------- DATA PREP -----------------------------
df = xm30_cobra_export_weekly_extract.copy()

# keep your column names; standardize values
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df = df.dropna(subset=["DATE"]).copy()

df["COST-SET"] = (
    df["COST-SET"]
      .astype(str)
      .str.upper()
      .str.replace(r"\s+", "", regex=True)   # remove stray spaces
      .replace({"ACWP":"ACWP","BCWP":"BCWP","BCWS":"BCWS","ETC":"ETC"})
)
df["HOURS"] = pd.to_numeric(df["HOURS"], errors="coerce").fillna(0.0)

# Filter to the start date you actually use in Excel
df = df[df["DATE"] >= FILE_START_DATE].copy()

# as-of date is max in file after start-date filter
AS_OF = df["DATE"].max().normalize()
# If you want to cap to today, uncomment:
# if AS_OF > TODAY_CAP: AS_OF = TODAY_CAP

# ----------------------------- ACCOUNTING WEEKS -----------------------------
DAY_TO_NUM = {"MON":0,"TUE":1,"WED":2,"THU":3,"FRI":4,"SAT":5,"SUN":6}
anchor = DAY_TO_NUM[WEEK_START_DAY.upper()]

def week_start(s):
    dow = s.dt.dayofweek
    offset = (dow - anchor) % 7
    return s - pd.to_timedelta(offset, unit="D")

df["WEEK_START"] = week_start(df["DATE"])
status_week_start = week_start(pd.Series([AS_OF]))[0]
status_week_end   = status_week_start + pd.Timedelta(days=6)

# last FOUR_WK_LEN distinct accounting weeks ending at AS_OF
weeks_le_asof = (df.loc[df["DATE"] <= AS_OF, "WEEK_START"].drop_duplicates().sort_values())
last_k_weeks  = weeks_le_asof.tail(FOUR_WK_LEN).tolist()
fourwk_start  = min(last_k_weeks) if len(last_k_weeks) else status_week_start
fourwk_end    = status_week_end

# ----------------------------- HELPERS -----------------------------
def _pivot_sum_by_plug(dfx: pd.DataFrame) -> pd.DataFrame:
    """Rows = CHG#, Columns = COST-SET, Values = sum(HOURS); add TOTAL row; drop all-zero rows if requested."""
    if dfx.empty:
        return pd.DataFrame(columns=COST_ORDER, index=[])

    pt = (
        dfx.groupby(["CHG#","COST-SET"], as_index=False)["HOURS"].sum()
          .pivot(index="CHG#", columns="COST-SET", values="HOURS")
          .reindex(columns=COST_ORDER, fill_value=0)
          .fillna(0)
          .sort_index()
    )

    if DROP_ALL_ZERO_ROWS:
        pt = pt[(pt[COST_ORDER].sum(axis=1) != 0)]

    pt.loc["TOTAL"] = pt.sum(numeric_only=True, axis=0)
    return pt.round(4)

def detail_rows(dfx: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, cost_set: str) -> pd.DataFrame:
    """Return raw rows for 'clicked' cell like Excel details (window + specific COST-SET)."""
    mask = (dfx["DATE"] >= start) & (dfx["DATE"] <= end) & (dfx["COST-SET"] == cost_set)
    cols = [c for c in dfx.columns if c not in []]  # keep all original cols
    return dfx.loc[mask, cols].sort_values(["CHG#","DATE"])

# ----------------------------- BUILD TABLES -----------------------------
# 1) CUMULATIVE: FILE_START_DATE .. AS_OF
cum_window_start, cum_window_end = FILE_START_DATE, AS_OF
cum_pivot  = _pivot_sum_by_plug(df[(df["DATE"] >= cum_window_start) & (df["DATE"] <= cum_window_end)])

# 2) 4-WEEK: last 4 accounting weeks (end aligned to AS_OF week)
wk4_window_start, wk4_window_end = fourwk_start, fourwk_end
wk4_pivot  = _pivot_sum_by_plug(df[(df["DATE"] >= wk4_window_start) & (df["DATE"] <= wk4_window_end)])

# 3) STATUS PERIOD: accounting week containing AS_OF
stat_window_start, stat_window_end = status_week_start, status_week_end
stat_pivot = _pivot_sum_by_plug(df[(df["DATE"] >= stat_window_start) & (df["DATE"] <= stat_window_end)])

# ----------------------------- DISPLAY -----------------------------
print(f"As-of: {AS_OF.date()}  |  Week start={WEEK_START_DAY}")
print(f"  CUM window:   {cum_window_start.date()} → {cum_window_end.date()}")
print(f"  4WK window:   {wk4_window_start.date()} → {wk4_window_end.date()}")
print(f"  Status wk:    {stat_window_start.date()} → {stat_window_end.date()}\n")

print("CUMULATIVE (by CHG#)  — rows (excluding TOTAL):", len(cum_pivot.index) - 1)
display(cum_pivot)

print("\n4-WEEK (by CHG#)      — rows (excluding TOTAL):", len(wk4_pivot.index) - 1)
display(wk4_pivot)

print("\nSTATUS PERIOD (by CHG#) — rows (excluding TOTAL):", len(stat_pivot.index) - 1)
display(stat_pivot)

# ----------------------------- EXCEL-LIKE DETAILS -----------------------------
# Examples: uncomment any to see the “drill-through” rows like Excel
# Choose a COST-SET and it will show raw records inside that window for that cost-set.

# 1) Details for BCWS in the Status window (matches your first screenshot)
# display(detail_rows(df, stat_window_start, stat_window_end, "BCWS"))

# 2) Details for ETC in the Status window (matches your second screenshot)
# display(detail_rows(df, stat_window_start, stat_window_end, "ETC"))

# 3) Details for a specific COST-SET in the 4-week window
# display(detail_rows(df, wk4_window_start, wk4_window_end, "ACWP"))

# 4) Details for a specific COST-SET in the CUM window
# display(detail_rows(df, cum_window_start, cum_window_end, "BCWP"))
# ===========================================================================================