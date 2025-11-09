# ================= DASHBOARD TABLES (by CHG#): CUM • 4 WEEKS • STATUS =================
import pandas as pd
from IPython.display import display

# --------- CONFIG ---------
FILE_START_DATE = pd.Timestamp("2025-10-15")     # <-- your file's start
WEEK_START_DAY  = "MON"                          # change if your accounting week starts another day
FOUR_WK_LEN     = 4                              # number of accounting weeks in the "4WK" block
COST_ORDER      = ["ACWP","BCWP","BCWS","ETC"]

# OPTIONAL: cap the as-of date to today instead of the file max
# TODAY_CAP = pd.Timestamp("today").normalize()

# --------- LOAD DATA FROM MEMORY ---------
df = xm30_cobra_export_weekly_extract.copy()

# --------- CLEAN TYPES (keep your column names) ---------
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df = df.dropna(subset=["DATE"]).copy()
df["COST-SET"] = df["COST-SET"].astype(str).str.upper().str.strip()
df["HOURS"] = pd.to_numeric(df["HOURS"], errors="coerce").fillna(0.0)

# Restrict to file start date forward
df = df[df["DATE"] >= FILE_START_DATE].copy()

# AS_OF defaults to latest date present in the (filtered) file
AS_OF = df["DATE"].max().normalize()

# If you want to auto-limit to today's date, uncomment:
# if AS_OF > TODAY_CAP: AS_OF = TODAY_CAP

# --------- ACCOUNTING WEEK BOUNDARIES ---------
DAY_TO_NUM = {"MON":0,"TUE":1,"WED":2,"THU":3,"FRI":4,"SAT":5,"SUN":6}
anchor = DAY_TO_NUM[WEEK_START_DAY.upper()]

def week_start(s):
    dow = s.dt.dayofweek
    offset = (dow - anchor) % 7
    return s - pd.to_timedelta(offset, unit="D")

df["WEEK_START"] = week_start(df["DATE"])
status_week_start = week_start(pd.Series([AS_OF]))[0]
status_week_end   = status_week_start + pd.Timedelta(days=6)

# Last FOUR_WK_LEN distinct accounting weeks ending at AS_OF
weeks_le_asof = (df.loc[df["DATE"] <= AS_OF, "WEEK_START"]
                   .drop_duplicates()
                   .sort_values())
last_k_weeks = weeks_le_asof.tail(FOUR_WK_LEN).tolist()

# --------- HELPERS ---------
def pivot_by_plug(dfx):
    """Rows = CHG#, Cols = COST-SET (ACWP/BCWP/BCWS/ETC), Values = SUM(HOURS)."""
    if dfx.empty:
        return pd.DataFrame(columns=COST_ORDER, index=[]).astype(float)
    pt = (dfx.groupby(["CHG#","COST-SET"], as_index=False)["HOURS"].sum()
             .pivot(index="CHG#", columns="COST-SET", values="HOURS")
             .reindex(columns=COST_ORDER, fill_value=0)
             .fillna(0)
             .sort_index())
    pt.loc["TOTAL"] = pt.sum(numeric_only=True, axis=0)
    return pt.round(4)

# --------- 1) CUMULATIVE (to AS_OF) — use your proven cumsum+last logic ---------
# Sort then build cumulative HOURS per CHG# & COST-SET
df_sorted = df.sort_values(["CHG#","COST-SET","DATE"]).copy()
df_sorted["HOURS_CUM"] = df_sorted.groupby(["CHG#","COST-SET"], sort=False)["HOURS"].cumsum()

# Take the last cumulative value per CHG# × COST-SET up to AS_OF
cum_last = (df_sorted[df_sorted["DATE"] <= AS_OF]
            .groupby(["CHG#","COST-SET"], as_index=False)["HOURS_CUM"].max())

cum_pivot = (cum_last
             .pivot(index="CHG#", columns="COST-SET", values="HOURS_CUM")
             .reindex(columns=COST_ORDER, fill_value=0)
             .fillna(0)
             .sort_index())
cum_pivot.loc["TOTAL"] = cum_pivot.sum(numeric_only=True, axis=0)
cum_pivot = cum_pivot.round(4)

# --------- 2) LAST FOUR ACCOUNTING WEEKS ---------
df_4wk = df[df["WEEK_START"].isin(last_k_weeks) & (df["DATE"] <= AS_OF)]
wk4_pivot = pivot_by_plug(df_4wk)

# --------- 3) CURRENT STATUS PERIOD (week containing AS_OF) ---------
df_status = df[(df["WEEK_START"] == status_week_start) & (df["DATE"] <= AS_OF)]
stat_pivot = pivot_by_plug(df_status)

# --------- DISPLAY ---------
print(f"As-of: {AS_OF.date()}  |  Status week: {status_week_start.date()}–{status_week_end.date()}  (week start={WEEK_START_DAY})\n")

print("CUMULATIVE (by CHG#):")
display(cum_pivot)

print("\n4-WEEK (last 4 accounting weeks, by CHG#):")
display(wk4_pivot)

print("\nCURRENT STATUS PERIOD (by CHG#):")
display(stat_pivot)
# ================================================================================