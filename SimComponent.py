# ================= THREE DASHBOARD TABLES: CUM • 4WK • STATUS (by CHG#) =================
import pandas as pd
from IPython.display import display

# -------- settings you can tweak --------
AS_OF = None                 # e.g., '2025-10-15'; if None, uses max DATE in data
WEEK_START_DAY = "MON"       # one of: MON/TUE/WED/THU/FRI/SAT/SUN
FOUR_WK_LEN = 4              # length of "4WK" window (in weeks)
COST_ORDER = ["ACWP","BCWP","BCWS","ETC"]

# -------- use your in-memory DataFrame --------
df = xm30_cobra_export_weekly_extract.copy()

# -------- minimal cleaning (keep your column names) --------
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df = df.dropna(subset=["DATE"]).copy()
df["COST-SET"] = df["COST-SET"].astype(str).str.upper().str.strip()
df["HOURS"] = pd.to_numeric(df["HOURS"], errors="coerce").fillna(0.0)

# -------- period boundaries --------
if AS_OF is None:
    AS_OF = df["DATE"].max().normalize()
else:
    AS_OF = pd.Timestamp(AS_OF).normalize()

DAY_TO_NUM = {"MON":0,"TUE":1,"WED":2,"THU":3,"FRI":4,"SAT":5,"SUN":6}
anchor = DAY_TO_NUM[WEEK_START_DAY.upper()]

def week_start(s):  # align to accounting week
    dow = s.dt.dayofweek
    offset = (dow - anchor) % 7
    return s - pd.to_timedelta(offset, unit="D")

df["WEEK_START"] = week_start(df["DATE"])
status_week_start = week_start(pd.Series([AS_OF]))[0]
status_week_end   = status_week_start + pd.Timedelta(days=6)
fourwk_start = AS_OF - pd.Timedelta(weeks=FOUR_WK_LEN) + pd.Timedelta(days=1)

# -------- helper: CHG# x COST-SET pivot (sum of HOURS) --------
def make_pivot(dfx):
    pt = (dfx
          .groupby(["CHG#","COST-SET"], as_index=False)["HOURS"].sum()
          .pivot(index="CHG#", columns="COST-SET", values="HOURS")
          .reindex(columns=COST_ORDER, fill_value=0)
          .fillna(0))
    # optional: totals row like Excel’s grand total
    pt.loc["TOTAL"] = pt.sum(numeric_only=True, axis=0)
    return pt.round(2)

# -------- build the three dashboard tables --------
# 1) CUM: everything up to AS_OF
cum_pivot = make_pivot(df[df["DATE"] <= AS_OF])

# 2) 4WK: last FOUR_WK_LEN calendar weeks ending AS_OF (inclusive)
wk4_mask = (df["DATE"] >= fourwk_start) & (df["DATE"] <= AS_OF)
wk4_pivot = make_pivot(df[wk4_mask])

# 3) STATUS PERIOD: accounting week that contains AS_OF
stat_pivot = make_pivot(df[df["WEEK_START"] == status_week_start])

# -------- show results --------
print(f"As-of: {AS_OF.date()}  |  Status week: {status_week_start.date()}–{status_week_end.date()}  (start={WEEK_START_DAY})\n")

print("CUMULATIVE (by CHG#):")
display(cum_pivot)

print("\n4-WEEK (by CHG#):")
display(wk4_pivot)

print("\nCURRENT STATUS PERIOD (by CHG#):")
display(stat_pivot)
# ==============================================================================