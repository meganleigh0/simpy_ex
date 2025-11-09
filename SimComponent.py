# ================== DASHBOARD PIVOTS (by CHG#): CUM • 4 WEEKS • STATUS ==================
import pandas as pd
from IPython.display import display

# ---------------- CONFIG ----------------
FILE_START_DATE = pd.Timestamp("2025-10-15")  # anchor to your file's first valid date
WEEK_START_DAY  = "MON"                       # MON/TUE/WED/THU/FRI/SAT/SUN
FOUR_WK_LEN     = 4                           # number of accounting weeks for the 4WK table
COST_ORDER      = ["ACWP","BCWP","BCWS","ETC"]

# Optional: cap AS_OF to today instead of file max
# TODAY_CAP = pd.Timestamp("today").normalize()

# ---------------- DATA ----------------
df = xm30_cobra_export_weekly_extract.copy()

# Keep your column names; only clean types
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df = df.dropna(subset=["DATE"]).copy()
df["COST-SET"] = df["COST-SET"].astype(str).str.upper().str.strip()
df["HOURS"] = pd.to_numeric(df["HOURS"], errors="coerce").fillna(0.0)

# filter from file start onward
df = df[df["DATE"] >= FILE_START_DATE].copy()

# as-of is the latest date present (post-filter)
AS_OF = df["DATE"].max().normalize()
# If you want to cap by today, uncomment:
# if AS_OF > TODAY_CAP: AS_OF = TODAY_CAP

# ---------------- ACCOUNTING WEEKS ----------------
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
fourwk_start  = min(last_k_weeks)
fourwk_end    = status_week_end  # end aligned to end of the as-of week

# ---------------- HELPERS ----------------
FLOW_SETS = {"ACWP","BCWP"}      # sum over window
CUM_SET   = "BCWS"               # cumulative → use delta(last at end, last before start)
PTI_SET   = "ETC"                # point-in-time → last at end

def _last_at_or_before(g, cutoff):
    """g is a group (already filtered to one CHG# x COST-SET); return last value ≤ cutoff, else 0."""
    g2 = g[g["DATE"] <= cutoff]
    if g2.empty:
        return 0.0
    return float(g2.sort_values("DATE")["HOURS"].iloc[-1])

def window_aggregate(df, start, end):
    """
    For each CHG# x COST-SET:
      - ACWP/BCWP: sum HOURS where DATE in [start, end]
      - BCWS: last_at(end) - last_at(start - ε)
      - ETC:  last_at(end)
    Returns long frame with columns: CHG#, COST-SET, VALUE
    """
    # slice once for performance for the FLOW sums
    mask_window = (df["DATE"] >= start) & (df["DATE"] <= end)
    flow = (df[mask_window & df["COST-SET"].isin(FLOW_SETS)]
              .groupby(["CHG#","COST-SET"], as_index=False)["HOURS"].sum()
              .rename(columns={"HOURS":"VALUE"}))

    # cumulative (BCWS) and point-in-time (ETC) need last values
    # work per CHG# and per COST-SET to avoid double counting
    pieces = [flow]

    for cs in [CUM_SET, PTI_SET]:
        sub = df[df["COST-SET"] == cs][["CHG#","COST-SET","DATE","HOURS"]]
        if sub.empty:
            continue
        out_rows = []
        for (chg, _), g in sub.groupby(["CHG#","COST-SET"], sort=False):
            v_end = _last_at_or_before(g, end)
            if cs == CUM_SET:
                v_start = _last_at_or_before(g, start - pd.Timedelta(nanoseconds=1))
                val = v_end - v_start
            else:  # ETC
                val = v_end
            if val != 0.0:
                out_rows.append((chg, cs, val))
        pieces.append(pd.DataFrame(out_rows, columns=["CHG#","COST-SET","VALUE"]))

    if pieces:
        long = pd.concat(pieces, ignore_index=True)
    else:
        long = pd.DataFrame(columns=["CHG#","COST-SET","VALUE"])

    # Ensure all cost-set columns exist later
    return long

def pivot_by_plug(long_df):
    """Rows = CHG#, Cols = COST-SET with COST_ORDER; add TOTAL row."""
    if long_df.empty:
        return pd.DataFrame(columns=COST_ORDER, index=[])

    pt = (long_df.pivot(index="CHG#", columns="COST-SET", values="VALUE")
                  .reindex(columns=COST_ORDER, fill_value=0)
                  .fillna(0)
                  .sort_index())
    pt.loc["TOTAL"] = pt.sum(numeric_only=True, axis=0)
    return pt.round(4)

# ---------------- BUILD TABLES ----------------
# 1) CUMULATIVE (start at file start to AS_OF)
cum_long  = window_aggregate(df, start=FILE_START_DATE, end=AS_OF)
cum_pivot = pivot_by_plug(cum_long)

# 2) 4-WEEK (last 4 accounting weeks)
wk4_long  = window_aggregate(df, start=fourwk_start, end=fourwk_end)
wk4_pivot = pivot_by_plug(wk4_long)

# 3) STATUS PERIOD (the accounting week containing AS_OF)
stat_long  = window_aggregate(df, start=status_week_start, end=status_week_end)
stat_pivot = pivot_by_plug(stat_long)

# ---------------- SHOW ----------------
print(f"As-of: {AS_OF.date()}  |  Status week: {status_week_start.date()}–{status_week_end.date()}  (week start={WEEK_START_DAY})\n")

print("CUMULATIVE (by CHG#):")
display(cum_pivot)

print("\n4-WEEK (last 4 accounting weeks, by CHG#):")
display(wk4_pivot)

print("\nCURRENT STATUS PERIOD (by CHG#):")
display(stat_pivot)
# =========================================================================================