# ================== THREE DASHBOARD TABLES (by CHG#): CUM • 4 WEEKS • STATUS ==================
import pandas as pd
from IPython.display import display

# ---------------- CONFIG ----------------
FILE_START_DATE = pd.Timestamp("2025-10-15")   # <-- your file's start
WEEK_START_DAY  = "MON"                        # MON/TUE/WED/THU/FRI/SAT/SUN
FOUR_WK_LEN     = 4                            # number of accounting weeks in the 4WK block
COST_ORDER      = ["ACWP","BCWP","BCWS","ETC"]

# Optional: cap AS_OF to today's date
# TODAY_CAP = pd.Timestamp("today").normalize()

# ---------------- DATA (uses your in-memory DF) ----------------
df = xm30_cobra_export_weekly_extract.copy()

# Keep your column names; just clean types/values and normalize COST-SET labels a bit
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df = df.dropna(subset=["DATE"]).copy()
df["COST-SET"] = (
    df["COST-SET"].astype(str).str.upper().str.replace(r"\s+", "", regex=True)
    .replace({"BCWS": "BCWS", "ACWP": "ACWP", "BCWP": "BCWP", "ETC": "ETC"})
)
df["HOURS"] = pd.to_numeric(df["HOURS"], errors="coerce").fillna(0.0)

# -------- anchor cumulative to the file's start (CUM = 2025-10-15..AS_OF) --------
df = df[df["DATE"] >= FILE_START_DATE].copy()
AS_OF = df["DATE"].max().normalize()
# If you want to auto-limit to "today", un-comment:
# if AS_OF > TODAY_CAP: AS_OF = TODAY_CAP

# ---------------- ACCOUNTING WEEK BOUNDARIES ----------------
DAY_TO_NUM = {"MON":0,"TUE":1,"WED":2,"THU":3,"FRI":4,"SAT":5,"SUN":6}
anchor = DAY_TO_NUM[WEEK_START_DAY.upper()]

def week_start(s):
    dow = s.dt.dayofweek
    offset = (dow - anchor) % 7
    return s - pd.to_timedelta(offset, unit="D")

df["WEEK_START"] = week_start(df["DATE"])
status_week_start = week_start(pd.Series([AS_OF]))[0]
status_week_end   = status_week_start + pd.Timedelta(days=6)

# last FOUR_WK_LEN distinct accounting weeks (ending at AS_OF)
weeks_le_asof = (df.loc[df["DATE"] <= AS_OF, "WEEK_START"].drop_duplicates().sort_values())
last_k_weeks  = weeks_le_asof.tail(FOUR_WK_LEN).tolist()
fourwk_start  = min(last_k_weeks)
fourwk_end    = status_week_end  # align to end of AS_OF week

# ---------------- MIXED-AGG HELPERS ----------------
FLOW_SETS = {"ACWP","BCWP"}   # summed over window
CUM_SET   = "BCWS"            # cumulative → delta over window
PTI_SET   = "ETC"             # point-in-time → last at end

def _last_at_or_before(g, cutoff):
    """Return last HOURS value ≤ cutoff within group g (already one CHG# x COST-SET); 0.0 if none."""
    g2 = g[g["DATE"] <= cutoff]
    if g2.empty:
        return 0.0
    return float(g2.sort_values("DATE")["HOURS"].iloc[-1])

def window_aggregate(df_win, start, end):
    """
    Build long table with mixed aggregation for a window [start, end].
    Returns columns: CHG#, COST-SET, VALUE
    """
    pieces = []

    # FLOW: ACWP, BCWP sum across the window
    flow_mask = (df_win["DATE"] >= start) & (df_win["DATE"] <= end) & (df_win["COST-SET"].isin(FLOW_SETS))
    if flow_mask.any():
        flow = (df_win[flow_mask]
                .groupby(["CHG#","COST-SET"], as_index=False)["HOURS"].sum()
                .rename(columns={"HOURS":"VALUE"}))
        pieces.append(flow)

    # BCWS (cumulative delta) and ETC (last-at-end)
    for cs in [CUM_SET, PTI_SET]:
        sub = df_win[df_win["COST-SET"] == cs][["CHG#","COST-SET","DATE","HOURS"]]
        if sub.empty:
            continue
        rows = []
        for (chg, _), g in sub.groupby(["CHG#","COST-SET"], sort=False):
            v_end = _last_at_or_before(g, end)
            if cs == CUM_SET:
                v_start = _last_at_or_before(g, start - pd.Timedelta(nanoseconds=1))
                val = v_end - v_start
            else:  # ETC
                val = v_end
            if val != 0.0:
                rows.append((chg, cs, val))
        if rows:
            pieces.append(pd.DataFrame(rows, columns=["CHG#","COST-SET","VALUE"]))

    if pieces:
        long = pd.concat(pieces, ignore_index=True)
    else:
        long = pd.DataFrame(columns=["CHG#","COST-SET","VALUE"])
    return long

def pivot_by_plug(long_df):
    """Rows = CHG#, Cols = COST-SET; add TOTAL row; keep all four columns."""
    pt = (long_df.pivot(index="CHG#", columns="COST-SET", values="VALUE")
                  .reindex(columns=COST_ORDER, fill_value=0)
                  .fillna(0)
                  .sort_index())
    pt.loc["TOTAL"] = pt.sum(numeric_only=True, axis=0)
    return pt.round(4)

# ---------------- BUILD THE THREE TABLES ----------------
# 1) CUM (FILE_START_DATE .. AS_OF)
cum_long  = window_aggregate(df, start=FILE_START_DATE, end=AS_OF)
cum_pivot = pivot_by_plug(cum_long)

# 2) Last 4 accounting weeks (fourwk_start .. fourwk_end)
wk4_long  = window_aggregate(df, start=fourwk_start, end=fourwk_end)
wk4_pivot = pivot_by_plug(wk4_long)

# 3) Status period (status_week_start .. status_week_end)
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
# ======================================================================================