import os
import pandas as pd

# ------------------------------
# CONFIG
# ------------------------------
PROGRAM_ID    = "XM30"
SNAPSHOT_DATE = "2025-11-13"   # or use date.today().strftime("%Y-%m-%d")

outdir = "pbi_exports"
os.makedirs(outdir, exist_ok=True)
filename = f"{PROGRAM_ID}_{SNAPSHOT_DATE}.xlsx"
filepath = os.path.join(outdir, filename)

# ------------------------------
# HELPERS
# ------------------------------

def ensure_subteam_column(df):
    """
    Ensure SUB_TEAM is a real column and is the FIRST column.
    Handles cases where SUB_TEAM is the index or missing.
    """
    if df is None:
        return None

    df = df.copy()

    # If no SUB_TEAM column, but index seems to be subteam codes, use index
    if "SUB_TEAM" not in df.columns:
        # take index values as SUB_TEAM
        subteam = df.index.astype(str)
        df.insert(0, "SUB_TEAM", subteam)
    else:
        # make sure it's first column
        cols = ["SUB_TEAM"] + [c for c in df.columns if c != "SUB_TEAM"]
        df = df[cols]

    # reset index so it doesn't appear in Excel
    df.reset_index(drop=True, inplace=True)
    return df


def ensure_metric_column(df, metric_col_name="METRIC"):
    """
    Ensure a METRIC column exists (for SPI/CPI) and is the FIRST column.
    Assumes index currently holds the metric names (SPI, CPI).
    """
    if df is None:
        return None

    df = df.copy()

    if metric_col_name not in df.columns:
        metric_vals = df.index.astype(str)
        df.insert(0, metric_col_name, metric_vals)
    else:
        cols = [metric_col_name] + [c for c in df.columns if c != metric_col_name]
        df = df[cols]

    df.reset_index(drop=True, inplace=True)
    return df

# ------------------------------
# EXPORT
# ------------------------------

with pd.ExcelWriter(filepath, engine="openpyxl") as writer:

    # Cost Performance (CPI)
    if "cost_performance_tbl" in globals():
        df_cost = ensure_subteam_column(cost_performance_tbl)
        df_cost.to_excel(writer, sheet_name="cost_performance", index=False)

    # Schedule Performance (SPI)
    if "schedule_performance_tbl" in globals():
        df_sched = ensure_subteam_column(schedule_performance_tbl)
        df_sched.to_excel(writer, sheet_name="schedule_performance", index=False)

    # EVMS Metrics (SPI/CPI rows)
    if "evms_metrics_tbl" in globals():
        df_evms = ensure_metric_column(evms_metrics_tbl, metric_col_name="METRIC")
        df_evms.to_excel(writer, sheet_name="evms_metrics", index=False)

    # Labor Hours
    if "labor_tbl" in globals():
        df_labor = ensure_subteam_column(labor_tbl)
        df_labor.to_excel(writer, sheet_name="labor_hours", index=False)

    # Monthly Labor
    if "labor_monthly_tbl" in globals():
        df_labor_m = ensure_subteam_column(labor_monthly_tbl)
        df_labor_m.to_excel(writer, sheet_name="labor_monthly", index=False)

    # Optional meta sheet
    meta = pd.DataFrame({
        "Program": [PROGRAM_ID],
        "SnapshotDate": [SNAPSHOT_DATE]
    })
    meta.to_excel(writer, sheet_name="meta", index=False)

print(f"✅ Saved Power BI export to: {filepath}")