import os
import pandas as pd

# ------------------------------
# CONFIG: set these for each run
# ------------------------------
PROGRAM_ID    = "XM30"
SNAPSHOT_DATE = "2025-11-13"   # or use date.today().strftime("%Y-%m-%d")

# ------------------------------
# Helper to ensure SUB_TEAM is a column
# ------------------------------
def ensure_subteam_column(df, index_name="SUB_TEAM"):
    """
    Returns a copy of df where SUB_TEAM is a normal column.
    Handles the case where SUB_TEAM is currently the index.
    """
    if df is None:
        return None

    df = df.copy()

    # If SUB_TEAM is the index name, reset it to a column
    if df.index.name == index_name:
        df.reset_index(inplace=True)
    # If SUB_TEAM is part of a MultiIndex, also reset
    elif index_name in (df.index.names or []):
        df.reset_index(inplace=True)

    # If after this there is still no SUB_TEAM column, but there is an 'index' column,
    # rename it as a last resort
    if index_name not in df.columns and "index" in df.columns:
        df.rename(columns={"index": index_name}, inplace=True)

    return df

# ------------------------------
# Export all tables to ONE Excel file
# ------------------------------
outdir = "pbi_exports"
os.makedirs(outdir, exist_ok=True)

filename = f"{PROGRAM_ID}_{SNAPSHOT_DATE}.xlsx"
filepath = os.path.join(outdir, filename)

with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
    # Cost Performance
    if "cost_performance_tbl" in globals():
        df_cost = ensure_subteam_column(cost_performance_tbl)
        df_cost.to_excel(writer, sheet_name="cost_performance", index=False)

    # Schedule Performance
    if "schedule_performance_tbl" in globals():
        df_sched = ensure_subteam_column(schedule_performance_tbl)
        df_sched.to_excel(writer, sheet_name="schedule_performance", index=False)

    # EVMS Metrics
    if "evms_metrics_tbl" in globals():
        df_evms = ensure_subteam_column(evms_metrics_tbl)
        df_evms.to_excel(writer, sheet_name="evms_metrics", index=False)

    # Labor Hours
    if "labor_tbl" in globals():
        df_labor = ensure_subteam_column(labor_tbl)
        df_labor.to_excel(writer, sheet_name="labor_hours", index=False)

    # Monthly Labor
    if "labor_monthly_tbl" in globals():
        df_labor_m = ensure_subteam_column(labor_monthly_tbl)
        df_labor_m.to_excel(writer, sheet_name="labor_monthly", index=False)

    # Optional metadata sheet
    meta = pd.DataFrame({
        "Program": [PROGRAM_ID],
        "SnapshotDate": [SNAPSHOT_DATE]
    })
    meta.to_excel(writer, sheet_name="meta", index=False)

print(f"✅ Saved Power BI export to: {filepath}")