import pandas as pd
import os

def export_to_powerbi(program_id, snapshot_date,
                      cost_df, schedule_df, evms_df, labor_df, labor_monthly_df):

    outdir = r"C:\EVMS_PBI"
    os.makedirs(outdir, exist_ok=True)

    filename = f"{program_id}_{snapshot_date}.xlsx"
    filepath = os.path.join(outdir, filename)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        cost_df.to_excel(writer, sheet_name="cost_performance", index=False)
        schedule_df.to_excel(writer, sheet_name="schedule_performance", index=False)
        evms_df.to_excel(writer, sheet_name="evms_metrics", index=False)
        labor_df.to_excel(writer, sheet_name="labor_hours", index=False)
        labor_monthly_df.to_excel(writer, sheet_name="labor_monthly", index=False)

        # metadata sheet for easy Power BI linking
        meta = pd.DataFrame({
            "Program": [program_id],
            "SnapshotDate": [snapshot_date]
        })
        meta.to_excel(writer, sheet_name="meta", index=False)

    print(f"Saved Power BI export: {filepath}")
    
    
    export_to_powerbi(
    "XM30",
    "2025-11-13",
    cost_performance_tbl,
    schedule_performance_tbl,
    evms_metrics_tbl,
    labor_tbl,
    labor_monthly_tbl
)
    