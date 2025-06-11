# %%
import pandas as pd

def build_mrr_pipeline(
    mrr_path: str,
    scm_path: str,
    open_order_path: str | None = None,
    statuses_to_remove: tuple[str, ...] = ("RETURN", "REJECTED", "CANCEL", "CANCELLED"),
) -> pd.DataFrame:
    """ETL pipeline for MRR → add derived cols → de‑scope rejected/returned material."""
    
    # ──────────────────── 1) LOAD & PREP MRR ────────────────────
    mrr = pd.read_excel(mrr_path, sheet_name="MRR")
    mrr.columns = mrr.columns.str.strip()          # tidy headers
    
    # Remove any stale derived cols left over from prior runs
    cols_to_drop = [c for c in ("Month", "Year", "Promise Date", "VPP Billing") if c in mrr.columns]
    mrr = mrr.drop(columns=cols_to_drop, errors="ignore")
    
    # Convert Need‑By to datetime and add Month / Year
    mrr["Need By Date"] = pd.to_datetime(mrr["Need By Date"], errors="coerce")
    mrr["Month"] = mrr["Need By Date"].dt.month
    mrr["Year"]  = mrr["Need By Date"].dt.year
    
    # Promise‑date flag (True/False)
    mrr["Promise Date"] = mrr["Order Promised Date"].notna()
    
    # ──────────────────── 2) LOAD & PREP SCM ────────────────────
    scm = (
        pd.read_excel(scm_path, skiprows=2)        # two header rows in your file
          .rename(columns=lambda x: x.strip())
          .loc[:, ["REQ #", "REQ STATUS"]]
    )
    
    # ──────────────────── 3) MERGE & FILTER ────────────────────
    merged = mrr.merge(scm, how="left",
                       left_on="MCPR Order Number", right_on="REQ #")
    
    # Normalise status text for filtering
    merged["REQ STATUS"] = merged["REQ STATUS"].str.upper().str.strip()
    
    mask_keep = ~merged["REQ STATUS"].isin([s.upper() for s in statuses_to_remove]) | merged["REQ STATUS"].isna()
    filtered = merged.loc[mask_keep].copy()
    
    # ──────────────────── 4) (OPTIONAL) VPP BILLING RATE ────────────────────
    if open_order_path:
        open_order = (
            pd.read_excel(open_order_path, sheet_name="Data")
              .rename(columns=lambda x: x.strip())
              .loc[:, ["Requisition Num", "VPP Billing Rate"]]
              .rename(columns={"Requisition Num": "MCPR Order Number",
                               "VPP Billing Rate": "VPP Billing"})
        )
        filtered = filtered.merge(open_order, how="left", on="MCPR Order Number")
    
    # Final tidy‑up (optional sort)
    filtered = (
        filtered.sort_values(["Year", "Month", "Item Number"], ignore_index=True)
                .convert_dtypes()                 # lighter dtypes
    )
    
    return filtered


# ──────────────────── RUN THE PIPELINE ────────────────────
final_df = build_mrr_pipeline(
    mrr_path="data/ARHC 448 2025 MRR.xlsx",
    scm_path="data/SCM Requisition Report.xlsx",
    open_order_path="data/05-27-2025_Open Order Report - ALL VB_w-PIVOTS.xlsx"
)

# Example: save to Excel
# final_df.to_excel("clean_MRR_output.xlsx", index=False)

final_df.head()