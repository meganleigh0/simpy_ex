# ---------------------------------------------------------------------
# MASTER PIPELINE :  Cognos MRR  ➜  SCM Requisition  ➜  Open‑Order Report
# ---------------------------------------------------------------------
import pandas as pd
from pathlib import Path

# ---------------------------- 1. LOAD --------------------------------
def load_excel(path, sheet_name=0, skiprows=None, usecols=None, debug_name=""):
    """Light wrapper around pd.read_excel with a quick shape printout."""
    df = pd.read_excel(path, sheet_name=sheet_name, skiprows=skiprows, usecols=usecols)
    print(f"[✓] {debug_name or Path(path).name}  →  {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df

MRR_DF           = load_excel(mrr_file,            sheet_name="MRR", debug_name="MRR")
SCM_REQ_DF       = load_excel(scm_requisition_file, skiprows=2,        debug_name="SCM Requisition")
OPEN_ORD_DF      = load_excel(open_order_file,     sheet_name="Data",  debug_name="Open‑Order Report")

# ---------------------------- 2. CLEAN --------------------------------
def standardize_cols(df):
    """Lower‑case & strip – avoids key errors during merges."""
    df.columns = df.columns.str.lower().str.strip()
    return df

MRR_DF, SCM_REQ_DF, OPEN_ORD_DF = map(standardize_cols, [MRR_DF, SCM_REQ_DF, OPEN_ORD_DF])

# Ensure critical columns are datetime
MRR_DF["receipt_date"]       = pd.to_datetime(MRR_DF["receipt_date"],       errors="coerce")
SCM_REQ_DF["req_creation_date"] = pd.to_datetime(SCM_REQ_DF["req_creation_date"], errors="coerce")

# ---------------------------- 3. FILTER --------------------------------
# 3‑A.  Latest received‑material date from MRR
latest_received = MRR_DF["receipt_date"].max()
print(f"Latest material receipt in MRR: {latest_received:%Y‑%m‑%d}")

# 3‑B.  SCM Reqs after that date, excluding rejected/returned
valid_status    = ~SCM_REQ_DF["req_status"].str.contains("reject|return", case=False, na=False)
SCM_REQ_FILT    = SCM_REQ_DF[
    (SCM_REQ_DF["req_creation_date"] >= latest_received) & valid_status
].copy()

# ---------------------------- 4. MATCH ---------------------------------
# 4‑A.  Match MCPR Order # (col J)  →  Req # (col B)
JOIN_KEYS_1 = ["mcpr_order_number", "req_#"]  # rename if your files differ
SCM_REQ_FILT.rename(columns={"req #": "req_#", "mcpr order number": "mcpr_order_number"}, inplace=True)

MRR_JOINED = (
    SCM_REQ_FILT
    .merge(MRR_DF, left_on="mcpr_order_number", right_on="req_#", how="inner",
           suffixes=("_req", "_mrr"))
)

# 4‑B.  “True planned replenishments & commitments” –‑
#       assume that exists as a flag column you can tweak here:
PLANNED_FLAG_COL = "commitment_type"   # <‑‑ change to the real column
planned_commit   = MRR_JOINED[MRR_JOINED[PLANNED_FLAG_COL].str.contains("planned|commit", case=False, na=False)]

# 4‑C.  Bring in open‑order lines (vendor progress payments)
OPEN_ORD_DF.rename(columns={"mcpr order number": "mcpr_order_number"}, inplace=True)

FINAL_DF = (
    planned_commit
    .merge(OPEN_ORD_DF,
           on="mcpr_order_number",
           how="left",
           suffixes=("", "_openord"))
)

# ---------------------------- 5. OUTPUT --------------------------------
out_path = "data/output/mrr_scm_openorder_pipeline.xlsx"
with pd.ExcelWriter(out_path, engine="xlsxwriter") as xl:
    MRR_DF.to_excel(xl,           sheet_name="RAW_MRR",            index=False)
    SCM_REQ_FILT.to_excel(xl,     sheet_name="FILTERED_SCM_REQ",   index=False)
    planned_commit.to_excel(xl,   sheet_name="PLANNED_COMMIT",     index=False)
    FINAL_DF.to_excel(xl,         sheet_name="FINAL_MERGE",        index=False)
print(f"[✓] Pipeline complete – outputs saved ➜  {out_path}")