# ---------------------------------------------------------------------
# MASTER PIPELINE :  Cognos MRR  ➜  SCM Requisition  ➜  Open‑Order Report
# ---------------------------------------------------------------------
import pandas as pd, re, sys
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. LOAD (unchanged)
def load(path, sheet_name=0, skiprows=None, debug=""):
    df = pd.read_excel(path, sheet_name=sheet_name, skiprows=skiprows)
    print(f"[✓] {debug or Path(path).name:<28}  {df.shape[0]:>7,} × {df.shape[1]}")
    return df

MRR_DF      = load(mrr_file,            sheet_name="MRR",  debug="MRR")
SCM_REQ_DF  = load(scm_requisition_file, skiprows=2,        debug="SCM Requisition")
OPEN_ORD_DF = load(open_order_file,     sheet_name="Data", debug="Open‑Order")

# -----------------------------------------------------------------------------
# 2. CLEAN  – unify headers and discover key columns
def std_cols(df):
    df.columns = (
        df.columns
          .str.lower()
          .str.strip()
          .str.replace(r'[^a-z0-9]+', '_', regex=True)     # spaces, #, /, etc.
          .str.replace(r'(^_+|_+$)', '', regex=True)       # trim leading/trailing _
    )
    return df

MRR_DF, SCM_REQ_DF, OPEN_ORD_DF = map(std_cols, [MRR_DF, SCM_REQ_DF, OPEN_ORD_DF])

# ---- find the receipt‑date column in MRR ----
date_candidates = [c for c in MRR_DF.columns if 'receipt' in c and 'date' in c]
if not date_candidates:
    sys.exit("❌ Couldn’t find a receipt‑date column in MRR.  Run MRR_DF.columns to inspect.")
receipt_col = date_candidates[0]
print(f"[i] Using '{receipt_col}' as MRR receipt date")

# ---- find requisition creation date in SCM report ----
req_date_candidates = [c for c in SCM_REQ_DF.columns if re.search(r'creation.*date', c)]
if not req_date_candidates:
    sys.exit("❌ Couldn’t find a requisition‑creation‑date column in SCM report.")
scm_date_col = req_date_candidates[0]
print(f"[i] Using '{scm_date_col}' as SCM requisition date")

# ---- normalise key names we’ll join on ----
def rename_if_exists(df, current_name, new_name):
    if current_name in df.columns: df.rename(columns={current_name: new_name}, inplace=True)

rename_if_exists(SCM_REQ_DF, 'req_',              'req_num')
rename_if_exists(MRR_DF,     'req_',              'req_num')
rename_if_exists(SCM_REQ_DF, 'mcpr_order_number', 'mcpr_order_number')
rename_if_exists(OPEN_ORD_DF,'mcpr_order_number', 'mcpr_order_number')

# -----------------------------------------------------------------------------
# 3. TYPE CONVERSION + FILTERS
MRR_DF[receipt_col]          = pd.to_datetime(MRR_DF[receipt_col], errors='coerce')
SCM_REQ_DF[scm_date_col]     = pd.to_datetime(SCM_REQ_DF[scm_date_col], errors='coerce')

latest_received = MRR_DF[receipt_col].max()
print(f"[i] Latest receipt in MRR  →  {latest_received:%Y‑%m‑%d}")

is_not_rejected = ~SCM_REQ_DF['req_status'].str.contains('reject|return', case=False, na=False)
SCM_REQ_FILT = SCM_REQ_DF[(SCM_REQ_DF[scm_date_col] >= latest_received) & is_not_rejected].copy()

# -----------------------------------------------------------------------------
# 4. MERGES
STEP1 = SCM_REQ_FILT.merge(
    MRR_DF,
    on       ='req_num',           # same name in both after rename
    how      ='inner',
    suffixes =('_scm', '_mrr')
)

# Only “true planned replenishments & commitments”
flag_col  = next((c for c in STEP1.columns if 'commitment' in c), None)
if flag_col:
    STEP1 = STEP1[STEP1[flag_col].str.contains('planned|commit', case=False, na=False)]

FINAL = STEP1.merge(OPEN_ORD_DF, on='mcpr_order_number', how='left')

# -----------------------------------------------------------------------------
# 5. EXPORT
out_path = "data/output/mrr_scm_openorder_pipeline.xlsx"
with pd.ExcelWriter(out_path, engine='xlsxwriter') as xl:
    MRR_DF.to_excel(xl,        'RAW_MRR',           index=False)
    SCM_REQ_FILT.to_excel(xl,  'FILTERED_SCM_REQ',  index=False)
    STEP1.to_excel(xl,         'PLANNED_COMMIT',    index=False)
    FINAL.to_excel(xl,         'FINAL_MERGE',       index=False)

print(f"[✓] Pipeline finished – results written to →  {out_path}")