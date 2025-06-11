# ──────────────────────────────────────────────────────────────────────────
# ALL‑IN‑ONE  |  MRR  +  SCM Requisition  +  Open‑Order Report  (v2)
# ──────────────────────────────────────────────────────────────────────────
import pandas as pd, re, sys
from pathlib import Path

# ──── USER CONFIG ────────────────────────────────────────────────────────
MRR_SHEET          = "MRR"          # ← adjust if your sheet name changes
OPEN_ORDER_SHEET   = "Data"
DATE_KEY_MRR       = "need_by_date" # col used for Month / Year labels
DATE_KEY_SCM       = "req_creation_date"
KEY_REQ_NUM        = "req_num"
KEY_MCPR           = "mcpr_order_number"
OUTPUT_FILE        = "data/output/mrr_scm_openorder_pipeline.xlsx"
# ─────────────────────────────────────────────────────────────────────────

def std_cols(df):
    """snake‑case all headers so merges don’t break when users rename things."""
    df.columns = (df.columns.str.lower()
                            .str.strip()
                            .str.replace(r'[^a-z0-9]+', '_', regex=True)
                            .str.replace(r'^_+|_+$', '', regex=True))
    return df

def load(path, sheet_name=0, skiprows=None, debug=""):
    df = pd.read_excel(path, sheet_name=sheet_name, skiprows=skiprows)
    print(f"[✓] {debug or Path(path).name:<30}  {df.shape[0]:>7,} × {df.shape[1]}")
    return std_cols(df)

def add_month_year(df, date_col):
    if date_col not in df.columns:
        print(f"[!] '{date_col}' not found – skipping Month/Year labels.")
        return df
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df["month"] = df[date_col].dt.month
    df["year"]  = df[date_col].dt.year
    return df

def add_promise_flag(df, source_col="order_promised_date"):
    if source_col not in df.columns:
        print(f"[!] '{source_col}' not found – skipping Promise‑Date flag.")
        return df
    df["promise_date_flag"] = df[source_col].notna().map({True: "yes", False: "no"})
    return df

# ──── 1. LOAD RAW FILES ──────────────────────────────────────────────────
MRR_DF      = load(mrr_file,            sheet_name=MRR_SHEET,  debug="MRR")
SCM_DF      = load(scm_requisition_file, skiprows=2,            debug="SCM Requisition")
OPEN_DF     = load(open_order_file,     sheet_name=OPEN_ORDER_SHEET, debug="Open‑Order")

# ──── 2. KEEP MRR “HELPER” COLS SAFE – strip then re‑add later ──────────
HELPER_COLS = ["month", "year", "vpp_billing", "promise_date_flag"]
MRR_HELPERS = MRR_DF[HELPER_COLS].copy() if set(HELPER_COLS) <= set(MRR_DF.columns) else None
MRR_DF = MRR_DF.drop(columns=[c for c in HELPER_COLS if c in MRR_DF.columns])

# ──── 3. TIDY IDENTIFIERS & DEDUPLICATE SCM ─────────────────────────────
for df in (MRR_DF, SCM_DF):
    rename_map = {c: KEY_REQ_NUM      for c in df.columns if re.fullmatch(r'req[_ ]?#', c)}
    rename_map.update({c: KEY_MCPR    for c in df.columns if 'mcpr' in c})
    df.rename(columns=rename_map, inplace=True)

OPEN_DF.rename(columns={c: KEY_MCPR for c in OPEN_DF.columns if 'mcpr' in c}, inplace=True)
OPEN_DF.rename(columns={c: KEY_REQ_NUM for c in OPEN_DF.columns if 'requisition' in c}, inplace=True)

# SCM datetime & dedup
SCM_DF[DATE_KEY_SCM] = pd.to_datetime(SCM_DF[DATE_KEY_SCM], errors='coerce')
SCM_DEDUP = (SCM_DF
             .sort_values(DATE_KEY_SCM)
             .drop_duplicates(KEY_REQ_NUM, keep='last'))

# ──── 4. MERGE  (1‑to‑1 guaranteed) ──────────────────────────────────────
STEP1 = (MRR_DF
         .merge(SCM_DEDUP[[KEY_REQ_NUM, KEY_MCPR, "req_status"]],
                on=KEY_REQ_NUM, how='left'))

FINAL = STEP1.merge(OPEN_DF[[KEY_MCPR, "vpp_billing_rate"]],
                    on=KEY_MCPR, how='left')

# ──── 5. RE‑ADD / GENERATE LABEL COLUMNS ────────────────────────────────
# regenerate Month & Year if they were removed or missing
FINAL = add_month_year(FINAL, DATE_KEY_MRR)
FINAL = add_promise_flag(FINAL, source_col="order_promised_date")

# if original helper cols existed keep original values (overwrite regenerated)
if MRR_HELPERS is not None:
    FINAL[MRR_HELPERS.columns] = MRR_HELPERS

# ──── 6. EXPORT  – raw + interim + final for audit trail ────────────────
OUTPUT_FILE = Path(OUTPUT_FILE)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as xl:
    MRR_DF.to_excel( xl, "RAW_MRR",           index=False)
    SCM_DEDUP.to_excel(xl, "SCM_DEDUP",       index=False)
    OPEN_DF.to_excel(xl, "OPEN_ORDER_RAW",    index=False)
    FINAL.to_excel(xl,  "FINAL_MERGE",        index=False)

print(f"[✓] Pipeline complete → {OUTPUT_FILE.resolve()}")