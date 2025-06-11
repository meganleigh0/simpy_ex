# ---------------------------------------------
# UPDATE BURDEN‑RATE MASTER FROM "OTHER RATES"
# ---------------------------------------------
import pandas as pd
from pathlib import Path

## ── CONFIG ──────────────────────────────────
SOURCE_PATH  = Path("/dbfs/FileStore/raw/other_rates.xlsx")
TARGET_PATH  = Path("/dbfs/FileStore/raw/burden_rate_import.xlsx")
SOURCE_SHEET = "Rates"          # sheet that holds the I‑to‑M yearly numbers
TARGET_SHEET = "BURDEN_RATE"    # sheet that holds the 28‑col master

# 1) mapping:  (pattern_in_description, pattern_in_pool) → column‑in‑target
MAPPING = {
    ("PSGA - CSSC G & A ALLOWABLE", "CSSC"): "G&A CSSC",
    ("DVGA - DIVISION GENERAL",      "GDLS"): "G&A GDLS",
    ("DeptAA ALLOWABLE REORDER",     None)  : "ROP",
    ("PRLS - GDLS PROCUREMENT",      "GDLS"): "AbatProcOH",
    ("FRT - FREIGHT",                None)  : "FREIGHT",
    ("GENERAL DYNAMICS LAND SYSTEMS",None)  : "PROCUREMENT",
    ("DeptAlN ALLOWABLE MAJOR END",  None)  : "MAJOR END ITEM",
    ("ALLOWABLE SUPPORT RATE",       None)  : "SUPPORT",
    ("ALLOWABLE CONTL TEST RATE",    None)  : "CONTL TEST",
}

# 2) rounding rules per column name
def _decimals(col):
    if "Labor" in col:            # “labor dollars” fields
        return 3
    if col.strip().upper() == "COM":
        return 6
    return 5                      # all other burden %’s

## ── LOAD WORKBOOKS ──────────────────────────
src = pd.read_excel(SOURCE_PATH, sheet_name=SOURCE_SHEET, header=0)
tgt = pd.read_excel(TARGET_PATH, sheet_name=TARGET_SHEET, header=0)

# If your source’s header row is offset (e.g. “Unnamed: 0”), fix once:
if src.columns[0].startswith("Unnamed"):
    src.columns = src.iloc[0]
    src = src[1:].reset_index(drop=True)

## ── RESHAPE SOURCE TO LONG FORMAT ───────────
# Keep meta columns then melt the year columns
meta_cols = ["Burden Pool", "Description"]
year_cols = [c for c in src.columns if isinstance(c, (int, float, str)) and str(c).isdigit()]
src_long = src.melt(id_vars=meta_cols, value_vars=year_cols,
                    var_name="Year", value_name="Rate").dropna(subset=["Rate"])

## ── CORE UPDATE LOOP ────────────────────────
for (desc_pat, pool_pat), tgt_col in MAPPING.items():

    # --- 1. locate the relevant rows in the source ---
    mask = src_long["Description"].str.contains(desc_pat, case=False, na=False)
    if pool_pat:  # sometimes pool must match too
        mask &= src_long["Burden Pool"].str.contains(pool_pat, case=False, na=False)

    rows = src_long.loc[mask, ["Year", "Rate"]]
    if rows.empty:
        print(f"[warn] no data for mapping → {tgt_col}")
        continue

    # --- 2. shove those values into target ---
    for yr, val in rows.itertuples(index=False):
        idx = (tgt["Date"] == int(yr))  # target has a 'Date' (FY) col
        tgt.loc[idx, tgt_col] = round(val, _decimals(tgt_col))

    # --- 3. forward‑fill future fiscal years with last known value ---
    last_val = round(rows.sort_values("Year")["Rate"].iloc[-1], _decimals(tgt_col))
    future_mask = (tgt["Date"] > rows["Year"].max())
    tgt.loc[future_mask, tgt_col] = tgt.loc[future_mask, tgt_col].fillna(last_val)

## ── SAVE RESULT ─────────────────────────────
with pd.ExcelWriter(TARGET_PATH.with_stem(TARGET_PATH.stem + "_updated"),
                    engine="openpyxl", mode="w") as xls:
    tgt.to_excel(xls, sheet_name=TARGET_SHEET, index=False)

print("✅ burden‑rate workbook updated →", TARGET_PATH.with_stem(TARGET_PATH.stem + "_updated"))