# -*- coding: utf-8 -*-
import re
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

# ==================== CONFIG ====================
BURDEN_IMPORT_PATH = "data/BurdenRateImport.xlsx"   # your import template
RATES_TABLE_PATH   = "data/20220511FPRP_Burden.xlsx"  # your rates table (year + rate columns)
OUT_DIR = Path("out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

RATES_SHEET_HINT  = "burden"   # first sheet containing this (case-insensitive)

# Rounding
ROUND_COM = 6
ROUND_OTH = 5

# ==================== HARD-CODED MATRIX LOGIC (from image) ====================
# We rely on description and burden segment to infer applicability.
# Core idea from the matrix:
#  - Two blocks: “Spares Allocation does NOT apply” vs “Spares Allocation DOES apply”.
#  - G&A and COM are only applied when the row falls in the “DOES apply” block.
#  - ODC/Travel have G&A/COM applied as well (per bottom ODC block in the image).
#
# You can extend/tweak these lists/patterns easily.

# Patterns that mean “this line is a spares allocation line”
SPARES_KEYWORDS = (
    r"\bspares?\b", r"\bspare\b", r"\bspares allocation\b",
    r"\bintra[-\s]?segment deliverable spares\b",
    r"\bdeliverable spares\b", r"\bnon deliverable spares\b",
)

# Description tokens that we treat as ODC/Travel lines (bottom block in the matrix)
ODC_TOKENS  = (r"\bodc\b", r"\bother direct costs?\b")
TRVL_TOKENS = (r"\btrvl\b", r"\btravel\b")

# PCURE heuristics (lightweight, segment-aware — adjust as needed)
# If a rate column header contains "pcure gdls" we apply it when segment=GDLS; same for CSSC.
# “abated pcure” applied for spares rows (visible X’s in the spares block), otherwise off.
# If your headers differ slightly, the keyizer (below) keeps this robust.
PCURE_APPLIES_ON_SPARES_ONLY = True

# ==================== HELPERS ====================
def keyize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")

def round_half_up(x, places):
    if pd.isna(x): return np.nan
    q = Decimal("1").scaleb(-places)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))

def carry_forward(series_by_year: pd.Series, needed_years):
    if series_by_year is None or series_by_year.empty:
        return pd.Series(index=needed_years, dtype=float)
    s = pd.to_numeric(series_by_year, errors="coerce").dropna().sort_index()
    full_span = list(range(int(s.index.min()), max(int(s.index.max()), max(needed_years)) + 1))
    return s.reindex(full_span).ffill().reindex(needed_years)

def first_word_code(desc: str) -> str:
    if not isinstance(desc, str): return ""
    return re.sub(r"[^A-Za-z0-9]+", "", desc.strip().split()[0]).upper()

def bool_any(patterns, text):
    t = (text or "").lower()
    return any(re.search(p, t, flags=re.I) for p in patterns)

# ==================== LOAD: BURDEN IMPORT ====================
imp = pd.read_excel(BURDEN_IMPORT_PATH, dtype=object)
imp.columns = [str(c).strip() for c in imp.columns]
original_cols = imp.columns.tolist()  # preserve exact order for export

# Key columns (tolerant matching)
def pick_col(df, hints):
    lc = {c.lower(): c for c in df.columns}
    for h in hints:
        for k, orig in lc.items():
            if h in k: return orig
    raise ValueError(f"Could not find any of {hints} in {list(df.columns)}")

burden_col = pick_col(imp, ("burden",))
desc_col   = pick_col(imp, ("description", "desc"))
date_col   = pick_col(imp, ("date",))
eff_col    = pick_col(imp, ("effective", "effective date"))

# Rate columns: everything to the right of Date that looks like a rate
date_idx = imp.columns.get_loc(date_col)
candidate_rate_cols = imp.columns[date_idx+1:]
rate_cols = []
for c in candidate_rate_cols:
    sample = pd.to_numeric(imp[c], errors="coerce")
    if sample.notna().any() or re.search(r"(g&a|com|pcure|proc|award|fee|major|support|re.?order|spare)", c, flags=re.I):
        rate_cols.append(c)

rate_keys = {c: keyize(c) for c in rate_cols}

# Segment, tokens, year
imp["_seg"]  = imp[burden_col].astype(str).str.extract(r"(CSSC|GDLS)", flags=re.I, expand=False).str.upper().fillna("")
imp["_desc"] = imp[desc_col].astype(str)
imp["_code"] = imp["_desc"].apply(first_word_code)
imp["_year"] = pd.to_numeric(imp[date_col], errors="coerce").astype("Int64")

needed_years = sorted(imp["_year"].dropna().astype(int).unique().tolist())

# ==================== LOAD: RATES TABLE ====================
rt_xls = pd.ExcelFile(RATES_TABLE_PATH)
sheet = next((s for s in rt_xls.sheet_names if re.search(RATES_SHEET_HINT, s, flags=re.I)), rt_xls.sheet_names[0])
rt = pd.read_excel(rt_xls, sheet_name=sheet, dtype=object)
rt.columns = [str(c).strip() for c in rt.columns]

# Find year column
year_col = next((c for c in rt.columns if re.fullmatch(r"\d{4}|year|date", c, flags=re.I)), None)
if year_col is None:
    for c in rt.columns:
        vals = pd.to_numeric(rt[c], errors="coerce")
        if vals.notna().sum() and (vals.head(10).dropna().between(1900, 2100).all()):
            year_col = c; break
if year_col is None:
    raise ValueError("Could not find a Year/Date column in the rates table.")

rt[year_col] = pd.to_numeric(rt[year_col], errors="coerce").astype("Int64")
rt_keyed = rt.copy()
rt_keyed.columns = [year_col] + [keyize(c) for c in rt.columns if c != year_col]

# Build series per rate_key
ratekey_to_series = {}
for c in rt_keyed.columns:
    if c == year_col: continue
    s = rt_keyed.set_index(year_col)[c]
    s = pd.to_numeric(s, errors="coerce").dropna()
    if not s.empty:
        ratekey_to_series[c] = s

# Pre carry-forward
ratekey_to_cf = {rk: carry_forward(s, needed_years) for rk, s in ratekey_to_series.items()}

def value_for(rk: str, year: int):
    s = ratekey_to_cf.get(rk)
    if s is None: return np.nan
    return float(s.get(year, np.nan))

# ==================== MATRIX RULES ENGINE ====================
def applies_mask_for_row(seg: str, desc: str, code: str) -> dict:
    """
    Return a dict {rate_key: True/False} describing whether each rate column applies for this row,
    based on the hard-coded matrix logic visible in the image.
    """
    seg = (seg or "").upper()
    d = (desc or "")
    # Is this a 'Spares Allocation DOES apply' row?
    is_spares = bool_any(SPARES_KEYWORDS, d)

    # Is this ODC or TRAVEL?
    is_odc  = bool_any(ODC_TOKENS, d) or code.startswith("ODC")
    is_trvl = bool_any(TRVL_TOKENS, d) or code.startswith("TRVL")

    mask = {}
    for col, rk in rate_keys.items():
        rk_l = rk.lower()

        # G&A and COM only on spares or ODC/Travel (per matrix bottom block footnote)
        if "g_a_cssc" in rk_l or "ga_cssc" in rk_l:
            mask[rk] = (seg == "CSSC") and (is_spares or is_odc or is_trvl)
        elif "g_a_gdls" in rk_l or "ga_gdls" in rk_l:
            mask[rk] = (seg == "GDLS") and (is_spares or is_odc or is_trvl)
        elif re.search(r"\bcom_cssc\b", rk_l):
            mask[rk] = (seg == "CSSC") and (is_spares or is_odc or is_trvl)
        elif re.search(r"\bcom_gdls\b", rk_l):
            mask[rk] = (seg == "GDLS") and (is_spares or is_odc or is_trvl)

        # Spare Allocation column itself (if present): apply only when spares row
        elif "spare_allocation" in rk_l or "spares_allocation" in rk_l:
            mask[rk] = is_spares

        # PCURE per segment (visible columns in the matrix)
        elif "pcure_gdls" in rk_l:
            mask[rk] = (seg == "GDLS") and (is_spares if PCURE_APPLIES_ON_SPARES_ONLY else True)
        elif "pcure_cssc" in rk_l or "pcure_css" in rk_l:
            mask[rk] = (seg == "CSSC") and (is_spares if PCURE_APPLIES_ON_SPARES_ONLY else True)
        elif "abated_pcure" in rk_l or "abat_pcure" in rk_l or "abate_pcure" in rk_l:
            mask[rk] = is_spares  # X’s are shown in the spares block in the matrix

        # Reorder point / flags / misc columns (leave values as-is if rates table supplies them)
        elif re.search(r"re_?ord|reorder|flag|major_end|support|award|fee|proc", rk_l):
            mask[rk] = True  # keep data if rate exists; otherwise 0 will be written below

        else:
            # default: not applied unless a rate exists and we prefer to pass it through
            mask[rk] = True

    return mask

# ==================== UPDATE ====================
for idx, row in imp.iterrows():
    year = row["_year"]
    if pd.isna(year): 
        continue
    year = int(year)
    seg  = row["_seg"] if row["_seg"] else ""
    desc = row["_desc"]
    code = row["_code"]

    mask = applies_mask_for_row(seg, desc, code)

    for col in rate_cols:
        rk = rate_keys[col]
        # Only write a value if the column is known in the rates table
        if rk not in ratekey_to_cf:
            continue

        applies = bool(mask.get(rk, False))
        v = value_for(rk, year) if applies else 0.0

        # Rounding: COM to 6; others to 5
        places = ROUND_COM if re.search(r"\bcom\b", col, flags=re.I) else ROUND_OTH
        imp.at[idx, col] = round_half_up(v, places)

# Drop helpers and export in exact original shape
imp_export = imp.loc[:, original_cols]
out_path = OUT_DIR / "BurdenRateImportUpdate.xlsx"
imp_export.to_excel(out_path, sheet_name="update", index=False)
print(f"✅ BurdenRateImportUpdate → {out_path}")