# -*- coding: utf-8 -*-
import re
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

# ---------- PATHS ----------
RATESUM_PATH = "data/RATSUM.xlsx"            # your RateSum workbook
OTHER_SHEET  = "OTHER RATES"
COM_SHEET    = "COM SUMMARY"
BRI_PATH     = "data/BurdenRateImport.xlsx"  # your Burden Rate Import to update
OUT_DIR = Path("out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- ROUNDING ----------
ROUND_COM = 6
ROUND_OTH = 5

# ---------- HELPERS ----------
def round_half_up(x, places):
    if pd.isna(x): return np.nan
    q = Decimal("1").scaleb(-places)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))

def year_from_header(h):
    m = re.search(r"(?:CY)?\s*(20\d{2})", str(h))
    return int(m.group(1)) if m else None

def carry_forward(series_by_year: pd.Series, needed_years):
    if series_by_year is None or series_by_year.empty:
        return pd.Series(index=needed_years, dtype=float)
    s = pd.to_numeric(series_by_year, errors="coerce").dropna().sort_index()
    full = list(range(int(s.index.min()), max(int(s.index.max()), max(needed_years)) + 1))
    return s.reindex(full).ffill().reindex(needed_years)

def keyize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")

def pick_col(df, hints):
    lc = {c.lower(): c for c in df.columns}
    for h in hints:
        for k, orig in lc.items():
            if h in k: return orig
    raise ValueError(f"Missing any column like: {hints}")

# --------- MATRIX-ish RULES (from your matrix image) ----------
SPARES_PATTERNS = (r"\bspare\b", r"\bspares\b", r"\bspares allocation\b")
ODC_PATTERNS    = (r"\bodc\b", r"\bother direct costs?\b")
TRVL_PATTERNS   = (r"\btrvl\b", r"\btravel\b")

def is_spares(desc: str) -> bool:
    t = (desc or "").lower()
    return any(re.search(p, t) for p in SPARES_PATTERNS)

def is_odc_travel(desc: str) -> bool:
    t = (desc or "").lower()
    return any(re.search(p, t) for p in ODC_PATTERNS + TRVL_PATTERNS)

# =========================================================
# STEP 1 — Read OTHER RATES and COM SUMMARY from RATSUM.xlsx
# =========================================================
xls = pd.ExcelFile(RATESUM_PATH)

# OTHER RATES (skiprows=5 in your notebook)
other = pd.read_excel(xls, OTHER_SHEET, skiprows=5, dtype=object).dropna(how="all")
other = other.dropna(axis=1, how="all").fillna("").reset_index(drop=True)

# Build a clean "label" column like you did: Unnamed:1 + Unnamed:2
u1 = next((c for c in other.columns if "Unnamed: 1" in c), None)
u2 = next((c for c in other.columns if "Unnamed: 2" in c), None)
if u1 and u2:
    other["label"] = other[u1].astype(str).str.strip() + " " + other[u2].astype(str).str.strip()
elif u1:
    other["label"] = other[u1].astype(str).str.strip()
else:
    # fallback: first non-year column is label
    non_years = [c for c in other.columns if not year_from_header(c)]
    other["label"] = other[non_years[0]].astype(str).str.strip()

# year columns
year_cols = [c for c in other.columns if year_from_header(c)]
# map row -> canonical key
def canon_key(lbl: str) -> str:
    s = (lbl or "").lower()
    if "cssc" in s and "g & a" in s:       return "ga_cssc"
    if ("division general" in s or "dvga" in s) and "g & a" in s: return "ga_gdls"
    if "reorder point" in s:               return "reorder_point"
    if "total procurement" in s:           return "proc_total"
    if "procurement allowable overhead" in s and "freight" not in s: return "proc_prls"
    if "freight" in s and "allowable overhead" in s:                 return "proc_pfrt"
    if "major end-item" in s:              return "major_end_item"
    if "support rate" in s:                return "support"
    if "contl test" in s or "control test" in s: return "actr"  # probably not used here, but captured
    return ""

rows = {}
for _, r in other.iterrows():
    k = canon_key(r["label"])
    if not k: continue
    # collect year->value
    d = {}
    for c in year_cols:
        y = year_from_header(c)
        v = pd.to_numeric(r[c], errors="coerce")
        if pd.notna(v): d[y] = float(v)
    if d:
        rows[k] = pd.Series(d).sort_index()

# derive TOTAL PROCUREMENT if missing
if "proc_total" not in rows:
    if "proc_prls" in rows or "proc_pfrt" in rows:
        all_years = sorted(set(rows.get("proc_prls", pd.Series()).index.tolist()) |
                           set(rows.get("proc_pfrt", pd.Series()).index.tolist()))
        s_prls = rows.get("proc_prls", pd.Series(dtype=float)).reindex(all_years)
        s_pfrt = rows.get("proc_pfrt", pd.Series(dtype=float)).reindex(all_years)
        rows["proc_total"] = (s_prls.fillna(0) + s_pfrt.fillna(0)).dropna()

# COM SUMMARY (skiprows=5 mimics your notebook)
com = pd.read_excel(xls, COM_SHEET, skiprows=5, dtype=object).dropna(how="all")
com = com.dropna(axis=1, how="all").fillna("").reset_index(drop=True)
# find the first text column with the segment names
name_col = next((c for c in com.columns if "Unnamed" in c), com.columns[0])
com["name"] = com[name_col].astype(str).str.upper().str.strip()

# year columns for COM
com_year_cols = [c for c in com.columns if year_from_header(c)]
com_series = {}
for _, r in com.iterrows():
    nm = r["name"]
    if "GENERAL DYNAMICS LAND SYSTEM" in nm or "GDLS" in nm:
        key = "com_gdls"
    elif "CUSTOMER SERVICES AND SUPPORT" in nm or "CSSC" in nm:
        key = "com_cssc"
    else:
        continue
    d = {}
    for c in com_year_cols:
        y = year_from_header(c)
        v = pd.to_numeric(r[c], errors="coerce")
        if pd.notna(v): d[y] = float(v)
    if d:
        com_series[key] = pd.Series(d).sort_index()

# =========================================================
# STEP 2 — Read BurdenRateImport and update in place
# =========================================================
imp = pd.read_excel(BRI_PATH, dtype=object)
imp.columns = [str(c).strip() for c in imp.columns]
original_cols = imp.columns.tolist()

burden_col = pick_col(imp, ("burden",))
desc_col   = pick_col(imp, ("description", "desc"))
date_col   = pick_col(imp, ("date",))
# collect known target columns if present
col_ga_cssc = next((c for c in imp.columns if re.search(r"\bg&?a\b.*cssc", c, flags=re.I)), None)
col_ga_gdls = next((c for c in imp.columns if re.search(r"\bg&?a\b.*gdls", c, flags=re.I)), None)
col_rop     = next((c for c in imp.columns if re.search(r"\bro(p| re.?ord)", c, flags=re.I)), None)
col_proc_g  = next((c for c in imp.columns if re.search(r"proc.*o/?h.*g", c, flags=re.I)), None)
col_proc_c  = next((c for c in imp.columns if re.search(r"proc.*o/?h.*c", c, flags=re.I)), None)
col_major   = next((c for c in imp.columns if re.search(r"major.*end", c, flags=re.I)), None)
col_support = next((c for c in imp.columns if re.search(r"support", c, flags=re.I)), None)
col_com_cssc= next((c for c in imp.columns if re.search(r"\bcom\b.*cssc", c, flags=re.I)), None)
col_com_gdls= next((c for c in imp.columns if re.search(r"\bcom\b.*gdls", c, flags=re.I)), None)

# precompute needed years
imp["_year"] = pd.to_numeric(imp[date_col], errors="coerce").astype("Int64")
needed_years = sorted(imp["_year"].dropna().astype(int).unique().tolist())

# carry-forward each series to needed years
cf = {}
for k, s in rows.items():
    cf[k] = carry_forward(s, needed_years)
for k, s in com_series.items():
    cf[k] = carry_forward(s, needed_years)

def val(k, y):
    s = cf.get(k)
    if s is None: return np.nan
    return float(s.get(int(y), np.nan))

# Update loop (in place, keep shape)
for idx, r in imp.iterrows():
    y = r["_year"]
    if pd.isna(y): 
        continue
    y = int(y)
    seg = "CSSC" if "CSSC" in str(r[burden_col]).upper() else ("GDLS" if "GDLS" in str(r[burden_col]).upper() else "")
    dsc = str(r[desc_col] or "")

    # G&A + COM only on spares/ODC/travel rows (per matrix footnote)
    apply_gacm = is_spares(dsc) or is_odc_travel(dsc)

    # G&A
    if col_ga_cssc and seg == "CSSC":
        imp.at[idx, col_ga_cssc] = round_half_up(val("ga_cssc", y) if apply_gacm else 0.0, ROUND_OTH)
    if col_ga_gdls and seg == "GDLS":
        imp.at[idx, col_ga_gdls] = round_half_up(val("ga_gdls", y) if apply_gacm else 0.0, ROUND_OTH)

    # COM
    if col_com_cssc and seg == "CSSC":
        imp.at[idx, col_com_cssc] = round_half_up(val("com_cssc", y) if apply_gacm else 0.0, ROUND_COM)
    if col_com_gdls and seg == "GDLS":
        imp.at[idx, col_com_gdls] = round_half_up(val("com_gdls", y) if apply_gacm else 0.0, ROUND_COM)

    # Reorder Point (ROP) — apply on spares lines (visible in matrix header)
    if col_rop:
        imp.at[idx, col_rop] = round_half_up(val("reorder_point", y) if is_spares(dsc) else 0.0, ROUND_OTH)

    # Procurement OH — use TOTAL PROCUREMENT (PRLS+PFRT) for both segments when applicable
    if "proc_total" in cf:
        if col_proc_g and seg == "GDLS":
            imp.at[idx, col_proc_g] = round_half_up(val("proc_total", y), ROUND_OTH)
        if col_proc_c and seg == "CSSC":
            imp.at[idx, col_proc_c] = round_half_up(val("proc_total", y), ROUND_OTH)

    # Major End Item / Support — apply where relevant keywords appear
    if col_major:
        imp.at[idx, col_major] = round_half_up(val("major_end_item", y) if re.search(r"major|mei", dsc, flags=re.I) else 0.0, ROUND_OTH)
    if col_support:
        imp.at[idx, col_support] = round_half_up(val("support", y) if re.search(r"support", dsc, flags=re.I) else 0.0, ROUND_OTH)

# Export with identical shape/order
imp_export = imp.loc[:, original_cols]
out_path = OUT_DIR / "BurdenRateImportUpdate.xlsx"
imp_export.to_excel(out_path, sheet_name="update", index=False)
print(f"✅ BurdenRateImportUpdate → {out_path}")