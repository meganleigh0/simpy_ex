# -*- coding: utf-8 -*-
import re
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

# ---------------- PATHS ----------------
RATESUM_PATH = "data/RATSUM.xlsx"
OTHER_SHEET  = "OTHER RATES"
COM_SHEET    = "COM SUMMARY"
BRI_PATH     = "data/BurdenRateImport.xlsx"
OUT_DIR = Path("out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- ROUNDING ----------------
ROUND_COM = 6
ROUND_OTH = 5

# ---------------- HELPERS ----------------
def round_half_up(x, places):
    if pd.isna(x): return np.nan
    q = Decimal("1").scaleb(-places)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))

def year_from_header(h):
    m = re.search(r"(?:CY)?\s*(20\d{2})", str(h))
    return int(m.group(1)) if m else None

def carry_forward(series_by_year: pd.Series, needed_years):
    if series_by_year is None:
        return pd.Series(index=needed_years, dtype=float)
    s = pd.to_numeric(series_by_year, errors="coerce").dropna()
    if s.empty:
        return pd.Series(index=needed_years, dtype=float)
    s = s.sort_index()
    lo = int(s.index.min())
    hi = int(max(int(s.index.max()), max(needed_years)))
    full_index = list(range(lo, hi + 1))
    return s.reindex(full_index).ffill().reindex(needed_years)

def pick_col(df, hints):
    lc = {c.lower(): c for c in df.columns}
    for h in hints:
        for k, orig in lc.items():
            if h in k: return orig
    raise ValueError(f"Missing any column like: {hints}")

def sfill(series_like):
    """Return a 'string' dtype Series with NaNs replaced by empty string (no FutureWarning)."""
    s = pd.Series(series_like, copy=False)
    return s.astype("string").fillna("")

# Matrix-ish rules
SPARES_PATTERNS = (r"\bspare\b", r"\bspares\b", r"\bspares allocation\b")
ODC_PATTERNS    = (r"\bodc\b", r"\bother direct costs?\b")
TRVL_PATTERNS   = (r"\btrvl\b", r"\btravel\b")
def is_spares(desc: str) -> bool:
    t = (desc or "").lower()
    return any(re.search(p, t) for p in SPARES_PATTERNS)
def is_odc_travel(desc: str) -> bool:
    t = (desc or "").lower()
    return any(re.search(p, t) for p in ODC_PATTERNS + TRVL_PATTERNS)

# ---------------- STEP 1: read OTHER RATES (no frame-wide fillna) ----------------
xls = pd.ExcelFile(RATESUM_PATH)
other = (pd.read_excel(xls, OTHER_SHEET, skiprows=5, dtype=object)
           .dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True))

# Build robust label from the “Unnamed” text columns ONLY (string-cast + fill)
u1 = next((c for c in other.columns if "Unnamed: 1" in c), None)
u2 = next((c for c in other.columns if "Unnamed: 2" in c), None)
if u1 and u2:
    c1 = sfill(other[u1]).str.strip()
    c2 = sfill(other[u2]).str.strip()
    other["label"] = (c1 + " " + c2).str.strip()
else:
    non_years = [c for c in other.columns if not year_from_header(c)]
    other["label"] = sfill(other[non_years[0]]).str.strip()

year_cols = [c for c in other.columns if year_from_header(c)]

def canon_key(lbl: str) -> str:
    s = (lbl or "").lower()
    if "cssc" in s and "g & a" in s:                                         return "ga_cssc"
    if ("division general" in s or "dvga" in s) and "g & a" in s:            return "ga_gdls"
    if "reorder point" in s:                                                 return "reorder_point"
    if "total procurement" in s or ("allowable procurement rate" in s and "total" in s):
                                                                             return "proc_total"
    if "procurement allowable overhead rate" in s and "freight" not in s:    return "proc_prls"
    if "freight" in s and "allowable overhead rate" in s:                    return "proc_pfrt"
    if "major end-item" in s:                                                return "major_end_item"
    if "support rate" in s:                                                  return "support"
    return ""

rows = {}
for _, r in other.iterrows():
    k = canon_key(r["label"])
    if not k: continue
    vals = {year_from_header(c): pd.to_numeric(r[c], errors="coerce")
            for c in year_cols}
    s = pd.Series({y: float(v) for y, v in vals.items() if pd.notna(v)}).sort_index()
    if not s.empty:
        rows[k] = s

# derive TOTAL PROCUREMENT if needed
if "proc_total" not in rows and (("proc_prls" in rows) or ("proc_pfrt" in rows)):
    years = sorted(set(rows.get("proc_prls", pd.Series(dtype=float)).index) |
                   set(rows.get("proc_pfrt", pd.Series(dtype=float)).index))
    s_prls = rows.get("proc_prls", pd.Series(dtype=float)).reindex(years)
    s_pfrt = rows.get("proc_pfrt", pd.Series(dtype=float)).reindex(years)
    total = (s_prls.fillna(0) + s_pfrt.fillna(0)).dropna()
    if not total.empty:
        rows["proc_total"] = total

# ---------------- STEP 1b: read COM SUMMARY (no frame-wide fillna) ----------------
com = (pd.read_excel(xls, COM_SHEET, skiprows=5, dtype=object)
         .dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True))
name_col = next((c for c in com.columns if "Unnamed" in c), com.columns[0])
com["name"] = sfill(com[name_col]).str.upper().str.strip()
com_year_cols = [c for c in com.columns if year_from_header(c)]

com_series = {}
for _, r in com.iterrows():
    nm = r["name"]
    key = "com_gdls" if ("GENERAL DYNAMICS LAND SYSTEM" in nm or "GDLS" in nm) \
          else ("com_cssc" if ("CUSTOMER SERVICES AND SUPPORT" in nm or "CSSC" in nm) else None)
    if not key: 
        continue
    vals = {year_from_header(c): pd.to_numeric(r[c], errors="coerce")
            for c in com_year_cols}
    s = pd.Series({y: float(v) for y, v in vals.items() if pd.notna(v)}).sort_index()
    if not s.empty:
        com_series[key] = s

# ---------------- STEP 2: read BRI and compute needed years ----------------
imp = pd.read_excel(BRI_PATH, dtype=object)
imp.columns = [str(c).strip() for c in imp.columns]
original_cols = imp.columns.tolist()

burden_col = pick_col(imp, ("burden",))
desc_col   = pick_col(imp, ("description", "desc"))
date_col   = pick_col(imp, ("date",))

col_ga_cssc = next((c for c in imp.columns if re.search(r"\bg&?a\b.*cssc", c, flags=re.I)), None)
col_ga_gdls = next((c for c in imp.columns if re.search(r"\bg&?a\b.*gdls", c, flags=re.I)), None)
col_rop     = next((c for c in imp.columns if re.search(r"\bro(p| ?re.?ord)", c, flags=re.I)), None)
col_proc_g  = next((c for c in imp.columns if re.search(r"proc.*o/?h.*g", c, flags=re.I)), None)
col_proc_c  = next((c for c in imp.columns if re.search(r"proc.*o/?h.*c", c, flags=re.I)), None)
col_major   = next((c for c in imp.columns if re.search(r"major.*end", c, flags=re.I)), None)
col_support = next((c for c in imp.columns if re.search(r"support", c, flags=re.I)), None)
col_com_cssc= next((c for c in imp.columns if re.search(r"\bcom\b.*cssc", c, flags=re.I)), None)
col_com_gdls= next((c for c in imp.columns if re.search(r"\bcom\b.*gdls", c, flags=re.I)), None)

imp["_year"] = pd.to_numeric(imp[date_col], errors="coerce").astype("Int64")
needed_years = sorted(imp["_year"].dropna().astype(int).unique().tolist())
if not needed_years:
    other_years = [year_from_header(c) for c in other.columns if year_from_header(c)]
    com_years   = [year_from_header(c) for c in com.columns if year_from_header(c)]
    pool = sorted({y for y in other_years + com_years if y})
    needed_years = pool or [2022, 2023, 2024, 2025]

# pre-carry-forward
cf = {k: carry_forward(s, needed_years) for k, s in {**rows, **com_series}.items()}
def val(k, y):
    s = cf.get(k)
    return float(s.get(int(y), np.nan)) if s is not None else np.nan

# ---------------- STEP 3: update values (in place) ----------------
for idx, r in imp.iterrows():
    y = r["_year"]
    if pd.isna(y): 
        continue
    y = int(y)
    seg = "CSSC" if "CSSC" in str(r[burden_col]).upper() else ("GDLS" if "GDLS" in str(r[burden_col]).upper() else "")
    dsc = str(r[desc_col] or "")

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

    # ROP (only on spares)
    if col_rop:
        imp.at[idx, col_rop] = round_half_up(val("reorder_point", y) if is_spares(dsc) else 0.0, ROUND_OTH)

    # Procurement OH: TOTAL PROCUREMENT
    if "proc_total" in cf:
        if col_proc_g and seg == "GDLS":
            imp.at[idx, col_proc_g] = round_half_up(val("proc_total", y), ROUND_OTH)
        if col_proc_c and seg == "CSSC":
            imp.at[idx, col_proc_c] = round_half_up(val("proc_total", y), ROUND_OTH)

    # Major End-Item / Support (keyword-gated)
    if col_major:
        imp.at[idx, col_major] = round_half_up(val("major_end_item", y) if re.search(r"major|mei", dsc, flags=re.I) else 0.0, ROUND_OTH)
    if col_support:
        imp.at[idx, col_support] = round_half_up(val("support", y) if re.search(r"support", dsc, flags=re.I) else 0.0, ROUND_OTH)

# ---------------- STEP 4: export (exact same columns/order) ----------------
imp_export = imp.loc[:, original_cols]
out_path = OUT_DIR / "BurdenRateImportUpdate.xlsx"
imp_export.to_excel(out_path, sheet_name="update", index=False)
print(f"✅ BurdenRateImportUpdate → {out_path}")