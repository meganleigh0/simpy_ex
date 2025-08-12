# -*- coding: utf-8 -*-
import re
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

# ---------------- PATHS ----------------
RATESUM_PATH = "data/RATSUM.xlsx"           # source of OTHER RATES + COM SUMMARY
OTHER_SHEET  = "OTHER RATES"
COM_SHEET    = "COM SUMMARY"
BRI_PATH     = "data/BurdenRateImport.xlsx" # file to update
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
    """
    Safe CF: handles empty / all-NaN series without crashing.
    Returns a Series indexed by needed_years.
    """
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

def keyize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")

def pick_col(df, hints):
    lc = {c.lower(): c for c in df.columns}
    for h in hints:
        for k, orig in lc.items():
            if h in k: return orig
    raise ValueError(f"Missing any column like: {hints}")

# Matrix-ish rules from the image
SPARES_PATTERNS = (r"\bspare\b", r"\bspares\b", r"\bspares allocation\b")
ODC_PATTERNS    = (r"\bodc\b", r"\bother direct costs?\b")
TRVL_PATTERNS   = (r"\btrvl\b", r"\btravel\b")
def is_spares(desc: str) -> bool:
    t = (desc or "").lower()
    return any(re.search(p, t) for p in SPARES_PATTERNS)
def is_odc_travel(desc: str) -> bool:
    t = (desc or "").lower()
    return any(re.search(p, t) for p in ODC_PATTERNS + TRVL_PATTERNS)

# ---------------- STEP 1: read OTHER RATES & COM SUMMARY ----------------
xls = pd.ExcelFile(RATESUM_PATH)

# OTHER RATES (your screenshot shows skiprows=5)
other = (pd.read_excel(xls, OTHER_SHEET, skiprows=5, dtype=object)
           .dropna(how="all").dropna(axis=1, how="all").fillna("").reset_index(drop=True))

# Build a robust label column
u1 = next((c for c in other.columns if "Unnamed: 1" in c), None)
u2 = next((c for c in other.columns if "Unnamed: 2" in c), None)
if u1 and u2:
    other["label"] = other[u1].astype(str).str.strip() + " " + other[u2].astype(str).str.strip()
else:
    non_years = [c for c in other.columns if not year_from_header(c)]
    other["label"] = other[non_years[0]].astype(str).str.strip()

year_cols = [c for c in other.columns if year_from_header(c)]

def canon_key(lbl: str) -> str:
    s = (lbl or "").lower()
    if "cssc" in s and "g & a" in s:                                         return "ga_cssc"
    if ("division general" in s or "dvga" in s) and "g & a" in s:            return "ga_gdls"
    if "reorder point" in s:                                                 return "reorder_point"
    if "total procurement" in s:                                             return "proc_total"
    if "allowable procurement rate" in s and "total" in s:                   return "proc_total"
    if "procurement allowable overhead rate" in s and "freight" not in s:    return "proc_prls"
    if "freight" in s and "allowable overhead rate" in s:                    return "proc_pfrt"
    if "major end-item" in s:                                                return "major_end_item"
    if "support rate" in s:                                                  return "support"
    return ""

rows = {}
for _, r in other.iterrows():
    k = canon_key(r["label"])
    if not k: continue
    d = {}
    for c in year_cols:
        y = year_from_header(c)
        v = pd.to_numeric(r[c], errors="coerce")
        if pd.notna(v): d[y] = float(v)
    if d:
        rows[k] = pd.Series(d).sort_index()

# If TOTAL PROCUREMENT missing, derive PRLS+PFRT when available
if "proc_total" not in rows and (("proc_prls" in rows) or ("proc_pfrt" in rows)):
    years = sorted(set(rows.get("proc_prls", pd.Series(dtype=float)).index) |
                   set(rows.get("proc_pfrt", pd.Series(dtype=float)).index))
    s_prls = rows.get("proc_prls", pd.Series(dtype=float)).reindex(years)
    s_pfrt = rows.get("proc_pfrt", pd.Series(dtype=float)).reindex(years)
    total = (s_prls.fillna(0) + s_pfrt.fillna(0)).dropna()
    if not total.empty:
        rows["proc_total"] = total

# COM SUMMARY (skiprows=5 matches your notebook)
com = (pd.read_excel(xls, COM_SHEET, skiprows=5, dtype=object)
         .dropna(how="all").dropna(axis=1, how="all").fillna("").reset_index(drop=True))
name_col = next((c for c in com.columns if "Unnamed" in c), com.columns[0])
com["name"] = com[name_col].astype(str).str.upper().str.strip()
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

# ---------------- STEP 2: read BRI and compute needed years ----------------
imp = pd.read_excel(BRI_PATH, dtype=object)
imp.columns = [str(c).strip() for c in imp.columns]
original_cols = imp.columns.tolist()

burden_col = pick_col(imp, ("burden",))
desc_col   = pick_col(imp, ("description", "desc"))
date_col   = pick_col(imp, ("date",))

# locate target columns if present
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
# robust fallback if BRI has no usable years
if not needed_years:
    other_years = [year_from_header(c) for c in other.columns if year_from_header(c)]
    com_years   = [year_from_header(c) for c in com.columns if year_from_header(c)]
    pool = sorted({y for y in other_years + com_years if y})
    needed_years = pool or [2022, 2023, 2024, 2025]

# pre-carry-forward sources to needed_years (safe even if empty)
cf = {}
for k, s in rows.items():
    cf[k] = carry_forward(s, needed_years)
for k, s in com_series.items():
    cf[k] = carry_forward(s, needed_years)

def val(k, y):
    s = cf.get(k)
    if s is None: return np.nan
    return float(s.get(int(y), np.nan))

# ---------------- STEP 3: update values (in place) ----------------
for idx, r in imp.iterrows():
    y = r["_year"]
    if pd.isna(y): 
        continue
    y = int(y)
    seg = "CSSC" if "CSSC" in str(r[burden_col]).upper() else ("GDLS" if "GDLS" in str(r[burden_col]).upper() else "")
    dsc = str(r[desc_col] or "")

    # G&A + COM only on spares / ODC / travel lines (per matrix footnote)
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

    # Reorder Point (ROP) — only on spares rows
    if col_rop:
        imp.at[idx, col_rop] = round_half_up(val("reorder_point", y) if is_spares(dsc) else 0.0, ROUND_OTH)

    # Procurement OH — use TOTAL PROCUREMENT for both segments
    if "proc_total" in cf:
        if col_proc_g and seg == "GDLS":
            imp.at[idx, col_proc_g] = round_half_up(val("proc_total", y), ROUND_OTH)
        if col_proc_c and seg == "CSSC":
            imp.at[idx, col_proc_c] = round_half_up(val("proc_total", y), ROUND_OTH)

    # Major End-Item / Support — gated by keywords in Description
    if col_major:
        imp.at[idx, col_major] = round_half_up(val("major_end_item", y) if re.search(r"major|mei", dsc, flags=re.I) else 0.0, ROUND_OTH)
    if col_support:
        imp.at[idx, col_support] = round_half_up(val("support", y) if re.search(r"support", dsc, flags=re.I) else 0.0, ROUND_OTH)

# ---------------- STEP 4: export with exact same columns/order ----------------
imp_export = imp.loc[:, original_cols]
out_path = OUT_DIR / "BurdenRateImportUpdate.xlsx"
imp_export.to_excel(out_path, sheet_name="update", index=False)
print(f"✅ BurdenRateImportUpdate → {out_path}")