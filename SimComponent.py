# -*- coding: utf-8 -*-
import re, calendar, datetime as dt
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

# ---------- CONFIG ----------
RATESUM_PATH = "data/RATSUM.xlsx"
SIMPLIFIED_SHEET = "SIMPLIFIED RATES NON-PSPL"
SIMPLIFIED_SKIPROWS = 5
OTHER_RATES_SHEET = "OTHER RATES"

RATEBAND_PATH = "data/RateBandImport.xlsx"
OUT_DIR = Path("out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

VT_CODE = "VT"
ABRAMS_TOKENS = ("abrams", "vt abrams", "vtabrams")
ROUNDING = {"labor dollars": 3, "burdens": 5, "com": 6}

# ---------- helpers ----------
def round_half_up(x, places):
    if pd.isna(x): return np.nan
    q = Decimal("1").scaleb(-places)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))

def year_from_header(h):
    if h is None or (isinstance(h, float) and np.isnan(h)): return None
    m = re.search(r"(?:^|\D)(20\d{2})(?:\D|$)", str(h))
    return int(m.group(1)) if m else None

def normalize_band_code(s):
    s = str(s).strip()
    two = s[:2]
    if two.isdigit(): return two.zfill(2)
    return re.sub(r"[^A-Za-z0-9]", "", two).upper()

def is_valid_band(code):
    code = normalize_band_code(code)
    return (code == VT_CODE) or bool(re.match(r'^(?=.*\d)[A-Z0-9]{2}$', code))  # 2 chars, contains a digit

def pick_business_unit_col(cols):
    lc = {c.lower(): c for c in cols}
    for k, orig in lc.items():
        if "business" in k and "gdls" in k: return orig
    if "Unnamed: 1" in cols: return "Unnamed: 1"
    raise ValueError("Could not find BUSINESS UNIT GDLS column")

def find_col(df, hints):
    lc = {c.lower(): c for c in df.columns}
    for h in hints:
        for k, orig in lc.items():
            if h in k: return orig
    raise ValueError(f"Missing any column like: {hints}")

# --- explicit date parsing (no inference warnings) ---
def _parse_date_cell(x, role="start"):
    """Parse 'MM/YYYY', 'MM/DD/YYYY', or Excel serials. role='start' -> first day, 'end' -> last day."""
    if pd.isna(x): return pd.NaT
    if isinstance(x, (pd.Timestamp, dt.date, dt.datetime)): return pd.Timestamp(x)

    # Excel serial number
    if isinstance(x, (int, float)) and not np.isnan(x):
        try:
            return pd.to_datetime(x, unit="D", origin="1899-12-30")
        except Exception:
            pass

    s = str(x).strip()

    # MM/YYYY
    m = re.fullmatch(r"(\d{1,2})[/-](\d{4})", s)
    if m:
        mo, yr = int(m.group(1)), int(m.group(2))
        day = 1 if role == "start" else calendar.monthrange(yr, mo)[1]
        return pd.Timestamp(yr, mo, day)

    # MM/DD/YYYY (or 2-digit year)
    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", s)
    if m:
        mo, da, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if yr < 100: yr += 2000 if yr < 50 else 1900
        return pd.Timestamp(yr, mo, da)

    return pd.NaT

def parse_date_col(series, role="start"):
    return series.apply(lambda v: _parse_date_cell(v, role))

# =========================================================
# STEP 1 — Build combined_rates (Simplified + ACTR for VT)
# =========================================================
xls = pd.ExcelFile(RATESUM_PATH)

# Simplified
simp = pd.read_excel(xls, SIMPLIFIED_SHEET, skiprows=SIMPLIFIED_SKIPROWS, dtype=object)
simp.columns = [str(c).strip() for c in simp.columns]
bu_col = pick_business_unit_col(simp.columns)
simp.rename(columns={bu_col: "BUSINESS UNIT GDLS"}, inplace=True)
if str(simp.iloc[0]["BUSINESS UNIT GDLS"]).strip().lower().startswith("business"):
    simp = simp.iloc[1:].reset_index(drop=True)
simp = simp.loc[:, ~pd.Index(simp.columns).str.contains("Unnamed", case=False)]
year_cols = [c for c in simp.columns if year_from_header(c)]
if not year_cols: raise ValueError("No year columns found in Simplified Rates.")

simp["BUSINESS UNIT GDLS"] = simp["BUSINESS UNIT GDLS"].astype(str).str.strip()
simp["rate_band_code"] = simp["BUSINESS UNIT GDLS"].map(normalize_band_code)
simp_long = (simp.melt(id_vars=["BUSINESS UNIT GDLS","rate_band_code"],
                       value_vars=year_cols, var_name="year_col", value_name="rate_value")
               .assign(year=lambda d: d["year_col"].map(year_from_header),
                       rate_value=lambda d: pd.to_numeric(d["rate_value"], errors="coerce"))
               .dropna(subset=["rate_band_code","year"])
               .loc[:, ["rate_band_code","year","rate_value"]])
simp_long["source"] = "Simplified"

# Other Rates -> Allowable Control/Contl Test Rate
other = pd.read_excel(xls, OTHER_RATES_SHEET, header=None)
year_cols_idx, years = [], []
for j in range(other.shape[1]):
    block = " ".join(other.iloc[0:10, j].astype(str).tolist())
    m = re.search(r"CY\s*(20\d{2})", block, flags=re.IGNORECASE)
    if m: year_cols_idx.append(j); years.append(int(m.group(1)))
if not years:
    for j in range(other.shape[1]):
        block = " ".join(other.iloc[0:10, j].astype(str).tolist())
        m = re.search(r"(20\d{2})", block)
        if m: year_cols_idx.append(j); years.append(int(m.group(1)))
if not years: raise ValueError("Year columns not found in OTHER RATES.")

actr_row_idx = None
pat = re.compile(r"ALLOWABLE.*(CONTL|CONTROL).*TEST.*RATE", re.IGNORECASE)
for i in range(other.shape[0]):
    row_text = " ".join([str(x) for x in other.iloc[i, :].tolist() if pd.notna(x)])
    if pat.search(row_text): actr_row_idx = i; break
if actr_row_idx is None: raise ValueError("Missing 'ALLOWABLE ... TEST RATE' row.")

vt_rows = []
for j, y in zip(year_cols_idx, years):
    val = pd.to_numeric(other.iat[actr_row_idx, j], errors="coerce")
    if pd.notna(val): vt_rows.append({"rate_band_code": VT_CODE, "year": int(y), "rate_value": float(val), "source": "ACTR"})
vt_df = pd.DataFrame(vt_rows)

combined_rates = (pd.concat([simp_long, vt_df], ignore_index=True)
                    .dropna(subset=["rate_band_code","year"])
                    .astype({"year": int})
                    .sort_values(["rate_band_code","year","source"])
                    .drop_duplicates(subset=["rate_band_code","year"], keep="first")
                    .reset_index(drop=True))

# =========================================================
# STEP 2 — Update RateBandImport IN PLACE (no extra columns)
# =========================================================
rb = pd.read_excel(RATEBAND_PATH, dtype=object)
rb.columns = [str(c).strip() for c in rb.columns]
original_cols = rb.columns.tolist()

rate_band_col = find_col(rb, ("rate band","rateband"))
start_col     = find_col(rb, ("start", "effective start"))
end_col       = find_col(rb, ("end", "effective end"))
desc_col      = next((c for c in rb.columns if "desc" in c.lower() or "program" in c.lower() or "platform" in c.lower()), None)

base_col      = next((c for c in rb.columns if "base rate" in c.lower()), None)
ld_col        = next((c for c in rb.columns if "labor dollars" in c.lower()), None)
bur_col       = next((c for c in rb.columns if "burden" in c.lower()), None)
com_col       = next((c for c in rb.columns if re.search(r"\bcom\b", c.lower())), None)

rb["_band_raw"]      = rb[rate_band_col].astype(str).str.strip()
rb["rate_band_code"] = rb["_band_raw"].map(normalize_band_code)

# EXPLICIT date parsing (removes 'could not infer format' warnings)
rb["_start"] = parse_date_col(rb[start_col], role="start")
rb["_end"]   = parse_date_col(rb[end_col],   role="end")

rb = rb.dropna(subset=["_start","_end"]).copy()
rb["_year"] = rb["_start"].dt.year.astype(int)
rb["_is_vt_abrams"] = (rb["rate_band_code"].eq(VT_CODE) |
                       (rb[desc_col].astype(str).str.lower().str.contains("|".join(ABRAMS_TOKENS), na=False)
                        if desc_col else False))

# lookup with carry-forward
band_to_series = {b: g.set_index("year")["rate_value"].sort_index()
                  for b, g in combined_rates.groupby("rate_band_code")}

def lookup_rate(band, year):
    s = band_to_series.get(band)
    if s is None or s.empty: return np.nan
    if year in s.index: return float(s.loc[year])
    prev = s.index[s.index <= year]
    return float(s.loc[int(prev.max())]) if len(prev) else np.nan

vals = rb.apply(lambda r: lookup_rate(r["rate_band_code"], int(r["_year"])), axis=1)

# assign + round (keep shape)
if base_col:
    rb.loc[vals.notna(), base_col] = [
        round_half_up(v, 0 if vt else 3)
        for v, vt in zip(vals[vals.notna()], rb.loc[vals.notna(), "_is_vt_abrams"])
    ]
if ld_col:
    rb.loc[vals.notna(), ld_col]  = [round_half_up(v, ROUNDING["labor dollars"]) for v in vals[vals.notna()]]
if bur_col:
    rb.loc[vals.notna(), bur_col] = [round_half_up(v, ROUNDING["burdens"]) for v in vals[vals.notna()]]
if com_col:
    rb.loc[vals.notna(), com_col] = [round_half_up(v, ROUNDING["com"]) for v in vals[vals.notna()]]

rb_export = rb.loc[:, original_cols]  # exact same columns/order as import

# ---- write + enforce Excel number formats (so you see ≥ required decimals) ----
update_path = OUT_DIR / "RateBandImportUpdate.xlsx"
with pd.ExcelWriter(update_path, engine="openpyxl") as xw:
    rb_export.to_excel(xw, sheet_name="update", index=False)
    ws = xw.book["update"]

    def col_idx(colname):
        return rb_export.columns.get_loc(colname) + 1 if colname in rb_export.columns else None

    base_idx = col_idx(base_col)
    ld_idx   = col_idx(ld_col)
    bur_idx  = col_idx(bur_col)
    com_idx  = col_idx(com_col)

    if ld_idx:
        for r in range(2, ws.max_row + 1): ws.cell(r, ld_idx).number_format  = "0.000"
    if bur_idx:
        for r in range(2, ws.max_row + 1): ws.cell(r, bur_idx).number_format = "0.00000"
    if com_idx:
        for r in range(2, ws.max_row + 1): ws.cell(r, com_idx).number_format = "0.000000"

    if base_idx:
        vt_flags = rb["_is_vt_abrams"].to_list()
        for i, is_vt in enumerate(vt_flags, start=2):
            ws.cell(i, base_idx).number_format = "0" if is_vt else "0.000"

print(f"✅ Wrote update identical to import columns → {update_path}")

# =========================================================
# STEP 3 — Comparison (two sheets only; sub-headers removed)
# =========================================================
bands_cr = set(code for code in combined_rates["rate_band_code"].dropna().map(normalize_band_code).unique() if is_valid_band(code))
rb_full = pd.read_excel(RATEBAND_PATH, dtype=object)
rb_full["rate_band_code"] = rb_full[find_col(rb_full, ("rate band","rateband"))].map(normalize_band_code)
bands_rb = set(code for code in rb_full["rate_band_code"].dropna().unique() if is_valid_band(code))

only_in_ratesum   = sorted(list(bands_cr - bands_rb))
only_in_ratebands = sorted(list(bands_rb - bands_cr))

compare_path = OUT_DIR / "comparison_report.xlsx"
with pd.ExcelWriter(compare_path, engine="openpyxl") as xw:
    pd.DataFrame({"rate_band_code": only_in_ratesum}).to_excel(
        xw, sheet_name="in_ratesum_not_in_ratebands", index=False)
    pd.DataFrame({"rate_band_code": only_in_ratebands}).to_excel(
        xw, sheet_name="in_ratebands_not_in_ratesum", index=False)

print(f"✅ Wrote cleaned two-tab comparison → {compare_path}")
print(f"   In RateSum not in RateBands: {len(only_in_ratesum)} | In RateBands not in RateSum: {len(only_in_ratebands)}")