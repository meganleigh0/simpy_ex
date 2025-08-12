# -*- coding: utf-8 -*-
import re
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
RATESUM_PATH = "data/RATSUM.xlsx"
SIMPLIFIED_SHEET = "SIMPLIFIED RATES NON-PSPL"
SIMPLIFIED_SKIPROWS = 5
OTHER_RATES_SHEET = "OTHER RATES"

RATEBAND_PATH = "data/RateBandImport.xlsx"
OUT_DIR = Path("out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

VT_CODE = "VT"
ABRAMS_TOKENS = ("abrams", "vt abrams", "vtabrams")

# rounding rules
ROUNDING = {"labor dollars": 3, "burdens": 5, "com": 6}

# ---------------- helpers ----------------
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

# ---------------- STEP 1: Build combined_rates from RATSUM ----------------
xls = pd.ExcelFile(RATESUM_PATH)

# Simplified Rates
simp = pd.read_excel(xls, SIMPLIFIED_SHEET, skiprows=SIMPLIFIED_SKIPROWS, dtype=object)
simp.columns = [str(c).strip() for c in simp.columns]
bu_col = pick_business_unit_col(simp.columns)
simp.rename(columns={bu_col: "BUSINESS UNIT GDLS"}, inplace=True)

# drop repeated header row if present
if str(simp.iloc[0]["BUSINESS UNIT GDLS"]).strip().lower().startswith("business"):
    simp = simp.iloc[1:].reset_index(drop=True)

simp = simp.loc[:, ~pd.Index(simp.columns).str.contains("Unnamed", case=False)]
year_cols = [c for c in simp.columns if year_from_header(c)]
if not year_cols: raise ValueError("No year columns found in Simplified Rates.")

simp["BUSINESS UNIT GDLS"] = simp["BUSINESS UNIT GDLS"].astype(str).str.strip()
simp["rate_band_code"] = simp["BUSINESS UNIT GDLS"].map(normalize_band_code)
simp_long = (simp.melt(id_vars=["BUSINESS UNIT GDLS","rate_band_code"],
                       value_vars=year_cols, var_name="year_col", value_name="rate_value")
                .assign(year=lambda d: d["year_col"].map(year_from_header))
                .dropna(subset=["rate_band_code","year"])
                .assign(rate_value=lambda d: pd.to_numeric(d["rate_value"], errors="coerce"))
                .loc[:, ["rate_band_code","year","rate_value"]])
simp_long["source"] = "Simplified"

# Other Rates -> Allowable Control/Contl Test Rate (ACTR)
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
if not years: raise ValueError("Could not find year columns in OTHER RATES.")

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

# combined & dedup per band-year
combined_rates = (pd.concat([simp_long, vt_df], ignore_index=True)
                    .dropna(subset=["rate_band_code","year"])
                    .astype({"year": int})
                    .sort_values(["rate_band_code","year","source"])
                    .drop_duplicates(subset=["rate_band_code","year"], keep="first")
                    .reset_index(drop=True))

# ---------------- STEP 2: Load RateBandImport and UPDATE IN PLACE ----------------
rb = pd.read_excel(RATEBAND_PATH, dtype=object)
rb.columns = [str(c).strip() for c in rb.columns]
original_cols = rb.columns.tolist()  # keep exact order for export

rate_band_col = find_col(rb, ("rate band","rateband"))
start_col     = find_col(rb, ("start","effective start"))
end_col       = find_col(rb, ("end","effective end"))
desc_col      = next((c for c in rb.columns if "desc" in c.lower() or "program" in c.lower() or "platform" in c.lower()), None)

base_col      = next((c for c in rb.columns if "base rate" in c.lower()), None)
ld_col        = next((c for c in rb.columns if "labor dollars" in c.lower()), None)
bur_col       = next((c for c in rb.columns if "burden" in c.lower()), None)
com_col       = next((c for c in rb.columns if re.search(r"\bcom\b", c.lower())), None)

# normalize keys
rb["_band_raw"]      = rb[rate_band_col].astype(str).str.strip()
rb["rate_band_code"] = rb["_band_raw"].map(normalize_band_code)
rb["_start"] = pd.to_datetime(rb[start_col], errors="coerce")
rb["_end"]   = pd.to_datetime(rb[end_col], errors="coerce")
rb = rb.dropna(subset=["_start","_end"]).copy()
rb["_year"] = rb["_start"].dt.year.astype(int)  # assume 1 row per year (your file shows this)
rb["_is_vt_abrams"] = (rb["rate_band_code"].eq(VT_CODE) |
                       (rb[desc_col].astype(str).str.lower().str.contains("|".join(ABRAMS_TOKENS), na=False)
                        if desc_col else False))

# fast lookup map (with carry-forward)
band_to_series = {b: g.set_index("year")["rate_value"].sort_index()
                  for b, g in combined_rates.groupby("rate_band_code")}

def lookup_rate(band, year):
    s = band_to_series.get(band)
    if s is None or s.empty: return np.nan
    if year in s.index: return float(s.loc[year])
    prev = s.index[s.index <= year]
    return float(s.loc[int(prev.max())]) if len(prev) else np.nan

# compute updated values (no extra columns; keep shape)
vals = rb.apply(lambda r: lookup_rate(r["rate_band_code"], int(r["_year"])), axis=1)

# if lookup missing, keep original; otherwise apply rounding rules
if base_col:
    rb.loc[vals.notna(), base_col] = [
        round_half_up(v, 0 if vt else 3)
        for v, vt in zip(vals[vals.notna()], rb.loc[vals.notna(), "_is_vt_abrams"])
    ]

if ld_col:
    rb.loc[vals.notna(), ld_col] = [round_half_up(v, ROUNDING["labor dollars"]) for v in vals[vals.notna()]]

if bur_col:
    rb.loc[vals.notna(), bur_col] = [round_half_up(v, ROUNDING["burdens"]) for v in vals[vals.notna()]]

if com_col:
    rb.loc[vals.notna(), com_col] = [round_half_up(v, ROUNDING["com"]) for v in vals[vals.notna()]]

# drop helper columns before export; preserve EXACT original structure/order
rb_export = rb.loc[:, original_cols]

# ---------------- STEP 3: Clean comparison report (no sub-headers) ----------------
# Build RB band set
bands_rb = set(rb["rate_band_code"].dropna().unique())

# Filter RateSum bands to those that are either in RB or exactly two digits (true numeric bands) or VT
cr_bands = (combined_rates["rate_band_code"].dropna().map(normalize_band_code))
mask_valid = (cr_bands.isin(bands_rb)) | (cr_bands.str.match(r"^\d{2}$")) | (cr_bands.eq(VT_CODE))
bands_cr_filtered = set(cr_bands[mask_valid].unique())

only_in_ratesum   = sorted(list(bands_cr_filtered - bands_rb))
only_in_ratebands = sorted(list(bands_rb - bands_cr_filtered))

# details tabs (optional, but helpful)
ratesum_details = (combined_rates
    .assign(rate_band_code=lambda d: d["rate_band_code"].map(normalize_band_code))
    .query("rate_band_code in @bands_cr_filtered")
    .groupby("rate_band_code", as_index=False)
    .agg(min_year=("year","min"), max_year=("year","max"), rows=("year","size"))
    .sort_values("rate_band_code"))

ratebands_details = (rb
    .groupby("rate_band_code", as_index=False)
    .agg(rows=("rate_band_code","size"))
    .sort_values("rate_band_code"))

# ---------------- SAVE ----------------
update_path  = OUT_DIR / "RateBandImportUpdate.xlsx"
compare_path = OUT_DIR / "comparison_report.xlsx"

with pd.ExcelWriter(update_path, engine="openpyxl") as xw:
    rb_export.to_excel(xw, sheet_name="update", index=False)

with pd.ExcelWriter(compare_path, engine="openpyxl") as xw:
    pd.DataFrame({"rate_band_code": only_in_ratesum}).to_excel(
        xw, sheet_name="in_ratesum_not_in_ratebands", index=False)
    pd.DataFrame({"rate_band_code": only_in_ratebands}).to_excel(
        xw, sheet_name="in_ratebands_not_in_ratesum", index=False)
    ratesum_details.to_excel(xw, sheet_name="ratesum_details", index=False)
    ratebands_details.to_excel(xw, sheet_name="ratebands_details", index=False)

print(f"✅ Wrote update identical to import columns → {update_path}")
print(f"✅ Wrote cleaned comparison report        → {compare_path}")
print(f"   In RateSum not in RateBands: {len(only_in_ratesum)} | In RateBands not in RateSum: {len(only_in_ratebands)}")