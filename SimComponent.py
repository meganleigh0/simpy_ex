import re
from pathlib import Path
import pandas as pd
import numpy as np

RATESUM_PATH = "data/RATSUM.xlsx"
SIMPLIFIED_SHEET = "SIMPLIFIED RATES NON-PSPL"
SIMPLIFIED_SKIPROWS = 5
OTHER_RATES_SHEET = "OTHER RATES"

# --- helpers -------------------------------------------------------
def pick_business_unit_col(cols):
    """Find the 'BUSINESS UNIT _ GDLS' column by tokens (robust to underscores/spacing)."""
    lc = {c.lower(): c for c in cols}
    for k, orig in lc.items():
        if "business" in k and "gdls" in k:
            return orig
    # fallback seen in your notebook
    if "Unnamed: 1" in cols:
        return "Unnamed: 1"
    raise ValueError(f"Could not find BUSINESS UNIT GDLS column in: {list(cols)}")

def get_year_from_header(h):
    """Accepts '2024', 'CY2024', or ' # CY2024 ' -> 2024; else None."""
    if h is None or (isinstance(h, float) and np.isnan(h)):
        return None
    m = re.search(r"(?:^|\D)(20\d{2})(?:\D|$)", str(h))
    return int(m.group(1)) if m else None

def first_two_chars(s):
    s = str(s).strip()
    return s[:2] if s else ""

# --- load SIMPLIFIED RATES ----------------------------------------
xls = pd.ExcelFile(RATESUM_PATH)
simp = pd.read_excel(xls, SIMPLIFIED_SHEET, skiprows=SIMPLIFIED_SKIPROWS, dtype=object)
simp.columns = [str(c).strip() for c in simp.columns]

# normalize BU column name
bu_col = pick_business_unit_col(simp.columns)
simp.rename(columns={bu_col: "BUSINESS UNIT GDLS"}, inplace=True)

# drop repeated header row if present
if str(simp.iloc[0]["BUSINESS UNIT GDLS"]).strip().lower().startswith("business"):
    simp = simp.iloc[1:].reset_index(drop=True)

# drop 'Unnamed' junk columns
simp = simp.loc[:, ~pd.Index(simp.columns).str.contains("Unnamed", case=False)]

# find year columns (e.g., 2022, 2023, 2024, 2025)
year_cols = [c for c in simp.columns if get_year_from_header(c)]
if not year_cols:
    raise ValueError("No year columns found in Simplified Rates sheet.")

# derive rate band code from first two characters
simp["BUSINESS UNIT GDLS"] = simp["BUSINESS UNIT GDLS"].astype(str).str.strip()
simp["rate_band_code"] = simp["BUSINESS UNIT GDLS"].map(first_two_chars)

# melt to tidy
simp_long = simp.melt(
    id_vars=["BUSINESS UNIT GDLS", "rate_band_code"],
    value_vars=year_cols,
    var_name="year_col",
    value_name="rate_value"
)
simp_long["year"] = simp_long["year_col"].map(get_year_from_header)
simp_long["rate_value"] = pd.to_numeric(simp_long["rate_value"], errors="coerce")
simp_long = (simp_long
             .dropna(subset=["rate_band_code", "year"])
             .drop(columns=["year_col"])
             .sort_values(["rate_band_code", "year"]))

# keep first non-null per band-year
simp_long = (simp_long
             .drop_duplicates(subset=["rate_band_code", "year"], keep="first")
             .loc[:, ["rate_band_code", "year", "rate_value"]])
simp_long["source"] = "Simplified"

# --- load OTHER RATES (Allowable Control Test Rate) ----------------
# We'll scan for a row containing 'ALLOWABLE' and 'TEST' and 'RATE' (handles 'CONTL/CONTROL')
other = pd.read_excel(xls, OTHER_RATES_SHEET, header=None)

# determine which columns are CY-years
year_cols_idx = []
years = []
for j in range(other.shape[1]):
    y = get_year_from_header(other.iloc[0, j]) or get_year_from_header(other.iloc[1, j]) or get_year_from_header(other.iloc[2, j])
    # the "Rounded" header row in your screenshot typically places CY headers on a single row;
    # be permissive across the first few rows
    if y and str(other.iloc[0:5, j].astype(str).str.cat(sep=" ")).upper().find("CY") != -1:
        year_cols_idx.append(j)
        years.append(y)
# fallback: anywhere CY#### appears
if not years:
    for j in range(other.shape[1]):
        cell_block = " ".join(other.iloc[0:10, j].astype(str).tolist()).upper()
        m = re.search(r"CY\s*(20\d{2})", cell_block)
        if m:
            year_cols_idx.append(j)
            years.append(int(m.group(1)))

if not years:
    raise ValueError("Could not locate CY20xx year columns in OTHER RATES.")

# find the ACTR row index
actr_row_idx = None
pattern = re.compile(r"ALLOWABLE.*TEST.*RATE", re.IGNORECASE)  # matches 'ALLOWABLE CONTL TEST RATE' or 'CONTROL'
for i in range(other.shape[0]):
    row_txt = " ".join([str(x) for x in other.iloc[i, :].tolist() if pd.notna(x)])
    if pattern.search(row_txt):
        actr_row_idx = i
        break
if actr_row_idx is None:
    raise ValueError("Could not find 'ALLOWABLE ... TEST RATE' row in OTHER RATES.")

# build VT rows from ACTR
vt_rows = []
for j, y in zip(year_cols_idx, years):
    val = pd.to_numeric(other.iat[actr_row_idx, j], errors="coerce")
    if pd.notna(val):
        vt_rows.append({"rate_band_code": "VT", "year": int(y), "rate_value": float(val), "source": "ACTR"})

vt_df = pd.DataFrame(vt_rows)

# --- combined dataset for the importer ----------------------------
combined_rates = pd.concat([simp_long, vt_df], ignore_index=True).sort_values(["rate_band_code", "year"])
combined_rates.reset_index(drop=True, inplace=True)

# quick sanity prints (you can remove later)
print("Bands (sample):", combined_rates["rate_band_code"].drop_duplicates().head(10).tolist())
print("Years span:", combined_rates["year"].min(), "→", combined_rates["year"].max())
print("VT preview:\n", combined_rates[combined_rates["rate_band_code"] == "VT"].head())

# `combined_rates` is the single source of truth you’ll join to RateBandImport on (rate_band_code, year).
combined_rates.head(12)