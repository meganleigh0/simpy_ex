import re
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

# -------------------- config tied to your files --------------------
RATESUM_PATH = "data/RATSUM.xlsx"
SIMPLIFIED_SHEET = "SIMPLIFIED RATES NON-PSPL"   # from your screenshot
SIMPLIFIED_SKIPROWS = 5
OTHER_RATES_SHEET = "OTHER RATES"

RATEBAND_PATH = "data/RateBandImport.xlsx"
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True, parents=True)

ALLOWABLE_LABEL = "allowable control test rate"  # row label in Other Rates (case-insensitive)
VT_BAND_CODE = "VT"                              # treat these as ACTR
ABRAMS_TOKENS = ("abrams", "vtabrams")

ROUNDING = {  # applied if columns exist (case-insensitively)
    "labor dollars": 3,
    "burdens": 5,
    "com": 6,
}

# -------------------- small helpers --------------------
def first_two_chars(x):
    s = str(x).strip()
    return s[:2] if s else ""

def year_from_text(t):
    if t is None or (isinstance(t,float) and np.isnan(t)):
        return None
    m = re.search(r"(?:CY)?\s*(20\d{2})", str(t))
    return int(m.group(1)) if m else None

def round_half_up(x, places):
    if pd.isna(x): 
        return np.nan
    q = Decimal("1").scaleb(-places)  # 3 -> 0.001
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))

# -------------------- load: Simplified Rates --------------------
xls = pd.ExcelFile(RATESUM_PATH)

simp = pd.read_excel(xls, SIMPLIFIED_SHEET, skiprows=SIMPLIFIED_SKIPROWS, dtype=object)
simp.columns = [str(c).strip() for c in simp.columns]

# fix the Business Unit column name you showed (often "Unnamed: 1")
if "Unnamed: 1" in simp.columns:
    simp.rename(columns={"Unnamed: 1": "BUSINESS UNIT GDLS"}, inplace=True)

# remove duplicate “header” row if present
if str(simp.iloc[0].get("BUSINESS UNIT GDLS", "")).strip().lower() == "business unit gdls":
    simp = simp.iloc[1:].reset_index(drop=True)

# drop useless "Unnamed" columns
simp = simp.loc[:, ~pd.Index(simp.columns).astype(str).str.contains("Unnamed", case=False)]

# detect year columns like "CY2022", "# CY2023", "2024" etc.
year_cols = [c for c in simp.columns if year_from_text(c)]
if not year_cols:
    raise ValueError("No year columns found in Simplified Rates.")

simp["BUSINESS UNIT GDLS"] = simp["BUSINESS UNIT GDLS"].astype(str).str.strip()
simp["rate_band_code"] = simp["BUSINESS UNIT GDLS"].map(first_two_chars)

simp_long = simp.melt(
    id_vars=["BUSINESS UNIT GDLS", "rate_band_code"],
    value_vars=year_cols,
    var_name="year_col",
    value_name="value"
)
simp_long["year"] = simp_long["year_col"].map(year_from_text).astype("Int64")
simp_long["value"] = pd.to_numeric(simp_long["value"], errors="coerce")

# keep first non-null per (band, year)
simp_long = (simp_long
             .dropna(subset=["rate_band_code", "year"])
             .sort_values(["rate_band_code", "year"])
             .drop_duplicates(subset=["rate_band_code", "year"], keep="first"))

# map: (band, year) -> value
simp_map = simp_long.set_index(["rate_band_code", "year"])["value"].sort_index()

# -------------------- load: Other Rates (ACTR row) --------------------
# read without header; detect the year header row and the ACTR row by content
other = pd.read_excel(xls, OTHER_RATES_SHEET, header=None)

# find the row that holds years (needs >=2 valid years in that row)
year_row_idx = None
year_cols_idx = []
years_found = []
for i in range(len(other)):
    row = other.iloc[i]
    cols = []
    yrs = []
    for j, val in enumerate(row):
        y = year_from_text(val)
        if y:
            cols.append(j)
            yrs.append(y)
    if len(yrs) >= 2:   # reasonably sure we found the year header row
        year_row_idx = i
        year_cols_idx = cols
        years_found = yrs
        break
if year_row_idx is None:
    raise ValueError("Could not find a row with year headings in Other Rates.")

# find the row whose text contains the label
actr_row_idx = None
for i in range(len(other)):
    row_text = " ".join([str(v) for v in other.iloc[i].tolist() if isinstance(v, (str, int, float))]).lower()
    if ALLOWABLE_LABEL in row_text:
        actr_row_idx = i
        break
if actr_row_idx is None:
    raise ValueError(f"Could not find row containing '{ALLOWABLE_LABEL}' in Other Rates.")

# build series year -> ACTR value
actr_vals = []
for j, y in zip(year_cols_idx, years_found):
    val = pd.to_numeric(other.iat[actr_row_idx, j], errors="coerce")
    actr_vals.append((y, val))
actr_series = pd.Series({y: v for y, v in actr_vals}).sort_index()

# -------------------- load: Rate Band Import --------------------
rb = pd.read_excel(RATEBAND_PATH, dtype=object)
rb.columns = [str(c).strip() for c in rb.columns]

# detect key columns flexibly
def pick_col(df, hints):
    lc = {c.lower(): c for c in df.columns}
    for h in hints:
        for k, orig in lc.items():
            if h in k:
                return orig
    raise ValueError(f"Missing any column like: {hints}")

rate_band_col = pick_col(rb, ("rate band", "rateband"))
start_col     = pick_col(rb, ("start", "effective start"))
end_col       = pick_col(rb, ("end", "effective end"))
desc_col      = next((c for c in rb.columns if "desc" in c.lower() or "program" in c.lower() or "platform" in c.lower()), None)

# normalize
rb["_band_raw"] = rb[rate_band_col].astype(str).str.strip()
# left-pad digits to width 2 (e.g., "9" -> "09"), otherwise keep first two chars
def normalize_band(s):
    s = str(s).strip()
    if s.isdigit() and len(s) < 2:
        return s.zfill(2)
    return s[:2].upper()
rb["rate_band_code"] = rb["_band_raw"].map(normalize_band)

rb["_start"] = pd.to_datetime(rb[start_col], errors="coerce")
rb["_end"]   = pd.to_datetime(rb[end_col], errors="coerce")
if desc_col:
    rb["_desc"] = rb[desc_col].astype(str).str.lower()
else:
    rb["_desc"] = ""

rb = rb.dropna(subset=["_start", "_end"])
rb["_start_year"] = rb["_start"].dt.year.astype(int)
rb["_end_year"]   = rb["_end"].dt.year.astype(int)

# -------------------- comparison sets --------------------
bands_simplified = set(simp_long["rate_band_code"].unique())
bands_rb = set(rb["rate_band_code"].unique())
comp_in_simp_not_rb = pd.DataFrame({"rate_band_code": sorted(bands_simplified - bands_rb)})
comp_in_rb_not_simp = pd.DataFrame({"rate_band_code": sorted(bands_rb - bands_simplified)})

# -------------------- carry-forward helper --------------------
needed_years = list(range(int(rb["_start_year"].min()), int(rb["_end_year"].max()) + 1))

def cf(series_by_year, needed):
    if series_by_year is None or series_by_year.empty:
        return pd.Series(index=needed, dtype=float)
    s = series_by_year.sort_index()
    full = pd.Series(index=range(min(s.index), max(max(s.index), max(needed)) + 1), dtype=float)
    full.loc[s.index] = s.values
    full = full.ffill()
    return full.reindex(needed)

# build per-band series from simplified
band_to_series = {}
for b in bands_simplified:
    s = simp_map.xs(b) if (b,) in set(idx[0:1] for idx in simp_map.index) else None
    if s is None:
        continue
    band_to_series[b] = cf(s, needed_years)

actr_cf = cf(actr_series, needed_years)

# -------------------- update/expand per year --------------------
records = []
for idx, row in rb.iterrows():
    b = row["rate_band_code"]
    y0, y1 = int(row["_start_year"]), int(row["_end_year"])
    years = range(y0, y1 + 1)

    # ACTR override?
    desc = row["_desc"]
    use_actr = (b.upper() == VT_BAND_CODE) or any(t in desc for t in ABRAMS_TOKENS)

    s = actr_cf if use_actr else band_to_series.get(b, pd.Series(dtype=float))

    for y in years:
        val = np.nan if s is None or y not in s.index else s.loc[y]
        records.append({
            "Rate Band": row["_band_raw"],
            "rate_band_code": b,
            "Description": row.get(desc_col, np.nan) if desc_col else np.nan,
            "Start Date": row["_start"],
            "End Date": row["_end"],
            "Year": y,
            "Rate Value": val,
            "Source": "ACTR" if use_actr else "Simplified"
        })

update_df = pd.DataFrame.from_records(records)

# -------------------- rounding --------------------
# create rounded columns only if corresponding columns exist in the input
rb_cols_lc = {c.lower(): c for c in rb.columns}
for key, places in ROUNDING.items():
    # find the first column in the rate band file that matches this key
    match = next((rb_cols_lc[c] for c in rb_cols_lc if key in c), None)
    if match:
        update_df[match] = update_df["Rate Value"].apply(lambda v: round_half_up(v, places))

# always keep a generic rounded column (3 dp) for convenience
update_df["Rate Value (3dp)"] = update_df["Rate Value"].apply(lambda v: round_half_up(v, 3))

# -------------------- save --------------------
update_path = OUT_DIR / "rate_band_import_update.xlsx"
compare_path = OUT_DIR / "comparison_report.xlsx"

with pd.ExcelWriter(update_path, engine="openpyxl") as xw:
    update_df.to_excel(xw, sheet_name="rate_band_update", index=False)

with pd.ExcelWriter(compare_path, engine="openpyxl") as xw:
    comp_in_simp_not_rb.to_excel(xw, sheet_name="in_ratesum_not_in_ratebands", index=False)
    comp_in_rb_not_simp.to_excel(xw, sheet_name="in_ratebands_not_in_ratesum", index=False)

print(f"✅ Done.\n- Update: {update_path}\n- Compare: {compare_path}\n"
      f"- Bands in Simplified only: {len(comp_in_simp_not_rb)} | in RateBand only: {len(comp_in_rb_not_simp)}")