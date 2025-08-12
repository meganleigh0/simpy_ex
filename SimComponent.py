# -*- coding: utf-8 -*-
import re
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

# ------------ CONFIG ------------
RATESUM_PATH = "data/RATSUM.xlsx"
SIMPLIFIED_SHEET = "SIMPLIFIED RATES NON-PSPL"
SIMPLIFIED_SKIPROWS = 5
OTHER_RATES_SHEET = "OTHER RATES"

RATEBAND_PATH = "data/RateBandImport.xlsx"
OUT_DIR = Path("out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

VT_CODE = "VT"
ABRAMS_TOKENS = ("abrams", "vt abrams", "vtabrams")

ROUNDING = {"labor dollars": 3, "burdens": 5, "com": 6}

# ------------ HELPERS ------------
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

# =========================================================
# STEP 1 — Build combined_rates from RATSUM (Simplified + ACTR)
# =========================================================
xls = pd.ExcelFile(RATESUM_PATH)

# Simplified
simp = pd.read_excel(xls, SIMPLIFIED_SHEET, skiprows=SIMPLIFIED_SKIPROWS, dtype=object)
simp.columns = [str(c).strip() for c in simp.columns]
bu_col = pick_business_unit_col(simp.columns)
simp.rename(columns={bu_col: "BUSINESS UNIT GDLS"}, inplace=True)

# Drop repeated header row if present
if str(simp.iloc[0]["BUSINESS UNIT GDLS"]).strip().lower().startswith("business"):
    simp = simp.iloc[1:].reset_index(drop=True)

simp = simp.loc[:, ~pd.Index(simp.columns).str.contains("Unnamed", case=False)]
year_cols = [c for c in simp.columns if year_from_header(c)]
if not year_cols: raise ValueError("No year columns in Simplified sheet.")

simp["BUSINESS UNIT GDLS"] = simp["BUSINESS UNIT GDLS"].astype(str).str.strip()
simp["rate_band_code"] = simp["BUSINESS UNIT GDLS"].map(normalize_band_code)

simp_long = simp.melt(
    id_vars=["BUSINESS UNIT GDLS", "rate_band_code"],
    value_vars=year_cols, var_name="year_col", value_name="rate_value"
)
simp_long["year"] = simp_long["year_col"].map(year_from_header)
simp_long["rate_value"] = pd.to_numeric(simp_long["rate_value"], errors="coerce")
simp_long = (simp_long.dropna(subset=["rate_band_code","year"])
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

combined_rates = pd.concat([simp_long, vt_df], ignore_index=True)
combined_rates = (combined_rates.dropna(subset=["rate_band_code","year"])
                                .astype({"year": int})
                                .sort_values(["rate_band_code","year","source"])
                                .drop_duplicates(subset=["rate_band_code","year"], keep="first")
                                .reset_index(drop=True))

# =========================================================
# STEP 2 — Load RateBandImport, update values + rounding
# =========================================================
rb = pd.read_excel(RATEBAND_PATH, dtype=object)
rb.columns = [str(c).strip() for c in rb.columns]

rate_band_col = find_col(rb, ("rate band","rateband"))
start_col     = find_col(rb, ("start","effective start"))
end_col       = find_col(rb, ("end","effective end"))
desc_col      = next((c for c in rb.columns if "desc" in c.lower() or "program" in c.lower() or "platform" in c.lower()), None)
base_col      = next((c for c in rb.columns if "base rate" in c.lower()), None)
ld_col        = next((c for c in rb.columns if "labor dollars" in c.lower()), None)
bur_col       = next((c for c in rb.columns if "burden" in c.lower()), None)
com_col       = next((c for c in rb.columns if re.search(r"\bcom\b", c.lower())), None)

rb["_band_raw"]      = rb[rate_band_col].astype(str).str.strip()
rb["rate_band_code"] = rb["_band_raw"].map(normalize_band_code)
rb["_start"] = pd.to_datetime(rb[start_col], errors="coerce")
rb["_end"]   = pd.to_datetime(rb[end_col], errors="coerce")
rb = rb.dropna(subset=["_start","_end"]).copy()
rb["_start_year"] = rb["_start"].dt.year.astype(int)
rb["_end_year"]   = rb["_end"].dt.year.astype(int)
rb["_desc"] = rb[desc_col].astype(str).str.lower() if desc_col else ""

# fast lookup map (carry-forward)
band_to_series = {b: g.set_index("year")["rate_value"].sort_index()
                  for b, g in combined_rates.groupby("rate_band_code")}

def rate_for(band, year):
    s = band_to_series.get(band)
    if s is None or s.empty: return np.nan, None
    if year in s.index: return float(s.loc[year]), "exact"
    prev = s.index[s.index <= year]
    if len(prev)==0: return np.nan, None
    y0 = int(prev.max()); return float(s.loc[y0]), f"ffill_from_{y0}"

# expand each RB row to years
rows = []
for ridx, r in rb.iterrows():
    for y in range(int(r["_start_year"]), int(r["_end_year"]) + 1):
        val, how = rate_for(r["rate_band_code"], y)
        rows.append({"_ridx": ridx, "Year": y, "value_raw": val, "value_how": how})
rb_y = pd.DataFrame(rows)

# flag VT/Abrams — FIX: merge using rb index only (no '_ridx' column on rb)
rb["_is_vt_abrams"] = (rb["rate_band_code"] == VT_CODE) | rb["_desc"].str.contains("|".join(ABRAMS_TOKENS), na=False)
rb_y = rb_y.merge(rb[["_is_vt_abrams"]], left_on="_ridx", right_index=True, how="left")

def compute_outputs(val, is_vt):
    base_out = round_half_up(val, 0 if is_vt else 3)  # VT base rate = whole dollars
    out = {"_base_out": base_out}
    if ld_col:  out["LD_out"]  = round_half_up(val, ROUNDING["labor dollars"])
    if bur_col: out["BUR_out"] = round_half_up(val, ROUNDING["burdens"])
    if com_col: out["COM_out"] = round_half_up(val, ROUNDING["com"])
    return out

outs = rb_y.apply(lambda r: compute_outputs(r["value_raw"], bool(r["_is_vt_abrams"])), axis=1, result_type="expand")
rb_y = pd.concat([rb_y, outs], axis=1)

# fold back to RB rows (one per original per year)
rb_update = (rb.merge(rb_y.drop(columns=["value_how"]), left_index=True, right_on="_ridx", how="left"))

# write updated numeric columns
if base_col: rb_update[base_col] = rb_update["_base_out"]
if ld_col:   rb_update[ld_col]   = rb_update["LD_out"]
if bur_col:  rb_update[bur_col]  = rb_update["BUR_out"]
if com_col:  rb_update[com_col]  = rb_update["COM_out"]

# pretty order
lead_cols = [rate_band_col, "rate_band_code"] + ([desc_col] if desc_col else []) + [start_col, end_col, "Year"]
num_cols  = [c for c in [base_col, ld_col, bur_col, com_col] if c]
keep = [c for c in lead_cols + num_cols if c in rb_update.columns] + \
       [c for c in rb_update.columns if c not in lead_cols + num_cols and not c.startswith("_")]
rb_update = rb_update.loc[:, keep].sort_values([rate_band_col, "Year", start_col])

# =========================================================
# STEP 3 — Comparison report
# =========================================================
bands_cr = set(combined_rates["rate_band_code"].dropna().map(normalize_band_code).unique())
bands_rb = set(rb["rate_band_code"].dropna().unique())
only_in_ratesum   = sorted(list(bands_cr - bands_rb))
only_in_ratebands = sorted(list(bands_rb - bands_cr))

df_only_in_ratesum = (combined_rates.assign(rate_band_code=lambda d: d["rate_band_code"].map(normalize_band_code))
                      .query("rate_band_code in @only_in_ratesum")
                      .groupby("rate_band_code", as_index=False)
                      .agg(min_year=("year","min"), max_year=("year","max"), rows=("year","size"))
                      .sort_values("rate_band_code"))

df_only_in_ratebands = (rb.groupby("rate_band_code", as_index=False)
                          .agg(rows=("rate_band_code","size"))
                          .query("rate_band_code in @only_in_ratebands")
                          .sort_values("rate_band_code"))

# save
update_path  = OUT_DIR / "RateBandImportUpdate.xlsx"
compare_path = OUT_DIR / "comparison_report.xlsx"

with pd.ExcelWriter(update_path, engine="openpyxl") as xw:
    rb_update.to_excel(xw, sheet_name="update", index=False)

with pd.ExcelWriter(compare_path, engine="openpyxl") as xw:
    pd.DataFrame({"rate_band_code": only_in_ratesum}).to_excel(xw, sheet_name="in_ratesum_not_in_ratebands", index=False)
    pd.DataFrame({"rate_band_code": only_in_ratebands}).to_excel(xw, sheet_name="in_ratebands_not_in_ratesum", index=False)
    df_only_in_ratesum.to_excel(xw, sheet_name="ratesum_details", index=False)
    df_only_in_ratebands.to_excel(xw, sheet_name="ratebands_details", index=False)

print(f"✅ RateBandImportUpdate → {update_path}")
print(f"✅ comparison_report   → {compare_path}")
print(f"Bands: in RateSum not in RateBands = {len(only_in_ratesum)} | in RateBands not in RateSum = {len(only_in_ratebands)}")