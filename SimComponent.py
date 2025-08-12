import re
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

RATEBAND_PATH = "data/RateBandImport.xlsx"
OUT_DIR = Path("out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def round_half_up(x, places):
    if pd.isna(x): 
        return np.nan
    q = Decimal("1").scaleb(-places)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))

def first_two_chars(s):
    s = str(s).strip()
    return s[:2].upper() if s else ""

def pick_col(df, hints):
    lc = {c.lower(): c for c in df.columns}
    for h in hints:
        for k, orig in lc.items():
            if h in k:
                return orig
    raise ValueError(f"Missing any column like: {hints}")

# ---------- load the RateBandImport ----------
rb = pd.read_excel(RATEBAND_PATH, dtype=object)
rb.columns = [str(c).strip() for c in rb.columns]

rate_band_col = pick_col(rb, ("rate band", "rateband"))
start_col     = pick_col(rb, ("start", "effective start"))
end_col       = pick_col(rb, ("end", "effective end"))
desc_col      = next((c for c in rb.columns if "desc" in c.lower() or "program" in c.lower() or "platform" in c.lower()), None)

# numeric "base rate" style column (falls back if LD/Burdens/COM not present)
base_col = next((c for c in rb.columns if "base rate" in c.lower()), None)

# normalize key fields
rb["_band_raw"] = rb[rate_band_col].astype(str).str.strip()
rb["rate_band_code"] = rb["_band_raw"].map(first_two_chars)
rb["_start"] = pd.to_datetime(rb[start_col], errors="coerce")
rb["_end"]   = pd.to_datetime(rb[end_col], errors="coerce")
if desc_col:
    rb["_desc"] = rb[desc_col].astype(str).str.lower()
else:
    rb["_desc"] = ""

rb = rb.dropna(subset=["_start","_end"]).copy()
rb["_start_year"] = rb["_start"].dt.year.astype(int)
rb["_end_year"]   = rb["_end"].dt.year.astype(int)

# ---------- prep: fast lookup from combined_rates with carry-forward ----------
# Expecting combined_rates from Step 1: columns ['rate_band_code','year','rate_value','source']
if "combined_rates" not in globals():
    raise RuntimeError("`combined_rates` is not defined. Run Step 1 first to build it from RATSUM.")

# Keep the latest value per band-year (already deduped in Step 1)
cr = combined_rates.dropna(subset=["rate_band_code","year"]).copy()
cr["year"] = cr["year"].astype(int)

band_to_series = {}
for band, g in cr.groupby("rate_band_code"):
    s = g.set_index("year")["rate_value"].sort_index()
    band_to_series[band] = s

def rate_for(band, year):
    """Return value for band at year, carrying forward from the last <= year."""
    s = band_to_series.get(band)
    if s is None or s.empty:
        return np.nan, None
    # exact hit
    if year in s.index:
        return float(s.loc[year]), "exact"
    # carry-forward from last <= year
    prev_years = s.index[s.index <= year]
    if len(prev_years) == 0:
        return np.nan, None
    y0 = int(prev_years.max())
    return float(s.loc[y0]), f"ffill_from_{y0}"

# ---------- expand RB rows to years and attach values ----------
rows = []
for ridx, r in rb.iterrows():
    b = r["rate_band_code"]
    y0, y1 = int(r["_start_year"]), int(r["_end_year"])
    for y in range(y0, y1 + 1):
        val, how = rate_for(b, y)
        rows.append({
            "_ridx": ridx,
            "rate_band_code": b,
            "Year": y,
            "value_raw": val,
            "value_how": how
        })
rb_y = pd.DataFrame(rows)

# annotate VT/Abrams flag (for whole-dollar rounding)
rb["_is_vt_abrams"] = (rb["rate_band_code"] == "VT") | rb["_desc"].str.contains("abrams", na=False)

# join flags back onto per-year grid
rb_y = rb_y.merge(rb[["_ridx","_is_vt_abrams"]], left_on="_ridx", right_index=True, how="left")

# ---------- rounding + column mapping ----------
# Column targets (create only if present)
ld_col   = next((c for c in rb.columns if "labor dollars" in c.lower()), None)
bur_col  = next((c for c in rb.columns if "burden" in c.lower()), None)
com_col  = next((c for c in rb.columns if re.search(r"\bcom\b", c.lower())), None)

def compute_outputs(val, is_vt):
    # base rate logic: VT/Abrams -> whole dollars from ACTR; else 3dp
    base3 = round_half_up(val, 0 if is_vt else 3)
    out = {"_base_out": base3}
    if ld_col:  out["LD_out"]  = round_half_up(val, 3)
    if bur_col: out["BUR_out"] = round_half_up(val, 5)
    if com_col: out["COM_out"] = round_half_up(val, 6)
    return out

outs = rb_y.apply(lambda r: compute_outputs(r["value_raw"], bool(r["_is_vt_abrams"])), axis=1, result_type="expand")
rb_y = pd.concat([rb_y, outs], axis=1)

# ---------- fold back to RB shape (one row per original record per Year) ----------
# If original file already has one row per year, this is 1:1; if not, you'll get multiple rows per original record (one per year)
rb_update = (rb
    .drop(columns=[c for c in ["_start_year","_end_year"] if c in rb.columns])
    .merge(rb_y.drop(columns=["value_how"]), left_index=True, right_on="_ridx", how="left"))

# write updated numeric columns
if base_col:
    rb_update[base_col] = rb_update["_base_out"]
if ld_col:
    rb_update[ld_col] = rb_update["LD_out"]
if bur_col:
    rb_update[bur_col] = rb_update["BUR_out"]
if com_col:
    rb_update[com_col] = rb_update["COM_out"]

# nice ordering
lead_cols = [rate_band_col, "rate_band_code", desc_col] if desc_col else [rate_band_col, "rate_band_code"]
lead_cols += [start_col, end_col, "Year"]
num_cols = [c for c in [base_col, ld_col, bur_col, com_col] if c]
keep = [c for c in lead_cols + num_cols if c in rb_update.columns] + \
       [c for c in rb_update.columns if c not in lead_cols + num_cols and not c.startswith("_")]

rb_update = rb_update.loc[:, keep].sort_values([rate_band_col, "Year", start_col])

# ---------- export ----------
out_path = OUT_DIR / "RateBandImportUpdate.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
    rb_update.to_excel(xw, sheet_name="update", index=False)

# quick stats
n_vt = int(rb_update.loc[rb_update["rate_band_code"].eq("VT") | 
                         (rb_update.get(desc_col, "").astype(str).str.lower().str.contains("abrams", na=False)), :].shape[0])
print(f"✅ RateBandImportUpdate written to: {out_path}")
print(f"Rows updated: {rb_update.shape[0]} | VT/Abrams rows: {n_vt}")
rb_update.head(12)