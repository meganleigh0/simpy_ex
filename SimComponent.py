# ────────────────────────────────────────────────────────────────────────────────
# Pipeline: sync RateBandImport.xlsx with Simplified Rates in RATSUM.xlsx
# ────────────────────────────────────────────────────────────────────────────────
import pandas as pd, re
from pathlib import Path

# 1 ▌CONFIG ─────────────────────────────────────────────────────────────────────
RATE_SUM_PATH   = Path("data/RATSUM.xlsx")
RATE_BAND_PATH  = Path("data/RateBandImport.xlsx")
OUTPUT_PATH     = Path("data/RateBandImport_UPDATED.xlsx")     # written at end

SHEET_NAME      = "SIMPLIFIED RATES NON-PSPL"
SKIP_ROWS       = 5                                            # header offset

# 2 ▌LOAD + CLEAN RATSUM ────────────────────────────────────────────────────────
def load_simplified_rates(path: Path) -> pd.DataFrame:
    df = (pd.read_excel(path, sheet_name=SHEET_NAME, skiprows=SKIP_ROWS)
            .rename(columns={"Unnamed: 1": "BUSINESS UNIT GDLS"})   # real header
            .iloc[1:]                                              # drop dup row
            .reset_index(drop=True))

    # ensure all headers are strings *before* using .str methods
    df.columns = df.columns.map(lambda x: str(x).strip())

    # drop filler/blank columns
    df = df.loc[:, ~df.columns.str.contains("Unnamed", case=False, na=False)]

    # 2‑char band key (first two chars of the Business‑Unit string)
    df["rate_band"] = df["BUSINESS UNIT GDLS"].astype(str).str.extract(r"^(..)")
    df = df[df["rate_band"].str.len() == 2]                       # keep real rows

    # isolate the CY20xx columns (they start with “# CY” in this file)
    year_cols = [c for c in df.columns if re.match(r"#\s*CY\d{4}", c)]
    return df[["rate_band"] + year_cols]

simplified_rates = load_simplified_rates(RATE_SUM_PATH)

# 3 ▌LOAD RATE BAND IMPORT ──────────────────────────────────────────────────────
rate_band = pd.read_excel(RATE_BAND_PATH)
rate_band["Rate Band"] = rate_band["Rate Band"].astype(str).str.strip()

# 4 ▌MERGE / UPDATE ─────────────────────────────────────────────────────────────
for col in simplified_rates.columns:
    if col == "rate_band":
        continue
    # convert “# CY2024” → “CY2024” so naming matches RateBandImport
    target_col = re.sub(r"^#\s*", "", col)
    if target_col not in rate_band.columns:
        rate_band[target_col] = pd.NA                        # create if missing

    mapping = simplified_rates.set_index("rate_band")[col]
    rate_band[target_col] = (rate_band["Rate Band"]
                             .map(mapping)
                             .combine_first(rate_band[target_col]))

# 5 ▌ROUND & SAVE ───────────────────────────────────────────────────────────────
year_cols = [c for c in rate_band.columns if re.match(r"CY\d{4}", c)]
rate_band[year_cols] = (rate_band[year_cols]
                        .apply(pd.to_numeric, errors="coerce")
                        .round(5))                     # default: 5 decimals

rate_band.to_excel(OUTPUT_PATH, index=False)
print(f"✅  Updated file written → {OUTPUT_PATH.resolve()}")