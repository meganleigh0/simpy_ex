# ----------------------------------------------------------------------------------
# Pipeline: update Rate Band sheet with the “Simplified Rates” from RATSUM.xlsx
# ----------------------------------------------------------------------------------
import pandas as pd
from pathlib import Path

# ── 1. CONFIG  ────────────────────────────────────────────────────────────────────
RATE_SUM_PATH   = Path("data/RATSUM.xlsx")          # source file ①
RATE_BAND_PATH  = Path("data/RateBandImport.xlsx")  # source file ②
OUTPUT_PATH     = Path("data/RateBandImport_UPDATED.xlsx")  # ← will be written

SHEET_NAME      = "SIMPLIFIED RATES NON-PSPL"  # sheet that holds the clean rates
SKIP_ROWS       = 5                            # rows to skip before table starts

# ── 2. LOAD & CLEAN “SIMPLIFIED RATES”  ───────────────────────────────────────────
def load_rate_sum(fp: Path) -> pd.DataFrame:
    df = (pd.read_excel(fp, sheet_name=SHEET_NAME, skiprows=SKIP_ROWS)
            # rename the real header that sits in col B row 6
            .rename(columns={"Unnamed: 1": "BUSINESS UNIT GDLS"})
            .iloc[1:]                       # drop duplicate header row
            .reset_index(drop=True))

    # drop any leftover “Unnamed” columns
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]

    # create two‑letter “rate_band” code (first two chars of BUSINESS UNIT)
    df["rate_band"] = df["BUSINESS UNIT GDLS"].str.extract(r"^(..)")
    df = df[~df["rate_band"].isin({"B/", "na", "nan"})]  # remove sub‑headers

    # keep only the CY20xx numeric columns + the key
    year_cols = [c for c in df.columns if c.startswith("# CY")]
    keep_cols = ["rate_band"] + year_cols
    return df[keep_cols]

simplified_rates = load_rate_sum(RATE_SUM_PATH)

# ── 3. LOAD “RateBandImport.xlsx”  ────────────────────────────────────────────────
rate_band = pd.read_excel(RATE_BAND_PATH)
rate_band["Rate Band"] = rate_band["Rate Band"].astype(str)  # normalise dtype

# ── 4. UPDATE RATE BAND VALUES  ───────────────────────────────────────────────────
# match on the two‑letter code and overwrite CY20xx columns where a match exists
for col in simplified_rates.columns:
    if col == "rate_band":
        continue                                          # skip key column
    target_col = col.replace("# ", "")                    # "# CY2024" → "CY2024"
    if target_col not in rate_band.columns:
        # add missing year column so that the merge is loss‑less
        rate_band[target_col] = pd.NA

    # map() will align by index; combine_first() keeps existing non‑NA values
    mapping = simplified_rates.set_index("rate_band")[col]
    rate_band[target_col] = (rate_band["Rate Band"]
                             .map(mapping)
                             .combine_first(rate_band[target_col]))

# ── 5. ROUND & SAVE  ──────────────────────────────────────────────────────────────
year_cols = [c for c in rate_band.columns if c.startswith("CY20")]
rate_band[year_cols] = rate_band[year_cols].astype(float).round(5)  # 5‑dec default
rate_band.to_excel(OUTPUT_PATH, index=False)

print(f"✅  RateBandImport has been updated and written to → {OUTPUT_PATH.resolve()}")