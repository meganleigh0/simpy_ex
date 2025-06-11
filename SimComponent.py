# ────────────────────────────────────────────────────────────────────────────────
# Pipeline: sync RateBandImport with Simplified Rates + Allowable Control Test
#           (parses OTHER RATES exactly as in the user’s snippet)
# ────────────────────────────────────────────────────────────────────────────────
import pandas as pd, re
from pathlib import Path

# 1 ▌CONFIG ─────────────────────────────────────────────────────────────────────
RATE_SUM_PATH    = Path("data/RATSUM.xlsx")             # workbook containing both sheets
RATE_BAND_PATH   = Path("data/RateBandImport.xlsx")     # original import template
OUTPUT_PATH      = Path("data/RateBandImport_UPDATED.xlsx")
COMPARE_PATH     = Path("data/RateBand_Comparison.xlsx")

SHEET_SIMPLIFIED = "SIMPLIFIED RATES NON-PSPL"
SHEET_OTHER      = "OTHER RATES"          # sheet that has Allowable Control Test row
SKIP_SIMPLIFIED  = 5                      # rows to skip before header in SIMPLIFIED
VT_ABRAMS_CODE   = "VTABRAMS"             # Rate‑Band label inside RBI

# 2 ▌LOAD + CLEAN SIMPLIFIED RATES ─────────────────────────────────────────────
def load_simplified_rates(path: Path) -> pd.DataFrame:
    df = (pd.read_excel(path, sheet_name=SHEET_SIMPLIFIED, skiprows=SKIP_SIMPLIFIED)
            .rename(columns={"Unnamed: 1": "BUSINESS UNIT GDLS"})
            .iloc[1:]                                   # drop duplicated header row
            .reset_index(drop=True))
    df.columns = df.columns.map(lambda x: str(x).strip())
    df = df.loc[:, ~df.columns.str.contains("Unnamed", case=False, na=False)]
    df["rate_band"] = df["BUSINESS UNIT GDLS"].astype(str).str.extract(r"^(..)")
    df = df[df["rate_band"].str.len() == 2]
    year_cols = [c for c in df.columns if re.match(r"#\s*CY\d{4}", c)]
    return df[["rate_band"] + year_cols]

simplified_rates = load_simplified_rates(RATE_SUM_PATH)

# 3 ▌PARSE “OTHER RATES”  VIA YOUR ROW/COLUMN OFFSETS ──────────────────────────
def load_allowable_control_rate(path: Path) -> pd.Series:
    wb = pd.ExcelFile(path)
    other_rates = wb.parse(SHEET_OTHER, skiprows=0)

    # row 25 → rate values; row 4 → same row that holds the CY headers
    allowable_control_test_rate = other_rates.loc[25, 'Unnamed: 2':'Unnamed: 6']
    years_row                   = other_rates.loc[4,  'Unnamed: 2':'Unnamed: 6']

    # build a mapping:  {"CY2022": 5724.10 , ...}
    years  = years_row.squeeze().astype(str).str.strip()
    rates  = pd.to_numeric(allowable_control_test_rate.squeeze(), errors='coerce')
    return pd.Series(rates.values, index=years).rename_axis("Year")

allowable_rates = load_allowable_control_rate(RATE_SUM_PATH)

# 4 ▌LOAD RATE BAND IMPORT ─────────────────────────────────────────────────────
rate_band = pd.read_excel(RATE_BAND_PATH)
rate_band["Rate Band"] = rate_band["Rate Band"].astype(str).str.strip()

if VT_ABRAMS_CODE not in set(rate_band["Rate Band"]):
    rate_band = pd.concat(
        [rate_band, pd.DataFrame([{"Rate Band": VT_ABRAMS_CODE}])],
        ignore_index=True,
    )

# 5 ▌MERGE SIMPLIFIED RATES INTO RBI ───────────────────────────────────────────
for col in simplified_rates.columns:
    if col == "rate_band":
        continue
    tgt = re.sub(r"^#\s*", "", col)            # "# CY2024" → "CY2024"
    if tgt not in rate_band.columns:
        rate_band[tgt] = pd.NA

    mapping = simplified_rates.set_index("rate_band")[col]
    rate_band[tgt] = rate_band["Rate Band"].map(mapping).combine_first(rate_band[tgt])

# 6 ▌WRITE ALLOWABLE CONTROL TEST TO VT ABRAMS ────────────────────────────────
for yr, val in allowable_rates.items():
    if yr not in rate_band.columns:
        rate_band[yr] = pd.NA
    idx = rate_band["Rate Band"] == VT_ABRAMS_CODE
    rate_band.loc[idx, yr] = val

# 7 ▌ROUND & SAVE UPDATED RBI ─────────────────────────────────────────────────
year_cols = [c for c in rate_band.columns if re.match(r"CY\d{4}", c)]
rate_band[year_cols] = (rate_band[year_cols]
                        .apply(pd.to_numeric, errors="coerce")
                        .round(5))
rate_band.to_excel(OUTPUT_PATH, index=False)

# 8 ▌COMPARISON REPORT (“what’s missing”) ──────────────────────────────────────
simplified_set = set(simplified_rates["rate_band"])
rbi_set        = set(rate_band["Rate Band"])

pd.DataFrame(sorted(simplified_set - rbi_set), columns=["rate_band"]) \
  .to_excel(COMPARE_PATH, sheet_name="In_RateSum_only", index=False)
pd.DataFrame(sorted(rbi_set - simplified_set), columns=["rate_band"]) \
  .to_excel(COMPARE_PATH, sheet_name="In_RateBand_only", index=False)

print("✅  Done!")
print(f"   • Updated RBI  → {OUTPUT_PATH.resolve()}")
print(f"   • Diff report  → {COMPARE_PATH.resolve()}")