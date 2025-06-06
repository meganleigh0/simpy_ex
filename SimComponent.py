# ----------------------------------------------------------------------------------
#  Robust end‑to‑end pipeline for building ProPricer import files from Shelly’s RATSUM
# ----------------------------------------------------------------------------------
import pandas as pd, numpy as np, re, logging
from pathlib import Path
from datetime import date

# ════════════════════════════════════════════════
# CONFIG ─ tweak these five items when needed
# ════════════════════════════════════════════════
RATSUM_PATH  = Path("data/RATSUM.xlsx")           # Shelly’s workbook
OUTPUT_DIR   = Path("output")                     # where finished files go
YEARS        = range(2022, 2027)                  # CY columns to keep
LAB_ESC      = {2024: 1.02,  2025: 1.025, 2026: 1.03}   # overall escalation
BURD_ESC     = {2024: 1.015, 2025: 1.02,  2026: 1.02}   # burden‑specific

OUTPUT_DIR.mkdir(exist_ok=True)

# ════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO, format="%(asctime)s • %(levelname)s • %(message)s")

def load_sheet(xl: pd.ExcelFile, name: str, skiprows: int = 5) -> pd.DataFrame:
    logging.info(f"Loading sheet: {name}")
    return xl.parse(name, skiprows=skiprows)

def strip_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    # Remove Unnamed columns *regardless of dtype*
    keep = [c for c in df.columns if "unnamed" not in str(c).lower()]
    out  = df.loc[:, keep].copy()
    out.columns = out.columns.map(str)            # make every label a str
    return out

def normalise_business_unit(df: pd.DataFrame, final_name: str = "BUSINESS UNIT GDLS") -> pd.DataFrame:
    """
    Find the column that looks like the Business‑Unit label and rename it.
    Fallback: assume first column.
    """
    pattern = re.compile(r"business\s*unit", re.I)
    for col in df.columns:
        if pattern.search(col):
            return df.rename(columns={col: final_name})
    return df.rename(columns={df.columns[0]: final_name})

def add_rate_band(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Rate Band"] = df["BUSINESS UNIT GDLS"].str.extract(r"^(\w{2})", expand=False)
    return df

def apply_escalation(df: pd.DataFrame, factors: dict) -> pd.DataFrame:
    df = df.copy()
    for yr, f in factors.items():
        col = f"CY{yr}"
        if col in df.columns:
            df[col] = df[col] * f
    return df

def round_years(df: pd.DataFrame, precision: int, extra: dict = None) -> pd.DataFrame:
    years = [c for c in df.columns if c.startswith("CY")]
    if extra:  # allow special precision by row
        mask = df.eval(extra["mask"])
        df.loc[~mask, years] = df.loc[~mask, years].round(precision)
        df.loc[mask , years] = df.loc[ mask, years].round(extra["precision"])
    else:
        df[years] = df[years].round(precision)
    return df

def compare_sets(new_df: pd.DataFrame, path: Path, key: str):
    if not path.exists(): return
    old = pd.read_excel(path)[key].astype(str)
    new = new_df[key].astype(str)
    added, removed = sorted(set(new)-set(old)), sorted(set(old)-set(new))
    logging.info("∆ Added %s | Removed %s", added, removed)

# ════════════════════════════════════════════════
# Builders
# ════════════════════════════════════════════════
def build_rate_band(simplified: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    simp = normalise_business_unit(strip_unnamed(simplified))
    simp = add_rate_band(simp)
    simp = apply_escalation(simp, LAB_ESC)
    simp = round_years(simp, 3)

    ctl = strip_unnamed(other)
    ctl = ctl[ctl.iloc[:, 0].astype(str).str.contains("ALLOWABLE", case=False, na=False)]
    ctl = ctl.rename(columns={ctl.columns[0]: "Rate Band"})  # first col = 'ALLOWABLE ...'

    df  = pd.concat([simp, ctl], ignore_index=True, sort=False)
    df  = df[["Rate Band", "BUSINESS UNIT GDLS", *(f"CY{y}" for y in YEARS)]]

    df["Effective Date"] = date(YEARS[0], 1, 1)
    df["End Date"]       = date(YEARS[-1], 12, 31)
    df["Factor"]         = 1.0
    df["Step"]           = 12
    return df

def build_burden_rate(other: pd.DataFrame, com: pd.DataFrame) -> pd.DataFrame:
    bur = pd.concat([strip_unnamed(other), strip_unnamed(com)], ignore_index=True, sort=False)
    bur = bur.rename(columns=lambda c: "Rate Band" if "Burden Pool" in c else c)

    bur = apply_escalation(bur, BURD_ESC)

    # COM rows get 6‑dp rounding, others 5‑dp
    bur = round_years(bur, 5, extra=dict(mask="Description.str.contains('COM', na=False)", precision=6))

    bur["Effective Date"] = date(YEARS[0], 1, 1)
    bur["End Date"]       = date(YEARS[-1], 12, 31)
    bur["Factor"]         = 1.0
    bur["Step"]           = 12
    return bur

# ════════════════════════════════════════════════
# Driver
# ════════════════════════════════════════════════
def main():
    xl = pd.ExcelFile(RATSUM_PATH)

    simplified   = load_sheet(xl, "SIMPLIFIED RATES NON-PSPL")
    other_rates  = load_sheet(xl, "OTHER RATES")
    com_summary  = load_sheet(xl, "COM SUMMARY")

    rate_band_df   = build_rate_band(simplified, other_rates)
    burden_rate_df = build_burden_rate(other_rates, com_summary)

    rb_file = OUTPUT_DIR / "RateBandImport.xlsx"
    br_file = OUTPUT_DIR / "BurdenRateImport.xlsx"

    rate_band_df.to_excel(rb_file,   sheet_name="Rate Band Import",   index=False, engine="xlsxwriter")
    burden_rate_df.to_excel(br_file, sheet_name="Burden Rate Import", index=False, engine="xlsxwriter")

    compare_sets(rate_band_df,   rb_file, "Rate Band")
    compare_sets(burden_rate_df, br_file, "Rate Band")

    logging.info("✅ Done. Files saved in %s", OUTPUT_DIR.resolve())

# ════════════════════════════════════════════════
# Kick it off
# ════════════════════════════════════════════════
if __name__ == "__main__":
    main()