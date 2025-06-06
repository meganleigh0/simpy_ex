# ----------------------------------------------------------------------------------
#  Unified ProPricer Rate‑file Pipeline
#  • Reads Shelly’s RATSUM.xlsx
#  • Builds Rate Band Import & Burden Rate Import workbooks
#  • Handles rounding (labor = 3 dp, burdens = 5 dp, COM = 6 dp)
#  • Lets you plug in calendar‑year escalation factors
#  • Generates “what’s new / what’s missing” summaries
#  • Saves outputs in ./output/
# ----------------------------------------------------------------------------------
import pandas as pd
from pathlib import Path
import datetime as dt
import logging

# ── CONFIG ─────────────────────────────────────────────────────────────────────────
RATSUM_PATH   = Path("data/RATSUM.xlsx")              # Shelly’s file
OUTPUT_DIR    = Path("output")                        # where finished .xlsx files go
OUTPUT_DIR.mkdir(exist_ok=True)

YEARS = list(range(2022, 2027))                       # CY columns to keep (edit as needed)

overall_escalation = {2024: 1.020, 2025: 1.025, 2026: 1.030}   # labor escalation %
burden_escalation  = {2024: 1.015, 2025: 1.020, 2026: 1.020}   # burden‑specific %

ROUNDING = {"LABOR": 3, "BURDEN": 5, "COM": 6}         # decimal places

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")

# ── HELPERS ────────────────────────────────────────────────────────────────────────
def load_sheet(xl: pd.ExcelFile, name: str, skiprows: int = 5) -> pd.DataFrame:
    logging.info(f"Loading '{name}'")
    return xl.parse(name, skiprows=skiprows)

def strip_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.str.contains("Unnamed")].rename_axis(None, axis=1)

def add_rate_band(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Rate Band"] = df["BUSINESS UNIT GDLS"].str.extract(r"^(.{2})")
    return df

def escalate(df: pd.DataFrame, factors: dict) -> pd.DataFrame:
    df = df.copy()
    for yr, f in factors.items():
        col = f"CY{yr}"
        if col in df.columns:
            df[col] *= f
    return df

def round_years(df: pd.DataFrame, precision: int) -> pd.DataFrame:
    years = [c for c in df.columns if c.startswith("CY")]
    df[years] = df[years].round(precision)
    return df

def compare_to_previous(new_df: pd.DataFrame, existing_path: Path, key: str):
    if not existing_path.exists():
        logging.info(f"No previous {existing_path.name}; skipping comparison.")
        return
    old = pd.read_excel(existing_path)[key].astype(str)
    new = new_df[key].astype(str)
    logging.info(f"► Added:   {sorted(set(new) - set(old))}")
    logging.info(f"► Removed: {sorted(set(old) - set(new))}")

# ── BUILDERS ───────────────────────────────────────────────────────────────────────
def build_rate_band_import(simp: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    simp = add_rate_band(strip_unnamed(simp))
    simp = escalate(simp, overall_escalation)
    simp = round_years(simp, ROUNDING["LABOR"])

    # Allowable Control Test rows (only rows where column 0 == 'ALLOWABLE CONTROL TEST RATE')
    ctl = strip_unnamed(other)
    ctl = ctl[ctl.iloc[:,0].str.contains("ALLOWABLE CONTROL TEST", na=False)]

    df = pd.concat([simp, ctl], ignore_index=True, sort=False)
    df = df[["Rate Band", "BUSINESS UNIT GDLS", *[f"CY{y}" for y in YEARS]]]

    df["Effective Date"] = pd.Timestamp(dt.date(YEARS[0], 1, 1))
    df["End Date"]       = pd.Timestamp(dt.date(YEARS[-1],12,31))
    df["Factor"]         = 1.0
    df["Step"]           = 12
    return df

def build_burden_rate_import(other: pd.DataFrame, com: pd.DataFrame) -> pd.DataFrame:
    burdens = pd.concat([strip_unnamed(other), strip_unnamed(com)],
                        ignore_index=True, sort=False)
    burdens = escalate(burdens, burden_escalation)

    cy_cols = [f"CY{y}" for y in YEARS]
    is_com  = burdens["Description"].str.contains("COM", case=False, na=False)
    burdens.loc[~is_com, cy_cols] = burdens.loc[~is_com, cy_cols].round(ROUNDING["BURDEN"])
    burdens.loc[is_com , cy_cols] = burdens.loc[ is_com, cy_cols].round(ROUNDING["COM"])

    burdens = burdens.rename(columns={"Burden Pool": "Rate Band"})
    burdens["Effective Date"] = pd.Timestamp(dt.date(YEARS[0], 1, 1))
    burdens["End Date"]       = pd.Timestamp(dt.date(YEARS[-1],12,31))
    burdens["Factor"]         = 1.0
    burdens["Step"]           = 12
    return burdens

# ── DRIVER ─────────────────────────────────────────────────────────────────────────
def main():
    xl = pd.ExcelFile(RATSUM_PATH)

    simplified   = load_sheet(xl, "SIMPLIFIED RATES NON-PSPL")
    other_rates  = load_sheet(xl, "OTHER RATES")
    com_summary  = load_sheet(xl, "COM SUMMARY")

    rate_band_df   = build_rate_band_import(simplified, other_rates)
    burden_rate_df = build_burden_rate_import(other_rates, com_summary)

    rb_path = OUTPUT_DIR / "RateBandImport.xlsx"
    br_path = OUTPUT_DIR / "BurdenRateImport.xlsx"

    rate_band_df.to_excel(rb_path,   sheet_name="Rate Band Import",   index=False, engine="xlsxwriter")
    burden_rate_df.to_excel(br_path, sheet_name="Burden Rate Import", index=False, engine="xlsxwriter")

    compare_to_previous(rate_band_df,   rb_path, "Rate Band")
    compare_to_previous(burden_rate_df, br_path, "Rate Band")

    logging.info("✅ Pipeline complete. Files saved to %s", OUTPUT_DIR.resolve())

if __name__ == "__main__":
    main()