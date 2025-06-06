# ----------------------------------------------------------------------------------
#  Bullet‑proof ProPricer pipeline – single cell
# ----------------------------------------------------------------------------------
import pandas as pd, numpy as np, re, logging
from pathlib import Path
from datetime import date

# ╔═══════════════════════╗
# ║        CONFIG         ║
# ╚═══════════════════════╝
RATSUM_PATH = Path("data/RATSUM.xlsx")           # Shelly’s workbook
OUTPUT_DIR  = Path("output")                     # where finished files go
YEARS       = range(2022, 2027)                  # CY columns to keep
LAB_ESC     = {2024: 1.02,  2025: 1.025, 2026: 1.03}   # labor escalation factors
BURD_ESC    = {2024: 1.015, 2025: 1.02,  2026: 1.02}   # burden escalation factors
OUTPUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s • %(levelname)s • %(message)s")

def load_sheet(xl: pd.ExcelFile, name: str, skiprows: int = 5) -> pd.DataFrame:
    logging.info(f"Loading sheet: {name}")
    return xl.parse(name, skiprows=skiprows)

def strip_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if "unnamed" not in str(c).lower()]
    out  = df.loc[:, keep].copy()
    out.columns = out.columns.map(str)
    return out

def normalise_business_unit(df: pd.DataFrame,
                             final_name: str = "BUSINESS UNIT GDLS") -> pd.DataFrame:
    pat = re.compile(r"business\s*unit", re.I)
    for col in df.columns:
        if pat.search(col):
            return df.rename(columns={col: final_name})
    return df.rename(columns={df.columns[0]: final_name})

def add_rate_band(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ►► KEY SAFEGUARD ◄◄  – force string dtype before using .str
    df["BUSINESS UNIT GDLS"] = df["BUSINESS UNIT GDLS"].astype(str)
    df["Rate Band"] = df["BUSINESS UNIT GDLS"].str.extract(r"^(\w{2})", expand=False)
    return df

def apply_escalation(df: pd.DataFrame, factors: dict) -> pd.DataFrame:
    df = df.copy()
    for yr, factor in factors.items():
        col = f"CY{yr}"
        if col in df.columns:
            df[col] = df[col] * factor
    return df

def round_years(df: pd.DataFrame, precision: int,
                special: dict | None = None) -> pd.DataFrame:
    years = [c for c in df.columns if c.startswith("CY")]
    if special:
        mask = df.eval(special["mask"])
        df.loc[~mask, years] = df.loc[~mask, years].round(precision)
        df.loc[ mask, years] = df.loc[ mask, years].round(special["precision"])
    else:
        df[years] = df[years].round(precision)
    return df

def compare_sets(new_df: pd.DataFrame, old_path: Path, key: str):
    if not old_path.exists():
        return
    old = pd.read_excel(old_path)[key].astype(str)
    new = new_df[key].astype(str)
    added, removed = sorted(set(new) - set(old)), sorted(set(old) - set(new))
    logging.info("∆ %s added | %s removed", len(added), len(removed))
    if added:   logging.info("   added:   %s", added)
    if removed: logging.info("   removed: %s", removed)

# ════════════════════════════════════════════════
# Builders
# ════════════════════════════════════════════════
def build_rate_band(simplified: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    simp = strip_unnamed(simplified)
    simp = normalise_business_unit(simp)
    simp = add_rate_band(simp)
    simp = apply_escalation(simp, LAB_ESC)
    simp = round_years(simp, 3)

    ctl = strip_unnamed(other)
    ctl = ctl[ctl.iloc[:, 0].astype(str).str.contains("ALLOWABLE", case=False, na=False)]
    ctl = ctl.rename(columns={ctl.columns[0]: "Rate Band"})

    df  = pd.concat([simp, ctl], ignore_index=True, sort=False)
    df  = df[["Rate Band", "BUSINESS UNIT GDLS", *(f"CY{y}" for y in YEARS)]]

    df["Effective Date"] = date(YEARS[0], 1, 1)
    df["End Date"]       = date(YEARS[-1], 12, 31)
    df["Factor"]         = 1.0
    df["Step"]           = 12
    return df

def build_burden_rate(other: pd.DataFrame, com: pd.DataFrame) -> pd.DataFrame:
    bur = pd.concat([strip_unnamed(other), strip_unnamed(com)],
                    ignore_index=True, sort=False)
    bur = bur.rename(columns=lambda c: "Rate Band" if "Burden Pool" in c else c)
    bur = apply_escalation(bur, BURD_ESC)
    bur = round_years(
        bur, 5,
        special=dict(mask="Description.str.contains('COM', na=False)", precision=6)
    )
    bur["Effective Date"] = date(YEARS[0], 1, 1)
    bur["End Date"]       = date(YEARS[-1], 12, 31)
    bur["Factor"]         = 1.0
    bur["Step"]           = 12
    return bur

# ════════════════════════════════════════════════
# Driver
# ════════════════════════════════════════════════
def main() -> None:
    xl = pd.ExcelFile(RATSUM_PATH)

    simplified  = load_sheet(xl, "SIMPLIFIED RATES NON-PSPL")
    other_rates = load_sheet(xl, "OTHER RATES")
    com_summary = load_sheet(xl, "COM SUMMARY")

    rate_band_df   = build_rate_band(simplified, other_rates)
    burden_rate_df = build_burden_rate(other_rates, com_summary)

    rb_file = OUTPUT_DIR / "RateBandImport.xlsx"
    br_file = OUTPUT_DIR / "BurdenRateImport.xlsx"

    rate_band_df.to_excel(rb_file,   sheet_name="Rate Band Import",   index=False, engine="xlsxwriter")
    burden_rate_df.to_excel(br_file, sheet_name="Burden Rate Import", index=False, engine="xlsxwriter")

    compare_sets(rate_band_df,   rb_file, "Rate Band")
    compare_sets(burden_rate_df, br_file, "Rate Band")

    logging.info("✅ Pipeline complete – outputs saved in %s", OUTPUT_DIR.resolve())

# ════════════════════════════════════════════════
# Execute
# ════════════════════════════════════════════════
if __name__ == "__main__":
    main()