# -------------------------------------------------------------------------------
#  Auto‑detecting, one‑cell ProPricer pipeline
# -------------------------------------------------------------------------------
import pandas as pd, re, logging
from pathlib import Path
from datetime import date

# ╔══════════════════════════════════╗
# ║ USER SETTINGS (tweak as needed)  ║
# ╚══════════════════════════════════╝
RATSUM_PATH = Path("data/RATSUM.xlsx")       # Shelly’s workbook
OUTPUT_DIR  = Path("output")                 # where finished files go
LAB_ESC     = {2024: 1.02,  2025: 1.025}     # labor escalation %
BURD_ESC    = {2024: 1.015, 2025: 1.02}      # burden escalation %
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ════════════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s • %(levelname)s • %(message)s")

CY_RE = re.compile(r"^CY(\d{4})")

# ── utility helpers ─────────────────────────────────────────────────────
def load_sheet(xl: pd.ExcelFile, name: str, skiprows: int = 5):
    logging.info(f"Loading sheet: {name}")
    return xl.parse(name, skiprows=skiprows)

def strip_unnamed(df):
    keep = [c for c in df.columns if "unnamed" not in str(c).lower()]
    out  = df.loc[:, keep].copy()
    out.columns = out.columns.map(str)
    return out

def normalise_bu(df, final="BUSINESS UNIT GDLS"):
    pat = re.compile(r"business\s*unit", re.I)
    for col in df.columns:
        if pat.search(col):
            return df.rename(columns={col: final})
    return df.rename(columns={df.columns[0]: final})

def add_rate_band(df):
    df = df.copy()
    df["BUSINESS UNIT GDLS"] = df["BUSINESS UNIT GDLS"].astype(str)
    df["Rate Band"] = df["BUSINESS UNIT GDLS"].str.extract(r"^(\w{2})")
    return df

def apply_escalation(df, factors):
    df = df.copy()
    for yr, f in factors.items():
        # Match columns that start with CYyyyy (e.g., CY2024 or CY2024.1)
        cols = [c for c in df.columns if re.match(fr"^CY{yr}\b", c)]
        df[cols] = df[cols] * f
    return df

def pick_year_cols(df):
    """Return ordered list of CY‑columns actually present in the frame."""
    years = sorted({int(m.group(1)) for c in df.columns if (m := CY_RE.match(str(c)))})
    cols  = [c for y in years for c in df.columns if re.match(fr"^CY{y}\b", c)]
    return years, cols

# ── builders ────────────────────────────────────────────────────────────
def build_rate_band(simplified, other):
    simp = normalise_bu(strip_unnamed(simplified))
    simp = add_rate_band(simp)
    simp = apply_escalation(simp, LAB_ESC)

    ctl = strip_unnamed(other)
    ctl = ctl[ctl.iloc[:, 0].astype(str).str.contains("ALLOWABLE", case=False, na=False)]
    ctl = ctl.rename(columns={ctl.columns[0]: "Rate Band"})

    df  = pd.concat([simp, ctl], ignore_index=True, sort=False)

    years, cy_cols = pick_year_cols(df)
    df = df[["Rate Band", "BUSINESS UNIT GDLS", *cy_cols]]

    df["Effective Date"] = date(min(years), 1, 1)
    df["End Date"]       = date(max(years), 12, 31)
    df["Factor"]         = 1.0
    df["Step"]           = 12
    return df, years

def build_burden_rate(other, com, all_years):
    bur = pd.concat([strip_unnamed(other), strip_unnamed(com)],
                    ignore_index=True, sort=False)
    bur = bur.rename(columns=lambda c: "Rate Band" if "Burden Pool" in c else c)
    bur = apply_escalation(bur, BURD_ESC)

    # Round: COM rows 6dp, others 5dp
    com_mask = bur["Description"].str.contains("COM", case=False, na=False)
    for y in all_years:
        cols = [c for c in bur.columns if re.match(fr"^CY{y}\b", c)]
        bur.loc[~com_mask, cols] = bur.loc[~com_mask, cols].round(5)
        bur.loc[ com_mask, cols] = bur.loc[ com_mask, cols].round(6)

    bur["Effective Date"] = date(min(all_years), 1, 1)
    bur["End Date"]       = date(max(all_years), 12, 31)
    bur["Factor"]         = 1.0
    bur["Step"]           = 12
    return bur

# ── main driver ─────────────────────────────────────────────────────────
def main():
    xl = pd.ExcelFile(RATSUM_PATH)

    simplified   = load_sheet(xl, "SIMPLIFIED RATES NON-PSPL")
    other_rates  = load_sheet(xl, "OTHER RATES")
    com_summary  = load_sheet(xl, "COM SUMMARY")

    rb_df, yrs   = build_rate_band(simplified, other_rates)
    br_df        = build_burden_rate(other_rates, com_summary, yrs)

    missing_in_rb = [y for y in LAB_ESC if y not in yrs]
    if missing_in_rb:
        logging.warning("Workbook missing CY columns for years: %s", missing_in_rb)

    rb_file = OUTPUT_DIR / "RateBandImport.xlsx"
    br_file = OUTPUT_DIR / "BurdenRateImport.xlsx"
    rb_df.to_excel(rb_file, sheet_name="Rate Band Import",   index=False, engine="xlsxwriter")
    br_df.to_excel(br_file, sheet_name="Burden Rate Import", index=False, engine="xlsxwriter")

    logging.info("✅ Done. Files saved → %s", OUTPUT_DIR.resolve())

if __name__ == "__main__":
    main()