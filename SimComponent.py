# rate_files_automation.py
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List
import pandas as pd

# ============== CONFIG ==============
RATSUM_PATH = Path("data/RATSUM.xlsx")
RATE_BAND_IMPORT_PATH = Path("data/RateBandImport.xlsx")
BURDEN_RATE_IMPORT_PATH = Path("data/BurdenRateImport.xlsx")
OUT_DIR = Path("output")

SHEET_SIMPLIFIED = "SIMPLIFIED RATES NON-PSPL"
SHEET_OTHER = "OTHER RATES"
SHEET_OVERHEAD = "OVERHEAD RATES"
SHEET_COM = "COM SUMMARY"

SKIP_SIMPLIFIED = 5        # header rows before grid (your screenshot shows ~5)
SKIP_OTHER = 5
SKIP_OVERHEAD = 5
SKIP_COM = 5

# “VT ABRAMS” target in RateBandImport
VT_ABRAMS_BAND = "VT ABRAMS"

# Rounding rules
LABOR_ROUND = 3
BURDEN_ROUND = 5
COM_ROUND = 6

# If Simplified labels begin with a two-digit band code like "04 SCR PROD/…"
PAT_BAND_CODE = re.compile(r"^\s*([0-9A-Z]{2})\b")
# ====================================


def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).strip().lower()


def find_year_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        s = str(c).strip()
        if re.fullmatch(r"CY?\s*20\d{2}", s) or re.fullmatch(r"20\d{2}", s):
            cols.append(c)
    return cols


# ---------- RATSUM READERS ----------
def read_simplified(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SHEET_SIMPLIFIED, skiprows=SKIP_SIMPLIFIED)
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    # first text col is label; years are wide columns
    year_cols = find_year_cols(df)
    label_col = [c for c in df.columns if c not in year_cols][0]
    df = df.rename(columns={label_col: "Label"})
    # extract 2-char RateBand code
    df["RateBand"] = df["Label"].astype(str).str.extract(PAT_BAND_CODE)[0].fillna(df["Label"].astype(str))
    return df[["RateBand", "Label"] + year_cols]


def read_act_from_other_rates(path: Path) -> pd.Series:
    df = pd.read_excel(path, sheet_name=SHEET_OTHER, skiprows=SKIP_OTHER)
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)

    # Build a descriptor column by joining any leading Unnamed columns
    text_cols = [c for c in df.columns if str(c).startswith("Unnamed") or isinstance(df[c].iloc[0], str)]
    if text_cols:
        desc = df[text_cols].astype(str).apply(lambda r: " ".join(x.strip() for x in r if x and x != "nan"), axis=1)
        df.insert(0, "Desc", desc)
    else:
        df.insert(0, "Desc", df.iloc[:, 0].astype(str))

    mask = df["Desc"].str.upper().str.contains("ALLOWABLE") & df["Desc"].str.upper().str.contains("CONTROL") & df["Desc"].str.upper().str.contains("TEST")
    row = df.loc[mask]
    if row.empty:
        raise ValueError("Could not find 'Allowable Control Test' row in OTHER RATES")

    year_cols = find_year_cols(df)
    s = row.iloc[0][year_cols].copy()
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).round(BURDEN_ROUND)
    s.index = [f"CY{re.search(r'(20\d{2})', str(c)).group(1)}" for c in s.index]  # normalize keys
    s.name = "ACT"
    return s


def read_overhead_rates(path: Path) -> pd.DataFrame:
    """Return long table: Pool, Year, <burden columns> from OVERHEAD RATES."""
    df = pd.read_excel(path, sheet_name=SHEET_OVERHEAD, skiprows=SKIP_OVERHEAD)
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)

    # detect year column (it’s a single 'Date' or 'Year' col in your screenshot)
    year_col = None
    for c in df.columns:
        if str(c).strip().lower() in {"date", "year", "cy"}:
            year_col = c; break
    if not year_col:
        raise ValueError("OVERHEAD RATES: could not find 'Date/Year' column")

    # pool/description columns
    pool_col = [c for c in df.columns if str(c).strip().lower() in {"burden pool", "pool", "segment", "descr", "description"}][0]

    # burden columns are everything numeric after the metadata
    meta = {pool_col, year_col}
    burden_cols = [c for c in df.columns if c not in meta]

    # tidy
    keep = [pool_col, year_col] + burden_cols
    df = df[keep].copy()
    df.rename(columns={pool_col: "Pool", year_col: "Year"}, inplace=True)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    # numeric + rounding
    for c in burden_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if "COM" in str(c).upper():
            df[c] = df[c].round(COM_ROUND)
        else:
            df[c] = df[c].round(BURDEN_ROUND)
    return df


def read_com_summary(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SHEET_COM, skiprows=SKIP_COM)
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    year_cols = find_year_cols(df)
    # find the row for the company (your screenshot shows two rows; keep all non-empty)
    text_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    df["Pool"] = df[text_cols].astype(str).apply(lambda r: " ".join(x.strip() for x in r if x and x != "nan"), axis=1)
    long = df.melt(id_vars=["Pool"], value_vars=year_cols, var_name="CY", value_name="COM")
    long["Year"] = long["CY"].astype(str).str.extract(r"(20\d{2})").astype(int)
    long.drop(columns=["CY"], inplace=True)
    long["COM"] = pd.to_numeric(long["COM"], errors="coerce").round(COM_ROUND)
    return long[["Pool", "Year", "COM"]]


# ---------- WRITERS ----------
def update_rate_band_import(simplified: pd.DataFrame, act: pd.Series, rbi_path: Path, out_path: Path):
    rbi = pd.read_excel(rbi_path)

    # Identify "Rate Band" & "Base Rate/Base Rate Band" columns
    rb_col = next((c for c in rbi.columns if norm(c) in {"rateband", "rate_band"}), None)
    base_col = next((c for c in rbi.columns if norm(c).startswith("baserate")), None)
    start_col = next((c for c in rbi.columns if "start" in str(c).lower()), None)  # 01/2023
    if rb_col is None or base_col is None or start_col is None:
        raise ValueError("RateBandImport.xlsx is missing expected columns (Rate Band, Base Rate..., Start Date).")

    # map: (band_code, year) -> value
    year_cols = [c for c in simplified.columns if c not in {"RateBand", "Label"}]
    simp_long = simplified.melt(id_vars=["RateBand"], value_vars=year_cols, var_name="CY", value_name="Rate")
    simp_long["Year"] = simp_long["CY"].astype(str).str.extract(r"(20\d{2})").astype(int)
    simp_long["Rate"] = pd.to_numeric(simp_long["Rate"], errors="coerce").round(LABOR_ROUND)

    # update Base Rate based on Start Date year
    rbi["_year"] = pd.to_datetime(rbi[start_col], errors="coerce").dt.year
    key_to_rate = {(str(k), int(y)): v for k, y, v in zip(simp_long["RateBand"], simp_long["Year"], simp_long["Rate"])}

    def set_labor(row):
        k = (str(row[rb_col]).strip()[:2], int(row["_year"]))  # leftmost 2 chars are the code
        val = key_to_rate.get(k)
        if val is not None:
            row[base_col] = val
        return row

    rbi = rbi.apply(set_labor, axis=1)
    rbi.drop(columns=["_year"], inplace=True)

    # Ensure ACT year columns exist, then assign to VT ABRAMS row
    for cy in act.index:
        if cy not in rbi.columns:
            rbi[cy] = pd.NA
    vt_mask = rbi[rb_col].astype(str).map(norm) == norm(VT_ABRAMS_BAND)
    for cy, val in act.items():
        rbi.loc[vt_mask, cy] = float(val)

    # Round any CY#### numeric columns we touched (other-rate rounding)
    cy_cols = [c for c in rbi.columns if re.fullmatch(r"CY20\d{2}", str(c))]
    rbi[cy_cols] = rbi[cy_cols].apply(pd.to_numeric, errors="coerce").round(BURDEN_ROUND)

    out_path.parent.mkdir(exist_ok=True, parents=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        rbi.to_excel(xw, index=False, sheet_name="Rate Band Import")
    print(f"✅ Rate Band Import updated → {out_path.resolve()}")


def update_burden_rate_import(overhead_df: pd.DataFrame, com_df: pd.DataFrame, bri_path: Path, out_path: Path):
    bri = pd.read_excel(bri_path)

    # Required columns
    pool_col = next((c for c in bri.columns if norm(c) in {"burdenpool", "burden_pool"}), None)
    year_col = next((c for c in bri.columns if norm(c) in {"date", "year"}), None)
    if pool_col is None or year_col is None:
        raise ValueError("BurdenRateImport.xlsx is missing 'Burden Pool' or 'Date/Year' column.")

    # Numeric burden columns present in BRI
    burden_cols = [c for c in bri.columns if c not in {pool_col, year_col, "Description", "Effective Date", "GFM Flag", "IST Flag", "Misc Matl", "ROP", "Proc O/H G", "Proc O/H C"}]

    # Create a single table of target rates by (Pool, Year)
    target = overhead_df.copy()
    if not com_df.empty:
        # Join COM if there is a COM column in BRI
        potential_com_cols = [c for c in burden_cols if "COM" in str(c).upper()]
        if potential_com_cols:
            # build a (Pool, Year)->COM mapping and then fill intersecting columns
            com_map = com_df.set_index(["Pool", "Year"])["COM"].to_dict()
            # If multiple COM* columns exist, fill each with same COM value (you can specialize if needed)
            target["_COM_fill_"] = [com_map.get((p, y)) for p, y in zip(target["Pool"], target["Year"])]

    # Align dtypes
    bri[year_col] = pd.to_numeric(bri[year_col], errors="coerce").astype("Int64")

    # Index for quick assignment
    t_idx = {(norm(p), int(y)): r for p, y, r in zip(target["Pool"], target["Year"], target.index)}

    def set_burdens(row):
        key = (norm(row[pool_col]), int(row[year_col])) if pd.notna(row[year_col]) else None
        if key and key in t_idx:
            src = target.loc[t_idx[key]]
            for c in burden_cols:
                if c in src:
                    row[c] = src[c]
                elif c.upper().startswith("COM") and "_COM_fill_" in src:
                    row[c] = src["_COM_fill_"]
        return row

    bri = bri.apply(set_burdens, axis=1)

    # Final rounding: COM 6, others 5
    for c in burden_cols:
        bri[c] = pd.to_numeric(bri[c], errors="coerce")
        if "COM" in str(c).upper():
            bri[c] = bri[c].round(COM_ROUND)
        else:
            bri[c] = bri[c].round(BURDEN_ROUND)

    out_path.parent.mkdir(exist_ok=True, parents=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        bri.to_excel(xw, index=False, sheet_name="Burden Rate Import")
    print(f"✅ Burden Rate Import updated → {out_path.resolve()}")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    simplified = read_simplified(RATSUM_PATH)
    act = read_act_from_other_rates(RATSUM_PATH)
    overhead = read_overhead_rates(RATSUM_PATH)
    com_long = read_com_summary(RATSUM_PATH)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    update_rate_band_import(
        simplified=simplified,
        act=act,
        rbi_path=RATE_BAND_IMPORT_PATH,
        out_path=OUT_DIR / "RateBandImport_updated.xlsx"
    )
    update_burden_rate_import(
        overhead_df=overhead,
        com_df=com_long,
        bri_path=BURDEN_RATE_IMPORT_PATH,
        out_path=OUT_DIR / "BurdenRateImport_updated.xlsx"
    )
    print("🎉 Process complete.")