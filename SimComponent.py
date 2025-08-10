# rate_files_automation.py
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

# ========= CONFIG (edit as needed) =========
RATSUM_PATH = Path("data/RATSUM.xlsx")
RATE_BAND_IMPORT_PATH = Path("data/RateBandImport.xlsx")
BURDEN_RATE_IMPORT_PATH = Path("data/BurdenRateImport.xlsx")
OUT_DIR = Path("output")

SHEET_SIMPLIFIED = "SIMPLIFIED RATES NON-PSPL"
SHEET_OTHER      = "OTHER RATES"
SHEET_OVERHEAD   = "OVERHEAD RATES"
SHEET_COM        = "COM SUMMARY"

SKIP_SIMPLIFIED  = 5
SKIP_OTHER       = 5
SKIP_OVERHEAD    = 5
SKIP_COM         = 5

VT_ABRAMS_BAND   = "VT ABRAMS"

LABOR_ROUND = 3
BURDEN_ROUND = 5
COM_ROUND = 6

# If your labor band codes are the first two characters (e.g., "04 …")
PAT_BAND_CODE = re.compile(r"^\s*([0-9A-Z]{2})\b")
# ===========================================


# ---------- helpers ----------
def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).strip().lower()

def find_year_cols(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        s = str(c).strip()
        if re.fullmatch(r"CY?\s*20\d{2}", s) or re.fullmatch(r"20\d{2}", s):
            cols.append(c)
    return cols

def normalize_cy(col_name: str) -> str:
    m = re.search(r"(20\d{2})", str(col_name))
    return f"CY{m.group(1)}" if m else str(col_name)


# ---------- readers from RATSUM ----------
def read_simplified(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SHEET_SIMPLIFIED, skiprows=SKIP_SIMPLIFIED)
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    year_cols = find_year_cols(df)
    label_col = [c for c in df.columns if c not in year_cols][0]
    df = df.rename(columns={label_col: "Label"})
    df["RateBand"] = df["Label"].astype(str).str.extract(PAT_BAND_CODE)[0].fillna(df["Label"].astype(str))
    # normalize year col names to CY####
    ren = {c: normalize_cy(c) for c in year_cols}
    df = df.rename(columns=ren)
    return df[["RateBand", "Label"] + list(ren.values())]


def read_act_from_other_rates(path: Path) -> pd.Series:
    df = pd.read_excel(path, sheet_name=SHEET_OTHER, skiprows=SKIP_OTHER)
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)

    # Build a descriptor by joining any leading text/Unnamed columns
    text_like = [c for c in df.columns if str(c).startswith("Unnamed")]
    if text_like:
        df["Desc"] = df[text_like].astype(str).apply(
            lambda r: " ".join(x.strip() for x in r if x and x != "nan"), axis=1
        )
    else:
        df["Desc"] = df.iloc[:, 0].astype(str)

    # Flexible match: ALLOWABLE + (CONTROL|CONTL) + TEST (+ optional RATE)
    def is_act(txt: str) -> bool:
        t = re.sub(r"\s+", " ", str(txt or "")).upper()
        return ("ALLOWABLE" in t) and (("CONTROL" in t) or ("CONTL" in t)) and ("TEST" in t)
    row = df.loc[df["Desc"].map(is_act)]
    if row.empty:
        # Helpful debug — see what we parsed if still not found
        raise ValueError("Could not find Allowable Control Test row in OTHER RATES.")

    year_cols = find_year_cols(df)
    if not year_cols:
        raise ValueError("OTHER RATES: No CY year columns found.")

    # normalize headers to CY####
    ren = {c: normalize_cy(c) for c in year_cols}
    df = df.rename(columns=ren)
    s = row.iloc[0][list(ren.values())].copy()
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).round(BURDEN_ROUND)
    s.name = "ACT"
    return s


def read_overhead_rates(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SHEET_OVERHEAD, skiprows=SKIP_OVERHEAD)
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)

    # try common Year/Date labels
    year_col = next((c for c in df.columns if str(c).strip().lower() in {"date", "year"}), None)
    if not year_col:
        raise ValueError("OVERHEAD RATES: could not find 'Date/Year' column")

    # choose a pool/description column
    pool_col = next((c for c in df.columns if str(c).strip().lower() in {"burden pool","pool","segment","description","descr"}), None)
    if pool_col is None:
        pool_col = df.columns[0]

    meta = {pool_col, year_col}
    burden_cols = [c for c in df.columns if c not in meta]

    out = df[[pool_col, year_col] + burden_cols].copy()
    out.rename(columns={pool_col:"Pool", year_col:"Year"}, inplace=True)
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    for c in burden_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].round(COM_ROUND if "COM" in str(c).upper() else BURDEN_ROUND)
    return out


def read_com_summary(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SHEET_COM, skiprows=SKIP_COM)
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    year_cols = find_year_cols(df)
    ren = {c: normalize_cy(c) for c in year_cols}
    df = df.rename(columns=ren)

    # Build pool label from Unnamed cols if present
    text_like = [c for c in df.columns if str(c).startswith("Unnamed")]
    if text_like:
        df["Pool"] = df[text_like].astype(str).apply(
            lambda r: " ".join(x.strip() for x in r if x and x != "nan"), axis=1
        )
    else:
        df["Pool"] = df.iloc[:, 0].astype(str)

    long = df.melt(id_vars=["Pool"], value_vars=list(ren.values()), var_name="CY", value_name="COM")
    long["Year"] = long["CY"].astype(str).str.extract(r"(20\d{2})").astype(int)
    long["COM"] = pd.to_numeric(long["COM"], errors="coerce").round(COM_ROUND)
    return long[["Pool","Year","COM"]]


# ---------- writers ----------
def update_rate_band_import(simplified: pd.DataFrame, act: pd.Series, rbi_path: Path, out_path: Path):
    rbi = pd.read_excel(rbi_path)

    rb_col   = next((c for c in rbi.columns if norm(c) in {"rateband","rate_band"}), None)
    base_col = next((c for c in rbi.columns if norm(c).startswith("baserate")), None)
    start_col= next((c for c in rbi.columns if "start" in str(c).lower()), None)
    if not all([rb_col, base_col, start_col]):
        raise ValueError("RateBandImport.xlsx missing one of: Rate Band, Base Rate..., Start Date")

    # Map labor by (RateBandCode, Year)
    year_cols = [c for c in simplified.columns if c not in {"RateBand","Label"}]
    simp_long = simplified.melt(id_vars=["RateBand"], value_vars=year_cols, var_name="CY", value_name="Rate")
    simp_long["Year"] = simp_long["CY"].astype(str).str.extract(r"(20\d{2})").astype(int)
    simp_long["Rate"] = pd.to_numeric(simp_long["Rate"], errors="coerce").round(LABOR_ROUND)
    key_to_rate = {(str(b), int(y)): float(r) for b,y,r in zip(simp_long["RateBand"], simp_long["Year"], simp_long["Rate"])}

    rbi["_Year"] = pd.to_datetime(rbi[start_col], errors="coerce").dt.year
    def set_labor(row):
        code = str(row[rb_col]).strip()[:2]  # adjust if your code length differs
        yr = int(row["_Year"]) if pd.notna(row["_Year"]) else None
        if yr is not None:
            val = key_to_rate.get((code, yr))
            if val is not None:
                row[base_col] = val
        return row
    rbi = rbi.apply(set_labor, axis=1).drop(columns=["_Year"])

    # Ensure ACT year columns exist & assign to VT ABRAMS row
    for cy in act.index:
        if cy not in rbi.columns:
            rbi[cy] = pd.NA
    vt_mask = rbi[rb_col].astype(str).map(norm) == norm(VT_ABRAMS_BAND)
    for cy, val in act.items():
        rbi.loc[vt_mask, cy] = float(val)

    # Round CY#### columns as burden values
    cy_cols = [c for c in rbi.columns if re.fullmatch(r"CY20\d{2}", str(c))]
    rbi[cy_cols] = rbi[cy_cols].apply(pd.to_numeric, errors="coerce").round(BURDEN_ROUND)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        rbi.to_excel(xw, index=False, sheet_name="Rate Band Import")
    print(f"✅ Rate Band Import updated → {out_path.resolve()}")


def update_burden_rate_import(overhead_df: pd.DataFrame, com_df: pd.DataFrame, bri_path: Path, out_path: Path):
    bri = pd.read_excel(bri_path)

    pool_col = next((c for c in bri.columns if norm(c) in {"burdenpool","burden_pool"}), None)
    year_col = next((c for c in bri.columns if norm(c) in {"date","year"}), None)
    if not all([pool_col, year_col]):
        raise ValueError("BurdenRateImport.xlsx missing 'Burden Pool' or 'Date/Year'")

    burden_cols = [c for c in bri.columns if c not in {pool_col, year_col, "Description", "Effective Date"}]

    # Merge overhead + COM (fill any COM* columns with the COM value)
    target = overhead_df.copy()
    if not com_df.empty:
        com_map = com_df.set_index(["Pool","Year"])["COM"].to_dict()
        target["_COM_fill_"] = [com_map.get((p,y)) for p,y in zip(target["Pool"], target["Year"])]

    bri[year_col] = pd.to_numeric(bri[year_col], errors="coerce").astype("Int64")
    t_idx = {(norm(p), int(y)): r for p,y,r in zip(target["Pool"], target["Year"], target.index)}

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

    # Final rounding
    for c in burden_cols:
        bri[c] = pd.to_numeric(bri[c], errors="coerce")
        bri[c] = bri[c].round(COM_ROUND if "COM" in str(c).upper() else BURDEN_ROUND)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        bri.to_excel(xw, index=False, sheet_name="Burden Rate Import")
    print(f"✅ Burden Rate Import updated → {out_path.resolve()}")


# ---------- main ----------
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    simplified = read_simplified(RATSUM_PATH)
    act = read_act_from_other_rates(RATSUM_PATH)
    overhead = read_overhead_rates(RATSUM_PATH)
    com_long = read_com_summary(RATSUM_PATH)

    update_rate_band_import(
        simplified=simplified,
        act=act,
        rbi_path=RATE_BAND_IMPORT_PATH,
        out_path=OUT_DIR / "RateBandImport_updated.xlsx",
    )
    update_burden_rate_import(
        overhead_df=overhead,
        com_df=com_long,
        bri_path=BURDEN_RATE_IMPORT_PATH,
        out_path=OUT_DIR / "BurdenRateImport_updated.xlsx",
    )
    print("🎉 Process complete.")