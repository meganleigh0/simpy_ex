# rate_files_automation.py
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

# ========== PATHS (edit if needed) ==========
RATSUM_PATH = Path("data/RATSUM.xlsx")
RATE_BAND_IMPORT_PATH = Path("data/RateBandImport.xlsx")
BURDEN_RATE_IMPORT_PATH = Path("data/BurdenRateImport.xlsx")
OUT_DIR = Path("output")
# Target rate-band row to receive the Allowable Control Test values
VT_ABRAMS_BAND = "VT ABRAMS"
# ============================================

# Rounding rules
LABOR_ROUND  = 3
BURDEN_ROUND = 5
COM_ROUND    = 6

# Labor band code: if your band code is first 2 chars (e.g., "04 ..."), keep this
PAT_BAND_CODE = re.compile(r"^\s*([0-9A-Z]{2})\b")

def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).strip().lower()

def normalize_cy(col_name: str) -> str:
    m = re.search(r"(20\d{2})", str(col_name))
    return f"CY{m.group(1)}" if m else str(col_name)

def find_year_cols(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        s = str(c).strip()
        if re.fullmatch(r"CY?\s*20\d{2}", s) or re.fullmatch(r"20\d{2}", s):
            cols.append(c)
    return cols

def pick_sheet(xls: pd.ExcelFile, *must_include: str) -> str:
    """Pick the first sheet whose name contains all tokens (case-insensitive)."""
    tokens = [t.upper() for t in must_include if t]
    for name in xls.sheet_names:
        u = name.upper()
        if all(t in u for t in tokens):
            return name
    raise ValueError(
        f"Could not find a sheet containing tokens {tokens}. "
        f"Available: {xls.sheet_names}"
    )

def read_best_skiprows(xls: pd.ExcelFile, sheet: str, candidates=range(0, 12)) -> pd.DataFrame:
    """Try multiple skiprows; pick the one yielding the most CY columns."""
    best = None
    best_score = -1
    for k in candidates:
        df = pd.read_excel(xls, sheet_name=sheet, skiprows=k)
        df = df.dropna(how="all").dropna(axis=1, how="all")
        yc = find_year_cols(df)
        score = len(yc)
        if score > best_score:
            best, best_score = df.reset_index(drop=True), score
    if best is None:
        raise ValueError(f"Failed to read a usable table from sheet '{sheet}'")
    return best

# ---------- RATSUM readers (AUTO) ----------
def read_simplified(xls: pd.ExcelFile) -> pd.DataFrame:
    sheet = pick_sheet(xls, "SIMPLIFIED", "RATE")  # handles "SIMPLIFIED RATES NON-PSPL"
    df = read_best_skiprows(xls, sheet)
    year_cols = find_year_cols(df)
    if not year_cols:
        raise ValueError("SIMPLIFIED: did not detect CY year columns")
    label_col = [c for c in df.columns if c not in year_cols][0]
    df = df.rename(columns={label_col: "Label"})
    df["RateBand"] = df["Label"].astype(str).str.extract(PAT_BAND_CODE)[0].fillna(df["Label"].astype(str))
    # normalize year headers
    df = df.rename(columns={c: normalize_cy(c) for c in year_cols})
    cols = ["RateBand", "Label"] + [normalize_cy(c) for c in year_cols]
    return df[cols]

def read_other_rates_act(xls: pd.ExcelFile) -> pd.Series:
    sheet = pick_sheet(xls, "OTHER", "RATE")      # "OTHER RATES"
    df = read_best_skiprows(xls, sheet)
    # Build a descriptor by concatenating any leading text columns up to the first CY column
    year_cols = find_year_cols(df)
    if not year_cols:
        raise ValueError("OTHER RATES: did not detect CY year columns")
    first_year_idx = df.columns.get_loc(year_cols[0])
    text_block = df.columns[:first_year_idx] if first_year_idx > 0 else [df.columns[0]]
    desc = df[text_block].astype(str).apply(lambda r: " ".join(x for x in r if x and x != "nan").strip(), axis=1)
    df.insert(0, "Desc", desc)

    # Flexible match: ALLOWABLE + (CONTROL|CONTL) + TEST (+ optional RATE)
    def is_act(txt: str) -> bool:
        t = re.sub(r"\s+", " ", str(txt or "")).upper()
        return ("ALLOWABLE" in t) and (("CONTROL" in t) or ("CONTL" in t)) and ("TEST" in t)
    row = df.loc[df["Desc"].map(is_act)]
    if row.empty:
        raise ValueError("OTHER RATES: Could not find the Allowable Control/Contl Test row.")

    # Normalize year headers to CY####
    df = df.rename(columns={c: normalize_cy(c) for c in year_cols})
    s = row.iloc[0][[normalize_cy(c) for c in year_cols]].copy()
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).round(BURDEN_ROUND)
    s.name = "ACT"
    return s

def read_overhead(xls: pd.ExcelFile) -> pd.DataFrame:
    sheet = pick_sheet(xls, "OVERHEAD")
    df = read_best_skiprows(xls, sheet)
    # choose Year column
    year_col = next((c for c in df.columns if str(c).strip().lower() in {"date","year"}), None)
    if not year_col:
        # sometimes the year is already a dedicated column named CY#### — pivot wide to long then back if needed
        yc = find_year_cols(df)
        if yc:
            # fabricate a tidy frame as Pool x Year columns
            pool_col = df.columns[0]
            long = df.melt(id_vars=[pool_col], value_vars=yc, var_name="CY", value_name="value")
            long["Year"] = long["CY"].astype(str).str.extract(r"(20\d{2})").astype(int)
            # everything else are burden columns; if only 'value' exists, assume it is a single pool rate col
            tidy = long.pivot_table(index=[pool_col, "Year"], values="value", aggfunc="first").reset_index()
            tidy = tidy.rename(columns={pool_col: "Pool"})
            tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce").round(BURDEN_ROUND)
            return tidy.rename(columns={"value": "Rate"})
        raise ValueError("OVERHEAD: No 'Date/Year' column and no CY#### columns found.")

    pool_col = next((c for c in df.columns if str(c).strip().lower() in {"burden pool","pool","segment","description","descr"}), df.columns[0])
    meta = {pool_col, year_col}
    burden_cols = [c for c in df.columns if c not in meta]

    out = df[[pool_col, year_col] + burden_cols].copy()
    out.rename(columns={pool_col:"Pool", year_col:"Year"}, inplace=True)
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    for c in burden_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].round(COM_ROUND if "COM" in str(c).upper() else BURDEN_ROUND)
    return out

def read_com_summary(xls: pd.ExcelFile) -> pd.DataFrame:
    sheet = pick_sheet(xls, "COM", "SUMMARY")
    df = read_best_skiprows(xls, sheet)
    year_cols = find_year_cols(df)
    if not year_cols:
        return pd.DataFrame(columns=["Pool","Year","COM"])
    df = df.rename(columns={c: normalize_cy(c) for c in year_cols})

    # Build a Pool label from columns before first CY
    first_year_idx = df.columns.get_loc([c for c in df.columns if c in [normalize_cy(y) for y in year_cols]][0])
    text_block = df.columns[:first_year_idx] if first_year_idx > 0 else [df.columns[0]]
    df["Pool"] = df[text_block].astype(str).apply(lambda r: " ".join(x for x in r if x and x != "nan").strip(), axis=1)

    long = df.melt(id_vars=["Pool"], value_vars=[normalize_cy(c) for c in year_cols], var_name="CY", value_name="COM")
    long["Year"] = long["CY"].astype(str).str.extract(r"(20\d{2})").astype(int)
    long["COM"] = pd.to_numeric(long["COM"], errors="coerce").round(COM_ROUND)
    return long[["Pool","Year","COM"]]

# ---------- Writers ----------
def update_rate_band_import(simplified: pd.DataFrame, act: pd.Series, rbi_path: Path, out_path: Path):
    rbi = pd.read_excel(rbi_path)
    rb_col   = next((c for c in rbi.columns if norm(c) in {"rateband","rate_band"}), None)
    base_col = next((c for c in rbi.columns if norm(c).startswith("baserate")), None)
    start_col= next((c for c in rbi.columns if "start" in str(c).lower()), None)
    if not all([rb_col, base_col, start_col]):
        raise ValueError("RateBandImport.xlsx missing one of: Rate Band, Base Rate..., Start Date")

    # LONG labor table (RateBand x Year -> Rate)
    year_cols = [c for c in simplified.columns if c not in {"RateBand","Label"}]
    simp_long = simplified.melt(id_vars=["RateBand"], value_vars=year_cols, var_name="CY", value_name="Rate")
    simp_long["Year"] = simp_long["CY"].astype(str).str.extract(r"(20\d{2})").astype(int)
    simp_long["Rate"] = pd.to_numeric(simp_long["Rate"], errors="coerce").round(LABOR_ROUND)
    key_to_rate = {(str(b), int(y)): float(r) for b,y,r in zip(simp_long["RateBand"], simp_long["Year"], simp_long["Rate"])}

    # Use Start Date year to choose which base rate to write
    rbi["_Year"] = pd.to_datetime(rbi[start_col], errors="coerce").dt.year
    def set_labor(row):
        code = str(row[rb_col]).strip()[:2]  # adjust if your band code length differs
        yr = int(row["_Year"]) if pd.notna(row["_Year"]) else None
        if yr is not None:
            val = key_to_rate.get((code, yr))
            if val is not None:
                row[base_col] = val
        return row
    rbi = rbi.apply(set_labor, axis=1).drop(columns=["_Year"])

    # Ensure ACT columns exist and write values to VT ABRAMS
    for cy in act.index:
        if cy not in rbi.columns:
            rbi[cy] = pd.NA
    vt_mask = rbi[rb_col].astype(str).map(norm) == norm(VT_ABRAMS_BAND)
    for cy, val in act.items():
        rbi.loc[vt_mask, cy] = float(val)

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

    # Merge COM values (if present) to fill any COM* columns
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

# ---------------- Main ----------------
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xls = pd.ExcelFile(RATSUM_PATH)

    # Auto-read from RATSUM
    simplified = read_simplified(xls)
    act = read_other_rates_act(xls)
    overhead = read_overhead(xls)
    com_long = read_com_summary(xls)

    # Write both outputs
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