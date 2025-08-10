# rate_files_automation.py  (revised)
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

# -------- PATHS --------
RATSUM_PATH = Path("data/RATSUM.xlsx")
RATE_BAND_IMPORT_PATH = Path("data/RateBandImport.xlsx")
BURDEN_RATE_IMPORT_PATH = Path("data/BurdenRateImport.xlsx")
OUT_DIR = Path("output")

# -------- TARGET: VT ABRAMS --------
VT_MATCH = {
    # If your VT Abrams rows use a 2-char code (e.g., 'VB'), put them here.
    "codes": ["VB"],                       # [] if you don’t want to use code matching
    # Or match by description tokens appearing in the Description column:
    "desc_tokens_all": ["VT", "ABRAM"],    # all tokens must appear (case-insensitive)
    # Optional guard: only update rows where Rate Type equals this (set to None to disable)
    "require_rate_type": "Units",
}

# Rounding rules
LABOR_ROUND  = 3
BURDEN_ROUND = 5
COM_ROUND    = 6

PAT_BAND_CODE = re.compile(r"^\s*([0-9A-Z]{2})\b")

def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).strip().lower()

def normalize_cy(col_name: str) -> str:
    m = re.search(r"(20\d{2})", str(col_name))
    return f"CY{m.group(1)}" if m else str(col_name)

def find_year_cols(df: pd.DataFrame):
    return [c for c in df.columns if re.fullmatch(r"(CY)?\s*20\d{2}", str(c).strip())]

def pick_sheet(xls: pd.ExcelFile, *must_include: str) -> str:
    toks = [t.upper() for t in must_include if t]
    for name in xls.sheet_names:
        if all(t in name.upper() for t in toks):
            return name
    raise ValueError(f"No sheet found containing tokens {toks}. Have: {xls.sheet_names}")

def read_best_skiprows(xls: pd.ExcelFile, sheet: str, candidates=range(0, 12)) -> pd.DataFrame:
    best, best_score = None, -1
    for k in candidates:
        df = pd.read_excel(xls, sheet_name=sheet, skiprows=k).dropna(how="all").dropna(axis=1, how="all")
        score = len(find_year_cols(df))
        if score > best_score:
            best, best_score = df.reset_index(drop=True), score
    if best is None:
        raise ValueError(f"Could not parse usable table from '{sheet}'")
    return best

# ---------- RATSUM readers ----------
def read_simplified(xls: pd.ExcelFile) -> pd.DataFrame:
    sheet = pick_sheet(xls, "SIMPLIFIED", "RATE")
    df = read_best_skiprows(xls, sheet)
    year_cols = find_year_cols(df)
    label_col = [c for c in df.columns if c not in year_cols][0]
    df = df.rename(columns={label_col: "Label"})
    df["RateBand"] = df["Label"].astype(str).str.extract(PAT_BAND_CODE)[0].fillna(df["Label"].astype(str))
    df = df.rename(columns={c: normalize_cy(c) for c in year_cols})
    return df[["RateBand", "Label"] + [normalize_cy(c) for c in year_cols]]

def read_other_rates_act(xls: pd.ExcelFile) -> pd.Series:
    sheet = pick_sheet(xls, "OTHER", "RATE")
    df = read_best_skiprows(xls, sheet)
    year_cols = find_year_cols(df)
    first_year_idx = df.columns.get_loc(year_cols[0]) if year_cols else 1
    desc_block = df.columns[:first_year_idx] if first_year_idx > 0 else [df.columns[0]]
    df["Desc"] = df[desc_block].astype(str).apply(lambda r: " ".join(x for x in r if x and x != "nan").strip(), axis=1)

    def is_act(txt: str) -> bool:
        t = re.sub(r"\s+", " ", str(txt or "")).upper()
        return ("ALLOWABLE" in t) and (("CONTROL" in t) or ("CONTL" in t)) and ("TEST" in t)

    row = df.loc[df["Desc"].map(is_act)]
    if row.empty:
        raise ValueError("OTHER RATES: Could not find Allowable Control/Contl Test row.")

    df = df.rename(columns={c: normalize_cy(c) for c in year_cols})
    s = row.iloc[0][[normalize_cy(c) for c in year_cols]].copy()
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).round(BURDEN_ROUND)
    s.name = "ACT"
    return s

def read_overhead(xls: pd.ExcelFile) -> pd.DataFrame:
    sheet = pick_sheet(xls, "OVERHEAD")
    df = read_best_skiprows(xls, sheet)
    year_col = next((c for c in df.columns if str(c).strip().lower() in {"date","year"}), None)
    pool_col = next((c for c in df.columns if str(c).strip().lower() in {"burden pool","pool","segment","description","descr"}), df.columns[0])

    if not year_col:
        yc = find_year_cols(df)
        if yc:
            long = df.melt(id_vars=[pool_col], value_vars=yc, var_name="CY", value_name="Rate")
            long["Year"] = long["CY"].astype(str).str.extract(r"(20\d{2})").astype(int)
            long = long.rename(columns={pool_col:"Pool"})
            long["Rate"] = pd.to_numeric(long["Rate"], errors="coerce").round(BURDEN_ROUND)
            return long[["Pool","Year","Rate"]]
        raise ValueError("OVERHEAD: No Year/Date or CY#### columns found.")

    meta = {pool_col, year_col}
    burden_cols = [c for c in df.columns if c not in meta]
    out = df[[pool_col, year_col] + burden_cols].copy().rename(columns={pool_col:"Pool", year_col:"Year"})
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
    first_year_idx = df.columns.get_loc([normalize_cy(c) for c in year_cols][0])
    text_block = df.columns[:first_year_idx] if first_year_idx > 0 else [df.columns[0]]
    df["Pool"] = df[text_block].astype(str).apply(lambda r: " ".join(x for x in r if x and x != "nan").strip(), axis=1)
    long = df.melt(id_vars=["Pool"], value_vars=[normalize_cy(c) for c in year_cols], var_name="CY", value_name="COM")
    long["Year"] = long["CY"].astype(str).str.extract(r"(20\d{2})").astype(int)
    long["COM"] = pd.to_numeric(long["COM"], errors="coerce").round(COM_ROUND)
    return long[["Pool","Year","COM"]]

# ---------- helpers for RateBandImport ----------
def find_col(df: pd.DataFrame, must_have: list[str]) -> str | None:
    """find column whose normalized header contains all tokens in order"""
    for c in df.columns:
        u = norm(c)
        if all(t in u for t in must_have):
            return c
    return None

def build_vt_mask(df: pd.DataFrame, rb_col: str, desc_col: str | None, rate_type_col: str | None) -> pd.Series:
    mask = pd.Series(False, index=df.index)

    # by code
    if VT_MATCH["codes"]:
        code_set = {c.upper() for c in VT_MATCH["codes"]}
        mask = mask | df[rb_col].astype(str).str.strip().str.upper().isin(code_set)

    # by description tokens
    if desc_col and VT_MATCH["desc_tokens_all"]:
        toks = [t.upper() for t in VT_MATCH["desc_tokens_all"]]
        m2 = df[desc_col].astype(str).str.upper().apply(lambda s: all(t in s for t in toks))
        mask = mask | m2

    # optional rate type filter
    if VT_MATCH["require_rate_type"] and rate_type_col:
        want = VT_MATCH["require_rate_type"].upper()
        mask = mask & (df[rate_type_col].astype(str).str.upper() == want)

    return mask

# ---------- writers ----------
def update_rate_band_import(simplified: pd.DataFrame, act: pd.Series, rbi_path: Path, out_path: Path):
    rbi = pd.read_excel(rbi_path)

    rb_col   = find_col(rbi, ["rate","band"]) or find_col(rbi, ["rateband"])
    base_col = find_col(rbi, ["base","rate"])
    start_col= find_col(rbi, ["start","date"])
    desc_col = find_col(rbi, ["desc"])          # optional
    rt_col   = find_col(rbi, ["rate","type"])   # optional
    if not all([rb_col, base_col, start_col]):
        raise ValueError(f"RateBandImport.xlsx missing columns. Found: {[rb_col, base_col, start_col]}")

    # Labor → Base Rate by (2-char band code, Start Date year)
    year_cols = [c for c in simplified.columns if c not in {"RateBand","Label"}]
    simp_long = simplified.melt(id_vars=["RateBand"], value_vars=year_cols, var_name="CY", value_name="Rate")
    simp_long["Year"] = simp_long["CY"].astype(str).str.extract(r"(20\d{2})").astype(int)
    simp_long["Rate"] = pd.to_numeric(simp_long["Rate"], errors="coerce").round(LABOR_ROUND)
    key_to_rate = {(str(b), int(y)): float(r) for b,y,r in zip(simp_long["RateBand"], simp_long["Year"], simp_long["Rate"])}

    rbi["_Year"] = pd.to_datetime(rbi[start_col], errors="coerce").dt.year
    def set_labor(row):
        code = str(row[rb_col]).strip()[:2]  # adjust if your code is longer
        yr = int(row["_Year"]) if pd.notna(row["_Year"]) else None
        if yr is not None:
            v = key_to_rate.get((code, yr))
            if v is not None:
                row[base_col] = v
        return row
    rbi = rbi.apply(set_labor, axis=1).drop(columns=["_Year"])

    # ACT → VT ABRAMS rows only
    for cy in act.index:
        if cy not in rbi.columns:
            rbi[cy] = pd.NA
    vt_mask = build_vt_mask(rbi, rb_col, desc_col, rt_col)
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
    pool_col = find_col(bri, ["burden","pool"]) or find_col(bri, ["pool"])
    year_col = find_col(bri, ["date"]) or find_col(bri, ["year"])
    if not all([pool_col, year_col]):
        raise ValueError("BurdenRateImport.xlsx missing 'Burden Pool' or 'Date/Year'")

    burden_cols = [c for c in bri.columns if c not in {pool_col, year_col, "Description", "Effective Date"}]

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

    for c in burden_cols:
        bri[c] = pd.to_numeric(bri[c], errors="coerce")
        bri[c] = bri[c].round(COM_ROUND if "COM" in str(c).upper() else BURDEN_ROUND)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        bri.to_excel(xw, index=False, sheet_name="Burden Rate Import")
    print(f"✅ Burden Rate Import updated → {out_path.resolve()}")

# ---------- comparison reports ----------
def write_rateband_comparison(simplified: pd.DataFrame, rbi_path_or_df, out_path: Path):
    if isinstance(rbi_path_or_df, (str, Path)):
        rbi = pd.read_excel(rbi_path_or_df)
    else:
        rbi = rbi_path_or_df

    rb_col = find_col(rbi, ["rate","band"]) or find_col(rbi, ["rateband"])
    if rb_col is None:
        raise ValueError("RateBandImport.xlsx: couldn't find 'Rate Band'")

    # Sets of band codes
    in_ratsum = set(simplified["RateBand"].astype(str).str.strip().str[:2])
    in_rbi    = set(rbi[rb_col].dropna().astype(str).str.strip().str[:2])

    only_ratsum = sorted(in_ratsum - in_rbi)
    only_rbi    = sorted(in_rbi - in_ratsum)

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        pd.DataFrame(only_ratsum, columns=["RateBand"]).to_excel(xw, sheet_name="In_RatSum_only", index=False)
        pd.DataFrame(only_rbi, columns=["RateBand"]).to_excel(xw, sheet_name="In_RateBand_only", index=False)
    print(f"✅ Comparison report → {out_path.resolve()}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xls = pd.ExcelFile(RATSUM_PATH)

    simplified = read_simplified(xls)
    act = read_other_rates_act(xls)
    overhead = read_overhead(xls)
    com_long = read_com_summary(xls)

    rbi_out = OUT_DIR / "RateBandImport_updated.xlsx"
    update_rate_band_import(simplified, act, RATE_BAND_IMPORT_PATH, rbi_out)
    update_burden_rate_import(overhead, com_long, BURDEN_RATE_IMPORT_PATH, OUT_DIR / "BurdenRateImport_updated.xlsx")
    write_rateband_comparison(simplified, rbi_out, OUT_DIR / "RateBand_Comparison.xlsx")

    print("🎉 Process complete.")