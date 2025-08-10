# rate_files_automation.py
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

# ========= PATHS =========
RATSUM_PATH = Path("data/RATSUM.xlsx")
RATE_BAND_IMPORT_PATH = Path("data/RateBandImport.xlsx")
BURDEN_RATE_IMPORT_PATH = Path("data/BurdenRateImport.xlsx")
OUT_DIR = Path("output")
# Which rows in RateBandImport should receive ACT values:
VT_MATCH = {
    "codes": ["VB"],              # put VT Abrams 2-char codes here; [] to disable code matching
    "desc_tokens_all": ["VT", "ABRAM"],  # description must contain all tokens (case-insensitive)
    "require_rate_type": "Units",        # set to None to disable this guard
}

# ========= RULES =========
LABOR_ROUND  = 3
BURDEN_ROUND = 5
COM_ROUND    = 6

# ========== UTILITIES ==========
def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).strip().lower()

def normalize_cy(col_name: str) -> str:
    m = re.search(r"(20\d{2})", str(col_name))
    return f"CY{m.group(1)}" if m else str(col_name)

def find_year_cols(df: pd.DataFrame):
    return [c for c in df.columns if re.fullmatch(r"(CY)?\s*20\d{2}", str(c).strip())]

def pick_sheet(xls: pd.ExcelFile, *must_include: str) -> str:
    tokens = [t.upper() for t in must_include]
    for name in xls.sheet_names:
        if all(t in name.upper() for t in tokens):
            return name
    raise ValueError(f"No sheet found containing tokens {tokens}. Have: {xls.sheet_names}")

def read_best_skiprows(xls: pd.ExcelFile, sheet: str, candidates=range(0, 12)) -> pd.DataFrame:
    best, best_score = None, -1
    for k in candidates:
        df = pd.read_excel(xls, sheet_name=sheet, skiprows=k).dropna(how="all").dropna(axis=1, how="all")
        yc = find_year_cols(df)
        if len(yc) > best_score:
            best, best_score = df.reset_index(drop=True), len(yc)
    if best is None:
        raise ValueError(f"Could not parse table from '{sheet}'")
    return best

def extract_code(text: str) -> str | None:
    m = re.match(r"^\s*([0-9A-Z]{1,2})\b", str(text or "").upper())
    return m.group(1) if m else None

def find_col(df: pd.DataFrame, must_have: list[str]) -> str | None:
    for c in df.columns:
        u = norm(c)
        if all(t in u for t in must_have):
            return c
    return None

def build_vt_mask(df: pd.DataFrame, rb_col: str, desc_col: str | None, rate_type_col: str | None) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    if VT_MATCH["codes"]:
        code_set = {c.upper() for c in VT_MATCH["codes"]}
        mask = mask | df[rb_col].astype(str).str.strip().str.upper().isin(code_set)
    if desc_col and VT_MATCH["desc_tokens_all"]:
        toks = [t.upper() for t in VT_MATCH["desc_tokens_all"]]
        mask = mask | df[desc_col].astype(str).str.upper().apply(lambda s: all(t in s for t in toks))
    if VT_MATCH["require_rate_type"] and rate_type_col:
        want = VT_MATCH["require_rate_type"].upper()
        mask = mask & (df[rate_type_col].astype(str).str.upper() == want)
    return mask

# ========== READERS (auto-detect) ==========
def read_simplified(xls: pd.ExcelFile) -> pd.DataFrame:
    sheet = pick_sheet(xls, "SIMPLIFIED", "RATE")
    df = read_best_skiprows(xls, sheet)
    year_cols = find_year_cols(df)
    if not year_cols:
        raise ValueError("SIMPLIFIED: no CY columns found")

    # pick a label column (BUSINESS UNIT… / DESCRIPTION…)
    label_col = None
    for c in df.columns:
        if c not in year_cols and any(k in str(c).upper() for k in ["BUSINESS", "UNIT", "GDLS", "CONTRACT", "PRODUCT", "DESCRIPTION"]):
            label_col = c; break
    if label_col is None:
        label_col = [c for c in df.columns if c not in year_cols][0]

    df = df.rename(columns={c: normalize_cy(c) for c in year_cols})
    df = df.rename(columns={label_col: "Label"})
    df["Code"] = df["Label"].map(extract_code)
    df = df[df["Code"].notna()].copy()   # drop subheaders/blank lines

    keep = ["Code", "Label"] + [normalize_cy(c) for c in year_cols]
    return df[keep]

def read_other_rates_act(xls: pd.ExcelFile) -> pd.Series:
    sheet = pick_sheet(xls, "OTHER", "RATE")
    df = read_best_skiprows(xls, sheet)
    year_cols = find_year_cols(df)
    if not year_cols:
        raise ValueError("OTHER RATES: no CY columns found")

    first_year_idx = df.columns.get_loc(year_cols[0])
    text_block = df.columns[:first_year_idx] if first_year_idx > 0 else [df.columns[0]]
    df["Desc"] = df[text_block].astype(str).apply(lambda r: " ".join(x for x in r if x and x != "nan").strip(), axis=1)

    def is_act(txt: str) -> bool:
        t = re.sub(r"\s+", " ", str(txt or "")).upper()
        return ("ALLOWABLE" in t) and (("CONTROL" in t) or ("CONTL" in t)) and ("TEST" in t)
    row = df.loc[df["Desc"].map(is_act)]
    if row.empty:
        raise ValueError("OTHER RATES: Allowable Control/Contl Test row not found")

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
        raise ValueError("OVERHEAD: no Year/Date or CY columns found")

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

# ========== WRITERS ==========
def update_rate_band_import(simplified: pd.DataFrame, act: pd.Series, rbi_path: Path, out_path: Path):
    rbi = pd.read_excel(rbi_path)

    rb_col   = find_col(rbi, ["rate","band"]) or find_col(rbi, ["rateband"])
    base_col = find_col(rbi, ["base","rate"])
    start_col= find_col(rbi, ["start","date"])
    desc_col = find_col(rbi, ["desc"])          # optional
    rt_col   = find_col(rbi, ["rate","type"])   # optional
    if not all([rb_col, base_col, start_col]):
        raise ValueError("RateBandImport.xlsx missing one of: Rate Band, Base Rate..., Start Date")

    # Build mapping (Code, Year) -> Rate
    year_cols = [c for c in simplified.columns if c not in {"Code","Label"}]
    simp_long = simplified.melt(id_vars=["Code"], value_vars=year_cols, var_name="CY", value_name="Rate")
    simp_long["Year"] = simp_long["CY"].astype(str).str.extract(r"(20\d{2})").astype(int)
    simp_long["Rate"] = pd.to_numeric(simp_long["Rate"], errors="coerce").round(LABOR_ROUND)
    key_to_rate = {(str(code), int(y)): float(r) for code, y, r in zip(simp_long["Code"], simp_long["Year"], simp_long["Rate"])}

    rbi["_Year"] = pd.to_datetime(rbi[start_col], errors="coerce").dt.year
    def set_labor(row):
        code = extract_code(row[rb_col])
        yr   = int(row["_Year"]) if pd.notna(row["_Year"]) else None
        if code and yr is not None:
            v = key_to_rate.get((code, yr))
            if v is not None:
                row[base_col] = v
        return row
    rbi = rbi.apply(set_labor, axis=1).drop(columns=["_Year"])

    # Write ACT to VT ABRAMS rows only
    for cy in act.index:
        if cy not in rbi.columns:
            rbi[cy] = pd.NA
    vt_mask = build_vt_mask(rbi, rb_col, desc_col, rt_col)
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

# ========== COMPARISON REPORT ==========
def write_rateband_comparison(simplified: pd.DataFrame, rbi_path_or_df, out_path: Path):
    rbi = pd.read_excel(rbi_path_or_df) if isinstance(rbi_path_or_df, (str, Path)) else rbi_path_or_df
    rb_col = find_col(rbi, ["rate","band"]) or find_col(rbi, ["rateband"])
    if rb_col is None:
        raise ValueError("RateBandImport.xlsx: couldn't find 'Rate Band' column")

    ratsum_codes = set(simplified["Code"].astype(str).str.upper().str.strip())
    rbi_codes = set(rbi[rb_col].dropna().astype(str).map(lambda v: (extract_code(v) or "").upper()).str.strip())
    rbi_codes.discard("")

    only_ratsum = sorted(ratsum_codes - rbi_codes)
    only_rbi    = sorted(rbi_codes - ratsum_codes)

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        pd.DataFrame(only_ratsum, columns=["RateBand Code"]).to_excel(xw, sheet_name="In_RatSum_only", index=False)
        pd.DataFrame(only_rbi,    columns=["RateBand Code"]).to_excel(xw, sheet_name="In_RateBand_only", index=False)
    print(f"✅ Comparison report → {out_path.resolve()}")

# ========== MAIN ==========
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