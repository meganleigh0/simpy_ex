# rate_files_pipeline_final.py
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

# ---------------- PATHS ----------------
RATSUM_PATH = Path("data/RATSUM.xlsx")
RATE_BAND_IMPORT_PATH = Path("data/RateBandImport.xlsx")
BURDEN_RATE_IMPORT_PATH = Path("data/BurdenRateImport.xlsx")
OUT_DIR = Path("output")

# ---------------- VT ABRAMS targeting ----------------
VT_CODES = {"VB"}                 # add more codes if you have them
VT_DESC_TOKENS = {"VT", "ABRAM"} # ALL tokens must be present in Description
VT_REQUIRE_RATE_TYPE = "Units"    # set to None to disable

# ---------------- Rounding ----------------
LABOR_ROUND  = 3
BURDEN_ROUND = 5
COM_ROUND    = 6

# ---------------- Utilities ----------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).strip().lower()

def _normalize_cy(c) -> str:
    m = re.search(r"(20\d{2})", str(c))
    return f"CY{m.group(1)}" if m else str(c)

def _find_year_cols(df: pd.DataFrame):
    pat = r"(?:CY)?\s*20\d{2}"
    return [c for c in df.columns if re.fullmatch(pat, str(c).strip(), flags=re.I)]

def _pick_sheet(xls: pd.ExcelFile, *tokens: str) -> str:
    toks = [t.upper() for t in tokens]
    for s in xls.sheet_names:
        if all(t in s.upper() for t in toks):
            return s
    raise ValueError(f"No sheet with tokens {toks}. Available: {xls.sheet_names}")

def _read_best_skiprows(xls: pd.ExcelFile, sheet: str, tries=range(0, 12)) -> pd.DataFrame:
    best, score = None, -1
    for k in tries:
        df = pd.read_excel(xls, sheet_name=sheet, skiprows=k).dropna(how="all").dropna(axis=1, how="all")
        yc = _find_year_cols(df)
        if len(yc) > score:
            best, score = df.reset_index(drop=True), len(yc)
    if best is None:
        raise ValueError(f"Could not parse a grid from '{sheet}'")
    return best

def _collapse_duplicate_named_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    If multiple columns share the same name, merge them left→right by taking
    the first non-null per row.
    """
    out = pd.DataFrame(index=df.index)
    # dict.fromkeys preserves order of first appearance
    for name in dict.fromkeys(df.columns):
        block = df.loc[:, df.columns == name]  # DataFrame (1..n columns)
        if block.shape[1] == 1:
            out[name] = block.iloc[:, 0]
        else:
            out[name] = block.bfill(axis=1).iloc[:, 0]
    return out

def _derive_code_from_text(text) -> str | None:
    """
    EXACT rule: first two characters of BUSINESS UNIT GDLS.
    - keep only [0-9A-Z], uppercase
    - if single digit → pad to 2: '4' -> '04'
    """
    s = str(text or "").strip().upper()
    if not s:
        return None
    two = re.sub(r"[^0-9A-Z]", "", s[:2])
    if not two:
        return None
    if len(two) == 1 and two.isdigit():
        two = f"0{two}"
    return two[:2]

def _derive_code_from_rbi_cell(val) -> str | None:
    s = str("" if val is None else val).strip().upper()
    if s.isdigit():
        return s.zfill(2)[:2]
    return _derive_code_from_text(s)

def _find_col(df: pd.DataFrame, tokens: list[str]) -> str | None:
    for c in df.columns:
        if all(t in _norm(c) for t in tokens):
            return c
    return None

# ---------------- Readers ----------------
def read_simplified_with_codes(xls: pd.ExcelFile) -> pd.DataFrame:
    """
    Output columns:
      - rate_band : first two chars of BUSINESS UNIT GDLS (zero-padded)
      - label     : BUSINESS UNIT GDLS string
      - CY####    : numeric year columns
    """
    sheet = _pick_sheet(xls, "SIMPLIFIED", "RATE")
    df = _read_best_skiprows(xls, sheet)

    year_cols_raw = _find_year_cols(df)
    if not year_cols_raw:
        raise ValueError("SIMPLIFIED: no year columns found")

    # Identify BUSINESS UNIT GDLS (fallback to first non-year column)
    label_col = next((c for c in df.columns
                      if c not in year_cols_raw and "BUSINESS" in str(c).upper() and "UNIT" in str(c).upper()), None)
    if label_col is None:
        label_col = [c for c in df.columns if c not in year_cols_raw][0]

    # Normalize year headers and collapse duplicates
    df = df.rename(columns={c: _normalize_cy(c) for c in year_cols_raw})
    df = _collapse_duplicate_named_cols(df)

    # Build final list of year columns after collapse
    year_cols = [c for c in df.columns if re.fullmatch(r"CY20\d{2}", str(c))]

    # Rename label column (if it survived collapse with same name)
    if label_col not in df.columns:
        # find something that still looks like the label col
        label_guess = next((c for c in df.columns if "BUSINESS" in str(c).upper() and "UNIT" in str(c).upper()), None)
        label_col = label_guess or df.columns[0]
    df = df.rename(columns={label_col: "label"})

    # Derive 2-char code from BUSINESS UNIT GDLS
    df["rate_band"] = df["label"].map(_derive_code_from_text)
    df = df[df["rate_band"].notna()].copy()

    # Cast year columns to numeric
    for c in year_cols:
        df.loc[:, c] = pd.to_numeric(df[c], errors="coerce")

    return df[["rate_band", "label"] + year_cols]

def read_allowable_control_test(xls: pd.ExcelFile) -> pd.Series:
    sheet = _pick_sheet(xls, "OTHER", "RATE")
    df = _read_best_skiprows(xls, sheet)
    years = _find_year_cols(df)
    if not years:
        raise ValueError("OTHER RATES: no CY columns")

    first_year_idx = df.columns.get_loc(years[0])
    pre = df.columns[:first_year_idx] if first_year_idx > 0 else [df.columns[0]]
    df["desc"] = df[pre].astype(str).apply(lambda r: " ".join(x for x in r if x and x != "nan").strip(), axis=1)

    def _is_act(t: str) -> bool:
        u = re.sub(r"\s+", " ", str(t or "")).upper()
        return ("ALLOWABLE" in u) and (("CONTROL" in u) or ("CONTL" in u)) and ("TEST" in u)

    row = df.loc[df["desc"].map(_is_act)]
    if row.empty:
        raise ValueError("OTHER RATES: Allowable Control/Contl Test row not found")

    df = df.rename(columns={c: _normalize_cy(c) for c in years})
    df = _collapse_duplicate_named_cols(df)

    keep_years = [c for c in df.columns if re.fullmatch(r"CY20\d{2}", str(c))]
    s = row.iloc[0][keep_years].copy()
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).round(BURDEN_ROUND)
    s.name = "ACT"
    return s

def read_overhead(xls: pd.ExcelFile) -> pd.DataFrame:
    sheet = _pick_sheet(xls, "OVERHEAD")
    df = _read_best_skiprows(xls, sheet)
    ycol = next((c for c in df.columns if str(c).strip().lower() in {"date", "year"}), None)
    pcol = next((c for c in df.columns if str(c).strip().lower() in
                 {"burden pool","pool","segment","description","descr"}), df.columns[0])

    if not ycol:
        yc = _find_year_cols(df)
        if yc:
            long = df.melt(id_vars=[pcol], value_vars=yc, var_name="CY", value_name="Rate")
            long["Year"] = long["CY"].astype(str).str.extract(r"(20\d{2})").astype(int)
            long = long.rename(columns={pcol: "Pool"})
            long["Rate"] = pd.to_numeric(long["Rate"], errors="coerce").round(BURDEN_ROUND)
            return long[["Pool", "Year", "Rate"]]
        raise ValueError("OVERHEAD: no Year/Date or CY columns")

    meta = {pcol, ycol}
    bcols = [c for c in df.columns if c not in meta]
    out = df[[pcol, ycol] + bcols].copy().rename(columns={pcol: "Pool", ycol: "Year"})
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    for c in bcols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].round(COM_ROUND if "COM" in str(c).upper() else BURDEN_ROUND)
    return out

def read_com_summary(xls: pd.ExcelFile) -> pd.DataFrame:
    sheet = _pick_sheet(xls, "COM", "SUMMARY")
    df = _read_best_skiprows(xls, sheet)
    years = _find_year_cols(df)
    if not years:
        return pd.DataFrame(columns=["Pool", "Year", "COM"])

    df = df.rename(columns={c: _normalize_cy(c) for c in years})
    df = _collapse_duplicate_named_cols(df)

    keep_years = [c for c in df.columns if re.fullmatch(r"CY20\d{2}", str(c))]
    # figure out “pre-year” block again after collapse
    first_year_idx = df.columns.get_loc(keep_years[0])
    pre = df.columns[:first_year_idx] if first_year_idx > 0 else [df.columns[0]]
    df["Pool"] = df[pre].astype(str).apply(lambda r: " ".join(x for x in r if x and x != "nan").strip(), axis=1)

    long = df.melt(id_vars=["Pool"], value_vars=keep_years, var_name="CY", value_name="COM")
    long["Year"] = long["CY"].astype(str).str.extract(r"(20\d{2})").astype(int)
    long["COM"] = pd.to_numeric(long["COM"], errors="coerce").round(COM_ROUND)
    return long[["Pool", "Year", "COM"]]

# ---------------- Writers ----------------
def update_rate_band_import(simplified: pd.DataFrame, act: pd.Series, rbi_in: Path, rbi_out: Path):
    rbi = pd.read_excel(rbi_in)

    rb_col   = _find_col(rbi, ["rate","band"]) or _find_col(rbi, ["rateband"])
    base_col = _find_col(rbi, ["base","rate"])
    start_col= _find_col(rbi, ["start","date"])
    desc_col = _find_col(rbi, ["desc"])        # optional
    rt_col   = _find_col(rbi, ["rate","type"]) # optional
    if not all([rb_col, base_col, start_col]):
        raise ValueError("RBI missing Rate Band / Base Rate / Start Date")

    # Build (code, year) -> rate
    year_cols = [c for c in simplified.columns if c not in {"rate_band","label"}]
    long = simplified.melt(id_vars=["rate_band"], value_vars=year_cols, var_name="CY", value_name="Rate")
    long["Year"] = long["CY"].astype(str).str.extract(r"(20\d{2})").astype(int)
    long["Rate"] = pd.to_numeric(long["Rate"], errors="coerce").round(LABOR_ROUND)
    rate_map = {(rb, int(y)): float(r) for rb, y, r in zip(long["rate_band"], long["Year"], long["Rate"])}

    # Apply Base Rate by (code, StartDate year)
    rbi["_Year"] = pd.to_datetime(rbi[start_col], errors="coerce").dt.year
    def _put(row):
        code = _derive_code_from_rbi_cell(row[rb_col])
        yr = int(row["_Year"]) if pd.notna(row["_Year"]) else None
        if code and yr is not None:
            v = rate_map.get((code, yr))
            if v is not None:
                row[base_col] = round(v, LABOR_ROUND)
        return row
    rbi = rbi.apply(_put, axis=1).drop(columns=["_Year"])

    # ACT → VT ABRAMS rows only
    for cy in act.index:
        if cy not in rbi.columns:
            rbi[cy] = pd.NA

    code_match = rbi[rb_col].map(_derive_code_from_rbi_cell).isin(VT_CODES) if VT_CODES else pd.Series(False, index=rbi.index)
    desc_match = pd.Series(True, index=rbi.index)  # all-tokens must be present
    if VT_DESC_TOKENS and desc_col:
        up = rbi[desc_col].astype(str).str.upper()
        for tok in VT_DESC_TOKENS:
            desc_match &= up.str.contains(re.escape(tok), na=False)
    mask = code_match | desc_match
    if VT_REQUIRE_RATE_TYPE and rt_col:
        mask &= rbi[rt_col].astype(str).str.upper().eq(VT_REQUIRE_RATE_TYPE.upper())

    rbi.loc[mask, list(act.index)] = list(act.values)

    # Round CY#### burden columns
    cy_cols = [c for c in rbi.columns if re.fullmatch(r"CY20\d{2}", str(c))]
    rbi[cy_cols] = rbi[cy_cols].apply(pd.to_numeric, errors="coerce").round(BURDEN_ROUND)

    rbi_out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(rbi_out, engine="openpyxl") as xw:
        rbi.to_excel(xw, index=False, sheet_name="Rate Band Import")
    print(f"✅ RateBandImport updated → {rbi_out.resolve()}")

def update_burden_rate_import(overhead_df: pd.DataFrame, com_df: pd.DataFrame, bri_in: Path, bri_out: Path):
    bri = pd.read_excel(bri_in)
    pool_col = _find_col(bri, ["burden","pool"]) or _find_col(bri, ["pool"])
    year_col = _find_col(bri, ["date"]) or _find_col(bri, ["year"])
    if not all([pool_col, year_col]):
        raise ValueError("BRI missing Burden Pool / Date or Year")

    bcols = [c for c in bri.columns if c not in {pool_col, year_col, "Description", "Effective Date"}]

    target = overhead_df.copy()
    if not com_df.empty:
        cmap = com_df.set_index(["Pool","Year"])["COM"].to_dict()
        target["_COM_fill_"] = [cmap.get((p,y)) for p,y in zip(target["Pool"], target["Year"])]

    bri[year_col] = pd.to_numeric(bri[year_col], errors="coerce").astype("Int64")
    idx = {(_norm(p), int(y)): r for p, y, r in zip(target["Pool"], target["Year"], target.index)}

    def _apply(row):
        key = (_norm(row[pool_col]), int(row[year_col])) if pd.notna(row[year_col]) else None
        if key and key in idx:
            src = target.loc[idx[key]]
            for c in bcols:
                if c in src:
                    row[c] = src[c]
                elif c.upper().startswith("COM") and "_COM_fill_" in src:
                    row[c] = src["_COM_fill_"]
        return row
    bri = bri.apply(_apply, axis=1)

    for c in bcols:
        bri[c] = pd.to_numeric(bri[c], errors="coerce")
        bri[c] = bri[c].round(COM_ROUND if "COM" in str(c).upper() else BURDEN_ROUND)

    bri_out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(bri_out, engine="openpyxl") as xw:
        bri.to_excel(xw, index=False, sheet_name="Burden Rate Import")
    print(f"✅ BurdenRateImport updated → {bri_out.resolve()}")

# ---------------- Comparison (your exact rule) ----------------
def write_rateband_comparison(simplified_df: pd.DataFrame, rbi_path_or_df, out_path: Path):
    """
    Compare:
      - RATSUM: first-two-chars from BUSINESS UNIT GDLS ('rate_band')
      - RBI:    first-two-chars of 'Rate Band' after normalization
    """
    rbi = pd.read_excel(rbi_path_or_df) if isinstance(rbi_path_or_df, (str, Path)) else rbi_path_or_df
    rb_col = _find_col(rbi, ["rate","band"]) or _find_col(rbi, ["rateband"])
    if rb_col is None:
        raise ValueError("RBI: can't find Rate Band column")

    ratsum_codes = set(simplified_df["rate_band"].dropna().astype(str).str.upper().str.strip())
    rbi_codes = set(rbi[rb_col].dropna().map(_derive_code_from_rbi_cell))
    rbi_codes.discard(None)

    only_ratsum = sorted(ratsum_codes - rbi_codes)
    only_rbi    = sorted(rbi_codes - ratsum_codes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        pd.DataFrame(only_ratsum, columns=["RateBand Code"]).to_excel(xw, sheet_name="In_RatSum_only", index=False)
        pd.DataFrame(only_rbi,    columns=["RateBand Code"]).to_excel(xw, sheet_name="In_RateBand_only", index=False)
    print(f"✅ Comparison report → {out_path.resolve()}")

# ---------------- Main ----------------
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xls = pd.ExcelFile(RATSUM_PATH)

    simplified = read_simplified_with_codes(xls)      # ← Code from first two chars of BUSINESS UNIT GDLS
    act        = read_allowable_control_test(xls)     # ← ACT series
    overhead   = read_overhead(xls)                   # ← burden pools by year
    com_long   = read_com_summary(xls)                # ← COM (optional)

    rbi_out = OUT_DIR / "RateBandImport_updated.xlsx"
    update_rate_band_import(simplified, act, RATE_BAND_IMPORT_PATH, rbi_out)

    bri_out = OUT_DIR / "BurdenRateImport_updated.xlsx"
    update_burden_rate_import(overhead, com_long, BURDEN_RATE_IMPORT_PATH, bri_out)

    write_rateband_comparison(simplified, rbi_out, OUT_DIR / "RateBand_Comparison.xlsx")
    print("🎉 Process complete.")