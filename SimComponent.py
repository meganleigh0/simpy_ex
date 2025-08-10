# pipeline_ratefiles.py
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

# ---------- PATHS ----------
RATSUM_PATH = Path("data/RATSUM.xlsx")
RATE_BAND_IMPORT_PATH = Path("data/RateBandImport.xlsx")
BURDEN_RATE_IMPORT_PATH = Path("data/BurdenRateImport.xlsx")
OUT_DIR = Path("output")

# ---------- VT ABRAMS targeting ----------
VT_CODES = {"VB"}                 # 2-char codes; leave empty to skip code matching
VT_DESC_TOKENS = {"VT", "ABRAM"}  # ALL tokens must appear in description (case-insensitive)
VT_REQUIRE_RATE_TYPE = "Units"    # None to disable this guard

# ---------- Rounding ----------
LABOR_ROUND  = 3
BURDEN_ROUND = 5
COM_ROUND    = 6


# ====================== helpers ======================
def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).strip().lower()

def _normalize_cy(col: str) -> str:
    m = re.search(r"(20\d{2})", str(col))
    return f"CY{m.group(1)}" if m else str(col)

def _find_year_cols(df: pd.DataFrame):
    PAT = r"(CY)?\s*20\d{2}"
    return [c for c in df.columns if re.fullmatch(PAT, str(c).strip())]

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

def _derive_code_from_text(text) -> str | None:
    """
    EXACT rule you asked for: the first two characters of BUSINESS UNIT GDLS.
    - take first two visible characters
    - uppercase
    - if it's a single digit (e.g., '4'), pad to '04'
    - must be 1–2 of [0-9A-Z] to be considered valid
    """
    s = str(text or "").strip().upper()
    if not s:
        return None
    two = s[:2]
    # If the first char is a digit and second char is space (rare), pull next nonspace
    if len(two) == 2 and two[1] == " " and len(s) >= 3:
        two = two[0] + s[2]
    # Validate: keep alnum only
    two = re.sub(r"[^0-9A-Z]", "", two)
    if not two:
        return None
    if len(two) == 1 and two.isdigit():
        two = f"0{two}"
    return two[:2]

def _derive_code_from_rbi_cell(val) -> str | None:
    """Normalize RBI 'Rate Band' cell to the same 2-char rule."""
    s = str(val if val is not None else "").strip().upper()
    if s == "":
        return None
    # sometimes the cell is purely numeric (4 -> '04')
    if s.isdigit():
        return s.zfill(2)[:2]
    # otherwise, take first two visible chars (same as BUSINESS UNIT rule)
    return _derive_code_from_text(s)

def _find_col(df: pd.DataFrame, tokens: list[str]) -> str | None:
    for c in df.columns:
        if all(t in _norm(c) for t in tokens):
            return c
    return None


# ====================== read RATSUM (SIMPLIFIED) ======================
def read_simplified_with_codes(xls: pd.ExcelFile) -> pd.DataFrame:
    """
    Returns: DataFrame with columns
      - rate_band (exactly the first two characters of BUSINESS UNIT GDLS, normalized)
      - label (BUSINESS UNIT GDLS)
      - CY#### columns as numeric
    """
    sheet = _pick_sheet(xls, "SIMPLIFIED", "RATE")
    df = _read_best_skiprows(xls, sheet)
    years = _find_year_cols(df)
    if not years:
        raise ValueError("SIMPLIFIED: no CY columns found")

    # Locate BUSINESS UNIT GDLS column explicitly (fallback: first non-year text column)
    label_col = next(
        (c for c in df.columns if c not in years and "BUSINESS" in str(c).upper() and "UNIT" in str(c).upper()),
        None
    )
    if label_col is None:
        label_col = [c for c in df.columns if c not in years][0]

    # Normalize headers, derive code
    df = df.rename(columns={label_col: "label", **{c: _normalize_cy(c) for c in years}})
    df["rate_band"] = df["label"].map(_derive_code_from_text)

    # keep only rows that produced a valid 2-char code
    df = df[df["rate_band"].notna()].copy()

    # Numeric year columns
    year_cols = [c for c in df.columns if re.fullmatch(r"CY20\d{2}", str(c))]
    for c in year_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[["rate_band", "label"] + year_cols]


# ====================== read OTHER RATES → ACT ======================
def read_allowable_control_test(xls: pd.ExcelFile) -> pd.Series:
    sheet = _pick_sheet(xls, "OTHER", "RATE")
    df = _read_best_skiprows(xls, sheet)
    years = _find_year_cols(df)
    if not years:
        raise ValueError("OTHER RATES: no CY columns")

    first_year_idx = df.columns.get_loc(years[0])
    pre_year = df.columns[:first_year_idx] if first_year_idx > 0 else [df.columns[0]]
    df["desc"] = df[pre_year].astype(str).apply(lambda r: " ".join(x for x in r if x and x != "nan").strip(), axis=1)

    def _is_act(text: str) -> bool:
        t = re.sub(r"\s+", " ", str(text or "")).upper()
        return ("ALLOWABLE" in t) and (("CONTROL" in t) or ("CONTL" in t)) and ("TEST" in t)

    row = df.loc[df["desc"].map(_is_act)]
    if row.empty:
        raise ValueError("OTHER RATES: Allowable Control/Contl Test row not found")

    df = df.rename(columns={c: _normalize_cy(c) for c in years})
    s = row.iloc[0][[_normalize_cy(c) for c in years]].copy()
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).round(BURDEN_ROUND)
    s.name = "ACT"
    return s


# ====================== read OVERHEAD + COM SUMMARY ======================
def read_overhead(xls: pd.ExcelFile) -> pd.DataFrame:
    sheet = _pick_sheet(xls, "OVERHEAD")
    df = _read_best_skiprows(xls, sheet)
    ycol = next((c for c in df.columns if str(c).strip().lower() in {"date", "year"}), None)
    pcol = next((c for c in df.columns if str(c).strip().lower() in {"burden pool","pool","segment","description","descr"}), df.columns[0])

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
    first_year_idx = df.columns.get_loc([_normalize_cy(c) for c in years][0])
    pre_year = df.columns[:first_year_idx] if first_year_idx > 0 else [df.columns[0]]
    df["Pool"] = df[pre_year].astype(str).apply(lambda r: " ".join(x for x in r if x and x != "nan").strip(), axis=1)

    long = df.melt(id_vars=["Pool"], value_vars=[_normalize_cy(c) for c in years], var_name="CY", value_name="COM")
    long["Year"] = long["CY"].astype(str).str.extract(r"(20\d{2})").astype(int)
    long["COM"] = pd.to_numeric(long["COM"], errors="coerce").round(COM_ROUND)
    return long[["Pool", "Year", "COM"]]


# ====================== writers ======================
def update_rate_band_import(simplified: pd.DataFrame, act: pd.Series, rbi_in: Path, rbi_out: Path):
    rbi = pd.read_excel(rbi_in)

    rb_col   = _find_col(rbi, ["rate", "band"]) or _find_col(rbi, ["rateband"])
    base_col = _find_col(rbi, ["base", "rate"])
    start_col= _find_col(rbi, ["start", "date"])
    desc_col = _find_col(rbi, ["desc"])        # optional
    rt_col   = _find_col(rbi, ["rate", "type"])# optional
    if not all([rb_col, base_col, start_col]):
        raise ValueError("RBI is missing one of: Rate Band / Base Rate / Start Date")

    # Map (code, year) -> rate from simplified
    year_cols = [c for c in simplified.columns if c not in {"rate_band","label"}]
    long = simplified.melt(id_vars=["rate_band"], value_vars=year_cols, var_name="CY", value_name="Rate")
    long["Year"] = long["CY"].astype(str).str.extract(r"(20\d{2})").astype(int)
    long["Rate"] = pd.to_numeric(long["Rate"], errors="coerce").round(LABOR_ROUND)
    map_rate = {(rb, int(y)): float(r) for rb, y, r in zip(long["rate_band"], long["Year"], long["Rate"])}

    # Write base rates by (code, StartDate year)
    rbi["_Year"] = pd.to_datetime(rbi[start_col], errors="coerce").dt.year
    def _put(row):
        code = _derive_code_from_rbi_cell(row[rb_col])
        yr = int(row["_Year"]) if pd.notna(row["_Year"]) else None
        if code and yr is not None:
            v = map_rate.get((code, yr))
            if v is not None:
                row[base_col] = v
        return row
    rbi = rbi.apply(_put, axis=1).drop(columns=["_Year"])

    # ACT → VT ABRAMS rows only
    for cy in act.index:
        if cy not in rbi.columns:
            rbi[cy] = pd.NA

    vt_mask = pd.Series(True, index=rbi.index)
    if VT_CODES:
        vt_mask &= rbi[rb_col].map(_derive_code_from_rbi_cell).isin({c for c in VT_CODES})
    if VT_DESC_TOKENS and desc_col:
        toks = {t.upper() for t in VT_DESC_TOKENS}
        vt_mask &= rbi[desc_col].astype(str).str.upper().apply(lambda s: all(t in s for t in toks))
    if VT_REQUIRE_RATE_TYPE and rt_col:
        vt_mask &= rbi[rt_col].astype(str).str.upper().eq(VT_REQUIRE_RATE_TYPE.upper())

    rbi.loc[vt_mask, list(act.index)] = list(act.values)

    # Round CY#### burden cols
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
        raise ValueError("BRI is missing Burden Pool or Date/Year")

    bcols = [c for c in bri.columns if c not in {pool_col, year_col, "Description", "Effective Date"}]

    target = overhead_df.copy()
    if not com_df.empty:
        cmap = com_df.set_index(["Pool","Year"])["COM"].to_dict()
        target["_COM_fill_"] = [cmap.get((p,y)) for p,y in zip(target["Pool"], target["Year"])]

    bri[year_col] = pd.to_numeric(bri[year_col], errors="coerce").astype("Int64")
    idx = {( _norm(p), int(y) ): r for p, y, r in zip(target["Pool"], target["Year"], target.index)}

    def _apply(row):
        key = (_norm(row[pool_col]), int(row[year_col])) if pd.notna(row[year_col]) else None
        if key and key in idx:
            src = target.loc[idx[key]]
            for c in bcols:
                if c in src: row[c] = src[c]
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


# ====================== comparison (precisely your rule) ======================
def write_rateband_comparison(simplified_df: pd.DataFrame, rbi_path_or_df, out_path: Path):
    """
    Build sets ONLY from:
      - RATSUM: first two characters of BUSINESS UNIT GDLS (we already computed 'rate_band')
      - RBI:    normalize first two characters of the 'Rate Band' column
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


# ====================== main ======================
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xls = pd.ExcelFile(RATSUM_PATH)

    simplified = read_simplified_with_codes(xls)     # <- uses FIRST TWO CHARS of BUSINESS UNIT GDLS
    act = read_allowable_control_test(xls)
    overhead = read_overhead(xls)
    com_long = read_com_summary(xls)

    rbi_out = OUT_DIR / "RateBandImport_updated.xlsx"
    update_rate_band_import(simplified, act, RATE_BAND_IMPORT_PATH, rbi_out)

    bri_out = OUT_DIR / "BurdenRateImport_updated.xlsx"
    update_burden_rate_import(overhead, com_long, BURDEN_RATE_IMPORT_PATH, bri_out)

    write_rateband_comparison(simplified, rbi_out, OUT_DIR / "RateBand_Comparison.xlsx")
    print("🎉 Process complete.")