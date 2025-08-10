# rate_files_pipeline.py
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

# ------------ Paths ------------
RATSUM_PATH = Path("data/RATSUM.xlsx")
RATE_BAND_IMPORT_PATH = Path("data/RateBandImport.xlsx")
BURDEN_RATE_IMPORT_PATH = Path("data/BurdenRateImport.xlsx")
OUT_DIR = Path("output")

# ------------ Targeting for VT ABRAMS (ACT values) ------------
VT_CODES = {"VB"}                 # add any 2-char VT codes here; leave empty to skip code matching
VT_DESC_TOKENS = {"VT", "ABRAM"} # all tokens must appear in Description (case-insensitive)
VT_REQUIRE_RATE_TYPE = "Units"    # set to None to disable the guard

# ------------ Rounding rules ------------
LABOR_ROUND  = 3
BURDEN_ROUND = 5
COM_ROUND    = 6


# ================= Helpers =================
def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).strip().lower()

def normalize_cy(col_name: str) -> str:
    m = re.search(r"(20\d{2})", str(col_name))
    return f"CY{m.group(1)}" if m else str(col_name)

def find_year_cols(df: pd.DataFrame):
    return [c for c in df.columns if re.fullmatch(r"(CY)?\s*20\d{2}", str(c).strip())]

def pick_sheet(xls: pd.ExcelFile, *tokens: str) -> str:
    toks = [t.upper() for t in tokens]
    for name in xls.sheet_names:
        if all(t in name.upper() for t in toks):
            return name
    raise ValueError(f"No sheet with tokens {toks}. Available: {xls.sheet_names}")

def read_best_skiprows(xls: pd.ExcelFile, sheet: str, tries=range(0, 12)) -> pd.DataFrame:
    best, score = None, -1
    for k in tries:
        df = pd.read_excel(xls, sheet_name=sheet, skiprows=k).dropna(how="all").dropna(axis=1, how="all")
        if len(find_year_cols(df)) > score:
            best, score = df.reset_index(drop=True), len(find_year_cols(df))
    if best is None:
        raise ValueError(f"Could not parse a grid from '{sheet}'")
    return best

def extract_raw_code(text) -> str | None:
    m = re.match(r"^\s*([0-9A-Z]{1,2})\b", str(text or "").upper())
    return m.group(1) if m else None

def normalize_code(x) -> str | None:
    """First 1–2 chars; pad 1-digit numbers (4 -> '04')."""
    raw = extract_raw_code(x)
    if raw is None:
        return None
    if raw.isdigit() and len(raw) == 1:
        return f"0{raw}"
    return raw

def find_col(df: pd.DataFrame, tokens: list[str]) -> str | None:
    for c in df.columns:
        if all(t in norm(c) for t in tokens):
            return c
    return None


# ================= RATSUM readers (derive codes from BUSINESS UNIT GDLS) =================
def read_simplified(xls: pd.ExcelFile) -> pd.DataFrame:
    """Return columns: Code, Label, CY####...  (Code = first two chars of BUSINESS UNIT GDLS)"""
    sheet = pick_sheet(xls, "SIMPLIFIED", "RATE")
    df = read_best_skiprows(xls, sheet)
    years = find_year_cols(df)
    if not years:
        raise ValueError("SIMPLIFIED: no CY columns found")

    # find the BUSINESS UNIT column (or first non-year text column)
    label_col = next(
        (c for c in df.columns if c not in years and any(k in str(c).upper() for k in ["BUSINESS", "UNIT", "GDLS"])),
        None
    )
    if label_col is None:
        label_col = [c for c in df.columns if c not in years][0]

    df = df.rename(columns={c: normalize_cy(c) for c in years})
    df = df.rename(columns={label_col: "Label"})
    df["Code"] = df["Label"].map(normalize_code)
    # keep only true data rows (rows that actually have a two-char code)
    df = df[df["Code"].notna()].copy()

    return df[["Code", "Label"] + [normalize_cy(c) for c in years]]

def read_other_rates_act(xls: pd.ExcelFile) -> pd.Series:
    """Pull the ALLOWABLE CONTROL/CONTL TEST (RATE) row as a Series indexed by CY####."""
    sheet = pick_sheet(xls, "OTHER", "RATE")
    df = read_best_skiprows(xls, sheet)
    years = find_year_cols(df)
    if not years:
        raise ValueError("OTHER RATES: no CY columns found")

    # Build descriptor from columns before the first year col
    first_year_idx = df.columns.get_loc(years[0])
    pre_year_block = df.columns[:first_year_idx] if first_year_idx > 0 else [df.columns[0]]
    df["Desc"] = df[pre_year_block].astype(str).apply(lambda r: " ".join(x for x in r if x and x != "nan").strip(), axis=1)

    def is_act(txt: str) -> bool:
        t = re.sub(r"\s+", " ", str(txt or "")).upper()
        return ("ALLOWABLE" in t) and (("CONTROL" in t) or ("CONTL" in t)) and ("TEST" in t)

    row = df.loc[df["Desc"].map(is_act)]
    if row.empty:
        raise ValueError("OTHER RATES: could not find Allowable Control/Contl Test row")

    df = df.rename(columns={c: normalize_cy(c) for c in years})
    s = row.iloc[0][[normalize_cy(c) for c in years]].copy()
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).round(BURDEN_ROUND)
    s.name = "ACT"
    return s

def read_overhead(xls: pd.ExcelFile) -> pd.DataFrame:
    sheet = pick_sheet(xls, "OVERHEAD")
    df = read_best_skiprows(xls, sheet)
    ycol = next((c for c in df.columns if str(c).strip().lower() in {"date", "year"}), None)
    pcol = next((c for c in df.columns if str(c).strip().lower() in {"burden pool","pool","segment","description","descr"}), df.columns[0])

    if not ycol:
        yc = find_year_cols(df)
        if yc:
            long = df.melt(id_vars=[pcol], value_vars=yc, var_name="CY", value_name="Rate")
            long["Year"] = long["CY"].astype(str).str.extract(r"(20\d{2})").astype(int)
            long = long.rename(columns={pcol: "Pool"})
            long["Rate"] = pd.to_numeric(long["Rate"], errors="coerce").round(BURDEN_ROUND)
            return long[["Pool", "Year", "Rate"]]
        raise ValueError("OVERHEAD: no Year/Date or CY columns found")

    meta = {pcol, ycol}
    bcols = [c for c in df.columns if c not in meta]
    out = df[[pcol, ycol] + bcols].copy().rename(columns={pcol: "Pool", ycol: "Year"})
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    for c in bcols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].round(COM_ROUND if "COM" in str(c).upper() else BURDEN_ROUND)
    return out

def read_com_summary(xls: pd.ExcelFile) -> pd.DataFrame:
    sheet = pick_sheet(xls, "COM", "SUMMARY")
    df = read_best_skiprows(xls, sheet)
    years = find_year_cols(df)
    if not years:
        return pd.DataFrame(columns=["Pool", "Year", "COM"])

    df = df.rename(columns={c: normalize_cy(c) for c in years})
    first_year_idx = df.columns.get_loc([normalize_cy(c) for c in years][0])
    pre_year = df.columns[:first_year_idx] if first_year_idx > 0 else [df.columns[0]]
    df["Pool"] = df[pre_year].astype(str).apply(lambda r: " ".join(x for x in r if x and x != "nan").strip(), axis=1)

    long = df.melt(id_vars=["Pool"], value_vars=[normalize_cy(c) for c in years], var_name="CY", value_name="COM")
    long["Year"] = long["CY"].astype(str).str.extract(r"(20\d{2})").astype(int)
    long["COM"] = pd.to_numeric(long["COM"], errors="coerce").round(COM_ROUND)
    return long[["Pool", "Year", "COM"]]


# ================= Writers =================
def update_rate_band_import(simplified: pd.DataFrame, act: pd.Series, rbi_in: Path, rbi_out: Path):
    rbi = pd.read_excel(rbi_in)

    rb_col   = find_col(rbi, ["rate","band"]) or find_col(rbi, ["rateband"])
    base_col = find_col(rbi, ["base","rate"])
    start_col= find_col(rbi, ["start","date"])
    desc_col = find_col(rbi, ["desc"])        # optional
    rt_col   = find_col(rbi, ["rate","type"]) # optional
    if not all([rb_col, base_col, start_col]):
        raise ValueError("RBI missing column(s): Rate Band / Base Rate / Start Date")

    # Build (Code, Year) → Rate from Simplified
    year_cols = [c for c in simplified.columns if c not in {"Code", "Label"}]
    simp_long = simplified.melt(id_vars=["Code"], value_vars=year_cols, var_name="CY", value_name="Rate")
    simp_long["Year"] = simp_long["CY"].astype(str).str.extract(r"(20\d{2})").astype(int)
    simp_long["Rate"] = pd.to_numeric(simp_long["Rate"], errors="coerce").round(LABOR_ROUND)
    map_rate = {(normalize_code(code), int(y)): float(r)
                for code, y, r in zip(simp_long["Code"], simp_long["Year"], simp_long["Rate"])}

    # Apply by Start Date year + normalized RBI code
    rbi["_Year"] = pd.to_datetime(rbi[start_col], errors="coerce").dt.year
    def _set_labor(row):
        code = normalize_code(row[rb_col])
        yr = int(row["_Year"]) if pd.notna(row["_Year"]) else None
        if code and yr is not None:
            v = map_rate.get((code, yr))
            if v is not None:
                row[base_col] = v
        return row
    rbi = rbi.apply(_set_labor, axis=1).drop(columns=["_Year"])

    # Write ACT only to VT ABRAMS rows
    for cy in act.index:
        if cy not in rbi.columns:
            rbi[cy] = pd.NA

    vt_mask = pd.Series(True, index=rbi.index)
    if VT_CODES:
        vt_mask &= rbi[rb_col].map(normalize_code).isin({normalize_code(c) for c in VT_CODES})
    if VT_DESC_TOKENS and desc_col:
        toks = {t.upper() for t in VT_DESC_TOKENS}
        vt_mask |= rbi[desc_col].astype(str).str.upper().apply(lambda s: toks.issubset(set(s.split())))
    if VT_REQUIRE_RATE_TYPE and rt_col:
        vt_mask &= rbi[rt_col].astype(str).str.upper().eq(VT_REQUIRE_RATE_TYPE.upper())

    rbi.loc[vt_mask, list(act.index)] = list(act.values)

    # Round CY#### burden columns
    cy_cols = [c for c in rbi.columns if re.fullmatch(r"CY20\d{2}", str(c))]
    rbi[cy_cols] = rbi[cy_cols].apply(pd.to_numeric, errors="coerce").round(BURDEN_ROUND)

    rbi_out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(rbi_out, engine="openpyxl") as xw:
        rbi.to_excel(xw, index=False, sheet_name="Rate Band Import")
    print(f"✅ RateBandImport updated → {rbi_out.resolve()}")

def update_burden_rate_import(overhead_df: pd.DataFrame, com_df: pd.DataFrame, bri_in: Path, bri_out: Path):
    bri = pd.read_excel(bri_in)
    pool_col = find_col(bri, ["burden","pool"]) or find_col(bri, ["pool"])
    year_col = find_col(bri, ["date"]) or find_col(bri, ["year"])
    if not all([pool_col, year_col]):
        raise ValueError("BRI missing Burden Pool / Date or Year")

    bcols = [c for c in bri.columns if c not in {pool_col, year_col, "Description", "Effective Date"}]

    target = overhead_df.copy()
    if not com_df.empty:
        cmap = com_df.set_index(["Pool", "Year"])["COM"].to_dict()
        target["_COM_fill_"] = [cmap.get((p, y)) for p, y in zip(target["Pool"], target["Year"])]

    bri[year_col] = pd.to_numeric(bri[year_col], errors="coerce").astype("Int64")
    idx = {(norm(p), int(y)): r for p, y, r in zip(target["Pool"], target["Year"], target.index)}

    def _apply(row):
        key = (norm(row[pool_col]), int(row[year_col])) if pd.notna(row[year_col]) else None
        if key and key in idx:
            src = target.loc[idx[key]]
            for c in bcols:
                if c in src: row[c] = src[c]
                elif c.upper().startswith("COM") and "_COM_fill_" in src: row[c] = src["_COM_fill_"]
        return row
    bri = bri.apply(_apply, axis=1)

    for c in bcols:
        bri[c] = pd.to_numeric(bri[c], errors="coerce")
        bri[c] = bri[c].round(COM_ROUND if "COM" in str(c).upper() else BURDEN_ROUND)

    bri_out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(bri_out, engine="openpyxl") as xw:
        bri.to_excel(xw, index=False, sheet_name="Burden Rate Import")
    print(f"✅ BurdenRateImport updated → {bri_out.resolve()}")

# ================= Comparison (codes are derived only from BUSINESS UNIT GDLS) =================
def write_rateband_comparison(ratsum_simplified: pd.DataFrame, rbi_path_or_df, out_path: Path):
    """Set diff based on *derived* Code from BUSINESS UNIT GDLS (RATSUM) vs RBI Rate Band column."""
    rbi = pd.read_excel(rbi_path_or_df) if isinstance(rbi_path_or_df, (str, Path)) else rbi_path_or_df
    rb_col = find_col(rbi, ["rate","band"]) or find_col(rbi, ["rateband"])
    if rb_col is None:
        raise ValueError("RBI: cannot find Rate Band column")

    # Build normalized sets
    ratsum_codes = set(ratsum_simplified["Code"].map(normalize_code).dropna())
    rbi_codes = set(rbi[rb_col].dropna().map(normalize_code).dropna())

    only_ratsum = sorted(ratsum_codes - rbi_codes)
    only_rbi    = sorted(rbi_codes - ratsum_codes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        pd.DataFrame(only_ratsum, columns=["RateBand Code"]).to_excel(xw, sheet_name="In_RatSum_only", index=False)
        pd.DataFrame(only_rbi,    columns=["RateBand Code"]).to_excel(xw, sheet_name="In_RateBand_only", index=False)
    print(f"✅ Comparison report → {out_path.resolve()}")

# ================= Main =================
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xls = pd.ExcelFile(RATSUM_PATH)

    simplified = read_simplified(xls)          # Code ← first 2 chars of BUSINESS UNIT GDLS
    act = read_other_rates_act(xls)            # Allowable Control/Contl Test (Rate) by CY#### 
    overhead = read_overhead(xls)              # Burden pools by Year
    com_long = read_com_summary(xls)           # COM by Year

    rbi_out = OUT_DIR / "RateBandImport_updated.xlsx"
    update_rate_band_import(simplified, act, RATE_BAND_IMPORT_PATH, rbi_out)

    bri_out = OUT_DIR / "BurdenRateImport_updated.xlsx"
    update_burden_rate_import(overhead, com_long, BURDEN_RATE_IMPORT_PATH, bri_out)

    write_rateband_comparison(simplified, rbi_out, OUT_DIR / "RateBand_Comparison.xlsx")
    print("🎉 Process complete.")