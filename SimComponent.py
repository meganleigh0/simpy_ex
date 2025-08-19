# ================== Make/Buy Flip Pipeline (header + warnings fixed) ==================
from __future__ import annotations
import re
from glob import glob
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd

# ---------------------------- CONFIG -----------------------------------------
BASE = Path("data")                 # folder containing bronze_boms_<program>
OUT  = Path("mb_output"); OUT.mkdir(exist_ok=True)

# File naming templates (edit if yours differ)
PATTERN = {
    "tc_mbm":     "{base}/bronze_boms_{program}/{program}_mbom_tc_{date}.xlsm",
    "tc_ebom":    "{base}/bronze_boms_{program}/{program}_ebom_tc_{date}.xlsm",
    "oracle_mbm": "{base}/bronze_boms_{program}/{program}_mbom_oracle_{date}.xlsx",
}

# If an Oracle dump truly uses row 1 as header, list its date here (kept for compatibility)
NO_HEADER_DATES = {"02-12-2025","02-20-2025","02-26-2025","03-05-2025","03-17-2025"}

DATE_FMT = "%m-%d-%Y"
DATE_RE  = re.compile(r"_(\d{2}-\d{2}-\d{4})\.(?:xlsx|xlsm)$")

# Canonical header candidates (we compare after lowercasing + stripping non-alnum)
PART_CANDS  = ["PART_NUMBER","Part Number","PART NUMBER","Part_Number","PartNumber","Part No","PartNo"]
DESC_CANDS  = ["Item Name","Description","Part Description","Part Desc","ItemName"]
MABU_CANDS  = ["Make/Buy","Make or Buy","Make or Buy:","MAKE/BUY","MakeBuy"]
LEVEL_CANDS = ["# Level","Level","Levels","Structure Level","Indent","Indented Level","Lvl","#Level","Level #"]

def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())

PART_KEYS  = {_norm(x) for x in PART_CANDS}
DESC_KEYS  = {_norm(x) for x in DESC_CANDS}
MABU_KEYS  = {_norm(x) for x in MABU_CANDS}
LEVEL_KEYS = {_norm(x) for x in LEVEL_CANDS}

# ---------------------------- PATHS / DATES ----------------------------------
def _path_for(program: str, date: str, source: str) -> Path:
    return Path(PATTERN[source].format(base=BASE, program=program, date=date))

def available_dates(program: str, source: str) -> List[str]:
    pat = str(_path_for(program, "******", source)).replace("******", "*")
    dates = []
    for f in glob(pat):
        m = DATE_RE.search(f)
        if m: dates.append(m.group(1))
    return sorted(set(dates), key=lambda d: pd.to_datetime(d, format=DATE_FMT))

def infer_dates(program: str, sources: List[str], mode: str = "union") -> List[str]:
    sets = [set(available_dates(program, s)) for s in sources]
    if not sets: return []
    dates = set.union(*sets) if mode == "union" else set.intersection(*sets)
    return sorted(dates, key=lambda d: pd.to_datetime(d, format=DATE_FMT))

# ---------------------------- HEADER DETECTION --------------------------------
def _detect_header_row(sample: pd.DataFrame) -> Optional[int]:
    """
    Scan the first ~40 rows; consider a row a header if it contains a Part Number
    candidate AND at least one of (Description/MakeBuy/Level).
    """
    limit = min(40, len(sample))
    for i in range(limit):
        vals = [_norm(v) for v in sample.iloc[i].tolist()]
        s = set(vals)
        if (len(s & PART_KEYS) >= 1) and (len(s & (DESC_KEYS | MABU_KEYS | LEVEL_KEYS)) >= 1):
            return i
    return None

# ---------------------------- READERS / NORMALIZATION -------------------------
def _first_non_null_across_cols(df: pd.DataFrame) -> pd.Series:
    """Row-wise first non-null/blank among the provided columns (no applymap/bfill)."""
    arr = df.astype("object").to_numpy()
    out = []
    for row in arr:
        v = pd.NA
        for x in row:
            if x is not None and not (isinstance(x, float) and pd.isna(x)):
                if not (isinstance(x, str) and x.strip() == ""):
                    v = x
                    break
        out.append(v)
    return pd.Series(out, index=df.index, dtype="object")

def _coalesce_dupes(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Collapse duplicate columns named `name` by keeping the first non-null across them."""
    cols = [c for c in df.columns if c == name]
    if len(cols) > 1:
        df[name] = _first_non_null_across_cols(df[cols])
        df.drop(columns=cols[1:], inplace=True)
    return df

def _rename_to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename fuzzily matched headers to canonical names."""
    def _find(cands):
        norm_map = {_norm(c): c for c in raw.columns}
        for cand in cands:
            k = _norm(cand)
            if k in norm_map:
                return norm_map[k]
        return None

    part  = _find(PART_CANDS)
    desc  = _find(DESC_CANDS)
    mabu  = _find(MABU_CANDS)
    level = _find(LEVEL_CANDS)

    ren = {}
    if part:  ren[part]  = "PART_NUMBER"
    if desc:  ren[desc]  = "Description"
    if mabu:  ren[mabu]  = "Make/Buy"
    if level: ren[level] = "Levels"

    df = raw.rename(columns=ren)
    keep = [c for c in ["PART_NUMBER","Description","Make/Buy","Levels"] if c in df.columns]
    if not keep:
        return pd.DataFrame(columns=["PART_NUMBER","Description","Make/Buy","Levels"])
    df = df[keep].copy()

    for c in ["PART_NUMBER","Description","Make/Buy","Levels"]:
        if c in df.columns: df = _coalesce_dupes(df, c)
    return df

def _read_with_detect(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Read any Excel (xlsm/xlsx), detect the right sheet + header row."""
    try:
        xls = pd.ExcelFile(path, engine="openpyxl")
    except Exception as e:
        return None, f"open error: {e}"

    # Try each sheet: read without header first, detect header row, then rebuild with that row as header
    for sh in xls.sheet_names:
        try:
            sample = pd.read_excel(xls, sheet_name=sh, header=None, nrows=50)
        except Exception:
            continue
        hdr = _detect_header_row(sample)
        if hdr is None:
            continue

        # Read the full sheet once (header=None), then set columns from the detected header row
        full = pd.read_excel(xls, sheet_name=sh, header=None)
        header_values = full.iloc[hdr].astype(str).tolist()
        table = full.iloc[hdr+1:].copy()
        table.columns = header_values
        # Drop empty rows
        table = table.dropna(how="all")
        # If we found PART_NUMBER etc. after renaming, return
        canon = _rename_to_canonical(table)
        if "PART_NUMBER" in canon.columns:
            return canon, None

    return None, "could not find a valid header row/columns in any sheet"

def _normalize(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """Clean values and de-duplicate records by PART_NUMBER+Date."""
    if df.empty:
        return df

    # Clean Make/Buy if present
    if "Make/Buy" in df.columns:
        s = (df["Make/Buy"].astype(str).str.strip().str.lower()
             .replace({"m":"make","mk":"make","b":"buy","bu":"buy"}))
        df["Make/Buy"] = s.where(s.isin(["make","buy"]))

    # attach snapshot date
    df["Date"] = pd.to_datetime(date, format=DATE_FMT, errors="coerce")

    # drop blank part numbers
    if "PART_NUMBER" in df.columns:
        mask = df["PART_NUMBER"].astype(str).str.strip() != ""
        df = df.loc[mask]

    if {"PART_NUMBER","Date"}.issubset(df.columns):
        df = (df.sort_values(["PART_NUMBER","Date"])
                .drop_duplicates(["PART_NUMBER","Date"], keep="last"))

    return df.reset_index(drop=True)

# ---------------------------- LOAD / FLIP LOGIC ------------------------------
def _read_one(program: str, date: str, source: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    p = _path_for(program, date, source)
    if not p.exists():
        return None, "missing file"

    # robust read (sheet+header detection)
    raw, why = _read_with_detect(p)
    if raw is None:
        return None, why

    df = _normalize(raw, date)
    need = [c for c in ["PART_NUMBER","Make/Buy"] if c not in df.columns]
    if need:
        return None, f"missing required columns after normalize: {need}"
    if df.empty:
        return None, "no usable rows after normalize"
    return df, None

def load_boms(program: str, dates, sources, align_sources: bool=False) -> pd.DataFrame:
    if dates == "auto" or dates is None:
        dates = infer_dates(program, sources, mode="intersection" if align_sources else "union")

    skipped, frames = [], []
    for src in sources:
        for d in dates:
            df, reason = _read_one(program, d, src)
            if df is None:
                skipped.append((src, d, reason))
                continue
            df["Source"] = src
            frames.append(df)

    if skipped:
        note = " ; ".join(f"{s}:{d} -> {r}" for s, d, r in skipped)
        print(f"⚠️  Skipped {len(skipped)} snapshots: {note}")

    if not frames:
        # expected schema to avoid downstream key errors
        return pd.DataFrame(columns=["PART_NUMBER","Description","Levels","Make/Buy","Date","Source"])

    all_df = (pd.concat(frames, ignore_index=True)
                .sort_values(["Source","PART_NUMBER","Date"])
                .drop_duplicates(["Source","PART_NUMBER","Date"], keep="last")
                .reset_index(drop=True))
    return all_df

def detect_flips(all_df: pd.DataFrame):
    if all_df.empty:
        return all_df, all_df, all_df

    all_df = all_df.sort_values(["Source","PART_NUMBER","Date"])
    all_df["previous_status"] = all_df.groupby(["Source","PART_NUMBER"])["Make/Buy"].shift()

    mask = (all_df["Make/Buy"].notna() &
            all_df["previous_status"].notna() &
            (all_df["Make/Buy"] != all_df["previous_status"]))

    flip_log = (all_df.loc[mask, ["Source","PART_NUMBER","Description","Levels","Date","previous_status","Make/Buy"]]
                      .rename(columns={"Make/Buy":"new_status"})
                      .sort_values(["Source","PART_NUMBER","Date"])
                      .reset_index(drop=True))

    snapshot_summary = (flip_log.groupby(["Source","Date"])["PART_NUMBER"]
                        .nunique()
                        .rename("# num_parts_changed")
                        .reset_index()
                        .sort_values(["Source","Date"]))

    per_part_counts = (flip_log.groupby(["Source","PART_NUMBER"])
                       .size()
                       .rename("num_flips")
                       .reset_index()
                       .sort_values(["Source","num_flips","PART_NUMBER"], ascending=[True,False,True]))

    return flip_log, snapshot_summary, per_part_counts

def write_outputs(program: str, flip_log, snapshot_summary, per_part_counts):
    out = OUT / program
    out.mkdir(parents=True, exist_ok=True)
    flip_log.to_csv(out / "make_buy_flip_log.csv", index=False)
    snapshot_summary.to_csv(out / "make_buy_flip_summary_by_date.csv", index=False)
    per_part_counts.to_csv(out / "make_buy_flip_counts_by_part.csv", index=False)
    print(f"✅ Wrote outputs to {out}")

# --------------------------------- RUN ---------------------------------------
if __name__ == "__main__":
    program = "cuas"
    sources = ["tc_mbm"]      # choose: "tc_mbm", "tc_ebom", "oracle_mbm"
    dates   = "auto"          # or supply a list like ["03-05-2025","03-17-2025",...]

    all_boms = load_boms(program, dates, sources, align_sources=False)
    flip_log, snapshot_summary, per_part_counts = detect_flips(all_boms)
    write_outputs(program, flip_log, snapshot_summary, per_part_counts)

    # quick peek
    print(flip_log.head(10))
    print(snapshot_summary.tail(10))
    print(per_part_counts.head(10))