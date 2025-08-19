# ======================= Make/Buy Flip Pipeline (robust) ======================
# Requirements: pandas, openpyxl
from __future__ import annotations
import re, warnings
from glob import glob
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

# Silence noisy engine warnings (we avoid deprecated behavior below)
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ------------------------------ CONFIG ---------------------------------------
BASE = Path("data")                 # folder that contains bronze_boms_<program>
OUT  = Path("mb_output"); OUT.mkdir(exist_ok=True)

# File naming templates (adjust if yours differ)
PATTERN = {
    "tc_mbm":     "{base}/bronze_boms_{program}/{program}_mbom_tc_{date}.xlsm",
    "tc_ebom":    "{base}/bronze_boms_{program}/{program}_ebom_tc_{date}.xlsm",
    "oracle_mbm": "{base}/bronze_boms_{program}/{program}_mbom_oracle_{date}.xlsx",
}

# Oracle snapshots that already have headers on row 1 (no offset)
NO_HEADER_DATES = {"02-12-2025","02-20-2025","02-26-2025","03-05-2025","03-17-2025"}

DATE_FMT = "%m-%d-%Y"
DATE_RE  = re.compile(r"_(\d{2}-\d{2}-\d{4})\.(?:xlsx|xlsm)$")

# Candidate column names
PART_CANDS  = ["PART_NUMBER","Part Number","PART NUMBER","Part_Number","PartNumber","Part No","PartNo"]
DESC_CANDS  = ["Item Name","Description","Part Description","Part Desc","ItemName"]
MABU_CANDS  = ["Make/Buy","Make or Buy","Make or Buy:","MAKE/BUY","MakeBuy"]
LEVEL_CANDS = ["# Level","Level","Levels","Structure Level","Indent","Indented Level","Lvl","#Level","Level #"]

# ------------------------------ PATH / DATES ---------------------------------
def _path_for(program: str, date: str, source: str) -> Path:
    return Path(PATTERN[source].format(base=BASE, program=program, date=date))

def available_dates(program: str, source: str) -> list[str]:
    pat = str(_path_for(program, "******", source)).replace("******", "*")
    dates = []
    for f in glob(pat):
        m = DATE_RE.search(f)
        if m: dates.append(m.group(1))
    return sorted(set(dates), key=lambda d: pd.to_datetime(d, format=DATE_FMT))

def infer_dates(program: str, sources: list[str], mode: str = "union") -> list[str]:
    sets = [set(available_dates(program, s)) for s in sources]
    if not sets: return []
    dates = set.union(*sets) if mode == "union" else set.intersection(*sets)
    return sorted(dates, key=lambda d: pd.to_datetime(d, format=DATE_FMT))

# ------------------------------ HELPERS --------------------------------------
def _find_col(cols, candidates):
    norm = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in cols}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        if key in norm: return norm[key]
    return None

def _coalesce_dupes_no_warning(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """If duplicate canonical names exist, keep the first non-null without using bfill/ffill."""
    same = [c for c in df.columns if c == col]
    if len(same) > 1:
        block = df[same].astype("object")
        # Treat empty strings as NA
        block = block.mask(block.applymap(lambda x: isinstance(x, str) and x.strip() == ""))
        # First non-null value per row (leftmost)
        first_vals = block.apply(lambda r: next((v for v in r if pd.notna(v)), pd.NA), axis=1)
        df[col] = first_vals
        df.drop(columns=same[1:], inplace=True)
    return df

def _detect_header_row(sample: pd.DataFrame) -> Optional[int]:
    """Find a row likely to be the header by searching for a Part Number candidate."""
    for i in range(min(30, len(sample))):  # scan top 30 rows
        row_vals = [str(v) for v in sample.iloc[i].tolist()]
        if any(re.sub(r"[^a-z0-9]", "", v.lower()) in
               {re.sub(r"[^a-z0-9]", "", c.lower()) for c in PART_CANDS}
               for v in row_vals):
            return i
    return None

def _read_excel_dynamic(path: Path, oracle_source: bool, date: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Robust Excel reader:
    - If Oracle and date not in NO_HEADER_DATES -> try header=5 first
    - Else: auto-detect header row & sheet by scanning top rows for Part Number
    Returns (DataFrame, None) or (None, reason)
    """
    try:
        xls = pd.ExcelFile(path, engine="openpyxl")
    except Exception as e:
        return None, f"open error: {e}"

    # candidate (sheet, header_row) pairs to try
    attempts = []
    # 1) If we know Oracle offset, try that first on first sheet
    if oracle_source and date not in NO_HEADER_DATES:
        attempts.append((xls.sheet_names[0], 5))

    # 2) For each sheet, try auto-detect header row
    for sh in xls.sheet_names:
        sample = pd.read_excel(xls, sheet_name=sh, nrows=30, header=None)
        hdr = _detect_header_row(sample)
        if hdr is not None:
            attempts.append((sh, hdr))

    # 3) Fallback: first sheet, header=0
    attempts.append((xls.sheet_names[0], 0))

    # Try attempts until one yields a frame with a recognizable part column
    for sh, hdr in attempts:
        try:
            df = pd.read_excel(xls, sheet_name=sh, header=hdr)
        except Exception:
            continue
        # If this looks like a real table, return it
        if any(_find_col(df.columns, PART_CANDS) for _ in [0]):
            return df, None

    return None, "could not locate a header row with a Part Number column"

def _normalize(raw: pd.DataFrame, date: str) -> pd.DataFrame:
    """Rename to canonical names, collapse dups, clean values, and de-dupe."""
    part  = _find_col(raw.columns, PART_CANDS)
    desc  = _find_col(raw.columns, DESC_CANDS)
    mabu  = _find_col(raw.columns, MABU_CANDS)
    level = _find_col(raw.columns, LEVEL_CANDS)

    rename = {}
    if part:  rename[part]  = "PART_NUMBER"
    if desc:  rename[desc]  = "Description"
    if mabu:  rename[mabu]  = "Make/Buy"
    if level: rename[level] = "Levels"

    df = raw.rename(columns=rename)

    keep = [c for c in ["PART_NUMBER","Description","Make/Buy","Levels"] if c in df.columns]
    if not keep:
        return pd.DataFrame(columns=["PART_NUMBER","Description","Make/Buy","Levels","Date"])

    df = df[keep].copy()

    # Collapse duplicates safely (no FutureWarnings)
    for col in ["PART_NUMBER","Description","Make/Buy","Levels"]:
        if col in df.columns:
            df = _coalesce_dupes_no_warning(df, col)

    # Clean Make/Buy
    if "Make/Buy" in df.columns:
        df["Make/Buy"] = (
            df["Make/Buy"]
            .astype(str).str.strip().str.lower()
            .replace({
                "m":"make","b":"buy",
                "make":"make","buy":"buy",
                "mk":"make","bu":"buy"
            })
            .where(df["Make/Buy"].isin(["make","buy"]))
        )

    df["Date"] = pd.to_datetime(date, format=DATE_FMT, errors="coerce")

    # Remove blank parts and de-dupe per part/date
    if "PART_NUMBER" in df.columns:
        df = df[df["PART_NUMBER"].astype(str).str.strip().ne("")]
    df = (df.sort_values([c for c in ["PART_NUMBER","Date"] if c in df.columns])
            .drop_duplicates([c for c in ["PART_NUMBER","Date"] if c in df.columns], keep="last")
            .reset_index(drop=True))
    return df

# ------------------------------ IO / LOADING ---------------------------------
def _read_one(program: str, date: str, source: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    path = _path_for(program, date, source)
    if not path.exists():
        return None, "missing file"

    oracle = (source == "oracle_mbm")
    raw, why = _read_excel_dynamic(path, oracle_source=oracle, date=date)
    if raw is None:
        return None, why

    df = _normalize(raw, date)
    # Require PART_NUMBER + Make/Buy for flip detection
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
    for source in sources:
        for d in dates:
            df, reason = _read_one(program, d, source)
            if df is None:
                skipped.append((source, d, reason))
                continue
            df["Source"] = source
            frames.append(df)

    if skipped:
        msg = " ; ".join([f"{s}:{d} -> {r}" for s, d, r in skipped])
        print(f"⚠️  Skipped {len(skipped)} snapshots: {msg}")

    if not frames:
        return pd.DataFrame(columns=["PART_NUMBER","Description","Levels","Make/Buy","Date","Source"])

    all_df = (pd.concat(frames, ignore_index=True)
                .sort_values(["Source","PART_NUMBER","Date"])
                .drop_duplicates(["Source","PART_NUMBER","Date"], keep="last")
                .reset_index(drop=True))
    return all_df

# ------------------------------ FLIP DETECTION -------------------------------
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
                        .nunique().rename("# num_parts_changed").reset_index()
                        .sort_values(["Source","Date"]))

    per_part_counts = (flip_log.groupby(["Source","PART_NUMBER"])
                       .size().rename("num_flips").reset_index()
                       .sort_values(["Source","num_flips","PART_NUMBER"], ascending=[True,False,True]))

    return flip_log, snapshot_summary, per_part_counts

# ------------------------------ OUTPUTS --------------------------------------
def write_outputs(program: str, flip_log, snapshot_summary, per_part_counts):
    out_dir = OUT / program
    out_dir.mkdir(parents=True, exist_ok=True)
    flip_log.to_csv(out_dir / "make_buy_flip_log.csv", index=False)
    snapshot_summary.to_csv(out_dir / "make_buy_flip_summary_by_date.csv", index=False)
    per_part_counts.to_csv(out_dir / "make_buy_flip_counts_by_part.csv", index=False)
    print(f"✅ Wrote outputs to {out_dir}")

# ------------------------------ RUN EXAMPLE ----------------------------------
if __name__ == "__main__":
    program = "cuas"
    sources = ["tc_mbm"]            # choose from: "tc_mbm", "tc_ebom", "oracle_mbm"

    # A) auto-discover valid dates for the chosen sources
    dates = "auto"                  # set align_sources=True to require common dates across sources
    all_boms = load_boms(program, dates, sources, align_sources=False)

    # B) or specify dates explicitly (skips any that don't exist / normalize)
    # dates = ["03-05-2025","03-17-2025","04-02-2025","04-09-2025","04-16-2025","06-30-2025"]
    # all_boms = load_boms(program, dates, sources)

    flip_log, snapshot_summary, per_part_counts = detect_flips(all_boms)
    write_outputs(program, flip_log, snapshot_summary, per_part_counts)

    # quick peek
    print(flip_log.head(10))
    print(snapshot_summary.tail(10))
    print(per_part_counts.head(10))