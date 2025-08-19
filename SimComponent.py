# ======================= Make/Buy Flip Pipeline ===============================
# Requirements: pandas, openpyxl
import re
from glob import glob
from pathlib import Path
import pandas as pd

# ------------------------------ CONFIG ---------------------------------------
BASE = Path("data")                 # root folder that contains bronze_boms_<program>
OUT  = Path("mb_output"); OUT.mkdir(exist_ok=True)

# File naming templates (adjust if yours differ)
PATTERN = {
    "tc_mbm":     "{base}/bronze_boms_{program}/{program}_mbom_tc_{date}.xlsm",
    "tc_ebom":    "{base}/bronze_boms_{program}/{program}_ebom_tc_{date}.xlsm",
    "oracle_mbm": "{base}/bronze_boms_{program}/{program}_mbom_oracle_{date}.xlsx",
}

# Oracle snapshots that already have the real header on row 1 (no offset)
NO_HEADER_DATES = {"02-12-2025","02-20-2025","02-26-2025","03-05-2025","03-17-2025"}

DATE_FMT = "%m-%d-%Y"
DATE_RE  = re.compile(r"_(\d{2}-\d{2}-\d{4})\.(?:xlsx|xlsm)$")

# ------------------------------ PATH / DATES ---------------------------------
def _path_for(program: str, date: str, source: str) -> Path:
    return Path(PATTERN[source].format(base=BASE, program=program, date=date))

def available_dates(program: str, source: str) -> list[str]:
    """List dates that actually exist on disk for a given source."""
    pat = str(_path_for(program, "******", source)).replace("******", "*")
    dates = []
    for f in glob(pat):
        m = DATE_RE.search(f)
        if m: dates.append(m.group(1))
    dates = sorted(set(dates), key=lambda d: pd.to_datetime(d, format=DATE_FMT))
    return dates

def infer_dates(program: str, sources: list[str], mode: str = "union") -> list[str]:
    """Discover dates across sources. mode: 'union' or 'intersection'."""
    sets = [set(available_dates(program, s)) for s in sources]
    if not sets: return []
    dates = set.union(*sets) if mode == "union" else set.intersection(*sets)
    return sorted(dates, key=lambda d: pd.to_datetime(d, format=DATE_FMT))

# ------------------------------ NORMALIZATION --------------------------------
def _find_col(cols, candidates):
    """Case/space/punct-insensitive match; return the actual column name."""
    norm = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in cols}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        if key in norm: return norm[key]
    return None

def _coalesce_dupes(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """If duplicate column names exist, keep the first non-null across them."""
    dup_mask = (df.columns == col)
    if dup_mask.sum() > 1:
        block = df.loc[:, dup_mask]
        df[col] = block.bfill(axis=1).iloc[:, 0]
        df = df.drop(columns=block.columns[1:])
    return df

def _normalize(raw: pd.DataFrame, date: str) -> pd.DataFrame:
    """Rename to canonical names, collapse duplicates, clean values."""
    part  = _find_col(raw.columns, ["PART_NUMBER","Part Number","PART NUMBER","Part_Number"])
    desc  = _find_col(raw.columns, ["Item Name","Description","Part Description"])
    mabu  = _find_col(raw.columns, ["Make/Buy","Make or Buy","Make or Buy:"])
    level = _find_col(raw.columns, ["# Level","Level","Levels","Structure Level"])

    rename = {}
    if part:  rename[part]  = "PART_NUMBER"
    if desc:  rename[desc]  = "Description"
    if mabu:  rename[mabu]  = "Make/Buy"
    if level: rename[level] = "Levels"

    df = raw.rename(columns=rename)

    # keep only the columns we need (if present)
    keep = [c for c in ["PART_NUMBER","Description","Make/Buy","Levels"] if c in df.columns]
    df = df[keep].copy()

    # collapse any duplicated names produced by the rename step
    for col in ["PART_NUMBER","Description","Make/Buy","Levels"]:
        if col in df.columns:
            df = _coalesce_dupes(df, col)

    # clean values
    if "Make/Buy" in df.columns:
        df["Make/Buy"] = (
            df["Make/Buy"]
            .astype(str).str.strip().str.lower()
            .replace({"m": "make", "b": "buy"})
            .where(df["Make/Buy"].isin(["make","buy"]))
        )
    df["Date"] = pd.to_datetime(date, format=DATE_FMT, errors="coerce")

    # remove blanks & de-dupe by part/date
    if "PART_NUMBER" in df.columns:
        df = df[df["PART_NUMBER"].astype(str).str.strip().ne("")]
    df = (df.sort_values(["PART_NUMBER","Date"])
            .drop_duplicates(["PART_NUMBER","Date"], keep="last")
            .reset_index(drop=True))
    return df

# ------------------------------ IO / LOADING ---------------------------------
def _read_one(program: str, date: str, source: str) -> pd.DataFrame | None:
    """Read one snapshot; return None if file missing (handled upstream)."""
    path = _path_for(program, date, source)
    if not path.exists():
        return None
    header = 0
    if source == "oracle_mbm" and date not in NO_HEADER_DATES:
        header = 5  # historical Oracle dumps with title rows above headers
    raw = pd.read_excel(path, engine="openpyxl", header=header)
    return _normalize(raw, date)

def load_boms(program: str, dates, sources, align_sources: bool=False) -> pd.DataFrame:
    """
    dates: list[str] of 'MM-DD-YYYY' or 'auto'.
    align_sources=False -> union of available dates; True -> intersection.
    """
    if dates == "auto" or dates is None:
        dates = infer_dates(program, sources, mode="intersection" if align_sources else "union")

    missing, frames = [], []
    for source in sources:
        for d in dates:
            df = _read_one(program, d, source)
            if df is None:
                missing.append((source, d))
                continue
            df["Source"] = source
            frames.append(df)

    if missing:
        txt = ", ".join([f"{s}:{d}" for s, d in missing])
        print(f"⚠️  Skipped missing files ({len(missing)}): {txt}")

    if not frames:
        return pd.DataFrame(columns=["PART_NUMBER","Description","Levels","Make/Buy","Date","Source"])

    all_df = (pd.concat(frames, ignore_index=True)
                .sort_values(["Source","PART_NUMBER","Date"])
                .drop_duplicates(["Source","PART_NUMBER","Date"], keep="last")
                .reset_index(drop=True))
    return all_df

# ------------------------------ FLIP DETECTION -------------------------------
def detect_flips(all_df: pd.DataFrame):
    """Return: flip_log, per-date summary, per-part flip counts."""
    if all_df.empty:
        return all_df, all_df, all_df

    all_df = all_df.sort_values(["Source","PART_NUMBER","Date"])
    all_df["previous_status"] = all_df.groupby(["Source","PART_NUMBER"])["Make/Buy"].shift()

    mask = (
        all_df["Make/Buy"].notna()
        & all_df["previous_status"].notna()
        & (all_df["Make/Buy"] != all_df["previous_status"])
    )

    flip_log = (
        all_df.loc[mask, ["Source","PART_NUMBER","Description","Levels","Date","previous_status","Make/Buy"]]
              .rename(columns={"Make/Buy":"new_status"})
              .sort_values(["Source","PART_NUMBER","Date"])
              .reset_index(drop=True)
    )

    snapshot_summary = (
        flip_log.groupby(["Source","Date"])["PART_NUMBER"]
                .nunique()
                .rename("# num_parts_changed")
                .reset_index()
                .sort_values(["Source","Date"])
    )

    per_part_counts = (
        flip_log.groupby(["Source","PART_NUMBER"])
                .size()
                .rename("num_flips")
                .reset_index()
                .sort_values(["Source","num_flips","PART_NUMBER"], ascending=[True,False,True])
    )
    return flip_log, snapshot_summary, per_part_counts

# ------------------------------ OUTPUTS --------------------------------------
def write_outputs(program: str, flip_log, snapshot_summary, per_part_counts):
    out_dir = OUT / program
    out_dir.mkdir(parents=True, exist_ok=True)
    flip_log.to_csv(out_dir / "make_buy_flip_log.csv", index=False)
    snapshot_summary.to_csv(out_dir / "make_buy_flip_summary_by_date.csv", index=False)
    per_part_counts.to_csv(out_dir / "make_buy_flip_counts_by_part.csv", index=False)
    print(f"✅ Wrote outputs to {out_dir}")

# ------------------------------ RUN EXAMPLES ---------------------------------
if __name__ == "__main__":
    # Pick your program and sources
    program = "cuas"
    sources = ["tc_mbm"]                 # any of: ["tc_mbm","tc_ebom","oracle_mbm"]

    # Option A: auto-discover dates that exist for these sources
    dates = "auto"                       # use align_sources=True to require common dates only
    all_boms = load_boms(program, dates, sources, align_sources=False)

    # Option B: or specify explicit dates (skips any missing with a warning)
    # dates = ["03-05-2025","03-17-2025","04-02-2025","04-09-2025","04-16-2025","06-30-2025"]
    # all_boms = load_boms(program, dates, sources)

    flip_log, snapshot_summary, per_part_counts = detect_flips(all_boms)
    write_outputs(program, flip_log, snapshot_summary, per_part_counts)

    # quick peek
    print(flip_log.head(10))
    print(snapshot_summary.tail(10))
    print(per_part_counts.head(10))