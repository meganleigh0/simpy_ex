# --- Make/Buy flip pipeline ---------------------------------------------------
# Requires: pandas (and openpyxl for .xlsx/.xlsm)
import re
from pathlib import Path
import pandas as pd

# _______________________________ CONFIG _______________________________________
BASE = Path("data")                                  # root folder
OUT  = Path("mb_output"); OUT.mkdir(exist_ok=True)

# file naming patterns by source (edit here if your names differ)
PATTERN = {
    "tc_mbm":      "{base}/bronze_boms_{program}/{program}_mbom_tc_{date}.xlsm",
    "tc_ebom":     "{base}/bronze_boms_{program}/{program}_ebom_tc_{date}.xlsm",
    "oracle_mbm":  "{base}/bronze_boms_{program}/{program}_mbom_oracle_{date}.xlsx",
}

# dates where Oracle files already have correct header row (skip header=5)
NO_HEADER_DATES = set([
    "02-12-2025","02-20-2025","02-26-2025","03-05-2025","03-17-2025"
])

# ___________________________ HELPER FUNCTIONS _________________________________
def _path_for(program:str, date:str, source:str)->Path:
    return Path(PATTERN[source].format(base=BASE, program=program, date=date))

def _find_col(cols, candidates):
    """Return the first matching column name (case/space/punct-insensitive)."""
    norm = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in cols}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        if key in norm:
            return norm[key]
    return None

def _normalize(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """Rename to canonical columns and clean values."""
    part  = _find_col(df.columns, ["PART_NUMBER","Part Number","PART NUMBER","Part_Number"])
    desc  = _find_col(df.columns, ["Item Name","Description","Part Description"])
    mabu  = _find_col(df.columns, ["Make/Buy","Make or Buy","Make or Buy:"])
    level = _find_col(df.columns, ["# Level","Level","Levels","Structure Level"])

    rename_map = {}
    if part:  rename_map[part]  = "PART_NUMBER"
    if desc:  rename_map[desc]  = "Description"
    if mabu:  rename_map[mabu]  = "Make/Buy"
    if level: rename_map[level] = "Levels"

    df = df.rename(columns=rename_map)
    keep = [c for c in ["PART_NUMBER","Description","Make/Buy","Levels"] if c in df.columns]
    df = df[keep].copy()

    # Clean Make/Buy and Date
    if "Make/Buy" in df.columns:
        df["Make/Buy"] = (
            df["Make/Buy"]
            .astype(str).str.strip().str.lower()
            .replace({"m":"make","b":"buy"})
            .where(df["Make/Buy"].isin(["make","buy"]))
        )
    df["Date"] = pd.to_datetime(date, format="%m-%d-%Y", errors="coerce")

    # Drop fully empty PART_NUMBERs, and de-dupe by part/date
    df = df[df["PART_NUMBER"].astype(str).str.strip().ne("")]
    df = df.sort_values(["PART_NUMBER","Date"]).drop_duplicates(["PART_NUMBER","Date"], keep="last")
    return df.reset_index(drop=True)

def _read_one(program: str, date: str, source: str) -> pd.DataFrame:
    """Read one snapshot robustly, handling Oracle header offsets."""
    path = _path_for(program, date, source)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    header = 0
    if source == "oracle_mbm" and date not in NO_HEADER_DATES:
        header = 5  # historical Oracle dumps with title rows above headers

    df = pd.read_excel(path, engine="openpyxl", header=header)
    return _normalize(df, date)

def load_boms(program: str, dates: list[str], sources: list[str]) -> pd.DataFrame:
    """Load and stack snapshots from selected sources."""
    frames = []
    for source in sources:
        for d in dates:
            df = _read_one(program, d, source)
            df["Source"] = source
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["PART_NUMBER","Description","Levels","Make/Buy","Date","Source"])
    all_df = pd.concat(frames, ignore_index=True)
    # If same part/date appears in multiple files of same source, keep last
    all_df = (all_df
              .sort_values(["Source","PART_NUMBER","Date"])
              .drop_duplicates(["Source","PART_NUMBER","Date"], keep="last")
              .reset_index(drop=True))
    return all_df

def detect_flips(all_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return flip_log, per-date summary, and per-part flip counts."""
    if all_df.empty:
        return all_df, all_df, all_df

    # For each Source separately
    all_df = all_df.sort_values(["Source","PART_NUMBER","Date"])
    all_df["prev_status"] = all_df.groupby(["Source","PART_NUMBER"])["Make/Buy"].shift()

    mask = (
        all_df["Make/Buy"].notna() &
        all_df["prev_status"].notna() &
        (all_df["Make/Buy"] != all_df["prev_status"])
    )

    flip_log = (all_df.loc[mask, ["Source","PART_NUMBER","Description","Levels","Date","prev_status","Make/Buy"]]
                    .rename(columns={"prev_status":"previous_status", "Make/Buy":"new_status"})
                    .sort_values(["Source","PART_NUMBER","Date"])
                    .reset_index(drop=True))

    snapshot_summary = (flip_log
                        .groupby(["Source","Date"])["PART_NUMBER"]
                        .nunique()
                        .rename("# num_parts_changed")
                        .reset_index()
                        .sort_values(["Source","Date"]))

    per_part_counts = (flip_log
                       .groupby(["Source","PART_NUMBER"])  # how often each part flips
                       .size()
                       .rename("num_flips")
                       .reset_index()
                       .sort_values(["Source","num_flips","PART_NUMBER"], ascending=[True,False,True]))

    return flip_log, snapshot_summary, per_part_counts

def write_outputs(program: str, flip_log: pd.DataFrame,
                  snapshot_summary: pd.DataFrame, per_part_counts: pd.DataFrame):
    out_dir = OUT / program
    out_dir.mkdir(parents=True, exist_ok=True)
    flip_log.to_csv(out_dir / "make_buy_flip_log.csv", index=False)
    snapshot_summary.to_csv(out_dir / "make_buy_flip_summary_by_date.csv", index=False)
    per_part_counts.to_csv(out_dir / "make_buy_flip_counts_by_part.csv", index=False)
    print(f"✅ Wrote outputs to {out_dir}")

# _________________________________ RUN ________________________________________
if __name__ == "__main__":
    # EXAMPLE: choose your program, dates, and sources
    program = "cuas"
    dates   = ["03-05-2025","03-17-2025","03-31-2025","04-02-2025","04-09-2025",
               "04-16-2025","04-22-2025","06-30-2025","07-07-2025","08-04-2025","08-11-2025"]
    sources = ["tc_mbm"]                     # any of: ["tc_mbm","tc_ebom","oracle_mbm"]

    all_boms = load_boms(program, dates, sources)
    flip_log, snapshot_summary, per_part_counts = detect_flips(all_boms)
    write_outputs(program, flip_log, snapshot_summary, per_part_counts)

    # (optional) quick peek
    print(flip_log.head(10))
    print(snapshot_summary.tail(10))
    print(per_part_counts.head(10))