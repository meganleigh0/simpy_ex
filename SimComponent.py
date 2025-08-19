# --- Make/Buy flip pipeline (with auto date discovery & safe missing-file handling)
import re
from glob import glob
from pathlib import Path
import pandas as pd

BASE = Path("data")
OUT  = Path("mb_output"); OUT.mkdir(exist_ok=True)

PATTERN = {
    "tc_mbm":     "{base}/bronze_boms_{program}/{program}_mbom_tc_{date}.xlsm",
    "tc_ebom":    "{base}/bronze_boms_{program}/{program}_ebom_tc_{date}.xlsm",
    "oracle_mbm": "{base}/bronze_boms_{program}/{program}_mbom_oracle_{date}.xlsx",
}

# Oracle files that DO NOT need header offset (i.e., use header=0)
NO_HEADER_DATES = {"02-12-2025","02-20-2025","02-26-2025","03-05-2025","03-17-2025"}

DATE_RE = re.compile(r"_(\d{2}-\d{2}-\d{4})\.(?:xlsx|xlsm)$")

def _path_for(program:str, date:str, source:str)->Path:
    return Path(PATTERN[source].format(base=BASE, program=program, date=date))

def available_dates(program: str, source: str) -> list[str]:
    """Find dates that actually exist on disk for a given source."""
    pat = str(_path_for(program, "******", source)).replace("******", "*")
    dates = []
    for f in glob(pat):
        m = DATE_RE.search(f)
        if m: dates.append(m.group(1))
    dates = sorted(set(dates), key=lambda d: pd.to_datetime(d, format="%m-%d-%Y"))
    return dates

def infer_dates(program: str, sources: list[str], mode: str = "union") -> list[str]:
    """mode='union' (default) or 'intersection' across the listed sources."""
    sets = [set(available_dates(program, s)) for s in sources]
    if not sets: return []
    dates = set.intersection(*sets) if mode == "intersection" else set.union(*sets)
    return sorted(dates, key=lambda d: pd.to_datetime(d, format="%m-%d-%Y"))

def _find_col(cols, candidates):
    norm = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in cols}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        if key in norm: return norm[key]
    return None

def _normalize(df: pd.DataFrame, date: str) -> pd.DataFrame:
    part  = _find_col(df.columns, ["PART_NUMBER","Part Number","PART NUMBER","Part_Number"])
    desc  = _find_col(df.columns, ["Item Name","Description","Part Description"])
    mabu  = _find_col(df.columns, ["Make/Buy","Make or Buy","Make or Buy:"])
    level = _find_col(df.columns, ["# Level","Level","Levels","Structure Level"])

    rename = {}
    if part:  rename[part]  = "PART_NUMBER"
    if desc:  rename[desc]  = "Description"
    if mabu:  rename[mabu]  = "Make/Buy"
    if level: rename[level] = "Levels"

    df = df.rename(columns=rename)
    keep = [c for c in ["PART_NUMBER","Description","Make/Buy","Levels"] if c in df.columns]
    df = df[keep].copy()

    if "Make/Buy" in df.columns:
        df["Make/Buy"] = (df["Make/Buy"].astype(str).str.strip().str.lower()
                          .replace({"m":"make","b":"buy"})
                          .where(df["Make/Buy"].isin(["make","buy"])))
    df["Date"] = pd.to_datetime(date, format="%m-%d-%Y", errors="coerce")
    df = df[df["PART_NUMBER"].astype(str).str.strip().ne("")]
    df = (df.sort_values(["PART_NUMBER","Date"])
            .drop_duplicates(["PART_NUMBER","Date"], keep="last")
            .reset_index(drop=True))
    return df

def _read_one(program: str, date: str, source: str) -> pd.DataFrame:
    path = _path_for(program, date, source)
    if not path.exists():
        return None  # handled upstream
    header = 0
    if source == "oracle_mbm" and date not in NO_HEADER_DATES:
        header = 5
    raw = pd.read_excel(path, engine="openpyxl", header=header)
    return _normalize(raw, date)

def load_boms(program: str, dates, sources, intersection: bool=False) -> pd.DataFrame:
    """dates: list[str] or 'auto'. If 'auto', discover dates (intersection=False -> union)."""
    if dates == "auto" or dates is None:
        dates = infer_dates(program, sources, mode="intersection" if intersection else "union")

    missing = []
    frames = []
    for source in sources:
        for d in dates:
            df = _read_one(program, d, source)
            if df is None:
                missing.append((source, d))
                continue
            df["Source"] = source
            frames.append(df)

    if missing:
        # helpful console note, not an exception
        miss_txt = ", ".join([f"{s}:{d}" for s, d in missing])
        print(f"⚠️ Skipped missing files ({len(missing)}): {miss_txt}")

    if not frames:
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
                        .nunique().rename("# num_parts_changed").reset_index()
                        .sort_values(["Source","Date"]))
    per_part_counts = (flip_log.groupby(["Source","PART_NUMBER"])
                       .size().rename("num_flips").reset_index()
                       .sort_values(["Source","num_flips"], ascending=[True,False]))
    return flip_log, snapshot_summary, per_part_counts

def write_outputs(program: str, flip_log, snapshot_summary, per_part_counts):
    out_dir = OUT / program; out_dir.mkdir(parents=True, exist_ok=True)
    flip_log.to_csv(out_dir / "make_buy_flip_log.csv", index=False)
    snapshot_summary.to_csv(out_dir / "make_buy_flip_summary_by_date.csv", index=False)
    per_part_counts.to_csv(out_dir / "make_buy_flip_counts_by_part.csv", index=False)
    print(f"✅ Wrote outputs to {out_dir}")

# ------------------------- Example run ----------------------------------------
if __name__ == "__main__":
    program = "cuas"
    # OPTION 1: let the code discover dates for the chosen sources:
    sources = ["tc_mbm"]                 # e.g., ["tc_mbm"], or ["tc_mbm","tc_ebom"], or ["oracle_mbm"]
    dates = "auto"                       # auto = discover (union). Use intersection=True to align sources.
    # OPTION 2: or give an explicit list and it will skip any missing safely:
    # dates = ["03-05-2025","03-17-2025","03-31-2025","04-02-2025","04-09-2025","04-16-2025","06-30-2025"]

    all_boms = load_boms(program, dates, sources, intersection=False)
    flip_log, snapshot_summary, per_part_counts = detect_flips(all_boms)
    write_outputs(program, flip_log, snapshot_summary, per_part_counts)

    print(flip_log.head(10))
    print(snapshot_summary.tail(10))
    print(per_part_counts.head(10))