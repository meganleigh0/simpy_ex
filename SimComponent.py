# ONE-RUN = ONE PROGRAM + ONE BOM SET (oracle | tc_mbom | tc_ebom)
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("data/bronze_boms")

# --- configure just once ---
NO_HEADER_DATES = set([
    "02-12-2025","02-20-2025","02-26-2025","03-05-2025","03-17-2025"  # your Oracle exceptions
])

COLMAP = {
    "Part Number": "PART_NUMBER", "PART_NUMBER": "PART_NUMBER", "Part Number*": "PART_NUMBER",
    "Part-Number": "PART_NUMBER", "PART_NUMBER.": "PART_NUMBER",
    "Item Name": "Description", "ITEM_NAME": "Description",
    "Make or Buy": "Make/Buy", "Make/Buy": "Make/Buy", "Make / Buy": "Make/Buy",
    "MAKE_OR_BUY": "Make/Buy", "Make /Buy": "Make/Buy", "Make/ Buy": "Make/Buy",
    "Level": "Levels", "# Level": "Levels", "# Levels": "Levels", "LEVEL": "Levels",
}
KEEP = ["PART_NUMBER","Description","Make/Buy","Levels","Date"]

def _paths(program: str, date: str, bom_set: str) -> Path:
    if bom_set == "oracle":
        return BASE / program / f"{program}_mbom_oracle_{date}.xlsx"
    if bom_set == "tc_mbom":
        return Path(f"data/bronze_boms_{program}") / f"{program}_mbom_tc_{date}.xlsm"
    if bom_set == "tc_ebom":
        return Path(f"data/bronze_boms_{program}") / f"{program}_ebom_tc_{date}.xlsm"
    raise ValueError("bom_set must be one of: 'oracle', 'tc_mbom', 'tc_ebom'")

def _cache_path(p: Path) -> Path:
    return p.with_suffix(p.suffix + ".parquet")

def _usecols(name) -> bool:
    return str(name).strip() in set(COLMAP.keys()) | set(["PART_NUMBER","Description","Make/Buy","Levels"])

def _read_excel_pruned(path: Path, header=None) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl", header=header, usecols=_usecols, dtype="object")

def _load_one(program: str, date: str, bom_set: str) -> pd.DataFrame:
    p = _paths(program, date, bom_set)
    if not p.exists():
        print(f"[skip] {bom_set} {date} -> file not found")
        return pd.DataFrame(columns=KEEP)

    pq = _cache_path(p)
    if pq.exists() and pq.stat().st_mtime >= p.stat().st_mtime:
        df = pd.read_parquet(pq)
    else:
        header = None
        if bom_set == "oracle" and date not in NO_HEADER_DATES:
            header = 5
        df = _read_excel_pruned(p, header=header).rename(columns=COLMAP, errors="ignore")
        df = df[[c for c in ["PART_NUMBER","Description","Make/Buy","Levels"] if c in df.columns]].copy()
        try:
            df.to_parquet(pq, index=False)
        except Exception:
            pass

    df["Date"] = pd.to_datetime(date)
    # normalize Make/Buy (make|buy only)
    s = (df.get("Make/Buy", pd.Series(dtype="object"))
            .astype(str).str.strip().str.lower().replace({"nan": np.nan}))
    df["Make/Buy"] = s.where(s.isin(["make","buy"]))
    # columns + order
    for c in KEEP:
        if c not in df.columns: df[c] = pd.NA
    return df[KEEP].drop_duplicates()

def load_bom_set(program: str, bom_set: str, dates: list[str] | None = None) -> pd.DataFrame:
    """Sequential, tight loop over dates for a single set only."""
    if dates is None or len(dates) == 0:
        # default = latest only (alphabetic dates like 'MM-DD-YYYY' sort correctly)
        dates = sorted([d.name.split("_")[-1].split(".")[0]
                        for d in (_paths(program, "00-00-0000", bom_set).parent).glob(f"{program}_*_{program.split()[0] if False else ''}*")])  # not robust; pass dates explicitly for reliability
        dates = dates[-1:]  # last only
    out = []
    for d in dates:
        out.append(_load_one(program, d, bom_set))
        if len(out) and len(out[-1]) == 0:
            print(f"[warn] empty after load: {bom_set} {d}")
    if not out:
        return pd.DataFrame(columns=KEEP)
    df = pd.concat(out, ignore_index=True)
    df = (df.sort_values(["PART_NUMBER","Date"])
            .drop_duplicates(subset=["PART_NUMBER","Date"], keep="last")
            .reset_index(drop=True))
    # add lag for flip detection (within this BOM set only)
    df["previous_status"] = df.groupby("PART_NUMBER")["Make/Buy"].shift(1)
    return df

def make_flip_log(df: pd.DataFrame, source_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flip log and per-snapshot counts for the **single** set you loaded."""
    if df.empty:
        return (pd.DataFrame(columns=["PART_NUMBER","Description","Levels","Date","previous_status","new_status","Source"]),
                pd.DataFrame(columns=["Date","num_parts_changed","Source"]))
    mask = df["Make/Buy"].notna() & df["previous_status"].notna() & df["Make/Buy"].ne(df["previous_status"])
    flips = (df.loc[mask, ["PART_NUMBER","Description","Levels","Date","previous_status","Make/Buy"]]
               .rename(columns={"Make/Buy":"new_status"})
               .assign(Source=source_label)
               .sort_values(["Date","PART_NUMBER"])
               .reset_index(drop=True))
    summary = (flips.groupby("Date", as_index=False)["PART_NUMBER"]
                    .nunique()
                    .rename(columns={"PART_NUMBER":"num_parts_changed"})
                    .assign(Source=source_label))
    return flips, summary

# --------- USAGE EXAMPLES (run one at a time) ----------
# 1) Load **m10** TC-MBOM only for two dates (fast, minimal memory):
# m10_tc = load_bom_set("m10", "tc_mbom", dates=["03-05-2025","03-17-2025"])
# flips, flip_summary = make_flip_log(m10_tc, "TC MBOM")
# flips.head(), flip_summary

# 2) Load **m10** Oracle MBOM only for the same dates:
# m10_oracle = load_bom_set("m10", "oracle", dates=["03-05-2025","03-17-2025"])
# o_flips, o_summary = make_flip_log(m10_oracle, "Oracle MBOM")

# 3) (Optional) Compare **two** sets for the same program/dates *after* you’ve loaded them:
def compare_two_sets(df_left: pd.DataFrame, df_right: pd.DataFrame, label_left: str, label_right: str) -> pd.DataFrame:
    if df_left.empty and df_right.empty:
        return pd.DataFrame(columns=["Date","metric","count"])
    all_dates = sorted(set(df_left["Date"]) | set(df_right["Date"]))
    rows = []
    for dt in all_dates:
        L = set(df_left.loc[df_left["Date"].eq(dt), "PART_NUMBER"])
        R = set(df_right.loc[df_right["Date"].eq(dt), "PART_NUMBER"])
        rows.append({"Date": dt, "metric": f"in_{label_left}_not_{label_right}", "count": len(L - R)})
        rows.append({"Date": dt, "metric": f"in_{label_right}_not_{label_left}", "count": len(R - L)})
        rows.append({"Date": dt, "metric": "common_both", "count": len(L & R)})
    return pd.DataFrame(rows).sort_values(["Date","metric"]).reset_index(drop=True)

# Example:
# cmp_oracle_vs_tc = compare_two_sets(m10_oracle, m10_tc, "oracle", "tc_mbom")
# cmp_oracle_vs_tc