# FAST UNIFIED BOM PIPELINE (Excel -> Parquet cache -> analysis)
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# CONFIG (keep your program/date lists here)
# -----------------------------
BASE = Path("data/bronze_boms")
# Example (replace with your real sets):
PROGRAM_CONFIG = {
    "m10": {
        "dates": [
            "03-05-2025","03-17-2025","03-26-2025","03-27-2025",
            "04-02-2025","04-09-2025","04-22-2025","05-07-2025",
            "06-04-2025","06-09-2025","06-23-2025","06-30-2025",
            "07-07-2025","07-21-2025","07-28-2025","08-04-2025","08-11-2025"
        ],
        "no_header_dates": set(["02-12-2025","02-20-2025","02-26-2025","03-05-2025","03-17-2025"]),
    },
    # "xm30": {...},
    # "cuas": {...}
}

# Harmonize names and keep only what we need
COLMAP = {
    "Part Number": "PART_NUMBER",
    "PART_NUMBER": "PART_NUMBER",
    "Part Number*": "PART_NUMBER",
    "Part-Number": "PART_NUMBER",
    "PART_NUMBER.": "PART_NUMBER",

    "Item Name": "Description",
    "ITEM_NAME": "Description",

    "Make or Buy": "Make/Buy",
    "Make/Buy": "Make/Buy",
    "Make / Buy": "Make/Buy",
    "MAKE_OR_BUY": "Make/Buy",
    "Make /Buy": "Make/Buy",
    "Make/ Buy": "Make/Buy",

    "Level": "Levels",
    "# Level": "Levels",
    "# Levels": "Levels",
    "LEVEL": "Levels",
}
WANTED = {"PART_NUMBER","Description","Make/Buy","Levels"}

# Prefer Arrow string if available (reduces memory/time on large text columns)
STRING_DTYPE = "string"
try:
    pd.Series(["x"], dtype="string[pyarrow]")
    STRING_DTYPE = "string[pyarrow]"
except Exception:
    pass

DTYPES = {
    "PART_NUMBER": STRING_DTYPE,
    "Description": STRING_DTYPE,
    "Make/Buy": STRING_DTYPE,
    "Levels": "Int64",
}

# -----------------------------
# HELPERS
# -----------------------------
def _cache_path(xl_path: Path) -> Path:
    return xl_path.with_suffix(xl_path.suffix + ".parquet")

def _usecols(name) -> bool:
    """Read only potentially relevant columns."""
    # we can't rely on exact header spelling; allow a superset
    return str(name).strip() in set(COLMAP.keys()) | WANTED

def _read_excel_pruned(path: Path, header=None) -> pd.DataFrame:
    if not path.exists(): 
        return pd.DataFrame(columns=list(WANTED))
    return pd.read_excel(
        path,
        engine="openpyxl",
        header=header,          # usually None or 5
        usecols=_usecols,       # <-- prune wide sheets at read time
        dtype="object"          # keep raw, we'll cast after rename
    )

def _load_with_cache(path: Path, header=None, date_str: str="") -> pd.DataFrame:
    """
    If a fresh parquet exists beside the Excel, load it; otherwise read Excel (pruned),
    normalize, then write parquet for next time.
    """
    pq = _cache_path(path)
    if pq.exists() and path.exists() and pq.stat().st_mtime >= path.stat().st_mtime:
        df = pd.read_parquet(pq)
        # stamp date (for cases where we cached without it)
        if "Date" not in df.columns and date_str:
            df["Date"] = pd.to_datetime(date_str)
        return df

    # read from Excel
    df = _read_excel_pruned(path, header=header)
    if df.empty:
        return pd.DataFrame(columns=["PART_NUMBER","Description","Make/Buy","Levels","Date"])

    df = (df.rename(columns=COLMAP, errors="ignore")
            .pipe(lambda d: d[[c for c in ["PART_NUMBER","Description","Make/Buy","Levels"] if c in d.columns]])
            .assign(Date=pd.to_datetime(date_str))
         )

    # cast to lean dtypes (avoids heavy object ops later)
    for c, dt in DTYPES.items():
        if c in df.columns:
            try:
                df[c] = df[c].astype(dt)
            except Exception:
                pass

    # persist parquet for next runs
    try:
        df.to_parquet(pq, index=False)
    except Exception:
        pass  # caching is a best-effort optimization

    return df

def _paths(program: str, date: str):
    oracle = BASE / program / f"{program}_mbom_oracle_{date}.xlsx"
    tc_m   = Path(f"data/bronze_boms_{program}") / f"{program}_mbom_tc_{date}.xlsm"
    tc_e   = Path(f"data/bronze_boms_{program}") / f"{program}_ebom_tc_{date}.xlsm"
    return oracle, tc_m, tc_e

def _clean_make_buy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.assign(previous_status=pd.Series(dtype=STRING_DTYPE))
    # operate only on the 5 columns we care about
    df = df[["PART_NUMBER","Description","Levels","Make/Buy","Date"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    # normalize make/buy
    s = (df["Make/Buy"]
            .astype("string")
            .str.strip().str.lower()
            .replace({"nan": pd.NA}))
    df["Make/Buy"] = s.where(s.isin(["make","buy"]))
    # drop dup snapshots per part/date (keep last to mimic your code)
    df = (df.sort_values(["PART_NUMBER","Date"])
            .drop_duplicates(subset=["PART_NUMBER","Date"], keep="last")
            .reset_index(drop=True))
    # lag
    df["previous_status"] = df.groupby("PART_NUMBER", sort=False)["Make/Buy"].shift(1)
    return df

def _flip_log_and_summary(df: pd.DataFrame, source: str):
    if df.empty:
        return (
            pd.DataFrame(columns=["PART_NUMBER","Description","Levels","Date","previous_status","new_status","Source"]),
            pd.DataFrame(columns=["Date","num_parts_changed","Source"]),
        )
    mask = df["Make/Buy"].notna() & df["previous_status"].notna() & df["Make/Buy"].ne(df["previous_status"])
    flips = (df.loc[mask, ["PART_NUMBER","Description","Levels","Date","previous_status","Make/Buy"]]
               .rename(columns={"Make/Buy":"new_status"})
               .assign(Source=source)
               .sort_values(["Date","PART_NUMBER"])
               .reset_index(drop=True))
    summary = (flips.groupby("Date", as_index=False)["PART_NUMBER"]
                    .nunique()
                    .rename(columns={"PART_NUMBER":"num_parts_changed"})
                    .assign(Source=source))
    return flips, summary

def _compare_sets_by_date(oracle_df, tc_m_df, tc_e_df):
    if oracle_df.empty and tc_m_df.empty and tc_e_df.empty:
        return pd.DataFrame(columns=["Date","metric","count"])
    dates = sorted(set(oracle_df["Date"]) | set(tc_m_df["Date"]) | set(tc_e_df["Date"]))
    out = []
    for d in dates:
        O = set(oracle_df.loc[oracle_df["Date"].eq(d), "PART_NUMBER"])
        M = set(tc_m_df.loc[tc_m_df["Date"].eq(d), "PART_NUMBER"])
        E = set(tc_e_df.loc[tc_e_df["Date"].eq(d), "PART_NUMBER"])
        out.append({"Date": d, "metric": "in_oracle_not_tc_m", "count": len(O-M)})
        out.append({"Date": d, "metric": "in_tc_m_not_oracle", "count": len(M-O)})
        out.append({"Date": d, "metric": "in_tc_e_not_tc_m", "count": len(E-M)})
        out.append({"Date": d, "metric": "common_all", "count": len(O & M & E)})
    return pd.DataFrame(out, dtype="Int64").sort_values(["Date","metric"]).reset_index(drop=True)

# -----------------------------
# CORE LOADER (parallel file reads)
# -----------------------------
def _load_one_date(program: str, date: str, no_headers: set):
    oracle_p, tc_m_p, tc_e_p = _paths(program, date)
    # choose header for Oracle dynamically
    o_header = None if date in no_headers else 5
    return (
        _load_with_cache(oracle_p, header=o_header, date_str=date),
        _load_with_cache(tc_m_p,     header=None,   date_str=date),
        _load_with_cache(tc_e_p,     header=None,   date_str=date),
    )

def load_program_fast(program: str):
    cfg = PROGRAM_CONFIG[program]
    dates = cfg["dates"]
    no_headers = cfg.get("no_header_dates", set())

    # parallelize I/O-bound Excel reads (GIL is released in I/O)
    results = []
    with ThreadPoolExecutor(max_workers=min(8, len(dates))) as pool:
        futures = [pool.submit(_load_one_date, program, d, no_headers) for d in dates]
        for f in futures:
            results.append(f.result())

    oracle = pd.concat([r[0] for r in results if r[0] is not None], ignore_index=True) if results else pd.DataFrame()
    tc_m   = pd.concat([r[1] for r in results if r[1] is not None], ignore_index=True) if results else pd.DataFrame()
    tc_e   = pd.concat([r[2] for r in results if r[2] is not None], ignore_index=True) if results else pd.DataFrame()

    # enforce column presence
    for df in (oracle, tc_m, tc_e):
        for c in ["PART_NUMBER","Description","Make/Buy","Levels","Date"]:
            if c not in df.columns:
                df[c] = pd.Series(dtype=DTYPES.get(c, "object"))

    # Clean + flips
    oracle_c = _clean_make_buy(oracle)
    tc_m_c   = _clean_make_buy(tc_m)
    tc_e_c   = _clean_make_buy(tc_e)

    o_flips, o_sum = _flip_log_and_summary(oracle_c, "Oracle MBOM")
    m_flips, m_sum = _flip_log_and_summary(tc_m_c,   "TC MBOM")
    e_flips, e_sum = _flip_log_and_summary(tc_e_c,   "TC EBOM")

    compare_tidy = _compare_sets_by_date(oracle, tc_m, tc_e)

    return {
        "oracle": oracle, "tc_mbom": tc_m, "tc_ebom": tc_e,
        "oracle_flips": o_flips, "tc_m_flips": m_flips, "tc_e_flips": e_flips,
        "flip_summary": pd.concat([o_sum, m_sum, e_sum], ignore_index=True),
        "compare_tidy": compare_tidy,
    }

# -----------------------------
# HOW YOU CALL IT (works like your previous loop)
# -----------------------------
# res = load_program_fast("m10")
# res["oracle_flips"].head()
# res["flip_summary"].head()
# res["compare_tidy"].head()