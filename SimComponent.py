# ================== Make↔Buy Flip Pipeline (final: safe cols, no warnings) ==================
from __future__ import annotations
import re
from glob import glob
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

# ---------- Config ----------
BASE = Path("data")
OUT  = Path("mb_output"); OUT.mkdir(exist_ok=True)

PATTERN = {
    "tc_mbm":     "{base}/bronze_boms_{program}/{program}_mbom_tc_{date}.xlsm",
    "tc_ebom":    "{base}/bronze_boms_{program}/{program}_ebom_tc_{date}.xlsm",
    "oracle_mbm": "{base}/bronze_boms_{program}/{program}_mbom_oracle_{date}.xlsx",
}

NO_HEADER_DATES = {"02-12-2025","02-20-2025","02-26-2025","03-05-2025","03-17-2025"}
DATE_FMT = "%m-%d-%Y"
DATE_RE  = re.compile(r"_(\d{2}-\d{2}-\d{4})\.(?:xlsx|xlsm)$")

PART_CANDS  = ["PART_NUMBER","Part Number","PART NUMBER","Part_Number","PartNumber","Part No","PartNo"]
DESC_CANDS  = ["Item Name","Description","Part Description","Part Desc","ItemName"]
MABU_CANDS  = ["Make/Buy","Make or Buy","Make or Buy:","MAKE/BUY","MakeBuy"]
LEVEL_CANDS = ["# Level","Level","Levels","Structure Level","Indent","Indented Level","Lvl","#Level","Level #"]

# ---------- Helpers ----------
def _path_for(program: str, date: str, source: str) -> Path:
    return Path(PATTERN[source].format(base=BASE, program=program, date=date))

def available_dates(program: str, source: str) -> list[str]:
    pat = str(_path_for(program, "******", source)).replace("******", "*")
    ds = []
    for f in glob(pat):
        m = DATE_RE.search(f)
        if m: ds.append(m.group(1))
    return sorted(set(ds), key=lambda d: pd.to_datetime(d, format=DATE_FMT))

def infer_dates(program: str, sources: list[str], mode: str="union") -> list[str]:
    sets = [set(available_dates(program, s)) for s in sources]
    if not sets: return []
    ds = set.intersection(*sets) if mode == "intersection" else set.union(*sets)
    return sorted(ds, key=lambda d: pd.to_datetime(d, format=DATE_FMT))

def _norm_colname(x) -> str:
    """Robust lowercasing for any dtype; NaN -> ''."""
    if pd.isna(x): return ""
    return re.sub(r"[^a-z0-9]", "", str(x).lower())

def _find_col(cols, candidates):
    norm = {_norm_colname(c): c for c in cols}
    for cand in candidates:
        key = _norm_colname(cand)
        if key in norm: return norm[key]
    return None

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure single string column index (handles ints, NaN, MultiIndex)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(str(p) for p in tup if not pd.isna(p)).strip() for tup in df.columns]
    else:
        df.columns = [str(c) if not pd.isna(c) else "" for c in df.columns]
    return df

def _coalesce_dupes(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """First non-null across duplicate columns — vectorized, no warnings."""
    same = [c for c in df.columns if c == col]
    if len(same) > 1:
        block = df[same].astype("object")
        # treat empty strings as NA, column-wise (no DataFrame.applymap)
        for c in same:
            s = block[c]
            mask_empty = s.apply(lambda v: isinstance(v, str) and v.strip() == "")
            block.loc[mask_empty, c] = pd.NA
        mask = block.notna().to_numpy()
        vals = block.to_numpy()
        any_row = mask.any(axis=1)
        first_idx = np.where(any_row, mask.argmax(axis=1), 0)
        first_vals = np.where(any_row, vals[np.arange(len(vals)), first_idx], pd.NA)
        df[col] = first_vals
        df.drop(columns=same[1:], inplace=True)
    return df

def _normalize(raw: pd.DataFrame, date: str) -> pd.DataFrame:
    raw = _flatten_columns(raw)

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
    for col in ["PART_NUMBER","Description","Make/Buy","Levels"]:
        if col in df.columns: df = _coalesce_dupes(df, col)

    if "Make/Buy" in df.columns:
        df["Make/Buy"] = (
            df["Make/Buy"].astype("string").str.strip().str.lower()
              .replace({"m":"make","mk":"make","b":"buy","bu":"buy"})
              .where(df["Make/Buy"].isin(["make","buy"]))
        )

    df["Date"] = pd.to_datetime(date, format=DATE_FMT, errors="coerce")
    if "PART_NUMBER" in df.columns:
        df = df[df["PART_NUMBER"].astype(str).str.strip().ne("")]

    df = (df.sort_values([c for c in ["PART_NUMBER","Date"] if c in df.columns])
            .drop_duplicates([c for c in ["PART_NUMBER","Date"] if c in df.columns], keep="last")
            .reset_index(drop=True))
    return df

# ---------- Smart Excel reader (try all sheets + header rows 0..10) ----------
def _read_excel_smart(path: Path, source: str, date: str) -> Tuple[pd.DataFrame|None, str|None]:
    try:
        xls = pd.ExcelFile(path, engine="openpyxl")
    except Exception as e:
        return None, f"open error: {e}"

    pref = []
    if source == "oracle_mbm" and date not in NO_HEADER_DATES:
        pref = [(xls.sheet_names[0], 5)]
    attempts = pref + [(sh, hdr) for sh in xls.sheet_names for hdr in range(0, 11)]

    last_reason = "no header produced required columns"
    for sh, hdr in attempts:
        try:
            raw = pd.read_excel(xls, sheet_name=sh, header=hdr)
        except Exception as e:
            last_reason = f"read error on {sh}@{hdr}: {e}"
            continue
        df = _normalize(raw, date)
        if {"PART_NUMBER","Make/Buy"}.issubset(df.columns) and not df.empty:
            df.attrs["chosen_sheet"] = sh
            df.attrs["chosen_header"] = hdr
            return df, None
    return None, last_reason

def _read_one(program: str, date: str, source: str):
    path = _path_for(program, date, source)
    if not path.exists():
        return None, "missing file"
    df, reason = _read_excel_smart(path, source, date)
    if df is None: return None, reason
    return df, None

def load_boms(program: str, dates, sources, align_sources: bool=False) -> pd.DataFrame:
    if dates == "auto" or dates is None:
        dates = infer_dates(program, sources, mode="intersection" if align_sources else "union")

    skipped, frames = [], []
    for src in sources:
        for d in dates:
            df, reason = _read_one(program, d, src)
            if df is None:
                skipped.append((src, d, reason)); continue
            df["Source"] = src
            df["sheet_used"] = df.attrs.get("chosen_sheet", "")
            df["header_row"] = df.attrs.get("chosen_header", -1)
            frames.append(df)

    if skipped:
        print("⚠️  Skipped", len(skipped), "snapshots:",
              "; ".join([f"{s}:{d} -> {r}" for s, d, r in skipped]))

    if not frames:
        return pd.DataFrame(columns=["PART_NUMBER","Description","Levels","Make/Buy","Date","Source"])

    return (pd.concat(frames, ignore_index=True)
              .sort_values(["Source","PART_NUMBER","Date"])
              .drop_duplicates(["Source","PART_NUMBER","Date"], keep="last")
              .reset_index(drop=True))

# ---------- Flip detection & outputs ----------
def detect_flips(all_df: pd.DataFrame):
    if all_df.empty: return all_df, all_df, all_df
    all_df = all_df.sort_values(["Source","PART_NUMBER","Date"])
    all_df["previous_status"] = all_df.groupby(["Source","PART_NUMBER"])["Make/Buy"].shift()

    mask = all_df["Make/Buy"].notna() & all_df["previous_status"].notna() & \
           (all_df["Make/Buy"] != all_df["previous_status"])

    flip_log = (all_df.loc[mask, ["Source","PART_NUMBER","Description","Levels","Date",
                                  "previous_status","Make/Buy","sheet_used","header_row"]]
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

def write_outputs(program: str, flip_log, snapshot_summary, per_part_counts):
    out = OUT / program; out.mkdir(parents=True, exist_ok=True)
    flip_log.to_csv(out / "make_buy_flip_log.csv", index=False)
    snapshot_summary.to_csv(out / "make_buy_flip_summary_by_date.csv", index=False)
    per_part_counts.to_csv(out / "make_buy_flip_counts_by_part.csv", index=False)
    print(f"✅ Wrote outputs to {out}")

# ---------- Example run ----------
if __name__ == "__main__":
    program = "cuas"
    sources = ["tc_mbm"]                 # "tc_mbm", "tc_ebom", "oracle_mbm"
    dates   = "auto"                     # or pass explicit list

    all_boms = load_boms(program, dates, sources, align_sources=False)
    flip_log, snapshot_summary, per_part_counts = detect_flips(all_boms)
    write_outputs(program, flip_log, snapshot_summary, per_part_counts)

    print(flip_log.head(10))
    print(snapshot_summary.tail(10))
    print(per_part_counts.head(10))