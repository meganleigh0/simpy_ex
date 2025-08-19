# === BOM Flip Log Pipeline (single program + single BOM set) ===
# Outputs a flip log with: PART_NUMBER, Description, previous_status, new_status, Date

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------
# USER SETTINGS (edit below)
# ---------------------------
PROGRAM = "m10"                              # e.g., "m10", "xm30", "cuas"
BOM_SET = "tc_mbom"                          # one of: "oracle" | "tc_mbom" | "tc_ebom"
DATES   = ["03-05-2025","03-17-2025"]        # keep this small while testing; add more once confirmed
SAVE_CSV = False                              # True to save CSV
OUTPUT_DIR = Path("mb_output") / PROGRAM

# Oracle-only: files that DO NOT have a header row at row 5 (so use default header)
NO_HEADER_DATES = {
    "02-12-2025","02-20-2025","02-26-2025","03-05-2025","03-17-2025"
}

# ---------------------------
# INTERNAL CONFIG
# ---------------------------
# Map common header variants to canonical names
COLMAP = {
    # Part number
    "Part Number": "PART_NUMBER",
    "PART_NUMBER": "PART_NUMBER",
    "Part Number*": "PART_NUMBER",
    "Part-Number": "PART_NUMBER",
    "PART_NUMBER.": "PART_NUMBER",

    # Description / Item name
    "Item Name": "Description",
    "ITEM_NAME": "Description",
    "DESCRIPTION": "Description",
    "Description": "Description",

    # Make/Buy
    "Make or Buy": "Make/Buy",
    "Make/Buy": "Make/Buy",
    "Make / Buy": "Make/Buy",
    "MAKE_OR_BUY": "Make/Buy",
    "Make /Buy": "Make/Buy",
    "Make/ Buy": "Make/Buy",

    # Level (not used in flip log, but harmless to capture)
    "Level": "Levels",
    "# Level": "Levels",
    "# Levels": "Levels",
    "LEVEL": "Levels",
}

# Only read columns that could map to what we need (keeps Excel parsing fast)
READABLE_COLS = set(COLMAP.keys()) | {"PART_NUMBER", "Description", "Make/Buy", "Levels"}
NEEDED_FOR_LOG = ["PART_NUMBER", "Description", "Make/Buy", "Date"]

BASE = Path("data/bronze_boms")

def _path(program: str, date: str, bom: str) -> Path:
    if bom == "oracle":
        return BASE / program / f"{program}_mbom_oracle_{date}.xlsx"
    if bom == "tc_mbom":
        return Path(f"data/bronze_boms_{program}") / f"{program}_mbom_tc_{date}.xlsm"
    if bom == "tc_ebom":
        return Path(f"data/bronze_boms_{program}") / f"{program}_ebom_tc_{date}.xlsm"
    raise ValueError("BOM_SET must be one of: 'oracle' | 'tc_mbom' | 'tc_ebom'")

def _usecols(name) -> bool:
    # openpyxl passes names as strings; tolerate odd spacing/casing
    return str(name).strip() in READABLE_COLS

def _read_one(program: str, date: str, bom: str) -> pd.DataFrame:
    """Read one snapshot for the chosen BOM set, prune columns, normalize Make/Buy."""
    p = _path(program, date, bom)
    if not p.exists():
        print(f"[MISS] {bom} {date} -> {p}")
        return pd.DataFrame(columns=NEEDED_FOR_LOG)

    # Oracle sometimes has the header starting at row 6 (index 5). Respect your exception dates.
    header = None
    if bom == "oracle" and date not in NO_HEADER_DATES:
        header = 5

    df = pd.read_excel(p, engine="openpyxl", header=header, usecols=_usecols, dtype="object")
    df = df.rename(columns=COLMAP, errors="ignore")

    # keep only columns that exist; add missing as <NA>
    present = [c for c in ["PART_NUMBER", "Description", "Make/Buy"] if c in df.columns]
    df = df[present].copy()
    for c in ["PART_NUMBER", "Description", "Make/Buy"]:
        if c not in df.columns:
            df[c] = pd.NA

    # basic cleanup
    df["PART_NUMBER"] = df["PART_NUMBER"].astype("string").str.strip()
    s = df["Make/Buy"].astype("string").str.strip().str.lower()
    df["Make/Buy"] = s.where(s.isin(["make", "buy"]))  # only keep make/buy, else NA
    df["Date"] = pd.to_datetime(date)

    # drop rows without part number for this snapshot; dedupe by part
    df = df.dropna(subset=["PART_NUMBER"]).drop_duplicates(subset=["PART_NUMBER"])
    return df[NEEDED_FOR_LOG]

def load_series(program: str, bom: str, dates: list[str]) -> pd.DataFrame:
    """Sequentially load the snapshots for one BOM set (fast & predictable)."""
    frames = [_read_one(program, d, bom) for d in dates]
    if not frames:
        return pd.DataFrame(columns=NEEDED_FOR_LOG + ["previous_status"])
    df = pd.concat(frames, ignore_index=True)
    # order so shift() reflects the earlier snapshot -> later snapshot progression
    df = (df.sort_values(["PART_NUMBER", "Date"])
            .drop_duplicates(subset=["PART_NUMBER", "Date"], keep="last")
            .reset_index(drop=True))
    df["previous_status"] = df.groupby("PART_NUMBER", sort=False)["Make/Buy"].shift(1)
    return df

def make_flip_log(df: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the flip rows where Make/Buy changed; Date is the snapshot where it flipped."""
    if df.empty:
        return pd.DataFrame(columns=["PART_NUMBER", "Description", "previous_status", "new_status", "Date"])
    m = df["Make/Buy"].notna() & df["previous_status"].notna() & df["Make/Buy"].ne(df["previous_status"])
    log = (df.loc[m, ["PART_NUMBER", "Description", "previous_status", "Make/Buy", "Date"]]
             .rename(columns={"Make/Buy": "new_status"})
             .sort_values(["Date", "PART_NUMBER"])
             .reset_index(drop=True))
    return log

# ---------------------------
# RUN
# ---------------------------
series_df = load_series(PROGRAM, BOM_SET, DATES)
flip_log  = make_flip_log(series_df)

print(f"{PROGRAM} • {BOM_SET} • snapshots={len(DATES)}  -> flips found: {len(flip_log)}")
display(flip_log.head(25))

if SAVE_CSV:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{PROGRAM}_{BOM_SET}_flip_log.csv"
    flip_log.to_csv(out_path, index=False)
    print(f"[saved] {out_path}")