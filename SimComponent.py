from pathlib import Path
import pandas as pd
import numpy as np

# --------- EDIT THESE ---------
PROGRAM = "m10"
BOM_SET = "tc_mbom"                    # "oracle" | "tc_mbom" | "tc_ebom"
DATES   = ["03-05-2025","03-17-2025","03-26-2025"]
NO_HEADER_DATES = {"02-12-2025","02-20-2025","02-26-2025","03-05-2025","03-17-2025"}  # Oracle only
SAVE_CSV = False
OUT_DIR = Path("mb_output")/PROGRAM
# ------------------------------

# Canonical names we want
TARGETS = {"part": "PART_NUMBER", "desc": "Description", "mb": "Make/Buy"}

# Flexible header resolver (handles weird spacing/variants)
def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def norm(s):
        s = str(s).strip()
        s = s.replace("\u00A0"," ")  # NBSP
        return " ".join(s.split())
    df = df.rename(columns=lambda c: norm(c))

    cols = {c.lower().replace(":", ""): c for c in df.columns}

    def pick(keys):
        for k in keys:
            k = k.lower()
            for col_lower, orig in cols.items():
                if k == col_lower:
                    return orig
        return None

    # try exacts then fuzzy contains
    part = pick(["PART_NUMBER","Part Number","Part-Number","Part Number*","PART_NUMBER."])
    if part is None:
        part = next((c for c in df.columns if "part" in c.lower() and "number" in c.lower()), None)

    desc = pick(["Description","Item Name","ITEM_NAME","DESCRIPTION"])
    if desc is None:
        desc = next((c for c in df.columns if "desc" in c.lower() or "item name" in c.lower()), None)

    mb = pick(["Make/Buy","Make or Buy","Make / Buy","MAKE_OR_BUY","Make /Buy","Make/ Buy"])
    if mb is None:
        mb = next((c for c in df.columns if "make" in c.lower() and "buy" in c.lower()), None)

    # build trimmed frame with guaranteed columns (fill NA if missing)
    out = pd.DataFrame()
    out[TARGETS["part"]] = df[part] if part in df.columns else pd.NA
    out[TARGETS["desc"]] = df[desc] if desc in df.columns else pd.NA
    out[TARGETS["mb"]]   = df[mb]   if mb   in df.columns else pd.NA
    return out

def _path(program: str, date: str, bom: str) -> Path:
    base = Path("data/bronze_boms")
    if bom == "oracle":
        return base/program/f"{program}_mbom_oracle_{date}.xlsx"
    if bom == "tc_mbom":
        return Path(f"data/bronze_boms_{program}")/f"{program}_mbom_tc_{date}.xlsm"
    if bom == "tc_ebom":
        return Path(f"data/bronze_boms_{program}")/f"{program}_ebom_tc_{date}.xlsm"
    raise ValueError("BOM_SET must be 'oracle'|'tc_mbom'|'tc_ebom'")

def _read_one(program: str, date: str, bom: str) -> pd.DataFrame:
    p = _path(program, date, bom)
    if not p.exists():
        print(f"[MISS] {bom} {date} -> {p}")
        return pd.DataFrame(columns=[TARGETS["part"], TARGETS["desc"], TARGETS["mb"], "Date"])

    header = None
    if bom == "oracle" and date not in NO_HEADER_DATES:
        header = 5

    # Read all columns; we’ll prune after (safer than usecols when headers vary)
    df_raw = pd.read_excel(p, engine="openpyxl", header=header, dtype="object")
    df = canonicalize_columns(df_raw)

    # Normalize
    df[TARGETS["part"]] = df[TARGETS["part"]].astype("string").str.strip()
    s = (df[TARGETS["mb"]].astype("string").str.strip().str.lower()
           .replace({"nan": pd.NA}))
    df[TARGETS["mb"]] = s.where(s.isin(["make","buy"]))

    df["Date"] = pd.to_datetime(date)

    # drop rows without part number; if dup part within a snapshot, keep last
    df = df.dropna(subset=[TARGETS["part"]]).drop_duplicates(subset=[TARGETS["part"]], keep="last")
    return df[[TARGETS["part"], TARGETS["desc"], TARGETS["mb"], "Date"]]

def load_series(program: str, bom: str, dates: list[str]) -> pd.DataFrame:
    frames = [_read_one(program, d, bom) for d in dates]
    if not frames:
        return pd.DataFrame(columns=[TARGETS["part"], TARGETS["desc"], TARGETS["mb"], "Date", "previous_status"])
    df = pd.concat(frames, ignore_index=True)
    df = (df.sort_values([TARGETS["part"], "Date"])
            .drop_duplicates(subset=[TARGETS["part"], "Date"], keep="last")
            .reset_index(drop=True))
    df["previous_status"] = df.groupby(TARGETS["part"], sort=False)[TARGETS["mb"]].shift(1)
    return df

def make_flip_log(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["PART_NUMBER","Description","previous_status","new_status","Date"])
    m = df[TARGETS["mb"]].notna() & df["previous_status"].notna() & df[TARGETS["mb"]].ne(df["previous_status"])
    log = (df.loc[m, [TARGETS["part"], TARGETS["desc"], "previous_status", TARGETS["mb"], "Date"]]
             .rename(columns={TARGETS["part"]:"PART_NUMBER", TARGETS["desc"]:"Description", TARGETS["mb"]:"new_status"})
             .sort_values(["Date","PART_NUMBER"])
             .reset_index(drop=True))
    return log

# ---------- DIAGNOSTICS (quick sanity checks) ----------
def diag_summary(df: pd.DataFrame):
    if df.empty:
        print("No rows loaded.")
        return
    print("Snapshots loaded:", df["Date"].dt.date.unique().tolist())
    print("\nPer-date counts (parts, non-null Make/Buy, value counts):")
    for d, grp in df.groupby(df["Date"].dt.date):
        mb_vc = grp[TARGETS["mb"]].value_counts(dropna=False).to_dict()
        print(f"  {d}: parts={grp[TARGETS['part']].nunique():6d}  mb_nonnull={grp[TARGETS['mb']].notna().sum():6d}  {mb_vc}")

# ---------------- RUN ----------------
series_df = load_series(PROGRAM, BOM_SET, DATES)
diag_summary(series_df)        # <- look here if flips seem missing
flip_log = make_flip_log(series_df)

print(f"\n{PROGRAM} • {BOM_SET} • snapshots={len(DATES)} -> flips: {len(flip_log)}")
display(flip_log.head(25))

if SAVE_CSV:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outp = OUT_DIR/f"{PROGRAM}_{BOM_SET}_flip_log.csv"
    flip_log.to_csv(outp, index=False)
    print(f"[saved] {outp}")