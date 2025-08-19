from pathlib import Path
import pandas as pd
import numpy as np

# --- you edit these when you run ---
PROGRAM = "m10"
BOM_SET = "tc_mbom"          # "oracle" | "tc_mbom" | "tc_ebom"
DATES   = ["03-05-2025","03-17-2025","03-26-2025"]  # only what you want to scan
NO_HEADER_DATES = {"02-12-2025","02-20-2025","02-26-2025","03-05-2025","03-17-2025"}  # oracle-only header exception

# ---------- minimal loader ----------
COLMAP = {
    "Part Number":"PART_NUMBER","PART_NUMBER":"PART_NUMBER","Part Number*":"PART_NUMBER",
    "Part-Number":"PART_NUMBER","PART_NUMBER.":"PART_NUMBER",
    "Item Name":"Description","ITEM_NAME":"Description",
    "Make or Buy":"Make/Buy","Make/Buy":"Make/Buy","Make / Buy":"Make/Buy","MAKE_OR_BUY":"Make/Buy",
    "Make /Buy":"Make/Buy","Make/ Buy":"Make/Buy",
    "Level":"Levels","# Level":"Levels","# Levels":"Levels","LEVEL":"Levels",
}
KEEP = ["PART_NUMBER","Description","Make/Buy","Date"]

def _path(program:str, date:str, bom:str)->Path:
    if bom=="oracle":
        return Path("data/bronze_boms")/program/f"{program}_mbom_oracle_{date}.xlsx"
    if bom=="tc_mbom":
        return Path(f"data/bronze_boms_{program}")/f"{program}_mbom_tc_{date}.xlsm"
    if bom=="tc_ebom":
        return Path(f"data/bronze_boms_{program}")/f"{program}_ebom_tc_{date}.xlsm"
    raise ValueError("bom must be 'oracle'|'tc_mbom'|'tc_ebom'")

def _usecols(name)->bool:
    return str(name).strip() in set(COLMAP.keys())|{"PART_NUMBER","Description","Make/Buy"}

def _read_one(program:str, date:str, bom:str)->pd.DataFrame:
    p = _path(program, date, bom)
    if not p.exists():
        print(f"[MISS] {bom} {date} -> {p}")
        return pd.DataFrame(columns=KEEP)
    header = None
    if bom=="oracle" and date not in NO_HEADER_DATES:
        header = 5
    df = pd.read_excel(p, engine="openpyxl", header=header, usecols=_usecols, dtype="object")
    df = df.rename(columns=COLMAP, errors="ignore")
    df = df[[c for c in ["PART_NUMBER","Description","Make/Buy"] if c in df.columns]].copy()
    df["Date"] = pd.to_datetime(date)
    # normalize Make/Buy
    s = (df.get("Make/Buy", pd.Series(dtype="object")).astype(str).str.strip().str.lower().replace({"nan":np.nan}))
    df["Make/Buy"] = s.where(s.isin(["make","buy"]))
    # drop exact dupes for this snapshot
    df = df.drop_duplicates(subset=["PART_NUMBER"])
    return df[KEEP]

def load_bom_series(program:str, bom:str, dates:list[str])->pd.DataFrame:
    parts = [_read_one(program, d, bom) for d in dates]
    if not parts:
        return pd.DataFrame(columns=KEEP)
    df = pd.concat(parts, ignore_index=True)
    # ensure time order for shift to work (flip will be recorded at the *current* row's Date)
    df = (df.sort_values(["PART_NUMBER","Date"])
            .drop_duplicates(subset=["PART_NUMBER","Date"], keep="last")
            .reset_index(drop=True))
    # lag previous status by part number
    df["previous_status"] = df.groupby("PART_NUMBER", sort=False)["Make/Buy"].shift(1)
    return df

def make_flip_log(df: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the columns requested; Date is the BOM where the flip occurred."""
    if df.empty:
        return pd.DataFrame(columns=["PART_NUMBER","Description","previous_status","new_status","Date"])
    m = df["Make/Buy"].notna() & df["previous_status"].notna() & df["Make/Buy"].ne(df["previous_status"])
    flip_log = (df.loc[m, ["PART_NUMBER","Description","previous_status","Make/Buy","Date"]]
                  .rename(columns={"Make/Buy":"new_status"})
                  .sort_values(["Date","PART_NUMBER"])
                  .reset_index(drop=True))
    return flip_log

# -------- run for the selection above --------
_series = load_bom_series(PROGRAM, BOM_SET, DATES)
flip_log = make_flip_log(_series)

print(f"{PROGRAM} • {BOM_SET} • dates={DATES}  -> flips: {len(flip_log)}")
display(flip_log.head(25))
# Optional save:
# flip_log.to_csv(f"mb_output/{PROGRAM}/{PROGRAM}_{BOM_SET}_flip_log.csv", index=False)