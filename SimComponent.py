from pathlib import Path
import pandas as pd
import numpy as np

# --- CONFIG you edit ---
PROGRAM = "m10"
DATES   = ["03-05-2025","03-17-2025"]   # keep this short while testing
NO_HEADER_DATES = {"02-12-2025","02-20-2025","02-26-2025","03-05-2025","03-17-2025"}  # Oracle only

# --- Helpers (tiny, fast, single-set) ---
COLMAP = {
    "Part Number":"PART_NUMBER","PART_NUMBER":"PART_NUMBER","Part Number*":"PART_NUMBER",
    "Part-Number":"PART_NUMBER","PART_NUMBER.":"PART_NUMBER",
    "Item Name":"Description","ITEM_NAME":"Description",
    "Make or Buy":"Make/Buy","Make/Buy":"Make/Buy","Make / Buy":"Make/Buy","MAKE_OR_BUY":"Make/Buy",
    "Make /Buy":"Make/Buy","Make/ Buy":"Make/Buy",
    "Level":"Levels","# Level":"Levels","# Levels":"Levels","LEVEL":"Levels",
}
KEEP = ["PART_NUMBER","Description","Make/Buy","Levels","Date"]

def _path(program:str, date:str, bom:str)->Path:
    if bom=="oracle":
        return Path("data/bronze_boms")/program/f"{program}_mbom_oracle_{date}.xlsx"
    if bom=="tc_mbom":
        return Path(f"data/bronze_boms_{program}")/f"{program}_mbom_tc_{date}.xlsm"
    if bom=="tc_ebom":
        return Path(f"data/bronze_boms_{program}")/f"{program}_ebom_tc_{date}.xlsm"
    raise ValueError("bom must be 'oracle'|'tc_mbom'|'tc_ebom'")

def _usecols(name)->bool:
    return str(name).strip() in set(COLMAP.keys())|{"PART_NUMBER","Description","Make/Buy","Levels"}

def _read(program:str, date:str, bom:str)->pd.DataFrame:
    p = _path(program,date,bom)
    if not p.exists():
        print(f"[MISS] {bom} {date} -> {p}")
        return pd.DataFrame(columns=KEEP)
    header = (None if bom!="oracle" or date in NO_HEADER_DATES else 5)
    df = pd.read_excel(p, engine="openpyxl", header=header, usecols=_usecols, dtype="object")
    df = df.rename(columns=COLMAP, errors="ignore")
    df = df[[c for c in ["PART_NUMBER","Description","Make/Buy","Levels"] if c in df.columns]].copy()
    df["Date"] = pd.to_datetime(date)
    s = (df.get("Make/Buy", pd.Series(dtype="object")).astype(str).str.strip().str.lower().replace({"nan":np.nan}))
    df["Make/Buy"] = s.where(s.isin(["make","buy"]))
    for c in KEEP:
        if c not in df.columns: df[c]=pd.NA
    return df[KEEP]

def load_bom_set(program:str, bom:str, dates:list[str])->pd.DataFrame:
    parts = [_read(program,d,bom) for d in dates]
    if not parts: return pd.DataFrame(columns=KEEP)
    df = pd.concat(parts, ignore_index=True)
    df = (df.sort_values(["PART_NUMBER","Date"])
            .drop_duplicates(subset=["PART_NUMBER","Date"], keep="last")
            .reset_index(drop=True))
    df["previous_status"] = df.groupby("PART_NUMBER")["Make/Buy"].shift()
    return df

def flips(df:pd.DataFrame, label:str):
    if df.empty:
        return pd.DataFrame(columns=["PART_NUMBER","Description","Levels","Date","previous_status","new_status","Source"]), \
               pd.DataFrame(columns=["Date","num_parts_changed","Source"])
    m = df["Make/Buy"].notna() & df["previous_status"].notna() & df["Make/Buy"].ne(df["previous_status"])
    log = (df.loc[m, ["PART_NUMBER","Description","Levels","Date","previous_status","Make/Buy"]]
             .rename(columns={"Make/Buy":"new_status"})
             .assign(Source=label)
             .sort_values(["Date","PART_NUMBER"])
             .reset_index(drop=True))
    summary = (log.groupby("Date",as_index=False)["PART_NUMBER"]
                  .nunique()
                  .rename(columns={"PART_NUMBER":"num_parts_changed"})
                  .assign(Source=label))
    return log, summary

def compare_two(df_left:pd.DataFrame, df_right:pd.DataFrame, left:str, right:str)->pd.DataFrame:
    if df_left.empty and df_right.empty:
        return pd.DataFrame(columns=["Date","metric","count"])
    dates = sorted(set(df_left["Date"])|set(df_right["Date"]))
    rows=[]
    for dt in dates:
        L=set(df_left.loc[df_left["Date"].eq(dt),"PART_NUMBER"])
        R=set(df_right.loc[df_right["Date"].eq(dt),"PART_NUMBER"])
        rows += [
            {"Date":dt,"metric":f"in_{left}_not_{right}","count":len(L-R)},
            {"Date":dt,"metric":f"in_{right}_not_{left}","count":len(R-L)},
            {"Date":dt,"metric":"common_both","count":len(L&R)},
        ]
    return pd.DataFrame(rows).sort_values(["Date","metric"]).reset_index(drop=True)

# --- RUN: exactly one program + two sets (kept small) ---
m10_tc      = load_bom_set(PROGRAM, "tc_mbom", DATES)
m10_oracle  = load_bom_set(PROGRAM, "oracle",  DATES)

print("tc_mbom shape:", m10_tc.shape, "oracle shape:", m10_oracle.shape)

tc_log, tc_sum = flips(m10_tc, "TC MBOM")
or_log, or_sum = flips(m10_oracle, "Oracle MBOM")

cmp_oracle_vs_tc = compare_two(m10_oracle, m10_tc, "oracle","tc_mbom")

display(tc_sum.sort_values("Date"))
display(or_sum.sort_values("Date"))
display(cmp_oracle_vs_tc.sort_values("Date"))