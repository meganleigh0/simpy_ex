import re, os, pandas as pd

# ---------- helper to pull MM-DD-YYYY from the file name ----------
def get_snapshot_date(path, pattern=r'(\d{2}-\d{2}-\d{4})'):
    return pd.to_datetime(re.search(pattern, os.path.basename(path)).group(1),
                          format='%m-%d-%Y')

# ---------- choose YOUR exact column names here ----------
# Oracle workbook columns
ORACLE_PART_COL = "Item No"
ORACLE_MB_COL   = "Make/Buy"

# TeamCenter workbook columns
TC_PART_COL = "PART_NUMBER"
TC_MB_COL   = "Make or Buy"

# ---------- build a list of tidy DataFrames ----------
frames = []

# 1️⃣  Oracle files
for path, df in oracle_mboms.items():
    sub = df[[ORACLE_PART_COL, ORACLE_MB_COL]].copy()
    sub.columns = ["part_number", "make_buy"]
    sub["make_buy"]      = sub["make_buy"].astype(str).str.strip().str[0].str.upper()
    sub["snapshot_date"] = get_snapshot_date(path)
    sub["system"]        = "Oracle"
    frames.append(sub)

# 2️⃣  TeamCenter files
for path, df in tc_mboms.items():
    sub = df[[TC_PART_COL, TC_MB_COL]].copy()
    sub.columns = ["part_number", "make_buy"]
    sub["make_buy"]      = sub["make_buy"].astype(str).str.strip().str[0].str.upper()
    sub["snapshot_date"] = get_snapshot_date(path)
    sub["system"]        = "TeamCenter"
    frames.append(sub)

# ---------- one big tidy table ----------
all_mboms_long = pd.concat(frames, ignore_index=True)

# quick peek
print(all_mboms_long.head())
print(f"\nTotal rows: {len(all_mboms_long):,}")