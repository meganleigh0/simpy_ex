import os, re, pandas as pd

# ------------ tweak if your date format differs ------------
DATE_RE = re.compile(r'(\d{2}-\d{2}-\d{4})')   # MM-DD-YYYY in file name

def snap_date(path):
    """Return a pandas Timestamp extracted from the file path."""
    return pd.to_datetime(DATE_RE.search(os.path.basename(path)).group(1),
                          format='%m-%d-%Y')

frames = []

# ---------- Oracle ----------
for path, df in oracle_mboms.items():
    tmp = df[['Item No', 'Make/Buy']].copy()      # ← your exact column names
    tmp.columns = ['part_number', 'make_buy']
    tmp['make_buy']      = tmp['make_buy'].astype(str).str.strip().str[0].str.upper()
    tmp['snapshot_date'] = snap_date(path)
    tmp['system']        = 'Oracle'
    frames.append(tmp)

# ---------- TeamCenter ----------
for path, df in tc_mboms.items():
    tmp = df[['PART_NUMBER', 'Make or Buy']].copy()   # ← your exact column names
    tmp.columns = ['part_number', 'make_buy']
    tmp['make_buy']      = tmp['make_buy'].astype(str).str.strip().str[0].str.upper()
    tmp['snapshot_date'] = snap_date(path)
    tmp['system']        = 'TeamCenter'
    frames.append(tmp)

# ---------- one big tidy table ----------
all_mboms_long = pd.concat(frames, ignore_index=True)

# quick sanity-check
print(all_mboms_long.head())
print(f"\nRows: {len(all_mboms_long):,} | Unique parts: {all_mboms_long.part_number.nunique():,}")