import os, re, pandas as pd

# ──────────────────────────────────────────────────────────────
# 1.  helper – pull MM-DD-YYYY from each dictionary key
# ──────────────────────────────────────────────────────────────
DATE_RE = re.compile(r'_(\d{2}-\d{2}-\d{4})')   # e.g. _06-23-2025

def extract_date(path: str) -> pd.Timestamp:
    m = DATE_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"No MM-DD-YYYY date in: {path}")
    return pd.to_datetime(m.group(1), format='%m-%d-%Y')

# ──────────────────────────────────────────────────────────────
# 2.  build one long DataFrame
# ──────────────────────────────────────────────────────────────
frames = []

# Oracle MBOMs  -------------------------------------------------
for path, df in oracle_mboms.items():
    sub = df[['PART_NUMBER', 'Make/Buy']].copy()
    sub.columns = ['part_number', 'make_buy']
    sub['make_buy']      = sub['make_buy'].astype(str).str.strip().str[0].str.upper()
    sub['snapshot_date'] = extract_date(path)
    sub['system']        = 'Oracle'
    frames.append(sub)

# TeamCenter MBOMs  --------------------------------------------
for path, df in tc_mboms.items():
    sub = df[['PART_NUMBER', 'Make or Buy']].copy()
    sub.columns = ['part_number', 'make_buy']
    sub['make_buy']      = sub['make_buy'].astype(str).str.strip().str[0].str.upper()
    sub['snapshot_date'] = extract_date(path)
    sub['system']        = 'TeamCenter'
    frames.append(sub)

# one big table
all_mboms_long = pd.concat(frames, ignore_index=True)

# ──────────────────────────────────────────────────────────────
# 3.  detect flips (M ↔ B) for every part
# ──────────────────────────────────────────────────────────────
all_mboms_long.sort_values(['system', 'part_number', 'snapshot_date'], inplace=True)

all_mboms_long['prev_flag'] = (
    all_mboms_long.groupby(['system', 'part_number'])['make_buy'].shift()
)

flips = all_mboms_long[
    all_mboms_long['prev_flag'].notna() &
    (all_mboms_long['make_buy'] != all_mboms_long['prev_flag'])
].copy()

flips['change_dir'] = flips['prev_flag'] + '→' + flips['make_buy']

# ──────────────────────────────────────────────────────────────
# 4.  weekly roll-up of flips
# ──────────────────────────────────────────────────────────────
flips['week_start'] = flips['snapshot_date'].dt.to_period('W').start_time

weekly_flip_summary = (
    flips.groupby(['system', 'week_start', 'change_dir'])
         .agg(n_parts=('part_number', 'nunique'),
              parts_changed=('part_number', lambda s: sorted(s.unique())))
         .reset_index()
         .sort_values(['system', 'week_start'])
)

# ──────────────────────────────────────────────────────────────
# 5.  quick sanity-check
# ──────────────────────────────────────────────────────────────
print("All MBOM rows:", len(all_mboms_long))
print("Flips detected:", len(flips))
print("\nSample weekly summary:")
display(weekly_flip_summary.head())