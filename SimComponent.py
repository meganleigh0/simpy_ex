import os, re, pandas as pd

# ───────── 1.  pull MM-DD-YYYY out of each filename ──────────
DATE_RE = re.compile(r'(\d{2}-\d{2}-\d{4})')     # 06-30-2025 etc.

def snapshot_date(path: str) -> pd.Timestamp | pd.NaT:
    m = DATE_RE.search(os.path.basename(path))
    return pd.to_datetime(m.group(1), format='%m-%d-%Y') if m else pd.NaT


# ───────── 2.  build one tidy table from both dictionaries ───
frames = []

for system, mboms in [('Oracle', oracle_mboms), ('TeamCenter', tc_mboms)]:
    for path, df in mboms.items():

        # find the two columns, ignoring upper/lower/spaces/slashes
        make_col = next(
            (c for c in df.columns
             if 'make' in c.lower() and 'buy' in c.lower()),
            None
        )
        part_col = next(
            (c for c in df.columns
             if 'part' in c.lower() and 'num'  in c.lower()),
            None
        )

        if make_col is None or part_col is None:
            print(f"⚠️  {os.path.basename(path)} skipped – column not found")
            continue

        sub = df[[part_col, make_col]].copy()
        sub.columns      = ['part_number', 'make_buy']
        sub['make_buy']  = sub['make_buy'].astype(str).str.strip().str[0].str.upper()
        sub['system']    = system
        sub['snapshot_date'] = snapshot_date(path)
        frames.append(sub)

all_mboms_long = pd.concat(frames, ignore_index=True)


# ───────── 3.  detect every Make↔Buy flip per part ───────────
all_mboms_long.sort_values(['system','part_number','snapshot_date'], inplace=True)

all_mboms_long['prev_flag'] = (
    all_mboms_long.groupby(['system','part_number'])['make_buy'].shift()
)

flips = all_mboms_long[
    all_mboms_long['prev_flag'].notna() &
    (all_mboms_long['make_buy'] != all_mboms_long['prev_flag'])
].copy()

flips['change_dir'] = flips['prev_flag'] + '→' + flips['make_buy']


# ───────── 4.  weekly roll-up of those flips ────────────────
flips['week_start'] = flips['snapshot_date'].dt.to_period('W').start_time

weekly_flip_summary = (
    flips.groupby(['system','week_start','change_dir'])
         .agg(n_parts=('part_number','nunique'),
              parts_changed=('part_number', lambda s: sorted(s.unique())))
         .reset_index()
         .sort_values(['system','week_start'])
)


# ───────── 5.  quick sanity-check (feel free to delete) ─────
print(f"All rows loaded : {len(all_mboms_long):,}")
print(f"Flips detected  : {len(flips):,}")
display(weekly_flip_summary.head())