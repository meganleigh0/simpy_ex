import os, re, pandas as pd

# ────────────────────────────────────────────────────────────────
# helper: pull MM-DD-YYYY from the file name
# ────────────────────────────────────────────────────────────────
DATE_RE = re.compile(r'(\d{2}-\d{2}-\d{4})')        # match 06-23-2025, etc.

def snap_date(path):
    m = DATE_RE.search(os.path.basename(path))
    return pd.to_datetime(m.group(1), format='%m-%d-%Y') if m else pd.NaT

# ────────────────────────────────────────────────────────────────
# helper: squash weird spacing / case in column labels
# ────────────────────────────────────────────────────────────────
def normalise_cols(df):
    """
    'Item No'  ->  'itemno'
    'Make or Buy' -> 'makeorbuy'
    """
    df = df.copy()
    df.columns = [re.sub(r'[\s_\-]+', '', str(c)).lower() for c in df.columns]
    return df

# ────────────────────────────────────────────────────────────────
# turn one of the dictionaries into a tidy long-form DataFrame
# ────────────────────────────────────────────────────────────────
def stack_mboms(mboms_dict, system):
    frames = []

    for path, df in mboms_dict.items():
        sdf = normalise_cols(df)

        # Oracle files say 'itemno', TeamCenter 'partnumber' → grab whichever exists
        part_col = next((c for c in sdf.columns if c in ('itemno', 'partnumber')), None)
        make_col = next((c for c in sdf.columns if 'make' in c and 'buy' in c), None)

        if part_col is None or make_col is None:
            print(f"⚠️  Skipping {os.path.basename(path)} – missing part/make columns")
            continue

        tmp = sdf[[part_col, make_col]].copy()
        tmp.columns = ['part_number', 'make_buy']
        tmp['make_buy']      = tmp['make_buy'].astype(str).str.strip().str[0].str.upper()
        tmp['snapshot_date'] = snap_date(path)
        tmp['system']        = system
        frames.append(tmp)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ────────────────────────────────────────────────────────────────
# 1. build the long table
# ────────────────────────────────────────────────────────────────
oracle_long = stack_mboms(oracle_mboms,   'Oracle')
tc_long     = stack_mboms(tc_mboms,       'TeamCenter')
all_mboms_long = pd.concat([oracle_long, tc_long], ignore_index=True)

# ────────────────────────────────────────────────────────────────
# 2. find status flips part-by-part
# ────────────────────────────────────────────────────────────────
all_mboms_long = all_mboms_long.dropna(subset=['snapshot_date'])              # safety
all_mboms_long = all_mboms_long.sort_values(['system','part_number','snapshot_date'])

# previous flag for each part
all_mboms_long['prev_flag'] = (
    all_mboms_long.groupby(['system','part_number'])['make_buy'].shift()
)

# rows where flag changed (M→B or B→M)
flips = all_mboms_long[
    all_mboms_long['prev_flag'].notna() &
    (all_mboms_long['make_buy'] != all_mboms_long['prev_flag'])
].copy()

flips['change_dir'] = flips['prev_flag'] + '→' + flips['make_buy']

# ────────────────────────────────────────────────────────────────
# 3. roll up by calendar week
# ────────────────────────────────────────────────────────────────
flips['week'] = flips['snapshot_date'].dt.to_period('W').apply(lambda r: r.start_time)

weekly_flip_summary = (flips
        .groupby(['system','week','change_dir'])
        .agg(parts_changed=('part_number', lambda s: sorted(s.unique())),
             n_parts       =('part_number', 'nunique'))
        .reset_index())

# ────────────────────────────────────────────────────────────────
# 4. quick sanity-check in the notebook
# ────────────────────────────────────────────────────────────────
print("\nAll MBOM rows:", len(all_mboms_long))
print("Flips detected :", len(flips))
print("\n=== Sample of weekly flip summary ===")
display(weekly_flip_summary.head())