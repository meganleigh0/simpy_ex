import os, re, pandas as pd, warnings

# ------------------------------------------------------------
# 1.  Helpers
# ------------------------------------------------------------
def get_snapshot_date(path, pattern=r'(\d{2}-\d{2}-\d{4})'):
    """
    Extract MM-DD-YYYY from filename and return a Timestamp.
    """
    m = re.search(pattern, os.path.basename(path))
    return pd.to_datetime(m.group(1), format='%m-%d-%Y') if m else pd.NaT

def stack_mboms(mboms_dict, system_name,
                part_aliases=("partnumber", "part-number", "part number",
                              "part_no", "partno", "itemno", "item_no",
                              "item number", "itemno."),
                make_aliases=("make/buy", "make_buy", "makebuy",
                              "make or buy", "makeorbuy")):
    """
    Turn a dict of {file: DataFrame} → one long DataFrame with
    part_number | make_buy | snapshot_date | system
    """
    frames = []

    for path, df in mboms_dict.items():
        # 1️⃣ normalise column labels → lowercase, no spaces, underscores, hyphens
        df = df.rename(columns=lambda c: re.sub(r'[\s_\-]+', '', str(c)).lower())

        # 2️⃣ locate the two columns we need
        part_col = next((c for c in df.columns if c in part_aliases), None)
        make_col = next((c for c in df.columns if c in make_aliases), None)

        if not part_col or not make_col:
            warnings.warn(f"Skipping {os.path.basename(path)} – "
                          f"couldn't find part or make/buy column")
            continue

        sub = (df[[part_col, make_col]].copy()
                 .rename(columns={part_col: 'part_number',
                                  make_col: 'make_buy'}))

        # 3️⃣ house-keeping
        sub['snapshot_date'] = get_snapshot_date(path)
        sub['system']        = system_name
        sub['make_buy']      = (sub['make_buy']
                                .astype(str)
                                .str.strip()
                                .str[0]         # first letter
                                .str.upper())   # → 'M' or 'B'

        frames.append(sub)

    return (pd.concat(frames, ignore_index=True)
            if frames else
            pd.DataFrame(columns=['part_number', 'make_buy',
                                  'snapshot_date', 'system']))

# ------------------------------------------------------------
# 2.  Build the long tables and combine
# ------------------------------------------------------------
oracle_long = stack_mboms(oracle_mboms, system_name='Oracle')
tc_long     = stack_mboms(tc_mboms,     system_name='TeamCenter')

all_mboms_long = pd.concat([oracle_long, tc_long], ignore_index=True)

# optional peek
print(all_mboms_long.head())
print(f"\nRows: {len(all_mboms_long):,} | Parts: {all_mboms_long['part_number'].nunique():,}")