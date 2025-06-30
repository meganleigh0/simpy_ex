import pandas as pd, os, re

# ------------------------------------------------------------
# 0.  Helpers
# ------------------------------------------------------------
def get_snapshot_date(path, pattern=r'(\d{2}-\d{2}-\d{4})'):
    """Pull MM-DD-YYYY out of the filename and turn into Timestamp."""
    date_str = re.search(pattern, os.path.basename(path)).group(1)
    return pd.to_datetime(date_str, format='%m-%d-%Y')

def to_long(boms_dict, part_col, make_col):
    """Stack snapshots into one tidy df with part, status, date."""
    frames = []
    for path, df in boms_dict.items():
        snap_date = get_snapshot_date(path)
        sub = (df[[part_col, make_col]]
               .rename(columns={part_col: 'part_number',
                                make_col: 'make_buy'}))
        sub['snapshot_date'] = snap_date
        frames.append(sub)
    long_df = (pd.concat(frames, ignore_index=True)
                 .dropna(subset=['part_number', 'make_buy']))
    # normalise flag → 'M' or 'B'
    long_df['make_buy'] = (long_df['make_buy']
                           .astype(str)
                           .str.strip()
                           .str[0].str.upper())
    return long_df

def status_changes(df_long):
    """Count Make/Buy flips for each part across time-ordered snapshots."""
    # sort so every part’s history is chronological
    df_long = df_long.sort_values(['part_number', 'snapshot_date'])
    
    # identify rows where the status differs from the PREVIOUS snapshot
    df_long['status_changed'] = (
        df_long.groupby('part_number')['make_buy']
               .apply(lambda s: s != s.shift())
    )

    # build summary
    agg = (df_long.groupby('part_number')
           .agg(first_status=('make_buy', 'first'),
                last_status =('make_buy', 'last'),
                n_switches  =('status_changed', lambda x: x.sum() - 1),
                change_dates=('snapshot_date',
                               lambda x: list(x[x.index[x.groupby(level=0).cumcount() > 0 & (df_long.loc[x.index, "status_changed"])] ])))
           .reset_index())
    return agg

# ------------------------------------------------------------
# 1.  Build long-form data for each system
#     (Adjust the column names below if yours differ!)
# ------------------------------------------------------------
oracle_long = to_long(oracle_mboms,
                      part_col='Item No',      # <-- oracle part-no column
                      make_col='Make/Buy')     # <-- oracle make/buy column

tc_long     = to_long(tc_mboms,
                      part_col='PART_NUMBER',  # <-- TC part-no column
                      make_col='Make or Buy')  # <-- TC make/buy column

# ------------------------------------------------------------
# 2.  Summaries
# ------------------------------------------------------------
oracle_switches = status_changes(oracle_long)
tc_switches     = status_changes(tc_long)

# Peek at the result
display(oracle_switches.head())
display(tc_switches.head())