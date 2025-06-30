import pandas as pd, os, re, warnings

# ------------------------------------------------------------------
# 0.  Quick peek – make sure you know the real column labels
# ------------------------------------------------------------------
sample_oracle = next(iter(oracle_mboms.values()))
sample_tc     = next(iter(tc_mboms.values()))
print("Oracle sample columns:\n", list(sample_oracle.columns), "\n")
print("TC sample columns:\n",     list(sample_tc.columns), "\n")

# ------------------------------------------------------------------
# 1.  Helper: pull date from filename
# ------------------------------------------------------------------
def get_snapshot_date(path, pattern=r'(\d{2}-\d{2}-\d{4})'):
    return pd.to_datetime(re.search(pattern, os.path.basename(path)).group(1),
                          format='%m-%d-%Y')

# ------------------------------------------------------------------
# 2.  Robust stacker
# ------------------------------------------------------------------
def to_long(boms_dict,
            part_aliases=("itemno", "item number", "partnumber", "part-number",
                          "part number", "part_no", "item_no"),
            make_aliases=("make/buy", "make or buy", "makebuy", "make_buy")):
    
    frames = []
    for path, df in boms_dict.items():
        # Normalise labels: "Part Number" -> "partnumber"
        norm_cols = {c: re.sub(r'[\s_\-]+', '', str(c)).lower() for c in df.columns}
        df = df.rename(columns=norm_cols)
        
        # Which columns are present?
        part_col = next((c for c in norm_cols.values() if c in part_aliases), None)
        make_col = next((c for c in norm_cols.values() if c in make_aliases), None)
        
        if not part_col or not make_col:
            warnings.warn(f"Skipping {os.path.basename(path)} – "
                          f'missing columns (got {list(df.columns)[:10]}…)')
            continue
        
        sub = df[[part_col, make_col]].copy()
        sub.columns = ["part_number", "make_buy"]
        sub["snapshot_date"] = get_snapshot_date(path)
        
        # Compact the Make/Buy flag to single letter
        sub["make_buy"] = sub["make_buy"].astype(str).str.strip().str[0].str.upper()
        frames.append(sub)
    
    if not frames:
        raise ValueError("No valid files found – check column aliases above.")
        
    return pd.concat(frames, ignore_index=True)

# ------------------------------------------------------------------
# 3.  Build long-form data
# ------------------------------------------------------------------
oracle_long = to_long(oracle_mboms)   # uses default alias lists
tc_long     = to_long(tc_mboms)

# ------------------------------------------------------------------
# 4.  Function to count flips
# ------------------------------------------------------------------
def status_changes(df_long):
    df_long = df_long.sort_values(["part_number", "snapshot_date"])
    df_long["changed"] = df_long.groupby("part_number")["make_buy"].apply(
        lambda s: s != s.shift())
    
    summary = (df_long.groupby("part_number")
               .agg(first_status=("make_buy", "first"),
                    last_status =("make_buy", "last"),
                    n_switches  =("changed", lambda s: s.sum()-1),
                    change_dates=("snapshot_date",
                                  lambda s: list(s[df_long.loc[s.index, "changed"]][1:])))
               .reset_index())
    return summary

# ------------------------------------------------------------------
# 5.  Summaries
# ------------------------------------------------------------------
oracle_switches = status_changes(oracle_long)
tc_switches     = status_changes(tc_long)

display(oracle_switches.head())
display(tc_switches.head())