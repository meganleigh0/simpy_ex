import pandas as pd

# --- 0.  starting point -------------------------------------------------
# df  already has columns:  PART_NUMBER | Make/Buy | Date

# --- 1.  basic hygiene --------------------------------------------------
df['Date']      = pd.to_datetime(df['Date'])                     # make sure Date is datetime
df['Make/Buy']  = (df['Make/Buy']
                  .str.strip()                                   # remove stray spaces
                  .str.lower().map({'make':'make', 'buy':'buy'}))# normalise wording

# keep one record per part per day (if duplicates exist, keep the LAST snapshot that day)
df = (df.sort_values(['PART_NUMBER', 'Date'])
        .drop_duplicates(subset=['PART_NUMBER', 'Date'],
                          keep='last'))

# --- 2.  detect status flips -------------------------------------------
# for each part, compare its status with the previous day it appeared
df['changed'] = (df.groupby('PART_NUMBER')['Make/Buy']
                   .apply(lambda s: s.ne(s.shift())))            # True where status ≠ previous

# first row of every part is always True in the test above; mask it out
df.loc[df.groupby('PART_NUMBER').head(1).index, 'changed'] = False

# --- 3.  summarise ------------------------------------------------------
switch_counts = (df[df['changed']]
                 .groupby('PART_NUMBER')
                 .size()
                 .rename('switch_count')
                 .reset_index()
                 .sort_values('switch_count', ascending=False))

# --- 4.  output ---------------------------------------------------------
print("Top parts by # Make↔Buy flips")
print(switch_counts.head(10))