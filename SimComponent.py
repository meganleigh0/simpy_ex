# ----------  TC MBOM — load, clean, concat  ----------
# Assumes `program` (e.g. "abrams") and `dates` (list/Series of snapshot strings) are already defined.

tc_mboms = []

for date in dates:
    path = f"data/bronze_boms/{program}/{program}_mbom_tc_{date}.xlsm"

    df = (
        pd.read_excel(path, engine="openpyxl")                         # read file
          .rename(                                                     # normalise critical column names
              columns={
                  "Part Number": "PART_NUMBER",
                  "PART_NUMBER": "PART_NUMBER",
                  "Make or Buy": "Make/Buy",
                  "Make/Buy": "Make/Buy"
              }
          )
    )

    # keep just the fields we care about (if they exist)
    wanted = ["PART_NUMBER", "Item Name", "Make/Buy", "Level"]
    df = df[[c for c in wanted if c in df.columns]].copy()

    df["Date"] = pd.to_datetime(date)                                  # add snapshot date

    df = df.loc[:, ~df.columns.duplicated(keep="first")]               # *guarantee* no dup-column names
    tc_mboms.append(df.reset_index(drop=True))

# one tidy DataFrame with every snapshot
all_tc_mboms = (
    pd.concat(tc_mboms, ignore_index=True)                             # stack rows
      .loc[:, lambda x: ~x.columns.duplicated(keep="first")]           # safety-check once more
)

display(all_tc_mboms.head())                                           # or print(all_tc_mboms)