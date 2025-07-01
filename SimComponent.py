# Databricks / Jupyter – run in ONE cell
import pandas as pd, glob, os, re

BASE_PATH = "data/bronze/boms"          # adjust if your root is different

def load_all_mboms(base_path: str = BASE_PATH) -> pd.DataFrame:
    """Return one long tidy DF: PART_NUMBER | make_buy | source | snapshot_date"""
    dfs = []
    for f in glob.glob(os.path.join(base_path, "mbom_*_*.csv")):
        # expect mbom_oracle_2025-06-30.csv or mbom_tc_2025-06-30.csv
        m = re.search(r"mbom_(oracle|tc)_(\d{4}-\d{2}-\d{2})", os.path.basename(f))
        if not m:
            continue                       # skip anything that doesn't match the pattern
        src, date_str = m.groups()
        snap_date = pd.to_datetime(date_str)

        df = pd.read_csv(f)
        # unify column names
        if src == "oracle":
            df = df.rename(columns={"Make/Buy": "make_buy"})
        else:                              # tc
            df = df.rename(columns={"Make or Buy": "make_buy"})

        df = df[["PART_NUMBER", "make_buy"]].copy()
        df["source"] = src
        df["snapshot_date"] = snap_date
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No MBOM files matched the expected patterns.")
    return pd.concat(dfs, ignore_index=True)

# ---------- LOAD ----------
combined = load_all_mboms()

# ---------- ORACLE vs TC COMPARISON *WITHIN* EACH SNAPSHOT ----------
same_day_match = (
    combined[combined["source"] == "oracle"]
    .merge(
        combined[combined["source"] == "tc"],
        on=["PART_NUMBER", "snapshot_date"],
        suffixes=("_oracle", "_tc"),
        how="outer",
    )
    .assign(match=lambda d: d["make_buy_oracle"] == d["make_buy_tc"])
)

mismatches = same_day_match[~same_day_match["match"]]

# ---------- MAKE-BUY FLIPS *OVER TIME* INSIDE EACH SOURCE ----------
combined = combined.sort_values(["PART_NUMBER", "source", "snapshot_date"])
combined["prev_make_buy"] = combined.groupby(["PART_NUMBER", "source"])["make_buy"].shift()
combined["flipped"] = combined["make_buy"] != combined["prev_make_buy"]

flips = combined[combined["flipped"]]

# ---------- OPTIONAL WEEKLY SUMMARY ----------
weekly_summary = (
    same_day_match.assign(week=lambda d: d["snapshot_date"].dt.isocalendar().week)
    .groupby("week")["match"]
    .agg(total_parts="size", mismatched_parts=lambda s: (~s).sum())
    .assign(percent_match=lambda d: 100 * (d["total_parts"] - d["mismatched_parts"]) / d["total_parts"])
    .reset_index()
)

# ---------- SAVE BACK TO THE LAKE (if desired) ----------
# spark_df = spark.createDataFrame(combined)   # Databricks: convert to Spark
# spark_df.write.format("delta").mode("overwrite").save("dbfs:/gold/match_history")

# ---- Quick sanity prints ----
print(f"Loaded {combined.shape[0]} total part-records across {combined['snapshot_date'].nunique()} snapshots")
print(f"{mismatches.shape[0]} mismatches between Oracle and TC on the same day")
print(f"{flips.shape[0]} make/buy flips within sources over time")