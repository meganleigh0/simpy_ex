import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession

# -------------------------------
# 1. Define your config
# -------------------------------
config = {
    "xm30": ["02-20-2025", "02-27-2025", "03-06-2025"],
    "xy22": ["03-01-2025", "03-08-2025"],
    "zq19": ["03-15-2025", "03-22-2025"]
}

# -------------------------------
# 2. Comparison and metric functions
# -------------------------------
def compare_bom_data(df_ebom, df_mbom_tc, df_mbom_oracle):
    group_df_ebom = pd.DataFrame(df_ebom.groupby("PART_NUMBER").agg({"Quantity": "sum", "Make or Buy": "first"})).reset_index()
    group_df_mbom_tc = pd.DataFrame(df_mbom_tc.groupby("PART_NUMBER").agg({"Quantity": "sum", "Make or Buy": "first"}))
    group_df_mbom_oracle = pd.DataFrame(df_mbom_oracle.groupby("PART_NUMBER").agg({"Quantity": "sum", "Make or Buy": "first"}))

    merged_df = group_df_ebom.merge(group_df_mbom_tc, on="PART_NUMBER", how="left", suffixes=("_ebom", "_mbom_tc"))
    combined_df = merged_df.merge(group_df_mbom_oracle, on="PART_NUMBER", how="left", suffixes=("", "_mbom_oracle"))

    combined_df["Match_EBOM_MBOM_TC"] = combined_df["Quantity_ebom"] == combined_df["Quantity_mbom_tc"]
    combined_df["Match_EBOM_MBOM_Oracle"] = combined_df["Quantity_ebom"] == combined_df["Quantity"]
    combined_df["Match_MBOM_TC_MBOM_Oracle"] = combined_df["Quantity_mbom_tc"] == combined_df["Quantity"]

    return combined_df

def calculate_bom_completion(combined_df, snapshot_date, variant_id):
    records = []
    for source, qty_col, match_flag in [
        ('TeamCenter', 'Quantity_mbom_tc', 'Match_EBOM_MBOM_TC'),
        ('Oracle', 'Quantity', 'Match_EBOM_MBOM_Oracle')
    ]:
        for mob in ['Make', 'Buy']:
            ebom_parts = combined_df[combined_df['Make or Buy_ebom'] == mob]
            total_parts = ebom_parts['PART_NUMBER'].nunique()
            matched_parts = ebom_parts[ebom_parts[match_flag] == True]['PART_NUMBER'].nunique()
            qty_mismatch_parts = ebom_parts[
                (ebom_parts[qty_col].notnull()) & (ebom_parts[match_flag] == False)
            ]['PART_NUMBER'].nunique()
            missing_parts = ebom_parts[ebom_parts[qty_col].isnull()]['PART_NUMBER'].nunique()
            percent_matched = (matched_parts / total_parts * 100) if total_parts > 0 else 0

            records.append({
                'snapshot_date': snapshot_date,
                'variant_id': variant_id,
                'source': source,
                'make_or_buy': mob,
                'total_parts': total_parts,
                'matched_parts': matched_parts,
                'quantity_mismatches': qty_mismatch_parts,
                'missing_parts': missing_parts,
                'percent_matched': percent_matched
            })

    return pd.DataFrame(records)

# -------------------------------
# 3. Process all snapshots and save
# -------------------------------
for program, dates in config.items():
    all_snapshots = []

    for snapshot_date in dates:
        variant_id = f"{program}_{snapshot_date}"
        print(f"Processing: {variant_id}")

        try:
            # Load silver data
            ebom_path = f"/Volumes/poc/default/silver_boms_{program}/cleaned_ebom_{snapshot_date}.csv"
            mbom_tc_path = f"/Volumes/poc/default/silver_boms_{program}/cleaned_mbom_tc_{snapshot_date}.csv"
            mbom_oracle_path = f"/Volumes/poc/default/silver_boms_{program}/cleaned_mbom_oracle_{snapshot_date}.csv"

            ebom_df = pd.read_csv(ebom_path)
            ebom_df.rename(columns={"Make/Buy": "Make or Buy", "Qty": "Quantity"}, inplace=True)
            mbom_tc_df = pd.read_csv(mbom_tc_path)
            mbom_oracle_df = pd.read_csv(mbom_oracle_path)

            # Compare & calculate snapshot
            combined_df = compare_bom_data(ebom_df, mbom_tc_df, mbom_oracle_df)
            snapshot_df = calculate_bom_completion(combined_df, snapshot_date, variant_id)
            all_snapshots.append(snapshot_df)

        except Exception as e:
            print(f"Error processing {variant_id}: {e}")
            continue

    # Save all snapshots for this program to Delta + SQL Table
    if all_snapshots:
        full_df = pd.concat(all_snapshots, ignore_index=True)
        spark_df = spark.createDataFrame(full_df)

        # Save to gold volume path
        gold_path = f"/Volumes/poc/default/gold_bom_snapshot/{program}"
        spark_df.write.format("delta").mode("append").save(gold_path)

        # Register SQL table using delta.`<path>` for dashboard access
        spark.sql(f"""
            CREATE OR REPLACE TABLE {program}_bom_completion_snapshot
            AS SELECT * FROM delta.`{gold_path}`
        """)

        print(f"Saved and updated SQL table: {program}_bom_completion_snapshot")

print("All programs processed and SQL tables registered.")


SELECT snapshot_date, source, make_or_buy, percent_matched
FROM xm30_bom_completion_snapshot
ORDER BY snapshot_date