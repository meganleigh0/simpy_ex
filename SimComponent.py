import pandas as pd
import os
from pyspark.sql import SparkSession

# -------------------------------
# 1. Define your variant/date config
# -------------------------------
config = [
    {"program": "xm30", "date": "02-20-2025"},
    {"program": "xy22", "date": "03-10-2025"},
    {"program": "zq19", "date": "03-25-2025"}
    # Add more if needed
]

# -------------------------------
# 2. Define comparison and match logic
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
# 3. Process all configs
# -------------------------------
for entry in config:
    program = entry["program"]
    snapshot_date = entry["date"]
    variant_id = f"{program}_{snapshot_date}"

    print(f"Processing: {variant_id}")

    # Load silver files
    ebom_path = f"/Volumes/poc/default/silver_boms_{program}/cleaned_ebom_{snapshot_date}.csv"
    mbom_tc_path = f"/Volumes/poc/default/silver_boms_{program}/cleaned_mbom_tc_{snapshot_date}.csv"
    mbom_oracle_path = f"/Volumes/poc/default/silver_boms_{program}/cleaned_mbom_oracle_{snapshot_date}.csv"

    # Read CSVs
    ebom_df = pd.read_csv(ebom_path)
    ebom_df.rename(columns={"Make/Buy": "Make or Buy", "Qty": "Quantity"}, inplace=True)
    mbom_tc_df = pd.read_csv(mbom_tc_path)
    mbom_oracle_df = pd.read_csv(mbom_oracle_path)

    # Compare BOMs
    combined_df = compare_bom_data(ebom_df, mbom_tc_df, mbom_oracle_df)

    # Calculate completion metrics
    snapshot_df = calculate_bom_completion(combined_df, snapshot_date, variant_id)

    # Save to Gold Layer
    snapshot_spark_df = spark.createDataFrame(snapshot_df)
    gold_path = f"/Volumes/poc/default/gold_bom_snapshot/{program}/{snapshot_date}"
    snapshot_spark_df.write.format("delta").mode("overwrite").save(gold_path)

    # Register table for dashboard access
    spark.sql(f"""
        CREATE OR REPLACE TABLE gold_{program}_bom_completion_snapshot
        USING DELTA
        LOCATION '{gold_path}'
    """)

print("All snapshots processed and saved to gold.")