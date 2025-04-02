import pandas as pd
import yaml
from datetime import datetime
from pyspark.sql import SparkSession

# Load config file
config_path = "/Workspace/Repos/your_user/config.yaml"  # update as needed
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

def calculate_bom_completion_detailed(combined_df, snapshot_date, variant_id):
    records, bars = [], []

    for source, qty_col, match_flag in [
        ('TeamCenter', 'quantity_mbom_tc', 'Match_EBOM_MBOM_TC'),
        ('Oracle', 'Quantity', 'Match_EBOM_MBOM_Oracle')
    ]:
        for mob in ['Make', 'Buy']:
            ebom_parts = combined_df[combined_df['Make_or_Buy_ebom'] == mob]
            total_parts = ebom_parts['PART_NUMBER'].nunique()
            matched_parts = ebom_parts[ebom_parts[match_flag] == True]['PART_NUMBER'].nunique()
            qty_mismatch_parts = ebom_parts[
                (ebom_parts[qty_col].notnull()) & (ebom_parts[match_flag] == False)
            ]['PART_NUMBER'].nunique()
            missing_parts = ebom_parts[ebom_parts[qty_col].isnull()]['PART_NUMBER'].nunique()
            percent_matched = (matched_parts / total_parts * 100) if total_parts > 0 else 0

            label = f"{source} - {mob}"
            bars.append({'label': label, 'value': percent_matched})
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

# Process all programs
for entry in config['programs']:
    program = entry['name']
    snapshot_date = entry['snapshot_date']
    variant_id = f"{program}_{snapshot_date}"

    # Construct path to clean BOM CSV in silver volume
    csv_path = f"/Volumes/POC/default/silver_boms_{program}/clean_bom_{snapshot_date}.csv"
    combined_df = pd.read_csv(csv_path)

    # Calculate snapshot metrics
    snapshot_df = calculate_bom_completion_detailed(combined_df, snapshot_date, variant_id)

    # Convert to Spark DataFrame
    snapshot_spark_df = spark.createDataFrame(snapshot_df)

    # Define gold layer path
    gold_path = f"/Volumes/POC/default/gold_bom_snapshot/{program}"

    # Save to Delta (overwrite old snapshot if exists)
    snapshot_spark_df.write.format("delta").mode("overwrite").save(gold_path)

    # Register as table for dashboard use
    spark.sql(f"""
        CREATE OR REPLACE TABLE gold_{program}_bom_completion_snapshot
        USING DELTA
        LOCATION '{gold_path}'
    """)

print("All snapshots processed and saved to gold layer.")