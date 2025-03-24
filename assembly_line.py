import os
import re
import glob
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

def parse_filename_date(filename_pattern):
    """
    Extracts the <month>_<day>_<year> portion from the filename using regex
    and converts it to a Python datetime. 
    Example filename: program_A_compare_all_1_20_25.csv  ->  01/20/2025
    """
    # Regex group to capture something like 1_20_25 or 12_2_25
    match = re.search(r'program_A_compare_all_(\d+_\d+_\d+)\.csv', filename_pattern)
    if not match:
        return None
    
    date_str = match.group(1)  # e.g. "1_20_25"
    # Parse into a datetime object
    try:
        parsed_date = datetime.strptime(date_str, '%m_%d_%y')
        return parsed_date
    except ValueError:
        return None

def compute_coverage_metrics(df):
    """
    Given a single 'compare_all' DataFrame, compute:
      - total parts
      - count of True for Template_Match_EBOM_TC
      - count of True for Template_Match_EBOM_Oracle
      - coverage (True/total) for TC and Oracle
    """
    total_parts = len(df)
    tc_true_count = df['Template_Match_EBOM_TC'].sum()            # sum of True == sum of 1.0 for booleans
    oracle_true_count = df['Template_Match_EBOM_Oracle'].sum()
    
    coverage_tc = tc_true_count / total_parts if total_parts > 0 else 0
    coverage_oracle = oracle_true_count / total_parts if total_parts > 0 else 0
    
    return total_parts, tc_true_count, oracle_true_count, coverage_tc, coverage_oracle

# -----------------------------------------------------------------------------
# 1. Read multiple CSVs matching a pattern, parse date, compute coverage
# -----------------------------------------------------------------------------
coverage_data = []
for file_path in glob.glob("program_A_compare_all_*.csv"):
    # Parse date from filename
    file_date = parse_filename_date(os.path.basename(file_path))
    if file_date is None:
        continue  # skip files that do not match the pattern
    
    # Read the DataFrame
    df = pd.read_csv(file_path)
    
    # Compute coverage metrics
    total, tc_true, oracle_true, coverage_tc, coverage_oracle = compute_coverage_metrics(df)
    
    coverage_data.append({
        'File': file_path,
        'Date': file_date,
        'TotalParts': total,
        'TC_TrueCount': tc_true,
        'Oracle_TrueCount': oracle_true,
        'Coverage_TC': coverage_tc,          # ratio 0.0 -> 1.0
        'Coverage_Oracle': coverage_oracle   # ratio 0.0 -> 1.0
    })

# If no files found, coverage_data remains empty
if not coverage_data:
    print("No matching 'program_A_compare_all_*.csv' files found.")
else:
    # -----------------------------------------------------------------------------
    # 2. Build a summary DataFrame of coverage vs. date
    # -----------------------------------------------------------------------------
    coverage_df = pd.DataFrame(coverage_data)
    coverage_df.sort_values('Date', inplace=True)

    # Convert coverage to percentage if desired
    coverage_df['Coverage_TC_pct'] = coverage_df['Coverage_TC'] * 100
    coverage_df['Coverage_Oracle_pct'] = coverage_df['Coverage_Oracle'] * 100

    print("Coverage DataFrame:")
    print(coverage_df)

    # -----------------------------------------------------------------------------
    # 3. Plot Weekly (or Daily) Coverage Trend as Bar Charts
    #    - One bar chart for TC, another for Oracle
    # -----------------------------------------------------------------------------
    
    # A) Teamcenter coverage bar chart
    plt.figure()
    plt.bar(coverage_df['Date'].astype(str), coverage_df['Coverage_TC_pct'])
    plt.title("Teamcenter Coverage Over Time")
    plt.xlabel("Date")
    plt.ylabel("Coverage (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()  # Avoid label cutoff
    plt.show()

    # B) Oracle coverage bar chart
    plt.figure()
    plt.bar(coverage_df['Date'].astype(str), coverage_df['Coverage_Oracle_pct'])
    plt.title("Oracle Coverage Over Time")
    plt.xlabel("Date")
    plt.ylabel("Coverage (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------------------
    # 4. Pie Chart Showing Oracle Coverage for the Most Recent Date
    # -----------------------------------------------------------------------------
    most_recent_date = coverage_df['Date'].max()
    row = coverage_df.loc[coverage_df['Date'] == most_recent_date].iloc[0]
    matched = row['Oracle_TrueCount']
    total_parts = row['TotalParts']
    not_matched = total_parts - matched

    plt.figure()
    plt.pie([matched, not_matched], labels=['Matched', 'Not Matched'], autopct='%1.1f%%')
    plt.title(f"Oracle Coverage on {most_recent_date.strftime('%m/%d/%y')}")
    plt.show()
