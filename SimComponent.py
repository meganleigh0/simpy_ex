import pandas as pd
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. Get Unique Parts from each DataFrame
# -----------------------------------------------------------------------------
# Assuming your DataFrames are:
# LATEST_RELEASE_TC (EBOM)
# LATEST_RELEASE_TC_MBOM (TeamCenter MBOM)
# oracle_df (Oracle MBOM)
# and each has a column "PART_NUMBER"

ebom_unique_parts = LATEST_RELEASE_TC["PART_NUMBER"].unique()
mbom_tc_unique_parts = LATEST_RELEASE_TC_MBOM["PART_NUMBER"].unique()
mbom_oracle_unique_parts = oracle_df["PART_NUMBER"].unique()

num_ebom = len(ebom_unique_parts)
num_mbom_tc = len(mbom_tc_unique_parts)
num_mbom_oracle = len(mbom_oracle_unique_parts)

# For combined total unique parts across *all* dataframes:
all_unique_parts = set(ebom_unique_parts).union(
    set(mbom_tc_unique_parts), 
    set(mbom_oracle_unique_parts)
)
num_all_unique = len(all_unique_parts)

# -----------------------------------------------------------------------------
# 2. Parse Suffixes (Optional)
# -----------------------------------------------------------------------------
# If you want to split PART_NUMBER like "1234_p" into "1234" and "p", you can:
def split_part_suffix(part_number):
    # Check if there's an underscore in the part_number
    if '_' in part_number:
        base, suffix = part_number.rsplit('_', 1)
        return base, suffix
    else:
        # No underscore, treat the entire part as base, no suffix
        return part_number, None

# Create new columns in each dataframe
LATEST_RELEASE_TC['BASE_PART'], LATEST_RELEASE_TC['SUFFIX'] = zip(*LATEST_RELEASE_TC['PART_NUMBER'].apply(split_part_suffix))
LATEST_RELEASE_TC_MBOM['BASE_PART'], LATEST_RELEASE_TC_MBOM['SUFFIX'] = zip(*LATEST_RELEASE_TC_MBOM['PART_NUMBER'].apply(split_part_suffix))
oracle_df['BASE_PART'], oracle_df['SUFFIX'] = zip(*oracle_df['PART_NUMBER'].apply(split_part_suffix))

# -----------------------------------------------------------------------------
# 3. Print Metrics / Summary
# -----------------------------------------------------------------------------
print("Number of unique EBOM parts:", num_ebom)
print("Number of unique MBOM (TeamCenter) parts:", num_mbom_tc)
print("Number of unique MBOM (Oracle) parts:", num_mbom_oracle)
print("Total unique parts across all sources:", num_all_unique)

# Optionally compute intersections if needed:
ebom_set = set(ebom_unique_parts)
mbom_tc_set = set(mbom_tc_unique_parts)
mbom_oracle_set = set(mbom_oracle_unique_parts)

common_ebom_mbom_tc = ebom_set.intersection(mbom_tc_set)
common_ebom_mbom_oracle = ebom_set.intersection(mbom_oracle_set)
common_mbom_tc_oracle = mbom_tc_set.intersection(mbom_oracle_set)
common_all_three = ebom_set.intersection(mbom_tc_set, mbom_oracle_set)

print("Intersection EBOM & MBOM (TeamCenter):", len(common_ebom_mbom_tc))
print("Intersection EBOM & MBOM (Oracle):", len(common_ebom_mbom_oracle))
print("Intersection MBOM (TeamCenter) & MBOM (Oracle):", len(common_mbom_tc_oracle))
print("Intersection across all three:", len(common_all_three))

# -----------------------------------------------------------------------------
# 4. Create a Plotly Visualization
# -----------------------------------------------------------------------------
# A simple bar chart of the counts:
data_to_plot = pd.DataFrame({
    'Source': [
        'EBOM', 
        'MBOM_TC', 
        'MBOM_Oracle',
        'All_Combined'
    ],
    'Unique_Count': [
        num_ebom,
        num_mbom_tc,
        num_mbom_oracle,
        num_all_unique
    ]
})

fig = px.bar(
    data_to_plot,
    x='Source',
    y='Unique_Count',
    title='Unique Part Counts from EBOM, MBOM (TC), and MBOM (Oracle)',
    text='Unique_Count'
)

# Improve layout (optional)
fig.update_layout(
    xaxis_title='Source',
    yaxis_title='Count of Unique Parts',
    uniformtext_minsize=8,
    uniformtext_mode='hide'
)

fig.show()
