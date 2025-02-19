import matplotlib.pyplot as plt
from matplotlib_venn import venn3

# Define the dataset
ebom_only = 4871
mbom_tc_only = 2205
mbom_oracle_only = 1657
ebom_tc_intersect = 1981
ebom_oracle_intersect = 1418
tc_oracle_intersect = 1429
all_three_intersect = 1218

# Adjusted values for the Venn diagram
ebom_tc_only = ebom_tc_intersect - all_three_intersect
ebom_oracle_only = ebom_oracle_intersect - all_three_intersect
tc_oracle_only = tc_oracle_intersect - all_three_intersect

# Create the Venn diagram
plt.figure(figsize=(6, 6))
venn = venn3(
    subsets=(
        ebom_only,       # Only in eBOM
        mbom_tc_only,    # Only in mBOM_TC
        ebom_tc_only,    # eBOM & mBOM_TC only (excluding all three)
        mbom_oracle_only,  # Only in mBOM_Oracle
        ebom_oracle_only
