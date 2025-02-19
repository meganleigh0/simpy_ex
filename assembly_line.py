
# Define the values
ebom = 4871
mbom_tc = 2205
mbom_oracle = 1657
intersect_ebom_tc = 1981
intersect_ebom_oracle = 1418
intersect_tc_oracle = 1429
intersect_all_three = 1218

# Create the Venn diagram
plt.figure(figsize=(6,6))
venn = venn3(
    subsets=(
        ebom,          # Only in eBOM
        mbom_tc,       # Only in mBOM_TC
        intersect_ebom_tc - intersect_all_three,  # eBOM & mBOM_TC only
        mbom_oracle,   # Only in mBOM_Oracle
        intersect_ebom_oracle - intersect_all_three,  # eBOM & mBOM_Oracle only
        intersect_tc_oracle - intersect_all_three,  # mBOM_TC & mBOM_Oracle only
        intersect_all_three  # All three
    ),
    set_labels=('eBOM', 'mBOM_TC', 'mBOM_Oracle')
)

# Display the diagram
plt.title("Venn Diagram of BOM Intersections")
plt.show()
