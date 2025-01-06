dist_df_10 = dist_df.copy()
dist_df_10['Total_Days_for_10'] = dist_df_10['Cycle_Time'] * 10

# Find the worst-case scenario (i.e., max) among all simulations
worst_case_10 = dist_df_10['Total_Days_for_10'].max()

# ------------------------------------------------------------------
# Altair chart: histogram of total days for 10 vehicles, 
# plus a vertical rule for the worst-case scenario
# ------------------------------------------------------------------
base_hist = (
    alt.Chart(dist_df_10)
    .mark_bar(opacity=0.5)
    .encode(
        x=alt.X('Total_Days_for_10:Q', bin=alt.Bin(maxbins=50), 
                title='Total Days (for 10 vehicles)'),
        y=alt.Y('count()', title='Count'),
        color='Plant:N',  # color by Plant
        tooltip=['Plant:N', 'Total_Days_for_10:Q']
    )
    .properties(
        width=600, 
        height=300,
        title='Monte Carlo Worst-Case Scenario: 10 Vehicles'
    )
)

# Add a vertical line (rule) at the worst-case value
rule = (
    alt.Chart(pd.DataFrame({'WorstCase': [worst_case_10]}))
    .mark_rule(color='red', strokeDash=[5, 5])
    .encode(x='WorstCase:Q')
    .properties()
)

chart_worst_case_10 = (base_hist + rule).interactive()

# Display in a notebook environment:
