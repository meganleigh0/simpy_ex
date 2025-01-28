import pandas as pd
import altair as alt

# Example data (your PlantDF DataFrame)
data = {
    'variant': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B', 'C'],
    'section': ['X', 'X', 'X', 'Y', 'Y', 'Y', 'Z', 'Z', 'Z'],
    'number_of_days_spent': [10, 12, 15, 8, 9, 12, 20, 18, 22]
}

# Create the PlantDF DataFrame
PlantDF = pd.DataFrame(data)

# Step 1: Group by section and variant, and calculate the average number of days spent
grouped_df = PlantDF.groupby(['section', 'variant'])['number_of_days_spent'].mean().reset_index()

# Step 2: Create a box plot for each section using Altair
box_plot = alt.Chart(grouped_df).mark_boxplot().encode(
    x=alt.X('section:N', title='Section'),  # Grouping by section
    y=alt.Y('number_of_days_spent:Q', title='Average Number of Days Spent'),  # Average days spent
    color=alt.Color('section:N', legend=None)  # Optional: color by section
).properties(
    title='Box Plot of Average Number of Days Spent by Variant per Section',
    width=600,
    height=400
)

# Show the plot
box_plot.display()