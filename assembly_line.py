import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# 1. Data Overview & Missing Values
# -------------------------------
# Display the first few rows, data types, and basic statistics of the dataset.
print("Head of Data:")
print(DATA.head())

print("\nData Info:")
print(DATA.info())

print("\nSummary Statistics:")
print(DATA.describe())

# Check for missing values in each column
print("\nMissing Values per Column:")
print(DATA.isnull().sum())

# -------------------------------
# 2. Feature Engineering
# -------------------------------
# (a) Create a column for per-unit price based on the quoted price for 28 units.
# This gives you an idea of the cost per individual piece.
if "Award Range 4 Quote Pricing (28 units)" in DATA.columns:
    DATA["Unit Price"] = DATA["Award Range 4 Quote Pricing (28 units)"] / 28

# (b) Create a Total Breakdown Price by summing the titanium and fabrication pricing components.
# This could be compared with the overall quoted price.
if "Titanium Material Pricing" in DATA.columns and "Matching and Fabrication Pricing" in DATA.columns:
    DATA["Total Breakdown Price"] = DATA["Titanium Material Pricing"] + DATA["Matching and Fabrication Pricing"]
    # If the breakdown values are given for 28 units, you might also compute a per‐unit breakdown price.
    DATA["Breakdown Unit Price"] = DATA["Total Breakdown Price"] / 28

# -------------------------------
# 3. Interactive Visualizations using Plotly
# -------------------------------

# (a) Distribution of the overall quote pricing (for 28 units)
fig1 = px.histogram(
    DATA, 
    x="Award Range 4 Quote Pricing (28 units)", 
    nbins=50, 
    title="Distribution of Award Range 4 Quote Pricing (28 units)"
)
fig1.show()

# (b) Box plot to inspect outliers and spread in the overall quote pricing.
fig2 = px.box(
    DATA, 
    y="Award Range 4 Quote Pricing (28 units)", 
    title="Box Plot of Award Range 4 Quote Pricing (28 units)"
)
fig2.show()

# (c) Scatter plot to check for relationship between Award Quantity and the overall quote pricing.
fig3 = px.scatter(
    DATA, 
    x="AWARD QTY", 
    y="Award Range 4 Quote Pricing (28 units)", 
    title="AWARD QTY vs. Award Range 4 Quote Pricing (28 units)",
    labels={"AWARD QTY": "Award Quantity", "Award Range 4 Quote Pricing (28 units)": "Quote Pricing (28 units)"}
)
fig3.show()

# (d) If there is a categorical column like "USAGE", let’s see its frequency.
if "USAGE" in DATA.columns:
    usage_counts = DATA["USAGE"].value_counts().reset_index()
    usage_counts.columns = ["USAGE", "Count"]
    fig4 = px.bar(
        usage_counts, 
        x="USAGE", 
        y="Count", 
        title="Frequency of Each USAGE Category"
    )
    fig4.show()

# (e) Correlation heatmap among numerical pricing and award quantity variables.
# This can help reveal relationships, for example whether higher award quantities correlate with higher quoted prices.
num_cols = ["Award Range 4 Quote Pricing (28 units)", "Titanium Material Pricing", "Matching and Fabrication Pricing", "AWARD QTY"]
corr_matrix = DATA[num_cols].corr()

fig5 = px.imshow(
    corr_matrix,
    text_auto=True,
    title="Correlation Heatmap among Numerical Variables"
)
fig5.show()

# (f) Scatter matrix plot to visualize pairwise relationships between numerical variables.
fig6 = px.scatter_matrix(
    DATA,
    dimensions=num_cols,
    title="Scatter Matrix of Numerical Variables"
)
fig6.show()