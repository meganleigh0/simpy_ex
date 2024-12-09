import numpy as np
import pandas as pd
import altair as alt
from scipy.stats import lognorm

# Parameters for the log-normal distribution
avg_num_days = 20  # Example mean
std_dev = 5        # Example standard deviation

# Log-normal parameters (log-transformed)
shape = std_dev / avg_num_days  # Coefficient of variation
scale = avg_num_days

# Generate data for plotting
quantiles = np.linspace(0.01, 0.99, 500)  # Quantiles from 1% to 99%
time_values = lognorm.ppf(quantiles, s=shape, scale=scale)

# Compute confidence intervals
lower_ci = lognorm.ppf(0.05, s=shape, scale=scale)  # 5th percentile
upper_ci = lognorm.ppf(0.95, s=shape, scale=scale)  # 95th percentile

# Prepare data for Altair
df = pd.DataFrame({
    'Quantiles': quantiles,
    'Time (Days)': time_values
})

ci_df = pd.DataFrame({
    'Quantiles': [0.05, 0.95],
    'Time (Days)': [lower_ci, upper_ci],
    'Label': ['Lower CI (5%)', 'Upper CI (95%)']
})

# Create the Altair plot
base = alt.Chart(df).mark_line().encode(
    x=alt.X('Quantiles', title='Cumulative Probability'),
    y=alt.Y('Time (Days)', title='Time to Produce Variants')
).properties(
    title='Log-Normal Distribution with Confidence Intervals'
)

# Add shaded confidence intervals
ci_area = alt.Chart(df).mark_area(opacity=0.3).encode(
    x='Quantiles',
    y=alt.Y('Time (Days):Q', aggregate='min', title=None),
    y2=alt.Y2(lower_ci, title=None),
)

# Add CI markers
ci_markers = alt.Chart(ci_df).mark_rule(color='red').encode(
    x='Quantiles:Q',
)