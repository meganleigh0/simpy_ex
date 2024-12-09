import numpy as np
import pandas as pd
import altair as alt
from scipy.stats import lognorm

# Parameters for the log-normal distribution
avg_num_days = 20  # Example mean
std_dev = 5        # Example standard deviation

# Log-normal parameters
shape = std_dev / avg_num_days  # Coefficient of variation
scale = avg_num_days

# Generate quantiles and PPF values
quantiles = np.linspace(0.01, 0.99, 500)  # Quantiles from 1% to 99%
ppf_values = lognorm.ppf(quantiles, s=shape, scale=scale)

# Prepare data for Altair
df = pd.DataFrame({
    'Quantiles': quantiles,
    'PPF Values (Days)': ppf_values
})

# Create the Altair plot
ppf_plot = alt.Chart(df).mark_line(color='blue').encode(
    x=alt.X('Quantiles', title='Cumulative Probability'),
    y=alt.Y('PPF Values (Days)', title='PPF (Days)')
).properties(
    title='Log-Normal Percent-Point Function (PPF)',
    width=600,
    height=400
)

ppf_plot.interactive()