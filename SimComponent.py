import numpy as np
import pandas as pd
import altair as alt
from scipy.stats import lognorm

def plot_ppf_with_confidence(station_name, mean, std_dev):
    # Calculate the shape and scale parameters for the log-normal distribution
    shape = np.sqrt(np.log(1 + (std_dev / mean)**2))
    scale = mean / np.sqrt(1 + (std_dev / mean)**2)
    
    # Generate quantiles and PPF values
    quantiles = np.linspace(0.01, 0.99, 500)  # Quantiles from 1% to 99%
    ppf_values = lognorm.ppf(quantiles, s=shape, scale=scale)
    
    # Prepare the data for the main line plot
    df = pd.DataFrame({
        'Quantiles': quantiles,
        'PPF Values (DAYS)': ppf_values
    })
    
    # Calculate the 80% confidence interval (10th to 90th percentiles)
    lower_bound = lognorm.ppf(0.1, s=shape, scale=scale)
    upper_bound = lognorm.ppf(0.9, s=shape, scale=scale)
    
    # Prepare data for the confidence interval shading
    shade_df = df[(df['PPF Values (DAYS)'] >= lower_bound) & (df['PPF Values (DAYS)'] <= upper_bound)]
    
    # Main line plot
    line_plot = alt.Chart(df).mark_line(color='blue').encode(
        x=alt.X('Quantiles', title='Cumulative Probability'),
        y=alt.Y('PPF Values (DAYS)', title='PPF (Days)')
    )
    
    # Shaded confidence interval
    shaded_area = alt.Chart(shade_df).mark_area(opacity=0.3, color='lightblue').encode(
        x=alt.X('Quantiles', title='Cumulative Probability'),
        y=alt.Y('PPF Values (DAYS)', title='PPF (Days)')
    )
    
    # Combine the line plot and shaded area
    plot = (shaded_area + line_plot).properties(
        title=f'{station_name} Log Normal Percent Point Function (PPF)',
        width=600,
        height=400
    )
    
    return plot

# Example usage
plot = plot_ppf_with_confidence("Hull Station 0", mean=3, std_dev=1)
plot.show()