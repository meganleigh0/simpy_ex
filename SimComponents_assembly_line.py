import pandas as pd
import numpy as np
from scipy.stats import lognorm
import altair as alt

# ---------------------------------------------------------------------------------
# 1. Example data (You can replace this with your own DataFrame)
# ---------------------------------------------------------------------------------
data = {
    'PLANT':   ['PlantA','PlantA','PlantA','PlantB','PlantB','PlantC'],
    'STATION': ['S1','S2','S3','S1','S2','S1'],
    'AVG_NUM_DAYS': [5.0, 6.7, 8.2, 3.5, 4.0, 10.0],
    'MEDIAN_NUM_DAYS': [5.0, 6.5, 8.0, 3.3, 3.8, 9.5],  # Not used directly for sampling
    'STD_DEV': [1.0, 1.5, 2.0, 1.2, 1.3, 2.2],
    'PARENT':  ['S2','S3','End','S2','End','End']      # Not used for sampling here
}

df = pd.DataFrame(data)

# ---------------------------------------------------------------------------------
# 2. Helper: Convert (mean, std) in real space -> (mu, sigma) in log space
# ---------------------------------------------------------------------------------
def get_lognorm_params(mean, std):
    """
    Given a desired mean and std (in real space), return (mu, sigma) parameters
    for the underlying normal distribution (log space) of a lognormal.
    
    mean = exp(mu + sigma^2 / 2)
    var  = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
    std  = sqrt(var)
    """
    var = std**2
    mu = np.log(mean**2 / np.sqrt(var + mean**2))
    sigma = np.sqrt(np.log(1 + var / (mean**2)))
    return mu, sigma

# ---------------------------------------------------------------------------------
# 3. Monte Carlo Simulation with Lognormal draws
# ---------------------------------------------------------------------------------
def run_monte_carlo_sims(df, num_simulations=10_000):
    results = {}

    for plant in df['PLANT'].unique():
        # Subset DataFrame for this plant
        plant_df = df[df['PLANT'] == plant]

        station_samples = []
        for _, row in plant_df.iterrows():
            mean = row['AVG_NUM_DAYS']
            std_dev = row['STD_DEV']
            
            # Convert (mean, std) in real space to (mu, sigma) for lognormal
            mu, sigma = get_lognorm_params(mean, std_dev)
            
            # Sample from lognormal using (s=sigma, scale=exp(mu))
            samples = lognorm.rvs(s=sigma, scale=np.exp(mu), size=num_simulations)
            
            station_samples.append(samples)
        
        station_stacks = np.stack(station_samples, axis=0)
        # The plant's "cycle time" is the max across station times
        cycle_times = np.max(station_stacks, axis=0)
        results[plant] = cycle_times
    
    return results

# ---------------------------------------------------------------------------------
# 4. Run the simulations
# ---------------------------------------------------------------------------------
monte_carlo_results = run_monte_carlo_sims(df, num_simulations=10_000)

# ---------------------------------------------------------------------------------
# 5. Flatten results for Altair
# ---------------------------------------------------------------------------------
records = []
for plant, times in monte_carlo_results.items():
    for t in times:
        records.append({'Plant': plant, 'Cycle_Time': t})
dist_df = pd.DataFrame(records)

# ---------------------------------------------------------------------------------
# 6. Chart #1: Histogram of cycle times, colored by Plant
# ---------------------------------------------------------------------------------
chart1 = (
    alt.Chart(dist_df)
    .mark_bar(opacity=0.5)
    .encode(
        x=alt.X('Cycle_Time:Q', bin=alt.Bin(maxbins=50), title='Cycle Time (Days)'),
        y=alt.Y('count()', title='Count'),
        color='Plant:N',
        tooltip=['Plant:N', 'Cycle_Time:Q']
    )
    .properties(width=600, height=300, title='Lognormal Monte Carlo: Cycle Time Distributions')
    .interactive()
)

# ---------------------------------------------------------------------------------
# 7. Chart #2: Interactive slider for Vehicle Qty -> total days estimate
#    This uses alt.Param (introduced in Altair 5).
# ---------------------------------------------------------------------------------
vehicle_qty = alt.binding_range(min=1, max=200, step=1, name='Vehicle Qty:')
vehicle_qty_param = alt.param(value=1, bind=vehicle_qty)

chart2 = (
    alt.Chart(dist_df)
    .transform_calculate(
        Est_Days='datum.Cycle_Time * vehicle_qty_param',
        vehicle_qty_param=vehicle_qty_param
    )
    .mark_bar(opacity=0.5)
    .encode(
        x=alt.X('Est_Days:Q', bin=alt.Bin(maxbins=50), title='Estimated Total Days'),
        y=alt.Y('count()', title='Count'),
        color='Plant:N',
        tooltip=['Plant:N', 'Est_Days:Q']
    )
    .properties(width=600, height=300, title='Estimated Total Days vs. Vehicle Quantity')
    .add_params(vehicle_qty_param)  # attach the slider
    .interactive()
)

# ---------------------------------------------------------------------------------
# 8. Display both charts (in Jupyter, just use the final line to see them side-by-side)
# ---------------------------------------------------------------------------------
chart1 & chart2
