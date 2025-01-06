import pandas as pd
import numpy as np
from scipy.stats import lognorm
import altair as alt

# ----------------------------------------------------------------------------
# 1. Example data
# ----------------------------------------------------------------------------
data = {
    'PLANT':   ['PlantA','PlantA','PlantA','PlantB','PlantB','PlantC'],
    'STATION': ['S1','S2','S3','S1','S2','S1'],
    'AVG_NUM_DAYS': [5.0, 6.7, 8.2, 3.5, 4.0, 10.0],
    'STD_DEV': [1.0, 1.5, 2.0, 1.2, 1.3, 2.2],
}
df = pd.DataFrame(data)

# ----------------------------------------------------------------------------
# 2. Helper: Convert (mean, std) in real space -> (mu, sigma) for lognormal
#    mean = exp(mu + sigma^2 / 2)
#    var  = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
# ----------------------------------------------------------------------------
def get_lognorm_params(mean, std):
    var = std**2
    # mu
    mu = np.log(mean**2 / np.sqrt(var + mean**2))
    # sigma
    sigma = np.sqrt(np.log(1 + var / mean**2))
    return mu, sigma

# ----------------------------------------------------------------------------
# 3. Monte Carlo Simulation with lognormal draws
#    - Groups by PLANT
#    - For each station, draws from lognormal with mean=AVG_NUM_DAYS, std=STD_DEV
#    - Takes the max across stations => cycle time
# ----------------------------------------------------------------------------
def run_monte_carlo_sims(df, num_simulations=10_000):
    results = {}
    for plant in df['PLANT'].unique():
        plant_df = df[df['PLANT'] == plant]

        station_samples = []
        for _, row in plant_df.iterrows():
            mean = row['AVG_NUM_DAYS']
            std_dev = row['STD_DEV']
            mu, sigma = get_lognorm_params(mean, std_dev)            
            samples = lognorm.rvs(s=sigma, scale=np.exp(mu), size=num_simulations)
            station_samples.append(samples)
        
        # shape: (num_stations, num_simulations)
        station_stacks = np.stack(station_samples, axis=0)
        # cycle time = max station time for each simulation
        cycle_times = np.max(station_stacks, axis=0)
        results[plant] = cycle_times
    
    return results

# ----------------------------------------------------------------------------
# 4. Run the simulation (adjust num_simulations if you run into large-data issues)
# ----------------------------------------------------------------------------
monte_carlo_results = run_monte_carlo_sims(df, num_simulations=5_000)

# ----------------------------------------------------------------------------
# 5. Flatten the results into a "long" DataFrame for charting
# ----------------------------------------------------------------------------
records = []
for plant, times in monte_carlo_results.items():
    for t in times:
        records.append({'Plant': plant, 'Cycle_Time': t})
dist_df = pd.DataFrame(records)

# ----------------------------------------------------------------------------
# 6. Chart #1: Histogram of cycle times, by Plant
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# 7. Create a cartesian product with vehicle quantities (1..200)
#    We'll use a small helper function for merging everything
# ----------------------------------------------------------------------------
def cartesian_product(df1, df2):
    df1['_tmp'] = 1
    df2['_tmp'] = 1
    out = pd.merge(df1, df2, on='_tmp').drop('_tmp', axis=1)
    df1.drop('_tmp', axis=1, inplace=True)
    df2.drop('_tmp', axis=1, inplace=True)
    return out

df_qty = pd.DataFrame({'vehicle_qty': range(1, 201)})
master_df = cartesian_product(dist_df, df_qty)
# master_df columns: [Plant, Cycle_Time, vehicle_qty]

# ----------------------------------------------------------------------------
# 8. We want an interactive slider. Altair 4.1.0 does not have "alt.param",
#    so we'll use a selection_single with a range binding:
# ----------------------------------------------------------------------------
slider = alt.selection_single(
    name='SelectedQty',       # name shown on chart
    fields=['vehicle_qty'],    # field to store selection
    bind=alt.binding_range(min=1, max=200, step=1, name='Vehicle QTY'),
    init={'vehicle_qty': 1}    # initial slider value
)

# ----------------------------------------------------------------------------
# 9. Chart #2: Filter rows based on slider selection, then compute total days
# ----------------------------------------------------------------------------
chart2 = (
    alt.Chart(master_df)
    .mark_bar(opacity=0.5)
    .encode(
        x=alt.X('days_est:Q', bin=alt.Bin(maxbins=50), title='Estimated Total Days'),
        y=alt.Y('count()', title='Count'),
        color='Plant:N',
        tooltip=['Plant:N', 'days_est:Q']
    )
    .transform_calculate(
        # For each row, multiply the cycle_time by the row's vehicle_qty
        days_est='datum.Cycle_Time * datum.vehicle_qty'
    )
    .add_selection(slider)
    .transform_filter(slider)
    .properties(width=600, height=300, title='Estimated Total Days vs. Vehicle Quantity')
    .interactive()
)

# ----------------------------------------------------------------------------
# 10. Display both charts side-by-side (if you're in a Jupyter environment)
# ----------------------------------------------------------------------------
chart1 & chart2
