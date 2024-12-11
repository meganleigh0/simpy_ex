import pandas as pd
import numpy as np
from scipy.stats import norm

# --------------------------
# Step 1: Load Data (Example)
# --------------------------
# Assume we have a CSV with columns:
# STATION, AVG_HOURS, STD_HOURS representing historical average and std dev for each station
data = {
    'STATION': ['Station_A', 'Station_B', 'Station_C'],
    'AVG_HOURS': [40, 50, 30],  # example averages
    'STD_HOURS': [5, 8, 4]      # example standard deviations
}
df = pd.DataFrame(data)

# --------------------------
# Step 2: Define Confidence
# --------------------------
confidence_level = 0.95

# --------------------------
# Step 3: Calculate Required Hours per Station at 95% Confidence
# --------------------------
df['HOURS_95TH'] = df.apply(lambda row: norm.ppf(confidence_level, loc=row['AVG_HOURS'], scale=row['STD_HOURS']), axis=1)

# --------------------------
# Step 4: Aggregate Stations (Naive Sum for Illustration)
# --------------------------
total_hours_95th_naive = df['HOURS_95TH'].sum()

# --------------------------
# Step 5: Monte Carlo Simulation for More Accurate Total Estimate
# --------------------------
N = 10000
sim_results = []
for _ in range(N):
    total = 0
    for _, r in df.iterrows():
        # Draw a random sample from the station's distribution
        total += norm.rvs(loc=r['AVG_HOURS'], scale=r['STD_HOURS'])
    sim_results.append(total)
sim_results = np.array(sim_results)
total_hours_95th_sim = np.percentile(sim_results, 95)

# --------------------------
# Print Results
# --------------------------
print("Per-Station 95th Percentile Hours:")
print(df[['STATION', 'HOURS_95TH']])
print(f"Naive Sum of 95th Percentile Hours: {total_hours_95th_naive:.2f} hours")
print(f"95th Percentile of Total Hours via Simulation: {total_hours_95th_sim:.2f} hours")

# You can now use these results to adjust resources (e.g., add stands or shifts)
# by recalculating the distributions and repeating the steps above.
