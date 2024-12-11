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


Data-Driven Labor Hour Predictions with Confidence

Problem:
We need to understand not just the average labor hours for our production stations but also how much time might be required under less-than-ideal conditions, so we can plan staffing and shifts confidently.

Approach:

Use Historical Data: Start with the current average and variation in station processing times.
Probability Analysis: Model the times as a distribution and pick a “confidence level” (e.g., 95%) to ensure we’re prepared for longer-than-average scenarios.
Find the Required Hours: Using statistical methods, determine the hours needed so that only a 5% chance exists of exceeding that time. This helps us plan for the “worst reasonable case.”
Adjust Resources: Experiment with adding more workstations or extending shifts to see how these changes reduce the required hours at a high confidence level.
Benefit:
This approach provides a data-backed, confidence-driven estimate of labor hours required, guiding resource planning and reducing the risk of production delays.

