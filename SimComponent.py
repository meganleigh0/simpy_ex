import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def run_monte_carlo_simulation(df, monthly_schedule, num_simulations=10000, confidence=0.90):
    """
    Run Monte Carlo simulations based on historical station data and a monthly schedule.

    Parameters:
    - df: pandas DataFrame with columns ['STATION', 'VARIANT', 'AVG_NUM_DAYS', 'STD_DEV']
    - monthly_schedule: dict like {'VARIANT_A': 10, 'VARIANT_B': 5}, representing quantities per variant
    - num_simulations: number of Monte Carlo iterations
    - confidence: confidence level for intervals (0.90 means 90% CI)
    
    Returns:
    Dictionary with results:
    {
        'total_days_distribution': numpy array of simulated total days for entire schedule,
        'confidence_interval': (lower_bound, upper_bound),
        'mean_estimate': float,
        'start_date': datetime (if start date provided),
        'predicted_end_date': datetime estimate
    }
    """

    # Extract unique variants from the schedule
    variants_in_schedule = monthly_schedule.keys()

    # We will sum the total days for the entire monthly schedule.
    # For each simulation, sum over all variants and their quantities.
    total_days_per_simulation = np.zeros(num_simulations)

    # Pre-group the data by variant for quicker access
    variant_groups = df.groupby('VARIANT')

    # Run simulations
    for variant, quantity in monthly_schedule.items():
        # Get station data for this variant
        if variant not in variant_groups.groups:
            raise ValueError(f"No data found for variant {variant} in the dataset.")
        variant_data = variant_groups.get_group(variant)

        # For each unit of this variant to be produced
        for _ in range(quantity):
            # Simulate the time at each station and sum
            # Assume stations are sequential: total time = sum of all stations for that vehicle
            # Draw samples from normal distributions
            for i, row in variant_data.iterrows():
                mean = row['AVG_NUM_DAYS']
                std = row['STD_DEV'] if not np.isnan(row['STD_DEV']) else 0.0
                # Sample num_simulations times, once per iteration at the end we'll add them all
                # Actually, it's more efficient to sample all at once for each step.
                station_samples = np.random.normal(loc=mean, scale=std, size=num_simulations)
                # Ensure no negative times
                station_samples = np.maximum(station_samples, 0)
                total_days_per_simulation += station_samples

    # Now we have a distribution of total days for the entire schedule across num_simulations runs
    # Compute statistics
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 - (1 - confidence) / 2) * 100
    lower_bound = np.percentile(total_days_per_simulation, lower_percentile)
    upper_bound = np.percentile(total_days_per_simulation, upper_percentile)
    mean_estimate = np.mean(total_days_per_simulation)

    # If you assume a start date, you can translate days into dates:
    # For example, assume the schedule starts on the first day of the month
    start_date = datetime.now()  # or any given start date
    predicted_end_mean = start_date + timedelta(days=mean_estimate)
    predicted_end_lower = start_date + timedelta(days=lower_bound)
    predicted_end_upper = start_date + timedelta(days=upper_bound)

    results = {
        'total_days_distribution': total_days_per_simulation,
        'confidence_interval_days': (lower_bound, upper_bound),
        'mean_estimate_days': mean_estimate,
        'start_date': start_date,
        'predicted_end_date_mean': predicted_end_mean,
        'predicted_end_date_lower': predicted_end_lower,
        'predicted_end_date_upper': predicted_end_upper
    }

    return results


# Example usage:
if __name__ == "__main__":
    # Load the dataset
    # The CSV should have columns: STATION, VARIANT, AVG_NUM_DAYS, STD_DEV
    df = pd.DataFrame({
        'STATION': ['STATION_1', 'STATION_2', 'STATION_1', 'STATION_2'],
        'VARIANT': ['VARIANT_A', 'VARIANT_A', 'VARIANT_B', 'VARIANT_B'],
        'AVG_NUM_DAYS': [5.0, 3.0, 4.0, 6.0],
        'STD_DEV': [1.0, 0.5, 1.5, 2.0]
    })

    # Monthly schedule: say we have 10 units of VARIANT_A and 5 units of VARIANT_B
    monthly_schedule = {
        'VARIANT_A': 10,
        'VARIANT_B': 5
    }

    # Run simulation
    results = run_monte_carlo_simulation(df, monthly_schedule, num_simulations=10000, confidence=0.90)

    print("Mean total days estimate:", results['mean_estimate_days'])
    print("90% Confidence Interval (days):", results['confidence_interval_days'])
    print("Predicted End Date (Mean):", results['predicted_end_date_mean'])
    print("Predicted End Date (Lower CI):", results['predicted_end_date_lower'])
    print("Predicted End Date (Upper CI):", results['predicted_end_date_upper'])
