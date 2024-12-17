import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def lognormal_params_from_stats(mean, std):
    """
    Given the arithmetic mean and standard deviation of a dataset,
    return the μ and σ parameters of the underlying normal distribution
    for the corresponding lognormal distribution.
    """
    if std <= 0:
        # If std is zero or not positive, assume no variability: 
        # This basically means deterministic = lognormal with σ=0.
        # μ = ln(mean)
        return np.log(mean), 0.0
    variance = std**2
    sigma_sq = np.log((variance / (mean**2)) + 1)
    sigma = np.sqrt(sigma_sq)
    mu = np.log(mean) - (sigma_sq / 2)
    return mu, sigma

def run_monte_carlo_simulation(
    df, 
    station_data_df,
    monthly_schedule, 
    num_simulations=10000, 
    confidence=0.90,
    start_date=datetime.now()
):
    """
    Run Monte Carlo simulations using a lognormal distribution for each station 
    and aggregate results at station, plant, and total levels.

    Parameters:
    - df: DataFrame with columns ['STATION', 'VARIANT', 'AVG_NUM_DAYS', 'STD_DEV'].
    - station_data_df: DataFrame with columns ['STATION','PARENT','PLANT'].
    - monthly_schedule: dict like {'VARIANT_A': 10, 'VARIANT_B': 5}.
    - num_simulations: Number of Monte Carlo iterations.
    - confidence: Confidence level for intervals (e.g., 0.90 for 90%).
    - start_date: datetime object representing when production starts.

    Returns:
    A dictionary with simulation results, including distributions and CI for 
    total, per-plant, and per-station aggregates.
    """

    # Group variant data for quick access
    variant_groups = df.groupby('VARIANT')

    # Create a mapping of STATION -> PLANT
    station_to_plant = station_data_df.set_index('STATION')['PLANT'].to_dict()

    # Identify all stations and plants involved
    all_stations = df['STATION'].unique()
    all_plants = station_data_df['PLANT'].unique()

    # Initialize arrays to accumulate results
    # We'll store total simulated days for entire schedule:
    total_days_per_simulation = np.zeros(num_simulations)
    # Store by station
    station_days = {station: np.zeros(num_simulations) for station in all_stations}
    # Store by plant
    plant_days = {plant: np.zeros(num_simulations) for plant in all_plants}

    # For each variant and quantity
    for variant, quantity in monthly_schedule.items():
        if variant not in variant_groups.groups:
            raise ValueError(f"No data found for variant {variant} in the dataset.")

        variant_data = variant_groups.get_group(variant)
        
        # For each unit of this variant
        for _ in range(quantity):
            # Simulate each station
            for _, row in variant_data.iterrows():
                station = row['STATION']
                mean = row['AVG_NUM_DAYS']
                std = row['STD_DEV']

                # Convert to lognormal parameters
                mu, sigma = lognormal_params_from_stats(mean, std)

                # Draw samples from lognormal distribution
                # np.random.lognormal uses parameters mean=mu, sigma=sigma where 
                # mean/sigma refer to the underlying normal distribution.
                samples = np.random.lognormal(mean=mu, sigma=sigma, size=num_simulations)
                # Add these samples to the totals
                total_days_per_simulation += samples
                station_days[station] += samples
                
                # Add to plant-level sums
                plant = station_to_plant.get(station, None)
                if plant is not None:
                    plant_days[plant] += samples

    # Compute confidence intervals and statistics
    def confidence_interval(data, conf):
        lower_percentile = (1 - conf) / 2 * 100
        upper_percentile = (1 - (1 - conf) / 2) * 100
        return np.percentile(data, lower_percentile), np.percentile(data, upper_percentile)

    total_ci = confidence_interval(total_days_per_simulation, confidence)
    total_mean = np.mean(total_days_per_simulation)

    # Compute results for stations
    station_results = {}
    for station in all_stations:
        dist = station_days[station]
        ci = confidence_interval(dist, confidence)
        mean_val = np.mean(dist)
        station_results[station] = {
            'mean_days': mean_val,
            'confidence_interval_days': ci
        }

    # Compute results for plants
    plant_results = {}
    for plant in all_plants:
        dist = plant_days[plant]
        ci = confidence_interval(dist, confidence)
        mean_val = np.mean(dist)
        plant_results[plant] = {
            'mean_days': mean_val,
            'confidence_interval_days': ci
        }

    # Convert mean and CI bounds into dates
    predicted_end_mean = start_date + timedelta(days=total_mean)
    predicted_end_lower = start_date + timedelta(days=total_ci[0])
    predicted_end_upper = start_date + timedelta(days=total_ci[1])

    results = {
        'total': {
            'mean_days': total_mean,
            'confidence_interval_days': total_ci,
            'predicted_end_date_mean': predicted_end_mean,
            'predicted_end_date_lower': predicted_end_lower,
            'predicted_end_date_upper': predicted_end_upper
        },
        'stations': station_results,
        'plants': plant_results,
        'distributions': {
            'total_days_distribution': total_days_per_simulation,
            'station_days_distribution': station_days,
            'plant_days_distribution': plant_days
        },
        'start_date': start_date
    }

    return results

# Example usage:
if __name__ == "__main__":
    # Example station-level dataset
    df = pd.DataFrame({
        'STATION': ['STATION_1', 'STATION_2', 'STATION_3', 'STATION_1', 'STATION_2', 'STATION_3'],
        'VARIANT': ['VARIANT_A', 'VARIANT_A', 'VARIANT_A', 'VARIANT_B', 'VARIANT_B', 'VARIANT_B'],
        'AVG_NUM_DAYS': [5.0, 3.0, 2.0, 4.0, 6.0, 5.0],
        'STD_DEV': [1.0, 0.5, 0.2, 1.5, 2.0, 1.0]
    })

    # Station data linking stations to plants
    station_data_df = pd.DataFrame({
        'STATION': ['STATION_1', 'STATION_2', 'STATION_3'],
        'PARENT': ['PARENT_1', 'PARENT_1', 'PARENT_2'],
        'PLANT': ['PLANT1', 'PAINT', 'VEHICLE']
    })

    monthly_schedule = {
        'VARIANT_A': 10,
        'VARIANT_B': 5
    }

    results = run_monte_carlo_simulation(
        df, 
        station_data_df, 
        monthly_schedule, 
        num_simulations=5000, 
        confidence=0.90,
        start_date=datetime(2025, 1, 1)
    )

    print("Total Mean Days:", results['total']['mean_days'])
    print("Total 90% CI (days):", results['total']['confidence_interval_days'])
    print("Station Results:")
    for st, val in results['stations'].items():
        print(f"  {st}: mean={val['mean_days']}, CI={val['confidence_interval_days']}")
    print("Plant Results:")
    for pl, val in results['plants'].items():
        print(f"  {pl}: mean={val['mean_days']}, CI={val['confidence_interval_days']}")
    print("Predicted End Date (Mean):", results['total']['predicted_end_date_mean'])
    print("Predicted End Date (Lower CI):", results['total']['predicted_end_date_lower'])
    print("Predicted End Date (Upper CI):", results['total']['predicted_end_date_upper'])
