def run_monte_carlo_sims(df, num_simulations=10_000):
    """
    Run Monte Carlo simulations for each station's timing within each plant,
    then compute the max station time (cycle time) for each plant.
    
    Returns:
    --------
    A dict of {plant: cycle_time_array} for each plant.
    """
    results = {}

    for plant in df['PLANT'].unique():
        # Subset DataFrame for this plant
        plant_df = df[df['PLANT'] == plant]

        station_samples = []
        for _, row in plant_df.iterrows():
            mean = row['AVG_NUM_DAYS']
            std_dev = row['STD_DEV']
            
            # Sample from normal distribution (can produce negative values if std_dev is large;
            # consider clipping or using a lognormal if that is a concern).
            samples = norm.rvs(loc=mean, scale=std_dev, size=num_simulations)
            
            station_samples.append(samples)
        
        # Combine all station samples for this plant
        # shape: (num_stations, num_simulations)
        station_stacks = np.stack(station_samples, axis=0)
        
        # The cycle time is the maximum station time across all stations for each simulation
        cycle_times = np.max(station_stacks, axis=0)  # shape: (num_simulations,)

        results[plant] = cycle_times
    
    return results
def create_altair_histogram(sim_results):
    """
    Create an interactive Altair histogram showing the distribution of cycle times
    for each plant (overlapped histograms distinguished by color).
    
    sim_results: dict of {plant: np.array of cycle times}
    
    Returns:
    --------
    alt.Chart object
    """
    # Convert dictionary to a tidy DataFrame
    records = []
    for plant, times in sim_results.items():
        for t in times:
            records.append({'Plant': plant, 'Cycle_Time': t})

    chart_df = pd.DataFrame(records)
    
    # Build histogram
    histogram = (
        alt.Chart(chart_df)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X('Cycle_Time:Q', bin=alt.Bin(maxbins=50), title='Cycle Time (Days)'),
            y=alt.Y('count()', title='Count'),
            color='Plant:N',
            tooltip=['Plant:N', 'Cycle_Time:Q']
        )
        .properties(width=600, height=300, title='Monte Carlo Cycle Time Distributions')
        .interactive()  # enables zooming/panning
    )
    
    return histogram
# 1. Run the simulation
monte_carlo_results = run_monte_carlo_sims(df, num_simulations=10_000)

# 2. Create the Altair chart
chart = create_altair_histogram(monte_carlo_results)

# 3. Display the chart (in a Jupyter notebook or compatible environment)
chart
