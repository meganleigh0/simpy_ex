import numpy as np
import pandas as pd
import altair as alt

def prepare_data_for_viz(results):
    """
    Prepare data from the simulation results dictionary into DataFrames suitable for visualization.
    """
    # Extract total distribution
    total_dist = results['distributions']['total_days_distribution']
    total_df = pd.DataFrame({
        'simulation_id': np.arange(len(total_dist)),
        'total_days': total_dist
    })

    # Station distributions
    station_data = []
    for station, dist in results['distributions']['station_days_distribution'].items():
        station_data.append(pd.DataFrame({
            'simulation_id': np.arange(len(dist)),
            'station': station,
            'days': dist
        }))
    station_df = pd.concat(station_data, ignore_index=True)

    # Plant distributions
    plant_data = []
    for plant, dist in results['distributions']['plant_days_distribution'].items():
        plant_data.append(pd.DataFrame({
            'simulation_id': np.arange(len(dist)),
            'plant': plant,
            'days': dist
        }))
    plant_df = pd.concat(plant_data, ignore_index=True)

    return total_df, station_df, plant_df

def create_total_distribution_chart(total_df, results):
    """
    Create a histogram of the total days distribution with mean and CI lines.
    """
    mean_days = results['total']['mean_days']
    lower_ci, upper_ci = results['total']['confidence_interval_days']

    base = alt.Chart(total_df).mark_bar(opacity=0.7).encode(
        alt.X('total_days:Q', bin=alt.Bin(maxbins=50), title='Total Days'),
        alt.Y('count()', title='Count of Simulations')
    )

    mean_line = alt.Chart(pd.DataFrame({'x': [mean_days]})).mark_rule(color='red').encode(x='x:Q')
    lower_ci_line = alt.Chart(pd.DataFrame({'x': [lower_ci]})).mark_rule(color='green').encode(x='x:Q')
    upper_ci_line = alt.Chart(pd.DataFrame({'x': [upper_ci]})).mark_rule(color='green').encode(x='x:Q')

    text_mean = alt.Chart(pd.DataFrame({'x': [mean_days], 'label': [f'Mean: {mean_days:.1f}']})).mark_text(
        align='left', dx=5, dy=-10, color='red'
    ).encode(x='x:Q', text='label:N')

    text_lower = alt.Chart(pd.DataFrame({'x': [lower_ci], 'label': [f'Lower CI: {lower_ci:.1f}']})).mark_text(
        align='left', dx=5, dy=-10, color='green'
    ).encode(x='x:Q', text='label:N')

    text_upper = alt.Chart(pd.DataFrame({'x': [upper_ci], 'label': [f'Upper CI: {upper_ci:.1f}']})).mark_text(
        align='left', dx=5, dy=-10, color='green'
    ).encode(x='x:Q', text='label:N')

    return (base + mean_line + lower_ci_line + upper_ci_line + text_mean + text_lower + text_upper).properties(
        title="Total Completion Time Distribution"
    )

def create_station_chart(station_df, results):
    """
    Create boxplot or violin plot by station to visualize the distribution of times per station.
    Also overlay mean and CI as text or lines.
    """

    # Create a summary table for stations
    station_summary = []
    for station, val in results['stations'].items():
        station_summary.append({
            'station': station,
            'mean_days': val['mean_days'],
            'lower_ci': val['confidence_interval_days'][0],
            'upper_ci': val['confidence_interval_days'][1]
        })
    station_summary_df = pd.DataFrame(station_summary)

    boxplot = alt.Chart(station_df).mark_boxplot().encode(
        x=alt.X('station:N', title='Station'),
        y=alt.Y('days:Q', title='Days'),
        color='station:N'
    )

    # Add mean and CI overlay using rule and point marks
    mean_ci_lines = alt.Chart(station_summary_df).mark_rule(color='red').encode(
        x='station:N',
        y='lower_ci:Q',
        y2='upper_ci:Q'
    )

    mean_points = alt.Chart(station_summary_df).mark_point(color='red', size=50).encode(
        x='station:N',
        y='mean_days:Q'
    )

    return (boxplot + mean_ci_lines + mean_points).properties(
        title="Distribution of Times by Station"
    )

def create_plant_chart(plant_df, results):
    """
    Similar to station chart, but grouped by plant.
    """
    # Create a summary table for plants
    plant_summary = []
    for plant, val in results['plants'].items():
        plant_summary.append({
            'plant': plant,
            'mean_days': val['mean_days'],
            'lower_ci': val['confidence_interval_days'][0],
            'upper_ci': val['confidence_interval_days'][1]
        })
    plant_summary_df = pd.DataFrame(plant_summary)

    boxplot = alt.Chart(plant_df).mark_boxplot().encode(
        x=alt.X('plant:N', title='Plant'),
        y=alt.Y('days:Q', title='Days'),
        color='plant:N'
    )

    mean_ci_lines = alt.Chart(plant_summary_df).mark_rule(color='red').encode(
        x='plant:N',
        y='lower_ci:Q',
        y2='upper_ci:Q'
    )

    mean_points = alt.Chart(plant_summary_df).mark_point(color='red', size=50).encode(
        x='plant:N',
        y='mean_days:Q'
    )

    return (boxplot + mean_ci_lines + mean_points).properties(
        title="Distribution of Times by Plant"
    )

def create_leadership_dashboard(results):
    """
    Create a set of Altair charts for leadership to see schedule predictions.
    Returns a dictionary of charts.
    """
    total_df, station_df, plant_df = prepare_data_for_viz(results)

    total_chart = create_total_distribution_chart(total_df, results)
    station_chart = create_station_chart(station_df, results)
    plant_chart = create_plant_chart(plant_df, results)

    return {
        'total_chart': total_chart,
        'station_chart': station_chart,
        'plant_chart': plant_chart
    }

# Example usage (assuming `results` is obtained from the Monte Carlo simulation above):
if __name__ == "__main__":
    # Suppose we have `results` from the simulation
    # Here we'd just run the previous simulation code
    # (Omitting the simulation code here for brevity, assume `results` is available)
    # results = run_monte_carlo_simulation(...)

    charts = create_leadership_dashboard(results)
    # Display in a Jupyter environment:
    charts['total_chart'].display()
    charts['station_chart'].display()
    charts['plant_chart'].display()
