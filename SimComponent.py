import pandas as pd
import altair as alt

# Ensure Altair can handle large datasets if needed
alt.data_transformers.disable_max_rows()

def create_simple_executive_visuals(results, station_data_df):
    # Extract total summary
    total_mean = results['total']['mean_days']
    total_lower, total_upper = results['total']['confidence_interval_days']
    predicted_end_mean = results['total']['predicted_end_date_mean']
    predicted_end_lower = results['total']['predicted_end_date_lower']
    predicted_end_upper = results['total']['predicted_end_date_upper']
    
    # Prepare plant summary data
    plant_order = ['PLANT1', 'PAINT', 'PLANT3', 'VEHICLE']
    plant_summary = []
    for plant, val in results['plants'].items():
        plant_summary.append({
            'plant': plant,
            'mean_days': val['mean_days'],
            'lower_ci': val['confidence_interval_days'][0],
            'upper_ci': val['confidence_interval_days'][1]
        })
    plant_summary_df = pd.DataFrame(plant_summary)
    # Ensure correct plant order
    plant_summary_df['plant'] = pd.Categorical(plant_summary_df['plant'], categories=plant_order, ordered=True)
    plant_summary_df = plant_summary_df.sort_values('plant')
    
    # Prepare station summary data
    station_summary = []
    for station, val in results['stations'].items():
        # Find plant for station from station_data_df
        station_row = station_data_df[station_data_df['STATION'] == station].iloc[0]
        plant = station_row['PLANT']
        station_summary.append({
            'station': station,
            'plant': plant,
            'mean_days': val['mean_days'],
            'lower_ci': val['confidence_interval_days'][0],
            'upper_ci': val['confidence_interval_days'][1]
        })
    station_summary_df = pd.DataFrame(station_summary)
    # Sort stations by plant order, then by station name
    station_summary_df['plant'] = pd.Categorical(station_summary_df['plant'], categories=plant_order, ordered=True)
    station_summary_df = station_summary_df.sort_values(['plant', 'station'])

    # ---------------------------------------
    # Create Charts
    # ---------------------------------------

    # 1. Simple total summary chart
    # Instead of a complex chart, let's just show a single bar with error bars.
    total_df = pd.DataFrame([{
        'category': 'Total Schedule',
        'mean_days': total_mean,
        'lower_ci': total_lower,
        'upper_ci': total_upper
    }])

    total_chart = alt.Chart(total_df).mark_bar(color='steelblue').encode(
        x=alt.X('category:N', title='', axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y('mean_days:Q', title='Days')
    ).properties(title="Overall Completion Prediction")

    error_bars_total = alt.Chart(total_df).mark_rule(color='black').encode(
        x='category:N',
        y='lower_ci:Q',
        y2='upper_ci:Q'
    )

    # Add a text annotation to show the predicted end date directly
    text_annotation = alt.Chart(pd.DataFrame({
        'category': ['Total Schedule'],
        'label': [f"Estimated Completion: {predicted_end_mean.date()} (90% CI: {predicted_end_lower.date()} - {predicted_end_upper.date()})"]
    })).mark_text(align='left', dx=5).encode(
        x='category:N',
        y=alt.value(0),  # near the top
        text='label:N'
    )

    total_combined = (total_chart + error_bars_total + text_annotation).properties(
        width=300, height=150
    )

    # 2. Plant-level chart: Bar chart with error bars
    plant_chart = alt.Chart(plant_summary_df).mark_bar(color='forestgreen').encode(
        x=alt.X('plant:N', title='Plant'),
        y=alt.Y('mean_days:Q', title='Mean Days')
    ).properties(
        title="By Plant: Mean and Confidence Interval",
        width=300,
        height=200
    )

    plant_error = alt.Chart(plant_summary_df).mark_rule(color='black').encode(
        x='plant:N',
        y='lower_ci:Q',
        y2='upper_ci:Q'
    )

    plant_combined = plant_chart + plant_error

    # 3. Station-level chart: Bar chart with error bars
    # We'll show stations in order and color by their plant to reflect grouping
    station_chart = alt.Chart(station_summary_df).mark_bar().encode(
        x=alt.X('station:N', title='Station', sort=None),
        y=alt.Y('mean_days:Q', title='Mean Days'),
        color=alt.Color('plant:N', title='Plant', sort=plant_order)
    ).properties(
        title="By Station: Mean and Confidence Interval",
        width=500,
        height=200
    )

    station_error = alt.Chart(station_summary_df).mark_rule(color='black').encode(
        x='station:N',
        y='lower_ci:Q',
        y2='upper_ci:Q'
    )

    station_combined = station_chart + station_error

    return total_combined, plant_combined, station_combined

# Example usage:
if __name__ == "__main__":
    # Assuming `results` and `station_data_df` from previous steps
    # total_combined, plant_combined, station_combined = create_simple_executive_visuals(results, station_data_df)
    # In a Jupyter environment:
    # total_combined.display()
    # plant_combined.display()
    # station_combined.display()
    pass
