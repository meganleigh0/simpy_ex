import pandas as pd
import plotly.express as px

# ---- 1) Load your station reference file ----
station_ref = pd.read_csv("station_data_ref.csv")  
# Columns: STATION, ANONYM, PLANT

# ---- 2) Merge into your filtered dataset ----
df_anon = filtered_data.merge(
    station_ref[["STATION", "ANONYM"]], 
    on="STATION", 
    how="left"
)

# ---- 3) Use ANONYM column for ordering ----
station_order_anon = station_ref["ANONYM"].tolist()

# ---- 4) Plot anonymized scatter ----
fig = px.scatter(
    df_anon,
    title="Anonymized Daily Status",
    x="DATE",
    y="VEHICLE",       # you can anonymize VEHICLE similarly if needed
    color="ANONYM",    # use anonymized station names
    category_orders={"ANONYM": station_order_anon},
)

fig.update_layout(width=1000, height=1000)
fig