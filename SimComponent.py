import pandas as pd
import random
import string
import plotly.express as px

# ---- 1) Helper to generate random 3-letter codes ----
def random_codes(n, prefix="Station", seed=42):
    random.seed(seed)  # reproducibility
    codes = set()
    while len(codes) < n:
        code = ''.join(random.choices(string.ascii_uppercase, k=3))
        codes.add(f"{prefix} {code}")
    return list(codes)

# ---- 2) Build mapping for STATION ----
uniq_stations = filtered_data["STATION"].unique()
station_codes = random_codes(len(uniq_stations), prefix="Station")
STATION_MAP = dict(zip(uniq_stations, station_codes))

# ---- 3) Build mapping for VEHICLE ----
uniq_vehicles = filtered_data["VEHICLE"].unique()
vehicle_codes = random_codes(len(uniq_vehicles), prefix="Product", seed=99)
VEHICLE_MAP = dict(zip(uniq_vehicles, vehicle_codes))

# ---- 4) Apply anonymization ----
df_anon = filtered_data.copy()
df_anon["STATION_ANON"] = df_anon["STATION"].map(STATION_MAP)
df_anon["VEHICLE_ANON"] = df_anon["VEHICLE"].map(VEHICLE_MAP)

# ---- 5) Plot anonymized scatter ----
fig = px.scatter(
    df_anon,
    title="Anonymized Daily Status",
    x="DATE",
    y="VEHICLE_ANON",
    color="STATION_ANON"
)

fig.update_layout(width=1000, height=1000)
fig