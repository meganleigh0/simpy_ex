import pandas as pd
import hashlib
import plotly.express as px

# ---- 1) Deterministic anonymizer (stable across runs/files) ----
def _stable_map(values, prefix="Station", pad=3, salt="change-me"):
    """Return a dict mapping each unique value -> prefix NNN using salted SHA256 ordering."""
    uniq = pd.Index(pd.unique(values))
    # Order uniques by salted hash so the mapping looks random but is deterministic
    ordered = sorted(uniq, key=lambda v: hashlib.sha256((salt + str(v)).encode()).hexdigest())
    labels  = [f"{prefix} {str(i).zfill(pad)}" for i in range(1, len(ordered) + 1)]
    return dict(zip(ordered, labels))

# ---- 2) Build mappings for your columns ----
STATION_MAP = _stable_map(filtered_data["STATION"], prefix="Station", pad=3, salt="your-secret-salt-1")
VEHICLE_MAP = _stable_map(filtered_data["VEHICLE"], prefix="Product", pad=3, salt="your-secret-salt-2")

# ---- 3) Apply mappings (keep new columns; avoid leaking originals when sharing) ----
df_anon = filtered_data.copy()
df_anon["STATION_ANON"] = df_anon["STATION"].map(STATION_MAP)
df_anon["VEHICLE_ANON"] = df_anon["VEHICLE"].map(VEHICLE_MAP)

# Optional: drop raw identifiers before exporting/sharing
# df_anon = df_anon.drop(columns=["STATION", "VEHICLE"])

# ---- 4) Preserve ordering for colors/legend using the anonymized stations ----
station_order_anon = (
    df_anon.drop_duplicates("STATION_ANON")
           .loc[:, "STATION_ANON"]
           .tolist()
)

# ---- 5) Plot the anonymized scatter ----
fig = px.scatter(
    df_anon,
    title="Anonymized Daily Status",
    x="DATE",
    y="VEHICLE_ANON",
    color="STATION_ANON",
    category_orders={"STATION_ANON": station_order_anon},
)

fig.update_layout(width=1000, height=1000)
fig