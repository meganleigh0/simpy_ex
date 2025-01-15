import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import numpy as np
from scipy.stats import gaussian_kde

# Example dataset
data = pd.DataFrame({
    "Station": ["A", "A", "B", "B", "C", "C", "D", "D"] * 11125,
    "Number": [3, 5, 8, 2, 7, 6, 5, 10] * 11125,
})

# Step 1: Compute variance for each station
station_variance = data.groupby("Station")["Number"].var().reset_index()
station_variance.columns = ["Station", "Variance"]

# Normalize the variance
scaler = StandardScaler()
station_variance["Normalized Variance"] = scaler.fit_transform(station_variance[["Variance"]])

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust clusters as needed
station_variance["Cluster"] = kmeans.fit_predict(station_variance[["Normalized Variance"]])

# Step 2a: Scatter plot of clusters
fig_scatter = px.scatter(
    station_variance,
    x="Station",
    y="Variance",
    color="Cluster",
    title="Variance Clustering by Station",
    labels={"Variance": "Variance", "Station": "Station", "Cluster": "Cluster"},
    template="plotly_white",
)
fig_scatter.show()

# Step 2b: Gaussian KDE for variance distribution
variance_values = station_variance["Variance"]
kde = gaussian_kde(variance_values)

x = np.linspace(variance_values.min(), variance_values.max(), 500)
y = kde(x)

fig_kde = px.area(
    x=x,
    y=y,
    title="Density of Variance Across Stations",
    labels={"x": "Variance", "y": "Density"},
    template="plotly_white",
)
fig_kde.show()