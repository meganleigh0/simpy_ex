import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import lognorm

section = 'Vehicle'
df = total_df[total_df['Section'] == section]

data = list(df['Days'])

# Fit lognormal distribution
params = lognorm.fit(data)
x = np.linspace(np.min(data), np.max(data), num=1000)
pdf_fitted = lognorm.pdf(x, params[0], loc=params[1], scale=params[2])

# Plot
fig = plt.figure(figsize=(10, 5))
plt.plot(x, pdf_fitted, label="Log-normal Fit", color='blue')  # Fit curve
plt.hist(data, bins=15, density=True, edgecolor='black', color='orange', alpha=0.6)  # Histogram
plt.xlabel('Number of Days')
plt.ylabel('Probability Density')
plt.title(f'Distribution of Days Spent at {section}')
plt.legend()
plt.show()