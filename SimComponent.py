import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# --- PIE CHARTS: BUY vs MAKE share by SEP parts for each org ---
# Uses mh from the previous cell (normalized/filled)
for org, g in mh.groupby('Usr_Org'):
    parts = (
        g.groupby('Make_Buy')['SEP_Part_Count']
         .sum()
         .reindex(['BUY','MAKE'])  # keep stable order
         .fillna(0)
    )
    if parts.sum() <= 0:
        continue  # skip orgs with no parts

    plt.figure()
    plt.pie(parts.values, labels=parts.index, autopct='%1.1f%%', startangle=90)
    plt.title(f'{org}: BUY vs MAKE (by SEP parts)')
    plt.axis('equal')  # perfect circle
    plt.show()

# --- QUICK PREDICTIONS: XM30 & M10 hours from SEP hours (org-level) ---
agg = (
    mh.groupby('Usr_Org', as_index=True)
      .agg(
          SEP_Hours=('SEP_CWS_Hours','sum'),
          XM30_Hours=('XM30_CWS_Hours','sum'),
          M10_Hours=('M10_CWS_Hours','sum')
      )
)

# Drop rows where we don't have targets
xm30_df = agg[['SEP_Hours','XM30_Hours']].replace(0, np.nan).dropna()
m10_df  = agg[['SEP_Hours','M10_Hours']].replace(0, np.nan).dropna()

def fit_and_report(X, y, label):
    if len(X) < 2:
        print(f'Not enough data to fit {label} model (need ≥2 orgs with data).')
        return None, None
    model = LinearRegression().fit(X, y)
    yhat  = model.predict(X)
    print(f'\n{label} ~ SEP_Hours')
    print(f'  coef:      {model.coef_[0]:.4f}')
    print(f'  intercept: {model.intercept_:.4f}')
    print(f'  R^2:       {r2_score(y, yhat):.4f}')
    print(f'  MAE:       {mean_absolute_error(y, yhat):.4f}')
    return model, yhat

# Fit XM30 from SEP
xm30_model, xm30_hat = (None, None)
if not xm30_df.empty:
    xm30_model, xm30_hat = fit_and_report(xm30_df[['SEP_Hours']].values, xm30_df['XM30_Hours'].values, 'XM30_Hours')

# Fit M10 from SEP
m10_model, m10_hat = (None, None)
if not m10_df.empty:
    m10_model, m10_hat = fit_and_report(m10_df[['SEP_Hours']].values, m10_df['M10_Hours'].values, 'M10_Hours')

# Helper functions for ad-hoc predictions
def predict_xm30(sep_hours: float) -> float:
    """Predict XM30 hours from SEP hours."""
    if xm30_model is None:
        raise ValueError("XM30 model not available (insufficient data).")
    return float(xm30_model.predict(np.array([[sep_hours]])).ravel()[0])

def predict_m10(sep_hours: float) -> float:
    """Predict M10 hours from SEP hours."""
    if m10_model is None:
        raise ValueError("M10 model not available (insufficient data).")
    return float(m10_model.predict(np.array([[sep_hours]])).ravel()[0])

# Example usage:
# predict_xm30(250.0), predict_m10(250.0)