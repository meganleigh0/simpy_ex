# ----------------------------------------------------------
# ONE‑CELL “FIRST VALUE” LOCATOR AND MAPPER
# ----------------------------------------------------------
import numpy as np
import pandas as pd

# ── SET‑UP ────────────────────────────────────────────────
ANCHOR_YEAR = 2022       # look only at 2022 rows in BURDEN_RATE
TOLERANCE   = 1e-6       # float comparison tolerance

# ── 1) isolate 2022 rows of BURDEN_RATE ───────────────────
br2022 = BURDEN_RATE[BURDEN_RATE['# Date'] == ANCHOR_YEAR].copy()

# numeric columns to scan
num_cols = br2022.select_dtypes(include='number').columns.tolist()

# ── 2) find the first non‑blank, non‑1 numeric value ──────
target_val = None
br_info    = None

for idx, row in br2022.iterrows():
    for col in num_cols:
        val = row[col]
        if pd.notna(val) and val != 1:
            target_val = val
            br_info = {
                'BR_RowIndex'   : idx,
                'BR_BurdenPool' : row['Burden Pool'],
                'BR_Description': row['Description'],
                'BR_Column'     : col,
                'Value'         : val
            }
            break
    if target_val is not None:
        break

if target_val is None:
    raise ValueError("No numeric value other than 1 found in 2022 BURDEN_RATE rows.")

# ── 3) search other_rates 2022 columns for the same value ─
if other_rates.index.name != 'Burden Pool':
    other_rates = other_rates.set_index('Burden Pool')

yr_cols = [c for c in other_rates.columns if str(ANCHOR_YEAR) in c]  # CY2022, CY2022.1, …

match_info = None
for col in yr_cols:
    mask = np.isclose(other_rates[col], target_val, atol=TOLERANCE)
    if mask.any():
        pool = other_rates.index[mask].tolist()[0]
        match_info = {
            'OR_BurdenPool' : pool,
            'OR_Column'     : col,
            'Value'         : target_val
        }
        break

if match_info is None:
    raise ValueError(f"Value {target_val} not found in any {ANCHOR_YEAR} column of other_rates.")

# ── 4) show the mapping ──────────────────────────────────
print(">>> FOUND IN BURDEN_RATE")
for k, v in br_info.items():
    print(f"{k:15}: {v}")

print("\n>>> MATCHES IN other_rates")
for k, v in match_info.items():
    print(f"{k:15}: {v}")