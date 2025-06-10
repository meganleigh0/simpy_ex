# -----------------------------------------------------------
# 5‑BIS.  ROUND VALUES BY TYPE  (run **after** the ffill loop)
# -----------------------------------------------------------
LABOR_DOLLAR_COLS = {            # 3‑decimals
    "Base Fee", "Goal Fee", "Profit/Fee"
}

# any column that starts with “COM ” will be 6 decimals
COM_PREFIX      = "COM "         # e.g.  "COM GDLS", "COM CSSC", …

# everything in long_rates['target_col'] that is NOT labor $ or COM* is a burden → 5 decimals
BURDEN_COLS = set(long_rates["target_col"].unique()) - LABOR_DOLLAR_COLS

for col in BURDEN_COLS:
    if col.startswith(COM_PREFIX):
        BURDEN_RATE[col] = pd.to_numeric(BURDEN_RATE[col], errors="coerce").round(6)
    else:
        BURDEN_RATE[col] = pd.to_numeric(BURDEN_RATE[col], errors="coerce").round(5)

for col in LABOR_DOLLAR_COLS:
    BURDEN_RATE[col] = pd.to_numeric(BURDEN_RATE[col], errors="coerce").round(3)