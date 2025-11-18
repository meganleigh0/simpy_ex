# -------------------------------------------------------------
# Program Manpower table (SHC only, 9/80 schedule)
#   Demand  = current-month BCWS hours / 9/80 available hours
#   Actual  = current-month ACWP hours / 9/80 available hours
#   Next Mo BCWS = next-month BCWS hours / 9/80 available hours
#   Next Mo ETC  = next-month ETC  hours / 9/80 available hours
# -------------------------------------------------------------

xl_cobra = pd.ExcelFile(DATA_PATH)
cobra_all = xl_cobra.parse(SHEET_NAME)
cobra_all["DATE"] = pd.to_datetime(cobra_all["DATE"], errors="coerce")

# SHC only
cobra_shc = cobra_all[cobra_all[GROUP_COL] == "SHC"].copy()
cobra_shc["YEAR"] = cobra_shc["DATE"].dt.year
cobra_shc["MONTH"] = cobra_shc["DATE"].dt.month

# --- use the last date that exists in the SHC data as the "current" month ---
last_date = cobra_shc["DATE"].max()
cur_year = int(last_date.year)
cur_month = int(last_date.month)

# compute next month/year
if cur_month == 12:
    next_year = cur_year + 1
    next_month = 1
else:
    next_year = cur_year
    next_month = cur_month + 1

# Sum hours by COST-SET for current and next month
cur_hours = (
    cobra_shc[
        (cobra_shc["YEAR"] == cur_year) & (cobra_shc["MONTH"] == cur_month)
    ]
    .groupby("COST-SET")["HOURS"]
    .sum()
)

next_hours = (
    cobra_shc[
        (cobra_shc["YEAR"] == next_year) & (cobra_shc["MONTH"] == next_month)
    ]
    .groupby("COST-SET")["HOURS"]
    .sum()
)

# Ensure we always have ACWP / BCWS / ETC keys
cur_hours = cur_hours.reindex(["ACWP", "BCWS", "ETC"], fill_value=0.0)
next_hours = next_hours.reindex(["ACWP", "BCWS", "ETC"], fill_value=0.0)

# 9/80 schedule hours
available_9_80 = {
    2024: [142, 160, 196, 156, 160, 191, 151, 160, 191, 160, 151, 155],
    2025: [124, 160, 200, 160, 160, 191, 152, 160, 191, 191, 160, 173],
    2026: [147, 160, 200, 160, 151, 195, 156, 160, 191, 160, 151, 151],
    2027: [147, 160, 192, 160, 151, 190, 151, 160, 191, 160, 151, 160],
    2028: [151, 160, 200, 160, 160, 191, 151, 160, 191, 160, 142, 160],
}

def get_available_hours(year, month):
    return available_9_80[year][month - 1]  # 0-based index

cur_avail = get_available_hours(cur_year, cur_month)
next_avail = get_available_hours(next_year, next_month)

# Convert hours to FTE
demand        = cur_hours["BCWS"] / cur_avail
actual        = cur_hours["ACWP"] / cur_avail
next_bcws_fte = next_hours["BCWS"] / next_avail
next_etc_fte  = next_hours["ETC"]  / next_avail

program_manpower_tbl = pd.DataFrame(
    {
        "SUB_TEAM": ["SHC"],
        "Demand": [round(demand, 1)],
        "Actual": [round(actual, 1)],
        "Next Month BCWS": [round(next_bcws_fte, 1)],
        "Next Month ETC": [round(next_etc_fte, 1)],
    }
)

program_manpower_tbl