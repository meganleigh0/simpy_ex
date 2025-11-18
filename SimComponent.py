# -------------------------------------------------------------
# Program Manpower table (SHC only, 9/80 schedule)
#   Demand  = current-month BCWS hours / 9/80 available hours
#   Actual  = current-month ACWP hours / 9/80 available hours
#   Next Mo BCWS = next-month BCWS hours / 9/80 available hours
#   Next Mo ETC  = next-month ETC  hours / 9/80 available hours
# -------------------------------------------------------------

# 1) Reload Cobra data WITHOUT the ANCHOR filter so we can see next month
xl_cobra = pd.ExcelFile(DATA_PATH)
cobra_all = xl_cobra.parse(SHEET_NAME)
cobra_all["DATE"] = pd.to_datetime(cobra_all["DATE"], errors="coerce")

# SHC only, bucket into calendar months
cobra_shc = cobra_all[cobra_all[GROUP_COL] == "SHC"].copy()
cobra_shc["PERIOD"] = cobra_shc["DATE"].dt.to_period("M")

cur_per  = ANCHOR.to_period("M")
next_per = cur_per + 1

# Sum hours by COST-SET for current and next month
cur_hours = (
    cobra_shc[cobra_shc["PERIOD"] == cur_per]
    .groupby("COST-SET")["HOURS"]
    .sum()
)
next_hours = (
    cobra_shc[cobra_shc["PERIOD"] == next_per]
    .groupby("COST-SET")["HOURS"]
    .sum()
)

# Ensure all required COST-SETs exist
for k in ["ACWP", "BCWS", "ETC"]:
    if k not in cur_hours:
        cur_hours.loc[k] = 0.0
    if k not in next_hours:
        next_hours.loc[k] = 0.0

# 2) 9/80 schedule available hours (from the OpPlan screenshot)
#    If you prefer to read from Excel, replace this dict with pd.read_excel logic.
available_9_80 = {
    2024: [142, 160, 196, 156, 160, 191, 151, 160, 191, 160, 151, 155],
    2025: [124, 160, 200, 160, 160, 191, 152, 160, 191, 191, 160, 173],
    2026: [147, 160, 200, 160, 151, 195, 156, 160, 191, 160, 151, 151],
    2027: [147, 160, 192, 160, 151, 190, 151, 160, 191, 160, 151, 160],
    2028: [151, 160, 200, 160, 160, 191, 151, 160, 191, 160, 142, 160],
}

def get_available_hours(period):
    year = period.year
    month_idx = period.month - 1  # 0-based index
    return available_9_80[year][month_idx]

cur_avail  = get_available_hours(cur_per)
next_avail = get_available_hours(next_per)

# 3) Convert hours to headcount (FTE) using available hours
demand       = cur_hours["BCWS"] / cur_avail
actual       = cur_hours["ACWP"] / cur_avail
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