# ─────────────────────────────────────────────────────────
#  HARD‑CODED “WHERE DOES IT GO?” TABLE
#     key  = ‘Burden Pool’ string in other_rates
#     val  = ( row‑id in BURDEN_RATE, numeric column in BURDEN_RATE )
# ─────────────────────────────────────────────────────────
MAP = {
    # ALLOWABLE G&A – CSSC
    "PSGA - CSSC G & A ALLOWABLE G & A RATE"
        : ("CSSC BURDEN RATES",                "G&A C$"),

    # DIVISION‑level G&A – CSSC
    "DVGA - DIVISION GENERAL & ADM ALLOWABLE G & A RATE"
        : ("CSSC BURDEN RATES",                "G&A G$"),

    # Re‑order‑point surcharge
    "DeptNA ALLOWABLE REORDER POINT RATE"
        : ("NON DELIVERABLE",                  "ROP"),

    # Procurement O/H – GDLS
    "PRLS - GDLS PROCUREMENT ALLOWABLE OVERHEAD RATE"
        : ("DELIVERABLE - NON‑PRODUCTION FACILITY", "Proc O/H"),

    # Freight O/H – GDLS & CSSC
    "PFRT - FREIGHT -- GDLS & CSSC ALLOWABLE OVERHEAD RATE"
        : ("PASS THRU",                        "Proc O/H"),

    # Corporate‑wide Procurement O/H
    "GENERAL DYNAMICS LAND SYSTEMS ALLOWABLE PROCUREMENT RATE"
        : ("DELIVERABLE - PRODUCTION FACILITY", "Proc O/H"),

    # Major End‑Item surcharge
    "DeptNA ALLOWABLE MAJOR END-ITEM RATE"
        : ("MAJOR END ITEMS",                  "MEI O/H"),

    # Support surcharge
    "ALLOWABLE SUPPORT RATE"
        : ("PASS THRU SPARES",                 "Support"),

    # Control‑test surcharge
    "ALLOWABLE CONTL TEST RATE"
        : ("OTHER DIRECT COST",                "Contl Test"),
}

# ─────────────────────────────────────────────────────────
#  MAIN UPDATE ROUTINE
# ─────────────────────────────────────────────────────────
def push_other_rates_into_burden_rate(
    other_rates: pd.DataFrame,
    burden_rate: pd.DataFrame,
    years      = (2022, 2023, 2024, 2025)
) -> pd.DataFrame:
    """
    Copy every CYxxxx value in *other_rates* to its mapped
    row/column in *burden_rate*.

    Returns a **new** DataFrame (does not mutate in place).
    """
    br = burden_rate.copy()

    # make look‑ups quick
    br.set_index(["Description", "# Date"], inplace=True, drop=False)

    for pool, (br_row_desc, br_col) in MAP.items():
        for yr in years:
            col_in_other = f"CY{yr}"
            if col_in_other not in other_rates.columns:
                continue                                    # that year isn't present
            new_val = other_rates.loc[pool, col_in_other]

            # locate the single row (Description == br_row_desc & # Date == yr)
            key = (br_row_desc, yr)
            if key not in br.index:
                raise KeyError(f"Row “{br_row_desc}” / year {yr} not found in BURDEN_RATE")

            br.at[key, br_col] = new_val

    # restore original positional index order
    br.reset_index(drop=True, inplace=True)
    return br


# ─────────────────────────────────────────────────────────
#  USAGE
# ─────────────────────────────────────────────────────────
BURDEN_RATE = push_other_rates_into_burden_rate(other_rates, BURDEN_RATE)

# (Optional) save the refreshed file
# BURDEN_RATE.to_excel("updated_BurdenRateImport.xlsx", index=False)