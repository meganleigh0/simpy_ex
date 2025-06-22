# Define the matrix based on the visible table in the image
data = [
    {"Material": "GFM1", "Segment": "GDLS", "GFM FLAG": "X"},
    {"Material": "GFM2 GDLS", "Segment": "GDLS", "GFM FLAG": "X"},
    {"Material": "IST London", "Segment": "CSSC"},
    {"Material": "IST Other", "Segment": "CSSC"},
    {"Material": "MATL", "Segment": "Reference Only"},
    {"Material": "MS1A1 GDLS", "Segment": "GDLS", "SPARES MATL": "X", "REORD POINT": "X", "PCURE GDLS": "X"},
    {"Material": "MS1A2 CSSC", "Segment": "CSSC", "SPARES MATL": "X", "REORD POINT": "X", "PCURE GDLS": "X", "G&A CSSC": "X"},
    {"Material": "MS1A2 GDLS", "Segment": "GDLS", "SPARES MATL": "X", "REORD POINT": "X", "PCURE GDLS": "X", "G&A GDLS": "X"},
    {"Material": "MS1A3 CSSC", "Segment": "CSSC", "SPARES MATL": "X", "REORD POINT": "X", "PCURE GDLS": "X", "G&A CSSC": "X"},
    {"Material": "MS1A3 GDLS", "Segment": "GDLS", "SPARES MATL": "X", "REORD POINT": "X", "PCURE GDLS": "X", "G&A GDLS": "X"},
    {"Material": "MS1A4 GDLS", "Segment": "GDLS", "SPARES MATL": "X", "REORD POINT": "X", "PCURE GDLS": "X"},
    {"Material": "MS1C", "Segment": "GDLS", "PCURE GDLS": "X"},
    {"Material": "MSPT GDLS", "Segment": "GDLS", "SPARES ALLOCATION": "X"},

    # Spares Allocation DOES apply
    {"Material": "IST London Spares", "Segment": "CSSC", "SPARES MATL": "X", "REORD POINT": "X", "PCURE GDLS": "X", "SPARE ALLOCATION": "X"},
    {"Material": "MS1D1 GDLS", "Segment": "GDLS", "SPARES MATL": "X", "REORD POINT": "X", "PCURE GDLS": "X", "SPARE ALLOCATION": "X"},
    {"Material": "MS1D2 GDLS", "Segment": "GDLS", "SPARES MATL": "X", "REORD POINT": "X", "PCURE GDLS": "X", "SPARE ALLOCATION": "X"},
    {"Material": "MS1D3 GDLS", "Segment": "GDLS", "SPARES MATL": "X", "REORD POINT": "X", "PCURE GDLS": "X", "SPARE ALLOCATION": "X"},
    {"Material": "MS1D4 GDLS", "Segment": "GDLS", "SPARES MATL": "X", "REORD POINT": "X", "PCURE GDLS": "X", "SPARE ALLOCATION": "X"},
    {"Material": "MS1C GDLS", "Segment": "GDLS", "PCURE GDLS": "X", "SPARE ALLOCATION": "X"},

    # ODCs
    {"Material": "ODC CSSC", "Segment": "CSSC", "ODC": "X"},
    {"Material": "ODC GDLS", "Segment": "GDLS", "ODC": "X"},
    {"Material": "TRVL CSSC", "Segment": "CSSC", "ODC": "X"},
    {"Material": "TRVL GDLS", "Segment": "GDLS", "ODC": "X"},
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the structured matrix
import ace_tools as tools; tools.display_dataframe_to_user(name="Cost Element Burdening Matrix", dataframe=df)