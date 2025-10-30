# --- BOM Quality Snapshot + Visuals (SEP & XM30; TeamCenter & Oracle) ---

import pandas as pd
import matplotlib.pyplot as plt

# ====== INPUTS (update the variable names here if yours differ) ======
# Raw BOMs
inputs = {
    ("XM30", "TeamCenter"): XM30_BOM,
    ("XM30", "Oracle"): XM30_O_BOM,
    ("SEP",  "TeamCenter"): SEP_V3_BOM,
    ("SEP",  "Oracle"): SEP_V3_O_BOM,
}

# Processed/merged BOMs (optional; shapes only)
processed = {
    "XM30": xm30_df,
    "SEP":  sep_df,
}

# ====== Helpers ======
def find_first_col(df, candidates):
    """Return the first matching column from candidates (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise KeyError(f"None of the columns {candidates} found. Available: {list(df.columns)[:10]}...")

def clean_makebuy(series):
    """Normalize Make/Buy values to MAKE or BUY (everything else becomes OTHER)."""
    s = series.astype(str).str.strip().str.upper().replace({
        "MAKE/BUY": "",  # sometimes the header leaks into data; harmless
    })
    s = s.replace({"M": "MAKE", "B": "BUY"})
    s = s.where(s.isin(["MAKE", "BUY"]), other="OTHER")
    return s

def bom_stats(df, program, system):
    # Try common column names across exports
    part_col = find_first_col(df, ["PART_NUMBER", "Part-Number", "Part Number", "PART NO", "ITEM"])
    makebuy_col = find_first_col(df, ["Make or Buy", "Make/Buy", "MAKE_OR_BUY", "MAKE BUY", "Make_Buy"])
    
    # Counts
    total_rows, total_cols = df.shape
    unique_parts = df[part_col].nunique(dropna=True)
    mb = clean_makebuy(df[makebuy_col]).value_counts(dropna=False)
    make_cnt = int(mb.get("MAKE", 0))
    buy_cnt  = int(mb.get("BUY", 0))
    other_cnt = int(mb.get("OTHER", 0))
    make_pct = make_cnt / max((make_cnt + buy_cnt + other_cnt), 1)
    buy_pct  = buy_cnt  / max((make_cnt + buy_cnt + other_cnt), 1)
    
    return {
        "Program": program,
        "System": system,  # TeamCenter vs Oracle
        "Rows": total_rows,
        "Columns": total_cols,
        "Unique Parts": unique_parts,
        "Make Count": make_cnt,
        "Buy Count": buy_cnt,
        "Other/Unknown Count": other_cnt,
        "Make %": round(make_pct, 4),
        "Buy %": round(buy_pct, 4),
    }

# ====== Build summary table ======
rows = []
for (program, system), df in inputs.items():
    rows.append(bom_stats(df, program, system))

summary_df = pd.DataFrame(rows).sort_values(["Program", "System"]).reset_index(drop=True)

# Optional: add processed shapes if you want them in the same table
proc_rows = []
for program, df in processed.items():
    r, c = df.shape
    proc_rows.append({
        "Program": program,
        "System": "Processed",
        "Rows": r,
        "Columns": c,
        "Unique Parts": pd.NA,
        "Make Count": pd.NA,
        "Buy Count": pd.NA,
        "Other/Unknown Count": pd.NA,
        "Make %": pd.NA,
        "Buy %": pd.NA,
    })
if proc_rows:
    summary_df = pd.concat([summary_df, pd.DataFrame(proc_rows)], ignore_index=True)

# Display the table nicely
display(summary_df)

# ====== Visuals ======
# 1) Make vs Buy counts by Program & System
mb_long = (summary_df
           .query("System != 'Processed'")
           .melt(id_vars=["Program", "System"],
                 value_vars=["Make Count", "Buy Count"],
                 var_name="Type", value_name="Count"))

# Create a combined label for easy grouping on the x-axis
mb_long["Group"] = mb_long["Program"] + " • " + mb_long["System"]

plt.figure(figsize=(10, 5))
for t in ["Make Count", "Buy Count"]:
    sub = mb_long[mb_long["Type"] == t]
    plt.bar(sub["Group"], sub["Count"], label=t, bottom=None if t=="Make Count" else None)  # bars will overlay; keep labels
# To avoid overlap, we can do side-by-side by offsetting x positions:
# For clarity, rebuild as side-by-side:
plt.clf()
groups = mb_long["Group"].unique()
x = range(len(groups))
make_counts = [int(mb_long[(mb_long["Group"]==g) & (mb_long["Type"]=="Make Count")]["Count"].sum()) for g in groups]
buy_counts  = [int(mb_long[(mb_long["Group"]==g) & (mb_long["Type"]=="Buy Count")]["Count"].sum())  for g in groups]

width = 0.35
plt.figure(figsize=(10,5))
plt.bar([i - width/2 for i in x], make_counts, width, label="Make")
plt.bar([i + width/2 for i in x], buy_counts,  width, label="Buy")
plt.xticks(list(x), groups, rotation=20, ha="right")
plt.ylabel("Count")
plt.title("Make vs Buy Counts by Program • System")
plt.legend()
plt.tight_layout()
plt.show()

# 2) Unique parts by Program & System
up = summary_df.query("System != 'Processed'")[["Program","System","Unique Parts"]].copy()
up["Group"] = up["Program"] + " • " + up["System"]

plt.figure(figsize=(8,4))
plt.bar(up["Group"], up["Unique Parts"])
plt.xticks(rotation=20, ha="right")
plt.ylabel("Unique Part Count")
plt.title("Unique Parts by Program • System")
plt.tight_layout()
plt.show()

# 3) Optional: Make/Buy percentages by Program & System
pct = summary_df.query("System != 'Processed'")[["Program","System","Make %","Buy %"]].copy()
pct["Group"] = pct["Program"] + " • " + pct["System"]
plt.figure(figsize=(8,4))
plt.bar(pct["Group"], pct["Make %"], label="Make %")
plt.bar(pct["Group"], pct["Buy %"], bottom=pct["Make %"], label="Buy %")
plt.xticks(rotation=20, ha="right")
plt.ylabel("Share")
plt.title("Make vs Buy Share by Program • System")
plt.legend()
plt.tight_layout()
plt.show()