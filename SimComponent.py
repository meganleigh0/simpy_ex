# ONE-CELL SOLUTION  ──────────────────────────────────────────────────────────
import pandas as pd, re
from pathlib import Path

# ── 1. CONFIG ────────────────────────────────────────────────────────────────
BASE_DIR   = Path("data/bronze_boms")      # change if your root differs
EXTENSIONS = (".csv", ".xls", ".xlsx", ".xlsm")

# ── 2. DISCOVER & LOAD ───────────────────────────────────────────────────────
def discover_files(bom_kind: str, source: str) -> list[Path]:
    """
    Returns every file whose name looks like
        <anything>/<bom_kind>_<source>_YYYY-MM-DD.<ext>
    under BASE_DIR, no matter how deeply nested.
    """
    pattern = re.compile(fr"{bom_kind}_{source}_(\d{{4}}-\d{{2}}-\d{{2}})")
    return [
        fp for fp in BASE_DIR.rglob(f"*{bom_kind}_{source}_*")
        if fp.suffix.lower() in EXTENSIONS and pattern.search(fp.name)
    ]

def load_bom_set(bom_kind: str, source: str) -> pd.DataFrame:
    """
    Loads *all* files in one long tidy frame:
        PART_NUMBER | make_buy | snapshot_date | source | bom_kind
    """
    frames = []
    for fp in discover_files(bom_kind, source):
        # pull date from filename
        date_txt = re.search(r"_(\d{4}-\d{2}-\d{2})", fp.stem).group(1)
        snap_date = pd.to_datetime(date_txt)

        # read file (Excel or CSV)
        if fp.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
            df = pd.read_excel(fp)
        else:
            df = pd.read_csv(fp)

        # ----- column cleanup -------------------------------------------------
        colmap = {c: c.strip().upper().replace(" ", "_") for c in df.columns}
        df.rename(columns=colmap, inplace=True)

        # normalise part-number column
        part_col = [c for c in df.columns if c.startswith("PART")][0]
        df.rename(columns={part_col: "PART_NUMBER"}, inplace=True)

        # normalise make/buy column
        if "MAKE/BUY" in df.columns:
            df.rename(columns={"MAKE/BUY": "make_buy"}, inplace=True)
        elif "MAKE_OR_BUY" in df.columns:
            df.rename(columns={"MAKE_OR_BUY": "make_buy"}, inplace=True)
        else:  # fall-back: grab first col containing "MAKE"
            mb_col = [c for c in df.columns if "MAKE" in c][0]
            df.rename(columns={mb_col: "make_buy"}, inplace=True)

        df = df[["PART_NUMBER", "make_buy"]].copy()
        df["make_buy"] = (df["make_buy"]
                          .astype(str)
                          .str.strip()
                          .str.upper()
                          .replace({"M": "MAKE", "B": "BUY"}))

        # meta columns
        df["snapshot_date"] = snap_date
        df["source"]        = source
        df["bom_kind"]      = bom_kind
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No files found for {bom_kind}-{source}")

    return pd.concat(frames, ignore_index=True)

# ── 3. LOAD EnBOMs ───────────────────────────────────────────────────────────
enbom_oracle = load_bom_set("ebom", "oracle")      # Engineering BOM, Oracle
enbom_tc     = load_bom_set("ebom", "tc")          # Engineering BOM, TeamCenter

# ── 4. HOW MANY MAKE/BUY FLIPS? ──────────────────────────────────────────────
def flip_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return # of flips per part + first/last status for quick triage."""
    df = df.sort_values(["PART_NUMBER", "snapshot_date"])
    df["prev_state"] = df.groupby("PART_NUMBER")["make_buy"].shift()
    df["flipped"]    = df["make_buy"] != df["prev_state"]

    return (df.groupby("PART_NUMBER")
              .agg(
                  n_snapshots = ("snapshot_date", "size"),
                  n_flips     = ("flipped", "sum"),
                  first_state = ("make_buy", "first"),
                  last_state  = ("make_buy", "last"),
              )
              .reset_index()
              .sort_values("n_flips", ascending=False))

oracle_flip_summary = flip_report(enbom_oracle)
tc_flip_summary     = flip_report(enbom_tc)

# ── 5. QUICK LOOK ────────────────────────────────────────────────────────────
print("Oracle EnBOM – top 10 parts by # flips:")
print(oracle_flip_summary.head(10), "\n")

print("TeamCenter EnBOM – top 10 parts by # flips:")
print(tc_flip_summary.head(10))

# ── 6. (OPTIONAL) SAVE TO GOLD LAYER ────────────────────────────────────────
# spark.createDataFrame(oracle_flip_summary) \
#      .write.format("delta").mode("overwrite") \
#      .save("dbfs:/gold/enbom_oracle_flip_summary")
#
# spark.createDataFrame(tc_flip_summary) \
#      .write.format("delta").mode("overwrite") \
#      .save("dbfs:/gold/enbom_tc_flip_summary")