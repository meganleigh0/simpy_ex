# -*- coding: utf-8 -*-
"""
Bulletproof Rates Pipeline

Inputs:
- ratesum_path: Excel with sheets "Simplified Rates" and "Other Rates"
  * Simplified Rates columns: "Business Unit GDLS", then one col per YEAR (e.g., 2023, 2024, ...)
  * Other Rates contains a row named "Allowable Control Test Rate" with year columns
- rate_band_path: Excel/CSV with columns including:
  * "Rate Band" (or similar), "Start Date", "End Date"
  * Optional description/name columns (to detect 'Abrams')
  * Optional measure columns (e.g., "Labor Dollars", "Burdens", "COM") for rounding

Outputs:
- rate_band_import_update.xlsx
- comparison_report.xlsx

Author: (you)
"""
from __future__ import annotations

import re
import sys
import json
import math
import logging
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Set

import pandas as pd
import numpy as np


# ------------------------- Configuration -------------------------

@dataclass
class PipelineConfig:
    # Sheet names (partial, case-insensitive ok)
    simplified_rates_sheet_hint: str = "simplified"
    other_rates_sheet_hint: str = "other"

    # Key phrases (case-insensitive)
    business_unit_col_hints: Tuple[str, ...] = ("business unit gdls", "business unit - gdls", "business unit")
    rate_band_col_hints: Tuple[str, ...] = ("rate band", "rateband", "band")
    start_date_col_hints: Tuple[str, ...] = ("start date", "start", "effective start")
    end_date_col_hints: Tuple[str, ...] = ("end date", "end", "effective end")
    description_col_hints: Tuple[str, ...] = ("description", "name", "program", "platform")

    # Row label to pull from Other Rates
    allowable_control_test_label: str = "allowable control test rate"

    # VT/Abrams override logic
    vt_codes: Set[str] = field(default_factory=lambda: {"VT"})
    abrams_phrase: str = "abrams"  # used against description/name columns

    # Rounding rules applied to output columns if present
    rounding_rules: Dict[str, int] = field(default_factory=lambda: {
        "labor dollars": 3,
        "burdens": 5,
        "com": 6,
    })

    # Output filenames
    out_update_filename: str = "rate_band_import_update.xlsx"
    out_compare_filename: str = "comparison_report.xlsx"

    # Logging
    log_level: int = logging.INFO


# ------------------------- Utilities -------------------------

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _lc(s: str) -> str:
    return str(s).strip().lower()

def _find_sheet_name(xls: pd.ExcelFile, hint: str) -> str:
    hint_lc = hint.lower()
    for name in xls.sheet_names:
        if hint_lc in name.lower():
            return name
    # fallback: exact if present
    for name in xls.sheet_names:
        if name.lower() == hint_lc:
            return name
    raise ValueError(f"Could not find a sheet matching hint '{hint}' in {xls.sheet_names}")

def _find_col(df: pd.DataFrame, hints: Iterable[str]) -> str:
    lc_map = {c.lower(): c for c in df.columns}
    for h in hints:
        for c_lc, c in lc_map.items():
            if h in c_lc:
                return c
    raise ValueError(f"Could not find any column matching hints: {hints}. Columns: {list(df.columns)}")

def _year_cols(df: pd.DataFrame) -> List[str]:
    years = []
    for c in df.columns:
        s = str(c).strip()
        if re.fullmatch(r"\d{4}", s):
            years.append(c)
    if not years:
        # also allow 'FY2025' patterns
        for c in df.columns:
            s = str(c).strip()
            m = re.fullmatch(r"(?:fy|FY)?(\d{4})", s)
            if m:
                years.append(c)
    if not years:
        raise ValueError("No year columns (e.g., 2024, 2025) found.")
    return years

def _first_two_letters(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Take first two alphabetic characters
    letters = re.findall(r"[A-Za-z]", text)
    return "".join(letters[:2]).upper()

def _to_decimal(val) -> Optional[Decimal]:
    if pd.isna(val):
        return None
    try:
        return Decimal(str(val))
    except Exception:
        return None

def _round_half_up(x: Optional[Decimal], places: int) -> Optional[Decimal]:
    if x is None:
        return None
    q = Decimal("1").scaleb(-places)  # e.g., places=3 -> Decimal('0.001')
    return x.quantize(q, rounding=ROUND_HALF_UP)

def _parse_date(val) -> pd.Timestamp:
    if pd.isna(val) or val == "":
        return pd.NaT
    return pd.to_datetime(val, errors="coerce")

def _safe_int_year(val) -> Optional[int]:
    if pd.isna(val):
        return None
    try:
        return int(val)
    except Exception:
        try:
            return int(float(val))
        except Exception:
            return None


# ------------------------- Core Steps -------------------------

def load_ratesum_sheets(ratesum_path: Path, cfg: PipelineConfig) -> Tuple[pd.DataFrame, pd.Series]:
    xls = pd.ExcelFile(ratesum_path)
    simp_name = _find_sheet_name(xls, cfg.simplified_rates_sheet_hint)
    other_name = _find_sheet_name(xls, cfg.other_rates_sheet_hint)

    df_simp = pd.read_excel(xls, sheet_name=simp_name, dtype=object)
    df_other = pd.read_excel(xls, sheet_name=other_name, dtype=object)

    df_simp = _normalize_cols(df_simp)
    df_other = _normalize_cols(df_other)

    # Identify columns
    bu_col = _find_col(df_simp, tuple(h for h in cfg.business_unit_col_hints))
    year_cols = _year_cols(df_simp)

    # Build simplified long (rate_band, year -> value)
    df_simp["_rate_band"] = df_simp[bu_col].astype(str).map(_first_two_letters)
    melted = df_simp.melt(
        id_vars=[bu_col, "_rate_band"],
        value_vars=year_cols,
        var_name="year_col",
        value_name="value",
    )
    # Normalize year as int
    melted["year"] = melted["year_col"].astype(str).str.extract(r"(\d{4})").iloc[:, 0].astype(int)
    # Deduplicate: if same band/year appears multiple times, take first non-null
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted = (melted
              .dropna(subset=["_rate_band", "year"])
              .sort_values([ "_rate_band", "year"])
              .drop_duplicates(subset=["_rate_band", "year"], keep="first"))

    # Create mapping band->year->value
    # We'll carry-forward within the full needed year span later.
    simp_map = (melted
                .set_index(["_rate_band", "year"])["value"]
                .sort_index())

    # Parse Other Rates -> Allowable Control Test Rate by year
    # Find the row whose "label cell" equals allowable_control_test_label
    df_other_lc = df_other.applymap(lambda x: _lc(x) if isinstance(x, str) else x)

    # Try to find a column that acts like a label column (first non-year column)
    possible_label_cols = [c for c in df_other.columns if c not in _year_cols(df_other)]
    actr_row = None
    for c in possible_label_cols:
        # find first row where this col matches the label
        mask = df_other_lc[c].astype(str).str.strip().str.lower().eq(cfg.allowable_control_test_label)
        if mask.any():
            actr_row = df_other.loc[mask].iloc[0]
            break
    if actr_row is None:
        # Fallback: search entire frame
        mask = df_other_lc.applymap(lambda v: isinstance(v, str) and v.strip().lower() == cfg.allowable_control_test_label)
        if mask.any().any():
            i, j = np.where(mask.values)
            actr_row = df_other.iloc[i[0]]
        else:
            raise ValueError(f"Could not find row '{cfg.allowable_control_test_label}' in Other Rates sheet.")

    other_year_cols = _year_cols(df_other)
    actr_by_year = {}
    for c in other_year_cols:
        y = int(re.search(r"(\d{4})", str(c)).group(1))
        val = pd.to_numeric(actr_row[c], errors="coerce")
        if not pd.isna(val):
            actr_by_year[y] = float(val)

    if not actr_by_year:
        raise ValueError("Allowable Control Test Rate row found, but no numeric year values detected.")

    actr_series = pd.Series(actr_by_year, name="allowable_control_test_rate").sort_index()

    return simp_map, actr_series


def load_rate_band(rate_band_path: Path, cfg: PipelineConfig) -> pd.DataFrame:
    p = Path(rate_band_path)
    if p.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        xls = pd.ExcelFile(p)
        # pick the first sheet unless 'rate' hint appears
        sheet = None
        for name in xls.sheet_names:
            if "rate" in name.lower():
                sheet = name
                break
        if sheet is None:
            sheet = xls.sheet_names[0]
        df_rb = pd.read_excel(xls, sheet_name=sheet, dtype=object)
    else:
        df_rb = pd.read_csv(p, dtype=object)

    df_rb = _normalize_cols(df_rb)

    # Identify key columns
    band_col = _find_col(df_rb, cfg.rate_band_col_hints)
    start_col = _find_col(df_rb, cfg.start_date_col_hints)
    end_col = _find_col(df_rb, cfg.end_date_col_hints)

    # Optional description column
    desc_col = None
    for h in cfg.description_col_hints:
        try:
            desc_col = _find_col(df_rb, (h,))
            break
        except Exception:
            continue

    # Normalize datatypes
    df_rb["_band"] = df_rb[band_col].astype(str).str.strip()
    df_rb["_band_code"] = df_rb["_band"].apply(lambda s: re.sub(r"[^A-Za-z]", "", s)[:2].upper())
    df_rb["_start"] = df_rb[start_col].apply(_parse_date)
    df_rb["_end"] = df_rb[end_col].apply(_parse_date)
    if desc_col:
        df_rb["_desc"] = df_rb[desc_col].astype(str).str.strip().str.lower()
    else:
        df_rb["_desc"] = ""

    # Validate dates
    bad_dates = df_rb[df_rb["_start"].isna() | df_rb["_end"].isna()]
    if not bad_dates.empty:
        logging.warning("Some rate band rows have invalid start/end dates; they will be skipped.")
        df_rb = df_rb.dropna(subset=["_start", "_end"])

    # Compute year spans
    df_rb["_start_year"] = df_rb["_start"].dt.year
    df_rb["_end_year"] = df_rb["_end"].dt.year

    # Keep original column names to round later (if present)
    df_rb.attrs["original_columns"] = list(df_rb.columns)

    return df_rb


def build_needed_years(df_rb: pd.DataFrame) -> List[int]:
    y_min = int(df_rb["_start_year"].min())
    y_max = int(df_rb["_end_year"].max())
    return list(range(y_min, y_max + 1))


def carry_forward(series_by_year: pd.Series, needed_years: List[int]) -> pd.Series:
    """Given a Series indexed by year with some gaps, return values for all needed_years,
    forward-filling using the last available <= year."""
    if series_by_year.empty:
        return pd.Series(index=needed_years, dtype=float)

    s = series_by_year.sort_index()
    # reindex with full span (from min to max provided), then ffill
    full_span = list(range(min(s.index), max(max(s.index), max(needed_years)) + 1))
    s_full = s.reindex(full_span)
    s_full = s_full.fillna(method="ffill")

    # Now pick only needed_years; if needed years precede the first known year, we cannot backfill—leave NaN
    out = s_full.reindex(needed_years)
    # For years < first provided year, leave NaN
    return out


def make_comparison_sets(df_simplified_bands: Set[str], df_rate_bands: Set[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    only_in_simplified = sorted(list(df_simplified_bands - df_rate_bands))
    only_in_ratebands = sorted(list(df_rate_bands - df_simplified_bands))
    a = pd.DataFrame({"rate_band_code": only_in_simplified})
    b = pd.DataFrame({"rate_band_code": only_in_ratebands})
    return a, b


def update_rates(
    df_rb: pd.DataFrame,
    simp_map: pd.Series,            # index: (band, year) -> value
    actr_series: pd.Series,         # index: year -> value
    cfg: PipelineConfig
) -> pd.DataFrame:
    needed_years = build_needed_years(df_rb)

    # Build per-band series from simplified map, then carry-forward
    bands_in_simplified = set([k[0] for k in simp_map.index])
    band_to_series: Dict[str, pd.Series] = {}
    for band in bands_in_simplified:
        s = simp_map.xs(band).sort_index()  # Series indexed by year
        band_to_series[band] = carry_forward(s, needed_years)

    # Carry-forward ACTR too
    actr_cf = carry_forward(actr_series, needed_years)

    # Make sets for comparison report
    bands_in_rb = set(df_rb["_band_code"].unique().tolist())
    comp_a, comp_b = make_comparison_sets(bands_in_simplified, bands_in_rb)

    # Expand rate band rows to per-year rows
    records = []
    for idx, row in df_rb.iterrows():
        band_code = row["_band_code"]
        desc = row.get("_desc", "")
        y0 = int(row["_start_year"])
        y1 = int(row["_end_year"])
        years = list(range(y0, y1 + 1))

        # Determine source series for this row
        use_actr = (band_code in cfg.vt_codes) or (cfg.abrams_phrase in str(desc).lower())

        for y in years:
            if use_actr:
                rate_val = actr_cf.get(y, np.nan)
            else:
                s = band_to_series.get(band_code)
                rate_val = s.get(y, np.nan) if s is not None else np.nan

            records.append({
                "row_id": idx,
                "rate_band_code": band_code,
                "year": y,
                "start_date": row["_start"],
                "end_date": row["_end"],
                "description": desc,
                "rate_value": rate_val
            })

    df_update = pd.DataFrame.from_records(records)

    # Apply rounding rules if the target file expects separate measure columns.
    # If your rate band import wants *one* column, we'll keep 'rate_value' and also
    # create optional columns if they exist in the original RB file.
    orig_cols = df_rb.attrs.get("original_columns", [])
    orig_cols_lc = [c.lower() for c in orig_cols]

    # Decide which measure columns exist; create them if present in input
    for colname, places in cfg.rounding_rules.items():
        # find matching column in original RB file (case-insensitive, contains)
        candidates = [orig_cols[i] for i, lc in enumerate(orig_cols_lc) if colname in lc]
        if candidates:
            out_col = candidates[0]  # use the first match
            # Round 'rate_value' to 'places' and put into this column
            df_update[out_col] = df_update["rate_value"].apply(lambda v: float(_round_half_up(_to_decimal(v), places)) if pd.notna(v) else np.nan)

    # Always keep a nicely rounded generic value too (3 decimals as general default)
    df_update["rate_value_rounded"] = df_update["rate_value"].apply(
        lambda v: float(_round_half_up(_to_decimal(v), 3)) if pd.notna(v) else np.nan
    )

    # Attach comparison frames for later save
    df_update.attrs["comparison_in_ratesum_not_in_ratebands"] = comp_a
    df_update.attrs["comparison_in_ratebands_not_in_ratesum"] = comp_b

    return df_update


def save_outputs(
    df_update: pd.DataFrame,
    cfg: PipelineConfig,
    out_dir: Path
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Update export
    update_path = out_dir / cfg.out_update_filename
    with pd.ExcelWriter(update_path, engine="openpyxl") as xw:
        # Keep a clean set of columns in logical order
        base_cols = ["rate_band_code", "year", "start_date", "end_date", "description", "rate_value", "rate_value_rounded"]
        measure_cols = [c for c in df_update.columns if c not in base_cols and c not in ("row_id")]
        cols = base_cols + measure_cols
        cols = [c for c in cols if c in df_update.columns]
        df_update.loc[:, cols].to_excel(xw, sheet_name="rate_band_update", index=False)

    # 2) Comparison export
    comp_path = out_dir / cfg.out_compare_filename
    comp_a = df_update.attrs.get("comparison_in_ratesum_not_in_ratebands", pd.DataFrame())
    comp_b = df_update.attrs.get("comparison_in_ratebands_not_in_ratesum", pd.DataFrame())

    with pd.ExcelWriter(comp_path, engine="openpyxl") as xw:
        comp_a.to_excel(xw, sheet_name="in_ratesum_not_in_ratebands", index=False)
        comp_b.to_excel(xw, sheet_name="in_ratebands_not_in_ratesum", index=False)

    return update_path, comp_path


# ------------------------- Main entrypoint -------------------------

def run_pipeline(
    ratesum_path: str,
    rate_band_path: str,
    out_dir: str = ".",
    config: Optional[PipelineConfig] = None
) -> Tuple[Path, Path]:
    cfg = config or PipelineConfig()
    logging.basicConfig(level=cfg.log_level, format="%(levelname)s: %(message)s")

    ratesum_path = Path(ratesum_path)
    rate_band_path = Path(rate_band_path)
    out_dir = Path(out_dir)

    logging.info("Loading RateSum sheets...")
    simp_map, actr_series = load_ratesum_sheets(ratesum_path, cfg)

    logging.info("Loading Rate Band file...")
    df_rb = load_rate_band(rate_band_path, cfg)

    logging.info("Updating rates across date ranges...")
    df_update = update_rates(df_rb, simp_map, actr_series, cfg)

    logging.info("Writing outputs...")
    update_path, comp_path = save_outputs(df_update, cfg, out_dir)

    logging.info(f"Done.\n - Update: {update_path}\n - Compare: {comp_path}")
    return update_path, comp_path


if __name__ == "__main__":
    # Example CLI usage:
    # python rates_pipeline.py "RateSum.xlsx" "RateBands.xlsx" --out "./out"
    import argparse

    parser = argparse.ArgumentParser(description="Run rates pipeline.")
    parser.add_argument("ratesum_path", help="Path to RateSum Excel file")
    parser.add_argument("rate_band_path", help="Path to Rate Band file (Excel or CSV)")
    parser.add_argument("--out", dest="out_dir", default=".", help="Output directory (default: current)")
    args = parser.parse_args()

    run_pipeline(args.ratesum_path, args.rate_band_path, args.out_dir)