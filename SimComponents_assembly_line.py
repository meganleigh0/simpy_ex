import pandas as pd
import difflib
from collections import defaultdict

# --- Helper Functions ---

def build_bom_tree(bom_df, parent_col='ParentID', part_col='PART_NUMBER', qty_col='Qty'):
    """
    Build a dictionary that maps a parent to a list of its (child, quantity) tuples.
    Rows with missing (NaN) ParentID are assumed to be the top-level assembly.
    """
    tree = defaultdict(list)
    for _, row in bom_df.iterrows():
        parent = row[parent_col]
        part = row[part_col]
        # Default quantity is 1 if not provided or if NaN
        qty = row[qty_col] if qty_col in row and pd.notna(row[qty_col]) else 1
        tree[parent].append((part, qty))
    return dict(tree)

def explode_bom(tree, parent, multiplier=1, exploded=None):
    """
    Recursively traverse the BOM tree to calculate the total (exploded) quantity of each part.
    
    Args:
        tree (dict): BOM tree as returned by build_bom_tree.
        parent: The current parent to start from (e.g. top-level assembly, usually NaN).
        multiplier (float): The accumulated multiplication factor for quantities.
        exploded (defaultdict): Dictionary to accumulate quantities.
    
    Returns:
        defaultdict: Mapping of part numbers to total required quantity.
    """
    if exploded is None:
        exploded = defaultdict(float)
    for child, qty in tree.get(parent, []):
        total_qty = multiplier * qty
        exploded[child] += total_qty
        # Recursively process the child to handle multi-level BOMs.
        explode_bom(tree, child, total_qty, exploded)
    return exploded

def get_unique_parts(bom_df, part_col='PART_NUMBER'):
    """Return the set of unique part numbers from a BOM DataFrame."""
    return set(bom_df[part_col].unique())

def map_parts(ebom_parts, mbom_parts, cutoff=0.6):
    """
    Map EBOM part numbers to MBOM part numbers using fuzzy matching.
    
    Args:
        ebom_parts (set): Set of EBOM part numbers.
        mbom_parts (set): Set of MBOM part numbers.
        cutoff (float): Matching threshold between 0 and 1.
    
    Returns:
        dict: Mapping of EBOM part number to best–matching MBOM part number (or None if no match).
    """
    mapping = {}
    for part in ebom_parts:
        matches = difflib.get_close_matches(part, mbom_parts, n=1, cutoff=cutoff)
        mapping[part] = matches[0] if matches else None
    return mapping

# --- Main Analysis in a Single Cell ---

def main():
    # Read your EBOM and MBOM CSV files.
    # Adjust file names and paths as needed. These CSVs should include at least:
    #   - A Parent identifier column (e.g. ParentID)
    #   - A Part number column (e.g. PART_NUMBER)
    #   - A Quantity column (e.g. Qty)
    ebom_df = pd.read_csv('ebom.csv')  # e.g. columns: ParentID, PART_NUMBER, Qty, etc.
    mbom_df = pd.read_csv('mbom.csv')  # e.g. similar structure

    # Build BOM trees from the dataframes.
    ebom_tree = build_bom_tree(ebom_df)
    mbom_tree = build_bom_tree(mbom_df)
    
    # Identify the top-level assembly.
    # Here we assume rows with a missing ParentID (NaN) are top-level.
    # (If your data designates top-level differently, adjust accordingly.)
    top_level = None
    # Note: If multiple top-level assemblies exist, you might need to loop through each.
    for parent in ebom_tree.keys():
        if pd.isna(parent):
            top_level = parent
            break
    
    # Compute exploded quantities (total required parts) for each BOM.
    ebom_exploded = explode_bom(ebom_tree, top_level, multiplier=1)
    mbom_exploded = explode_bom(mbom_tree, top_level, multiplier=1)

    # Get unique part numbers from each BOM.
    ebom_parts = get_unique_parts(ebom_df)
    mbom_parts = get_unique_parts(mbom_df)

    # Map EBOM parts to MBOM parts via fuzzy matching.
    part_mapping = map_parts(ebom_parts, mbom_parts, cutoff=0.6)

    # Compile a summary with mapping and quantity comparisons.
    results = []
    for ebom_part in ebom_parts:
        mbom_part = part_mapping.get(ebom_part)
        ebom_qty = ebom_exploded.get(ebom_part, 0)
        mbom_qty = mbom_exploded.get(mbom_part, 0) if mbom_part is not None else None
        ratio = (mbom_qty / ebom_qty) if ebom_qty and mbom_qty is not None else None
        results.append({
            'EBOM_Part': ebom_part,
            'Mapped_MBOM_Part': mbom_part,
            'EBOM_Exploded_Qty': ebom_qty,
            'MBOM_Exploded_Qty': mbom_qty,
            'Predicted_Ratio_MBOM_to_EBOM': ratio
        })

    df_results = pd.DataFrame(results)
    print("Mapping and Quantity Comparison:")
    print(df_results)

if __name__ == '__main__':
    main()