import pandas as pd

def normalize_part_numbers(bom_df, cross_ref_df, part_col='part_number', spec_col='specification', product_id_col='product_id'):
    """
    Replaces specifications in the part_number column of the BOM DataFrame with actual part numbers 
    using the consumables cross-reference DataFrame.
    
    Parameters:
    - bom_df: DataFrame containing the BOM (eBOM, mBOM-TC, mBOM-Oracle)
    - cross_ref_df: DataFrame with columns for part number, specification, and product ID
    - part_col: The name of the column in the BOM that contains the part numbers/specs
    - spec_col: The name of the column in cross_ref_df that contains specifications
    - product_id_col: Optional, in case you want to match on product ID too
    
    Returns:
    - A copy of the BOM DataFrame with normalized part numbers
    """
    # Create mapping from specification to part number
    spec_to_part = dict(zip(cross_ref_df[spec_col], cross_ref_df[part_col]))
    
    # Function to replace spec with part number if it matches
    def replace_spec(val):
        return spec_to_part.get(val, val)  # If no match, keep original

    # Replace in a copy of the BOM
    bom_df_copy = bom_df.copy()
    bom_df_copy[part_col] = bom_df_copy[part_col].apply(replace_spec)

    return bom_df_copy