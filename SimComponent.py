import pandas as pd

def replace_part_numbers(bom_df, cross_ref_df):
    """
    Replaces part numbers in the BOM DataFrame based on a cross-reference DataFrame.
    
    Parameters:
    - bom_df (pd.DataFrame): The BOM DataFrame containing a 'Part Number' column.
    - cross_ref_df (pd.DataFrame): The cross-reference DataFrame with 'Original' and 'Replacement' columns.
    
    Returns:
    - updated_bom_df (pd.DataFrame): BOM with updated part numbers.
    - mappings (dict): Dictionary of original -> replacement mappings that were applied.
    """
    # Make a copy to avoid modifying the original BOM
    updated_bom_df = bom_df.copy()
    
    # Create a dictionary from the cross-reference for fast lookup
    cross_ref_dict = dict(zip(cross_ref_df['Original'], cross_ref_df['Replacement']))
    
    # Track the mappings that were actually used
    mappings = {}

    def replace_part(part_number):
        if part_number in cross_ref_dict:
            mappings[part_number] = cross_ref_dict[part_number]
            return cross_ref_dict[part_number]
        return part_number

    # Apply the replacement
    updated_bom_df['Part Number'] = updated_bom_df['Part Number'].apply(replace_part)

    return updated_bom_df, mappings