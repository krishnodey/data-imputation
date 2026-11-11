import numpy as np
import pandas as pd

def analyze_missing_patterns(data, mask, feature_names):
    """
    Analyzes missing data ratio and pattern characteristics.
    """
    total_cells = mask.size
    missing_cells = total_cells - int(mask.sum())
    missing_percentage = (missing_cells / total_cells) * 100

    print("=== Missing Data Analysis ===")
    print(f"Total Cells: {total_cells}")
    print(f"Missing Cells: {missing_cells} ({missing_percentage:.2f}%)")
    
    # Column-wise missingness
    col_missing = {}
    for idx, col in enumerate(feature_names):
        col_mask = mask[:, idx]
        missing_count = len(col_mask) - int(col_mask.sum())
        col_missing[col] = {
            "Missing Count": missing_count,
            "Missing Percentage": (missing_count / len(col_mask)) * 100
        }
        
    df_missing_summary = pd.DataFrame(col_missing).T
    print("\nColumn-wise Missingness:")
    print(df_missing_summary)
    
    return df_missing_summary
