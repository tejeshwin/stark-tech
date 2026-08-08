import os
import sys
import pandas as pd

# Automatically resolve paths for robustness
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_enterprise_data() -> pd.DataFrame:
    """
    Directly loads the cleaned enterprise CSV dataset into a Pandas DataFrame.
    """
    candidates = [
        "data/cleaned_enterprise_data_final.csv",
        "cleaned_enterprise_data.csv",
        "data/cleaned_enterprise_data.csv"
    ]
    
    target_path = "cleaned_enterprise_data.csv"
    for c in candidates:
        if os.path.exists(c):
            target_path = c
            break
            
    try:
        df = pd.read_csv(target_path)
        return df
    except Exception as e:
        # Return an empty DataFrame or raise a clean error if loading fails
        raise RuntimeError(f"Critical Error: Failed to load enterprise dataset from {target_path}. Details: {str(e)}")

# Initialize the global dataframe instance for the data agent to use directly
try:
    enterprise_df = load_enterprise_data()
    DATA_LOAD_STATUS = f"Success: Loaded {len(enterprise_df):,} rows and {len(enterprise_df.columns)} columns."
except Exception as e:
    enterprise_df = pd.DataFrame()
    DATA_LOAD_STATUS = str(e)

if __name__ == "__main__":
    print(DATA_LOAD_STATUS)
    if not enterprise_df.empty:
        print("\nFirst 3 rows of direct dataset preview:")
        print(enterprise_df.head(3))