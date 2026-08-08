import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.config import get_dataset_path
except ImportError:
    def get_dataset_path():
        candidates = [
            "data/cleaned_enterprise_data_final.csv",
            "cleaned_enterprise_data.csv",
            "data/cleaned_enterprise_data.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return "cleaned_enterprise_data.csv"

def get_dataset_schema(file_path: str = "cleaned_enterprise_data.csv") -> str:
    """
    Pure Python deterministic tool (No AI Model).
    Reads the CSV file via Pandas and returns column names, data types, and row count as a string.
    """
    try:
        target_path = file_path if os.path.exists(file_path) else get_dataset_path()
        df = pd.read_csv(target_path)
        schema_info = f"--- ENTERPRISE DATASET SCHEMA ---\n"
        schema_info += f"File Path: {os.path.basename(target_path)}\n"
        schema_info += f"Total Records: {len(df):,} rows, {len(df.columns)} columns\n\n"
        schema_info += "Columns & Data Types:\n"
        for col, dtype in df.dtypes.items():
            null_cnt = df[col].isnull().sum()
            sample_val = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "N/A"
            if len(sample_val) > 35:
                sample_val = sample_val[:32] + "..."
            schema_info += f"- {col} ({dtype}) | Missing: {null_cnt} | Sample: {sample_val}\n"
        return schema_info
    except Exception as e:
        return f"Error reading dataset schema: {str(e)}"

if __name__ == "__main__":
    print(get_dataset_schema())