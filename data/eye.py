import pandas as pd

def get_dataset_schema(file_path: str = "cleaned_enterprise_data.csv") -> str:
    """
    Reads a CSV dataset and returns a list of its column names and their corresponding data types.
    Use this tool FIRST whenever you are asked to analyze data, so you understand the structure 
    of the dataset before writing any data manipulation code.
    
    Args:
        file_path (str): The path to the CSV file. Defaults to "cleaned_enterprise_data.csv".
        
    Returns:
        str: A formatted string detailing the dataset's columns and data types, or an error message.
    """
    try:
        # Load the clean dataset you generated earlier
        df = pd.read_csv(file_path)
        
        # Build a readable string of the schema for the LLM
        schema_info = "Dataset Schema for 'cleaned_enterprise_data.csv':\n"
        for col, dtype in df.dtypes.items():
            schema_info += f"- {col} (Data Type: {dtype})\n"            
        return schema_info
        
    except Exception as e:
        return f"Error reading the dataset. Please check the file path. Details: {str(e)}"
    