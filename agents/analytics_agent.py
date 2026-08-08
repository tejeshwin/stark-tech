import os
import sys
import pandas as pd
from google.adk.agents import LlmAgent

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.config import get_dataset_path
except ImportError:
    def get_dataset_path():
        candidates = ["data/cleaned_enterprise_data_final.csv", "cleaned_enterprise_data.csv", "data/cleaned_enterprise_data.csv"]
        for c in candidates:
            if os.path.exists(c): return c
        return "cleaned_enterprise_data.csv"

def execute_pandas_analysis(python_code: str) -> str:
    """
    Executes Python pandas code to analyze the enterprise CSV dataset.
    The dataset is pre-loaded as a DataFrame named 'df'.
    IMPORTANT: Save your final text/numerical output to a variable named 'analysis_result'.
    """
    try:
        df = pd.read_csv(get_dataset_path())
        local_vars = {'df': df, 'pd': pd, 'analysis_result': None}
        exec(python_code, {}, local_vars)
        result = local_vars.get('analysis_result')
        if result is None:
            return "Code ran successfully, but 'analysis_result' was not assigned."
        return str(result)
    except Exception as e:
        return f"Code execution crashed. Error details: {str(e)}. Fix the code and retry."

analytics_agent = LlmAgent(
    name="AnalyticsAgent",
    model="gemini-3.5-flash",
    description="Business Analysis & KPI Specialist",
    instruction="""You are the Analytics Agent (Math Engine).
    Use the execute_pandas_analysis tool to write and run Python code to crunch numbers.
    Always assume the dataframe is pre-loaded as 'df'. Assign your final answer to 'analysis_result'.""",
    tools=[execute_pandas_analysis]
)