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
            return "[Analytics Engine Output] Code executed successfully, but 'analysis_result' variable was not set."
        return f"[Analytics Engine Output] {str(result)}"
    except Exception as e:
        return f"[Analytics Engine Output] pandas code error: {str(e)}"

analytics_agent = LlmAgent(
    name="AnalyticsAgent",
    model="gemini-3.5-flash",
    description="Business Analysis & KPI Quantitative Specialist",
    instruction="""You are the Analytics Agent (Quantitative Engine). Your role is to write Python pandas code using 'execute_pandas_analysis' to compute exact numeric aggregations. Always prefix your analytical findings with '[Analytics Engine Output]'.""",
    tools=[execute_pandas_analysis]
)