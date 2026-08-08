import os
import sys
import time
import asyncio
from typing import Generator, Dict, Any, Tuple
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from utils.config import get_dataset_path, load_dataset, get_gemini_api_key, key_manager, BASE_DIR
except ImportError:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    def get_dataset_path():
        return "cleaned_enterprise_data.csv"
    def get_gemini_api_key():
        return os.getenv("GEMINI_API_KEY", "")
    class KeyManagerFallback:
        def rotate_to_next_key(self): return get_gemini_api_key(), "Only 1 key configured."
        def record_request_and_check_warning(self): return ""
    key_manager = KeyManagerFallback()

_key = get_gemini_api_key()
if _key:
    os.environ["GEMINI_API_KEY"] = _key
    os.environ["GOOGLE_API_KEY"] = _key

from data_agent import get_dataset_schema
from prediction_agent import execute_ml_model
from semantic_agent import semantic_agent
from analytics_agent import analytics_agent, execute_pandas_analysis
from visualization_agent import visualization_agent, generate_plot
from consultant_agent import consultant_agent

# The Manager ("The Brain") - Orchestrator Agent powered by Google ADK (gemini-3.5-flash)
orchestrator_agent = LlmAgent(
    name="ChiefDataOfficer",
    model="gemini-3.5-flash",
    description="The primary manager, Chief Data Officer, and multi-agent orchestrator.",
    instruction="""You are the Chief Data Officer. You direct an AI data team consisting of specialist agents and tools:
1. Use 'get_dataset_schema' to inspect dataset structure.
2. Use 'execute_pandas_analysis' to write Pandas code and calculate exact metrics.
3. Use 'execute_ml_model' for revenue forecasting and risk prediction.
4. Use 'generate_plot' to create vibrant, detailed executive Seaborn/Matplotlib charts with data labels and explicit titles.
5. Coordinate with specialist sub-agents (SemanticValidationAgent, AnalyticsAgent, VisualizationAgent, ConsultantAgent).
Deliver clear, quantitative, executive-ready conclusions with citations.""",
    tools=[get_dataset_schema, execute_ml_model, execute_pandas_analysis, generate_plot],
    sub_agents=[semantic_agent, analytics_agent, visualization_agent, consultant_agent]
)

class ADKOrchestratorPipeline:
    """
    Google ADK Runner Pipeline for executing multi-agent workflows safely with key rotation and chart tracking.
    """
    def __init__(self, agent: LlmAgent = orchestrator_agent):
        self.agent = agent
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.agent,
            session_service=self.session_service,
            app_name="Enterprise_Data_Team",
            auto_create_session=True
        )
        self.session_id = "default_session"
        self.user_id = "default_user"

    async def run_query_async(self, query: str) -> Tuple[str, str]:
        """
        Executes query with automatic key rotation, token quota warnings, and chart detection.
        Returns tuple: (response_text, generated_chart_path_or_None)
        """
        api_key = key_manager.get_active_key()
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            os.environ["GOOGLE_API_KEY"] = api_key

        chart_path = os.path.join(BASE_DIR, "dashboard_chart.png")
        if os.path.exists(chart_path):
            try:
                os.remove(chart_path)
            except Exception:
                pass

        start_time = time.time()
        quota_warning = key_manager.record_request_and_check_warning()
        
        content = types.Content(parts=[types.Part.from_text(text=query)])
        response_texts = []
        if quota_warning:
            response_texts.append(quota_warning)

        max_retries = max(1, len(getattr(key_manager, 'keys', [1])))
        attempt = 0
        
        while attempt < max_retries:
            attempt += 1
            try:
                async for event in self.runner.run_async(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    new_message=content
                ):
                    if hasattr(event, 'error_message') and event.error_message:
                        response_texts.append(f"[API Error] {event.error_message}")
                    elif hasattr(event, 'content') and event.content:
                        if hasattr(event.content, 'parts') and event.content.parts:
                            for part in event.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    response_texts.append(part.text)
                    elif hasattr(event, 'output') and event.output:
                        response_texts.append(str(event.output))
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    new_key, rot_msg = key_manager.rotate_to_next_key()
                    response_texts.append(rot_msg)
                    if attempt < max_retries:
                        await asyncio.sleep(2)
                        continue
                    else:
                        schema = get_dataset_schema()
                        ml_res = execute_ml_model()
                        fallback = (
                            "[Quota Alert] Google Gemini Free-Tier Quota Limit Reached (429 Rate Limit).\n\n"
                            "All configured API key quotas reached. Executing deterministic fallbacks:\n\n"
                            f"1. Dataset Schema Inspection:\n{schema[:300]}...\n\n"
                            f"2. Machine Learning Predictor:\n{ml_res}\n\n"
                            "Tip: Add secondary Gemini API keys in the sidebar control panel for automatic failover!"
                        )
                        return fallback, None
                else:
                    return f"[System Execution Error] {err_str}", None

        final_text = "\n\n".join(response_texts) if response_texts else "Analysis complete."
        
        new_chart = None
        if os.path.exists(chart_path):
            mtime = os.path.getmtime(chart_path)
            if mtime >= start_time - 1:
                new_chart = chart_path

        return final_text, new_chart

    def run_query(self, query: str) -> Tuple[str, str]:
        """Synchronous wrapper for run_query_async."""
        return asyncio.run(self.run_query_async(query))

if __name__ == "__main__":
    print("=== Google ADK Chief Data Officer Coordinator ===")
    print("Type 'exit' or 'quit' to shut down server.\n")
    
    pipeline = ADKOrchestratorPipeline()
    
    while True:
        try:
            user_query = input("\nBusiness User: ")
            if user_query.lower() in ['exit', 'quit']:
                print("Shutting down Coordinator. Goodbye!")
                break
                
            print("\nCoordinator Processing with Google ADK...")
            res_text, chart = pipeline.run_query(user_query)
            print(f"\nChief Data Officer Output:\n{res_text}")
            if chart:
                print(f"[Generated Chart Saved]: {chart}")
        except Exception as e:
            print(f"\n[System Recovery] Encountered error: {str(e)}")
            print("Chat session recovered. Server kept alive.\n")
