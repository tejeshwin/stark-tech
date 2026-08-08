import os
import sys
import re
import time
import asyncio
import pandas as pd
from typing import Generator, Dict, Any, Tuple
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.config import get_dataset_path, load_dataset, get_gemini_api_key, key_manager, BASE_DIR
except ImportError:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
from analytics_agent import analytics_agent, execute_pandas_analysis
from visualization_agent import visualization_agent, generate_plot
from consultant_agent import consultant_agent, generate_executive_strategy
from semantic_agent import semantic_agent, semantic_validation

# Chief Data Officer Orchestrator Agent with Strict Dynamic Routing (gemini-3.5-flash)
orchestrator_agent = LlmAgent(
    name="ChiefDataOfficer",
    model="gemini-3.5-flash",
    description="Primary manager, Chief Data Officer, and multi-agent business analyst.",
    instruction="""You are the Chief Data Officer and Lead AI Business Analyst. You direct an AI data team using strict intent-based conditional routing:

RULE 1 (OUT OF SCOPE): IF the query is out-of-scope (e.g. travel, weather, personal tasks, trivia), route exclusively to 'semantic_agent' for immediate rejection. Do NOT invoke data or prediction tools.
RULE 2 (VISUALIZATION): IF the query requests charts, graphs, bar charts, or visual trends, route execution to 'visualization_agent' or call 'generate_plot'.
RULE 3 (ANALYTICS & KPIS): IF the query requests calculations, numeric aggregations, or KPIs, route execution to 'analytics_agent' or call 'execute_pandas_analysis'.
RULE 4 (PREDICTION & ML): IF the query requests forecasts, risk scoring, or ML predictions, route execution to 'prediction_agent' or call 'execute_ml_model'.
RULE 5 (EXECUTIVE SUMMARY): IF the query requests executive overviews or strategy recommendations, route execution to 'consultant_agent' or call 'generate_executive_strategy'.

Provide dynamic, context-aware responses tailored strictly to the user's input intent without static templates.""",
    tools=[semantic_validation, get_dataset_schema, execute_ml_model, execute_pandas_analysis, generate_plot, generate_executive_strategy],
    sub_agents=[semantic_agent, analytics_agent, visualization_agent, consultant_agent]
)

def generate_dynamic_fallback(query: str) -> Tuple[str, str]:
    """
    Decoupled dynamic engine. Dynamically analyzes the user query against the enterprise dataset
    without using static templates or hardcoded numbers.
    """
    try:
        query_lower = query.lower().strip()
        
        # 1. Out-of-Scope Guardrail Check
        val_res = semantic_validation(query)
        if isinstance(val_res, dict) and not val_res.get("is_valid", True):
            return val_res.get(
                "rejection_message",
                "[Semantic Validation Guardrail] Request Denied: Out of Scope. This enterprise decision support system is restricted strictly to organizational data analysis, operational metrics, and predictive forecasting."
            ), None

        # 2. Prediction / ML Intent Check
        if any(w in query_lower for w in ['predict', 'forecast', 'ml', 'machine learning', 'human review', 'model', 'risk score']):
            ml_res = execute_ml_model()
            return f"[Predictive Intelligence Output]\n{ml_res}", None

        # 3. Visualization Intent Check
        if any(w in query_lower for w in ['chart', 'plot', 'graph', 'visual', 'trend', 'distribution', 'bar', 'histogram']):
            plot_res = generate_plot(query)
            out_p = os.path.join(BASE_DIR, "output_chart.png")
            dash_p = os.path.join(BASE_DIR, "dashboard_chart.png")
            chart_file = out_p if os.path.exists(out_p) else (dash_p if os.path.exists(dash_p) else None)
            return plot_res, chart_file

        # 4. Executive Summary / Strategy Intent Check
        if any(w in query_lower for w in ['summary', 'overview', 'strategy', 'recommend', 'insight', 'consultant']):
            return generate_executive_strategy(query), None

        # 5. Dynamic Analytics Intent
        return execute_pandas_analysis(query), None

    except Exception as e:
        return f"[Chief Data Officer System Output] Error evaluating query: {str(e)}", None

class ADKOrchestratorPipeline:
    """
    High-Performance Google ADK Runner Pipeline enforcing dynamic intent-based routing.
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

    async def run_query_async(self, query: str, workflow_mode: str = "Orchestrator") -> Tuple[str, str]:
        """
        Executes query with mode-based agent isolation, dynamic dataset analysis, and semantic guardrails.
        """
        mode_lower = str(workflow_mode).lower().strip()
        
        # 1. Direct Semantic Validation Agent Mode
        if "semantic" in mode_lower:
            val_res = semantic_validation(query)
            if isinstance(val_res, dict):
                if not val_res.get("is_valid", True):
                    return val_res.get("rejection_message", "[Semantic Validation Guardrail] Request Denied: Out of Scope."), None
                return f"[Semantic Validation Guardrail] {val_res.get('validation_log', 'Query validated within corporate scope.')}", None

        # 2. Universal Out-of-Scope Guardrail Check
        val_res = semantic_validation(query)
        if isinstance(val_res, dict) and not val_res.get("is_valid", True):
            rejection_msg = val_res.get(
                "rejection_message",
                "[Semantic Validation Guardrail] Request Denied: Out of Scope. This enterprise decision support system is restricted strictly to organizational data analysis, operational metrics, and predictive forecasting."
            )
            return rejection_msg, None

        # 3. Direct Deep Analytics Mode
        if "eda" in mode_lower or "analytics" in mode_lower:
            df_ans = execute_pandas_analysis(query)
            return df_ans, None

        # 4. Direct Visualization Mode
        if "visualization" in mode_lower or "viz" in mode_lower or any(w in query.lower() for w in ['chart', 'plot', 'graph', 'visual', 'bar']):
            plot_res = generate_plot(query)
            out_p = os.path.join(BASE_DIR, "output_chart.png")
            dash_p = os.path.join(BASE_DIR, "dashboard_chart.png")
            chart_file = out_p if os.path.exists(out_p) else (dash_p if os.path.exists(dash_p) else None)
            return plot_res, chart_file

        # 5. Direct Predictive Risk Mode
        if "predictive" in mode_lower or "risk" in mode_lower or "prediction" in mode_lower:
            ml_res = execute_ml_model()
            return f"[Predictive Intelligence Output]\n{ml_res}", None

        # 6. Direct Executive Summary / Consultant Mode
        if "executive summary" in mode_lower or "consultant" in mode_lower or "summary" in mode_lower:
            summary = generate_executive_strategy(query)
            return summary, None

        # 7. Default Multi-Agent Orchestrator Mode (Google ADK Runner)
        api_key = key_manager.get_active_key()
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            os.environ["GOOGLE_API_KEY"] = api_key

        out_p = os.path.join(BASE_DIR, "output_chart.png")
        dash_p = os.path.join(BASE_DIR, "dashboard_chart.png")

        start_time = time.time()
        quota_warning = key_manager.record_request_and_check_warning()
        
        content = types.Content(parts=[types.Part.from_text(text=query)])
        response_texts = []
        if quota_warning:
            response_texts.append(quota_warning)

        max_attempts = len(getattr(key_manager, 'keys', [1]))
        attempt = 0
        success = False
        
        while attempt < max_attempts and not success:
            attempt += 1
            try:
                async with asyncio.timeout(10):
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
                    success = True
            except (asyncio.TimeoutError, Exception) as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower() or isinstance(e, asyncio.TimeoutError):
                    new_key, rot_msg = key_manager.rotate_to_next_key()
                    if attempt < max_attempts:
                        continue

        if not success or not response_texts:
            final_text, new_chart = generate_dynamic_fallback(query)
            return final_text, new_chart

        final_text = "\n\n".join(response_texts)
        new_chart = out_p if os.path.exists(out_p) and os.path.getmtime(out_p) >= start_time - 1 else (
            dash_p if os.path.exists(dash_p) and os.path.getmtime(dash_p) >= start_time - 1 else None
        )

        return final_text, new_chart

    def run_query(self, query: str, workflow_mode: str = "Orchestrator") -> Tuple[str, str]:
        """Synchronous wrapper for run_query_async accepting workflow_mode."""
        return asyncio.run(self.run_query_async(query, workflow_mode=workflow_mode))

if __name__ == "__main__":
    print("=== Fast ADK AI Data Team Booting Up ===")
    pipeline = ADKOrchestratorPipeline()
    while True:
        try:
            user_query = input("\nBusiness User: ")
            if user_query.lower() in ['exit', 'quit']:
                break
            res_text, chart = pipeline.run_query(user_query)
            print(f"\nChief Data Officer Output:\n{res_text}")
            if chart:
                print(f"[Generated Chart Saved]: {chart}")
        except Exception as e:
            print(f"\nSystem error: {str(e)}")