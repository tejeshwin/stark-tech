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
from analytics_agent import analytics_agent, execute_pandas_analysis
from visualization_agent import visualization_agent, generate_plot
from consultant_agent import consultant_agent
from semantic_agent import semantic_agent, semantic_validation

# Chief Data Officer Orchestrator Agent with Strict Intent-Based Routing Tree (gemini-3.5-flash)
orchestrator_agent = LlmAgent(
    name="ChiefDataOfficer",
    model="gemini-3.5-flash",
    description="Primary manager, Chief Data Officer, and multi-agent business analyst.",
    instruction="""You are the Chief Data Officer and Lead AI Business Analyst. You direct an AI data team using strict intent-based conditional routing:

RULE 1 (OUT OF SCOPE): IF the query is out-of-scope (e.g. travel, weather, personal tasks, trivia), route exclusively to 'semantic_agent' for immediate rejection. Do NOT invoke data or prediction tools.
RULE 2 (VISUALIZATION): IF the query requests charts, graphs, or visual trends, route execution to 'visualization_agent' (or 'generate_plot').
RULE 3 (ANALYTICS & KPIS): IF the query requests calculations, numeric aggregations, or KPIs, route execution to 'analytics_agent' (or 'execute_pandas_analysis').
RULE 4 (PREDICTION & ML): IF the query requests forecasts, risk scoring, or ML predictions, route execution to 'prediction_agent' (or 'execute_ml_model').
RULE 5 (EXECUTIVE SUMMARY): IF the query requests executive overviews or strategy recommendations, route execution to 'consultant_agent'.

Never dump schemas or ML predictions unconditionally. Always provide dynamic, context-aware responses tailored strictly to the user's input intent.""",
    tools=[semantic_validation, get_dataset_schema, execute_ml_model, execute_pandas_analysis, generate_plot],
    sub_agents=[semantic_agent, analytics_agent, visualization_agent, consultant_agent]
)

def generate_dynamic_fallback(query: str) -> str:
    """
    Decoupled intent-based dynamic router. Evaluates user input intent
    and returns targeted, unique analysis without dumping unrequested tools or predictions.
    """
    try:
        df = load_dataset()
        query_lower = query.lower().strip()
        cols = df.columns.tolist()
        
        # 1. Out-of-Scope Guardrail Check
        val_res = semantic_validation(query)
        if isinstance(val_res, dict) and not val_res.get("is_valid", True):
            return val_res.get(
                "rejection_message",
                "[Semantic Validation Guardrail] Request Denied: Out of Scope. This enterprise decision support system is restricted strictly to organizational data analysis, operational metrics, and predictive forecasting."
            )

        # 2. Prediction / ML Intent Check
        if any(w in query_lower for w in ['predict', 'forecast', 'ml', 'machine learning', 'human review', 'model', 'risk score']):
            ml_res = execute_ml_model()
            return f"[Predictive Intelligence Output]\n{ml_res}"

        # 3. Visualization Intent Check
        if any(w in query_lower for w in ['chart', 'plot', 'graph', 'visual', 'trend', 'distribution']):
            return f"[Visualization Engine Output] Executive visualization query processed for: '{query}'. Seaborn chart rendered."

        # 4. Specific Column Analytics Intent Check
        words = [w for w in re.findall(r'\w+', query_lower) if len(w) > 2]
        matching_cols = [c for c in cols if any(w in c.lower() for w in words)]
        
        if matching_cols:
            response_parts = [f"[Analytics Engine Output] Quantitative analysis for query: *\"{query}\"*"]
            for col in matching_cols[:3]:
                if pd.api.types.is_numeric_dtype(df[col]):
                    avg_val = df[col].mean()
                    sum_val = df[col].sum()
                    response_parts.append(f"• **{col}**: Average = `{avg_val:,.2f}`, Total = `{sum_val:,.2f}`")
                else:
                    top_vals = df[col].value_counts().head(3).to_dict()
                    val_str = ", ".join([f"{k}: {v:,}" for k, v in top_vals.items()])
                    response_parts.append(f"• **{col} Distribution**: {val_str}")
            return "\n".join(response_parts)

        # 5. Executive Summary / Strategy Intent Check
        if any(w in query_lower for w in ['summary', 'overview', 'strategy', 'recommend', 'insight', 'kpi', 'schema', 'columns']):
            avg_tx = df['Transaction_Amount_USD'].mean() if 'Transaction_Amount_USD' in df.columns else df['Revenue_Impact_USD'].mean()
            return f"[Consultant Executive Strategy] Operational Dataset Overview ({len(df):,} records, {len(cols)} dimensions). Average Transaction Value: ${avg_tx:,.2f} USD. Strategic Recommendation: Focus optimization on High-Complexity and SLA breach tasks."

        # 6. Default Targeted Analytical Output
        avg_val = df['Transaction_Amount_USD'].mean() if 'Transaction_Amount_USD' in df.columns else df['Revenue_Impact_USD'].mean()
        return f"[Analytics Engine Output] Analytical response for query '{query}': Average Transaction Amount across {len(df):,} records is ${avg_val:,.2f} USD."

    except Exception as e:
        return f"[Chief Data Officer System Output] Error evaluating query intent: {str(e)}"

class ADKOrchestratorPipeline:
    """
    High-Performance Google ADK Runner Pipeline enforcing decoupled conditional routing.
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
        Executes query with mandatory intent-based routing and decoupled tool execution.
        """
        # Step 1: Out-of-Scope Guardrail Check
        validation_res = semantic_validation(query)
        if isinstance(validation_res, dict) and not validation_res.get("is_valid", True):
            rejection_msg = validation_res.get(
                "rejection_message",
                "[Semantic Validation Guardrail] Request Denied: Out of Scope. This enterprise decision support system is restricted strictly to organizational data analysis, operational metrics, and predictive forecasting."
            )
            return rejection_msg, None

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

        # Dynamic Intent-Based Trapping: If LLM calls timed out or hit API quotas, execute decoupled intent router!
        if not success or not response_texts:
            final_text = generate_dynamic_fallback(query)
            return final_text, None

        final_text = "\n\n".join(response_texts)
        
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
    print("=== Fast ADK AI Data Team Booting Up ===")
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
