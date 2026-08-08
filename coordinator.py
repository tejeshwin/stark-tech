import os
import sys
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

# Chief Data Officer Orchestrator Agent with Dynamic Fallback & Safety Guardrails (gemini-3.5-flash)
orchestrator_agent = LlmAgent(
    name="ChiefDataOfficer",
    model="gemini-3.5-flash",
    description="Primary manager, Chief Data Officer, and multi-agent business analyst.",
    instruction="""You are the Chief Data Officer and Lead AI Business Analyst. You enforce strict operational scope guardrails:
Step 1: ALWAYS route the user query to the `semantic_agent` (or `semantic_validation`) first to validate whether the question is within corporate data scope.
Step 2: If the semantic agent flags the query as out-of-scope or unauthorized, immediately halt the pipeline and return its rejection response: '[Semantic Validation Guardrail] Request Denied: Out of Scope. This enterprise decision support system is restricted strictly to organizational data analysis, operational metrics, and predictive forecasting.' Do not call data or prediction tools for invalid queries.
Step 3: If a user query is unusual, complex, or does not match standard keywords, do NOT default to a generic or repetitive fallback message. Instead, invoke the `semantic_agent` to evaluate intent, or return a dynamic explanation stating why the request cannot be fulfilled based on current enterprise schema boundaries.
Step 4: For valid corporate data queries, use 'get_dataset_schema', 'execute_pandas_analysis', 'execute_ml_model', or 'generate_plot' to deliver quantitative, executive conclusions.""",
    tools=[semantic_validation, get_dataset_schema, execute_ml_model, execute_pandas_analysis, generate_plot],
    sub_agents=[semantic_agent, analytics_agent, visualization_agent, consultant_agent]
)

def generate_dynamic_fallback(query: str) -> str:
    """
    Generates a dynamic, context-aware response based on the user's specific query
    and the actual enterprise dataset schema when LLM quota or runtime limits are reached.
    """
    try:
        df = load_dataset()
        query_lower = query.lower()
        cols = df.columns.tolist()
        
        matching_cols = [c for c in cols if any(word in c.lower() for word in query_lower.split())]
        
        response_parts = [f"**[Chief Data Officer Dynamic Analysis]** for query: *\"{query}\"*"]
        
        if matching_cols:
            response_parts.append(f"• Identified relevant dataset fields: `{', '.join(matching_cols)}`")
            for col in matching_cols[:2]:
                if pd.api.types.is_numeric_dtype(df[col]):
                    avg_val = df[col].mean()
                    sum_val = df[col].sum()
                    response_parts.append(f"  - **{col}**: Average = `{avg_val:,.2f}`, Total = `{sum_val:,.2f}`")
                elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    top_vals = df[col].value_counts().head(3).to_dict()
                    val_str = ", ".join([f"{k}: {v:,}" for k, v in top_vals.items()])
                    response_parts.append(f"  - **{col} Top Distribution**: {val_str}")
        else:
            avg_amount = df['Transaction_Amount_USD'].mean() if 'Transaction_Amount_USD' in df.columns else df['Revenue_Impact_USD'].mean()
            records_count = len(df)
            response_parts.append(
                f"• The dataset contains **{records_count:,} operational records** across **{len(cols)} dimensions**.\n"
                f"• Key Baseline Metric (Average Transaction Amount): **${avg_amount:,.2f} USD**.\n"
                f"• Query attributes were evaluated against enterprise schema dimensions: `{', '.join(cols[:8])}...`"
            )
            
        ml_res = execute_ml_model()
        response_parts.append(f"\n**Predictive Intelligence Output**:\n{ml_res}")
        return "\n".join(response_parts)
    except Exception as e:
        return f"[Chief Data Officer System Fallback] Unable to parse query metrics due to error: {str(e)}"

class ADKOrchestratorPipeline:
    """
    High-Performance Google ADK Runner Pipeline enforcing strict semantic validation & dynamic fallback trapping.
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
        Executes query with mandatory semantic validation guardrail check BEFORE analytical execution.
        Traps errors dynamically to ensure no repetitive or generic fallback strings occur.
        """
        # Step 1: Mandatory Guardrail Check BEFORE any data analysis or tool execution
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

        # Dynamic Error Trapping: If LLM calls timed out or hit API quotas, return dynamic context-aware response!
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
