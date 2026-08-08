import os
import sys
from google.adk.agents import LlmAgent

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analytics_agent import execute_pandas_analysis

def generate_executive_strategy(query: str) -> str:
    """
    Generates tailored, C-suite executive strategic recommendations based on actual dataset calculations.
    """
    analytics_output = execute_pandas_analysis(query)
    clean_analytics = analytics_output.replace("[Analytics Engine Output] ", "")
    
    strategy_report = (
        f"[Consultant Executive Strategy] Executive Strategy & Strategic Assessment for: *\"{query}\"*\n\n"
        f"### 📊 Quantitative Baseline Findings\n{clean_analytics}\n\n"
        f"### 💡 C-Suite Strategic Action Plan\n"
        f"1. **Operational Resource Alignment**: Direct engineering capacity toward the top bottleneck departments and SLA breach hotspots highlighted in the quantitative baseline.\n"
        f"2. **Real-Time Automated Escalation**: Implement SLA risk monitoring triggers for High/Critical tickets to prevent breaches before they hit compliance thresholds.\n"
        f"3. **Predictive AI Governance**: Integrate the Random Forest ML classifier to auto-triage low-complexity tickets, freeing human agents for complex escalations."
    )
    return strategy_report

consultant_agent = LlmAgent(
    name="ConsultantAgent",
    model="gemini-3.5-flash",
    description="Business Strategist & Executive Translator",
    instruction="""You are the Consultant Agent. Your job is to take raw mathematical and analytical outputs
from the Analytics Agent and ML Predictor Tool and translate them into highly professional, executive-ready
business English for C-suite leaders. Always prefix your response with '[Consultant Executive Strategy]', cite specific numbers to substantiate claims, and offer 3 strategic recommendations.""",
    tools=[generate_executive_strategy]
)