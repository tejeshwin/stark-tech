import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.adk.agents import LlmAgent

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from utils.config import get_dataset_path, BASE_DIR
except ImportError:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    def get_dataset_path():
        candidates = ["data/cleaned_enterprise_data_final.csv", "cleaned_enterprise_data.csv", "data/cleaned_enterprise_data.csv"]
        for c in candidates:
            if os.path.exists(c): return c
        return "cleaned_enterprise_data.csv"

def generate_plot(python_code: str) -> str:
    """
    Executes Python code to draw vibrant, detailed charts. 'df', 'plt', and 'sns' are already imported.
    Save the chart using plt.savefig('dashboard_chart.png', dpi=300, bbox_inches='tight').
    """
    try:
        df = pd.read_csv(get_dataset_path())
        chart_path = os.path.join(BASE_DIR, 'dashboard_chart.png')
        
        # Reset and prepare figure
        plt.close('all')
        sns.set_theme(style="whitegrid")
        
        local_vars = {'df': df, 'plt': plt, 'sns': sns, 'pd': pd}
        exec(python_code, {}, local_vars)
        
        if not os.path.exists(chart_path):
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        
        return f"Chart successfully generated and saved to {chart_path}!"
    except Exception as e:
        plt.close('all')
        return f"Plotting failed. Error details: {str(e)}"

visualization_agent = LlmAgent(
    name="VisualizationAgent",
    model="gemini-3.5-flash",
    description="Expert Data Visualization Artist for Executive Dashboards",
    instruction="""You are the Visualization Agent (Data Artist). Write Python code using matplotlib and seaborn to create publication-quality charts.

MANDATORY DESIGN RULES:
1. USE VIBRANT COLOR PALETTES: Use vibrant, multi-colored palettes like 'viridis', 'rocket', 'mako', 'crest', or 'Set2'. Never use single plain blue or gray bars!
2. ADD DATA LABELS ON DATA POINTS: Always call `ax.bar_label(ax.containers[0], fmt='$%','.2f')` or `ax.annotate()` to show exact values on top of bars/lines.
3. EXPLICIT TITLES & AXIS LABELS:
   - Chart Title: Bold, 14pt title (e.g. `ax.set_title("Department Revenue Impact", fontsize=14, fontweight='bold', pad=12)`).
   - X-Axis Label: Clear label with 45° rotated ticks if labels are long.
   - Y-Axis Label: Clear label including units (e.g., "Revenue Impact (USD $)", "Processing Time (Seconds)").
4. EXECUTIVE SUMMARY FOOTNOTE: Add a text box or footnote at the bottom explaining key takeaway (e.g. `fig.text(0.5, -0.05, "Key Takeaway: ...", ha='center', fontsize=10, style='italic')`).
5. ALWAYS CALL `plt.tight_layout()` and `plt.savefig('dashboard_chart.png', dpi=300, bbox_inches='tight')`.""",
    tools=[generate_plot]
)
