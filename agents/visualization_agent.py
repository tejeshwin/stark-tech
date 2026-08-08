import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.adk.agents import LlmAgent

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.config import get_dataset_path, BASE_DIR
except ImportError:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        
        plt.close('all')
        sns.set_theme(style="whitegrid")
        
        local_vars = {'df': df, 'plt': plt, 'sns': sns, 'pd': pd}
        exec(python_code, {}, local_vars)
        
        if not os.path.exists(chart_path):
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        
        return f"[Visualization Engine Output] Chart successfully generated and saved to {chart_path}!"
    except Exception as e:
        plt.close('all')
        return f"[Visualization Engine Output] Plotting failed. Error details: {str(e)}"

visualization_agent = LlmAgent(
    name="VisualizationAgent",
    model="gemini-3.5-flash",
    description="Expert Data Visualization Artist for Executive Dashboards",
    instruction="""You are the Visualization Agent (Data Artist). Write Python code using matplotlib and seaborn to create publication-quality charts. Always prefix your responses with '[Visualization Engine Output]'.

MANDATORY DESIGN RULES:
1. USE VIBRANT COLOR PALETTES: Use vibrant, multi-colored palettes like 'viridis', 'rocket', 'mako', 'crest', or 'Set2'.
2. ADD DATA LABELS ON DATA POINTS: Always call `ax.bar_label()` or `ax.annotate()`.
3. EXPLICIT TITLES & AXIS LABELS: Bold 14pt title, clear x/y labels including units.
4. EXECUTIVE SUMMARY FOOTNOTE: Add takeaway text box at bottom.
5. ALWAYS CALL `plt.tight_layout()` and `plt.savefig('dashboard_chart.png', dpi=300, bbox_inches='tight')`.""",
    tools=[generate_plot]
)