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

def generate_plot(python_code: str = "") -> str:
    """
    Executes Python matplotlib/seaborn plotting code on cleaned_enterprise_data.csv.
    Saves the physical output image as output_chart.png and dashboard_chart.png.
    """
    try:
        data_path = get_dataset_path()
        df = pd.read_csv(data_path)
        
        output_path = os.path.join(BASE_DIR, 'output_chart.png')
        dash_path = os.path.join(BASE_DIR, 'dashboard_chart.png')
        
        plt.close('all')
        sns.set_theme(style="whitegrid")
        
        if python_code and len(python_code.strip()) > 10:
            local_vars = {'df': df, 'plt': plt, 'sns': sns, 'pd': pd}
            exec(python_code, {}, local_vars)
        else:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            if 'Department' in df.columns and 'Revenue_Impact_USD' in df.columns:
                dept_rev = df.groupby('Department')['Revenue_Impact_USD'].sum().reset_index().sort_values('Revenue_Impact_USD', ascending=False)
                bars = sns.barplot(data=dept_rev, x='Department', y='Revenue_Impact_USD', hue='Department', palette='viridis', legend=False, ax=ax)
                ax.set_title("Department Total Revenue Impact ($ USD)", fontsize=13, fontweight='bold', pad=12, color='#1e3a8a')
                ax.set_ylabel("Revenue Impact ($ USD)", fontsize=10, fontweight='bold')
                for p in bars.patches:
                    h = p.get_height()
                    if h > 0:
                        ax.annotate(f'${h:,.0f}', (p.get_x() + p.get_width()/2., h), ha='center', va='bottom', fontsize=8.5, fontweight='bold')
            elif 'Priority_Level' in df.columns and 'SLA_Breached' in df.columns:
                sns.countplot(data=df, x='Priority_Level', hue='SLA_Breached', palette=['#1E3A8A', '#E11D48'], ax=ax)
                ax.set_title("Operational SLA Compliance by Priority Level", fontsize=13, fontweight='bold', pad=12)
            plt.xticks(rotation=25, ha='right')
            plt.tight_layout()
            
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(dash_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        
        return f"[Visualization Engine Output] Executive chart successfully generated, saved to {output_path} and rendered on dashboard."
    except Exception as e:
        plt.close('all')
        return f"[Visualization Engine Output] Plotting execution error: {str(e)}"

visualization_agent = LlmAgent(
    name="VisualizationAgent",
    model="gemini-3.5-flash",
    description="Expert Data Visualization Artist for Executive Dashboards",
    instruction="""You are the Visualization Agent (Data Artist). Your job is to execute Python plotting code using matplotlib and seaborn to render enterprise charts.

MANDATORY EXECUTION RULES:
1. Do NOT just write markdown or text descriptions. You MUST call the `generate_plot` tool to execute Python code.
2. Ensure the plot reads `cleaned_enterprise_data.csv` (pre-loaded as 'df').
3. Create the requested plot (bar chart, line chart, scatterplot, pie chart).
4. Save the plot using `plt.savefig("output_chart.png", dpi=300, bbox_inches='tight')` and `plt.close('all')`.
5. Use vibrant palettes ('viridis', 'rocket', 'mako', 'Set2'), add explicit titles, data labels, and axis titles. Always prefix responses with '[Visualization Engine Output]'.""",
    tools=[generate_plot]
)