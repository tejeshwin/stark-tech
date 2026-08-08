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

def generate_plot(query_or_code: str = "") -> str:
    """
    Executes Python matplotlib/seaborn plotting code tailored dynamically to the requested dataset columns and query intent.
    Saves physical high-resolution chart images as output_chart.png and dashboard_chart.png.
    """
    try:
        data_path = get_dataset_path()
        df = pd.read_csv(data_path)
        input_str = str(query_or_code).strip()
        query_lower = input_str.lower()
        
        output_path = os.path.join(BASE_DIR, 'output_chart.png')
        dash_path = os.path.join(BASE_DIR, 'dashboard_chart.png')
        
        plt.close('all')
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 4.5))

        # Check if direct Python code snippet was passed
        if any(kw in input_str for kw in ['plt.', 'sns.', 'ax.', 'fig,']):
            local_vars = {'df': df, 'plt': plt, 'sns': sns, 'pd': pd, 'ax': ax, 'fig': fig}
            exec(input_str, {}, local_vars)
        else:
            # Dynamic Intent-to-Chart Plotting Engine
            if any(w in query_lower for w in ['sla', 'breach', 'priority', 'level', 'ticket']):
                if 'Priority_Level' in df.columns and 'SLA_Breached' in df.columns:
                    prio_data = df.groupby(['Priority_Level', 'SLA_Breached']).size().reset_index(name='Count')
                    bars = sns.barplot(data=prio_data, x='Priority_Level', y='Count', hue='SLA_Breached', palette=['#1E3A8A', '#E11D48'], ax=ax)
                    ax.set_title("Operational SLA Compliance by Priority Level", fontsize=13, fontweight='bold', pad=12, color='#1e3a8a')
                    ax.set_xlabel("Priority Level", fontsize=10, fontweight='bold')
                    ax.set_ylabel("Ticket Volume", fontsize=10, fontweight='bold')

            elif any(w in query_lower for w in ['department', 'revenue', 'transaction', 'amount', 'usd', 'financial']):
                amt_col = 'Transaction_Amount_USD' if 'Transaction_Amount_USD' in df.columns else ('Revenue_Impact_USD' if 'Revenue_Impact_USD' in df.columns else None)
                if amt_col and 'Department' in df.columns:
                    dept_rev = df.groupby('Department')[amt_col].sum().reset_index().sort_values(amt_col, ascending=False)
                    bars = sns.barplot(data=dept_rev, x='Department', y=amt_col, hue='Department', palette='viridis', legend=False, ax=ax)
                    ax.set_title(f"Total {amt_col.replace('_', ' ')} by Department", fontsize=13, fontweight='bold', pad=12, color='#1e3a8a')
                    ax.set_ylabel("Total Amount ($ USD)", fontsize=10, fontweight='bold')
                    for p in bars.patches:
                        h = p.get_height()
                        if h > 0:
                            ax.annotate(f'${h:,.0f}', (p.get_x() + p.get_width()/2., h), ha='center', va='bottom', fontsize=8.5, fontweight='bold')

            elif any(w in query_lower for w in ['processing', 'duration', 'time', 'seconds', 'complexity', 'scatter']):
                if 'Query_Complexity_Score' in df.columns and 'Processing_Time_Sec' in df.columns:
                    sample = df.sample(min(1500, len(df)))
                    sns.scatterplot(data=sample, x='Query_Complexity_Score', y='Processing_Time_Sec', hue='SLA_Breached' if 'SLA_Breached' in df.columns else None, palette=['#0D9488', '#E11D48'], alpha=0.7, s=35, ax=ax)
                    ax.set_title("Processing Duration vs Query Complexity Score", fontsize=13, fontweight='bold', pad=12, color='#1e3a8a')
                    ax.set_xlabel("Query Complexity Score (1-10)", fontsize=10, fontweight='bold')
                    ax.set_ylabel("Processing Time (Seconds)", fontsize=10, fontweight='bold')

            elif any(w in query_lower for w in ['human', 'review', 'automation', 'risk', 'manual']):
                if 'Requires_Human_Review' in df.columns:
                    rev_df = df['Requires_Human_Review'].value_counts().reset_index()
                    rev_df.columns = ['Status', 'Count']
                    rev_df['Status'] = rev_df['Status'].map({True: 'Requires Human Review', False: 'Automated Resolution', 1: 'Requires Human Review', 0: 'Automated Resolution'})
                    bars = sns.barplot(data=rev_df, x='Status', y='Count', hue='Status', palette=['#10B981', '#F59E0B'], legend=False, ax=ax)
                    ax.set_title("Query Resolution Mode Distribution", fontsize=13, fontweight='bold', pad=12, color='#1e3a8a')

            else:
                amt_col = 'Transaction_Amount_USD' if 'Transaction_Amount_USD' in df.columns else df.columns[0]
                if 'Department' in df.columns:
                    dept_df = df.groupby('Department').size().reset_index(name='Query_Volume')
                    sns.barplot(data=dept_df, x='Department', y='Query_Volume', hue='Department', palette='mako', legend=False, ax=ax)
                    ax.set_title("Enterprise Operational Query Volume by Department", fontsize=13, fontweight='bold', pad=12, color='#1e3a8a')

            plt.xticks(rotation=20, ha='right')
            plt.tight_layout()

        # Save output image
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(dash_path, dpi=300, bbox_inches='tight')
        plt.close('all')

        return f"[Visualization Engine Output] Dynamic chart generated and saved to output_chart.png for query: *\"{input_str}\"*"
    except Exception as e:
        plt.close('all')
        return f"[Visualization Engine Output] Plot execution error: {str(e)}"

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