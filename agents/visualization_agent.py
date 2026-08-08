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
    Dynamically generates custom matplotlib/seaborn plots strictly based on the user's requested dataset columns.
    Extracts the exact columns specified in the input query and renders targeted visual charts.
    Saves high-resolution output images to output_chart.png and dashboard_chart.png.
    """
    try:
        data_path = get_dataset_path()
        df = pd.read_csv(data_path)
        input_str = str(query_or_code).strip()
        query_lower = input_str.lower()
        cols = df.columns.tolist()
        
        output_path = os.path.join(BASE_DIR, 'output_chart.png')
        dash_path = os.path.join(BASE_DIR, 'dashboard_chart.png')
        
        plt.close('all')
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8.5, 4.8))

        # 1. Direct Python code execution
        if any(kw in input_str for kw in ['plt.', 'sns.', 'ax.', 'fig,']):
            local_vars = {'df': df, 'plt': plt, 'sns': sns, 'pd': pd, 'ax': ax, 'fig': fig}
            exec(input_str, {}, local_vars)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.savefig(dash_path, dpi=300, bbox_inches='tight')
            plt.close('all')
            return f"[Visualization Engine Output] Executed custom Python plotting code. Saved chart to output_chart.png."

        # 2. Dynamic Column Extraction Engine: Find ALL dataset columns mentioned in user input
        matched_cols = []
        for col in cols:
            col_lower = col.lower()
            col_clean = col_lower.replace('_', ' ')
            col_words = [w for w in col_clean.split() if len(w) > 2]
            
            # Direct full or space/underscore match
            if col_lower in query_lower or col_clean in query_lower or col_lower.replace('_', '') in query_lower:
                if col not in matched_cols:
                    matched_cols.append(col)
            # Distinctive word match
            elif any(w in query_lower for w in col_words if w not in ['type', 'code', 'name', 'usd', 'sec', 'score', 'id', 'date', 'level']):
                if col not in matched_cols:
                    matched_cols.append(col)

        # 3. Dynamic Multi-Column Plotting Engine
        if len(matched_cols) >= 2:
            col1, col2 = matched_cols[0], matched_cols[1]
            
            col1_num = pd.api.types.is_numeric_dtype(df[col1])
            col2_num = pd.api.types.is_numeric_dtype(df[col2])

            if col1_num and col2_num:
                # 2 Numeric Columns -> Scatter Plot
                sample = df.sample(min(1500, len(df)))
                sns.scatterplot(data=sample, x=col1, y=col2, hue='SLA_Breached' if 'SLA_Breached' in df.columns else None, palette=['#0D9488', '#E11D48'], alpha=0.7, s=35, ax=ax)
                ax.set_title(f"{col2} vs {col1} Scatter Analysis", fontsize=13, fontweight='bold', pad=12, color='#1e3a8a')

            elif not col1_num and col2_num:
                # col1 Categorical, col2 Numeric -> Grouped Bar Plot
                top_cats = df[col1].value_counts().head(8).index
                sub_df = df[df[col1].isin(top_cats)]
                grp = sub_df.groupby(col1)[col2].mean().reset_index().sort_values(col2, ascending=False)
                bars = sns.barplot(data=grp, x=col1, y=col2, hue=col1, palette='viridis', legend=False, ax=ax)
                ax.set_title(f"Average {col2} by {col1}", fontsize=13, fontweight='bold', pad=12, color='#1e3a8a')
                for p in bars.patches:
                    h = p.get_height()
                    if h > 0:
                        ax.annotate(f'{h:,.1f}', (p.get_x() + p.get_width()/2., h), ha='center', va='bottom', fontsize=8.5, fontweight='bold')

            elif col1_num and not col2_num:
                # col1 Numeric, col2 Categorical -> Grouped Bar Plot
                top_cats = df[col2].value_counts().head(8).index
                sub_df = df[df[col2].isin(top_cats)]
                grp = sub_df.groupby(col2)[col1].mean().reset_index().sort_values(col1, ascending=False)
                bars = sns.barplot(data=grp, x=col2, y=col1, hue=col2, palette='mako', legend=False, ax=ax)
                ax.set_title(f"Average {col1} by {col2}", fontsize=13, fontweight='bold', pad=12, color='#1e3a8a')
                for p in bars.patches:
                    h = p.get_height()
                    if h > 0:
                        ax.annotate(f'{h:,.1f}', (p.get_x() + p.get_width()/2., h), ha='center', va='bottom', fontsize=8.5, fontweight='bold')

            else:
                # 2 Categorical Columns -> Stacked/Hue Countplot
                top_cats = df[col1].value_counts().head(6).index
                sub_df = df[df[col1].isin(top_cats)]
                sns.countplot(data=sub_df, x=col1, hue=col2, palette='rocket', ax=ax)
                ax.set_title(f"{col1} Breakdown by {col2}", fontsize=13, fontweight='bold', pad=12, color='#1e3a8a')

        elif len(matched_cols) == 1:
            col_target = matched_cols[0]
            if pd.api.types.is_numeric_dtype(df[col_target]):
                sns.histplot(df[col_target], kde=True, color='#1E3A8A', ax=ax)
                ax.set_title(f"Distribution Analysis of {col_target}", fontsize=13, fontweight='bold', pad=12, color='#1e3a8a')
                ax.set_xlabel(col_target, fontsize=10, fontweight='bold')
            else:
                top_cat = df[col_target].value_counts().head(8).reset_index()
                top_cat.columns = [col_target, 'Count']
                bars = sns.barplot(data=top_cat, x=col_target, y='Count', hue=col_target, palette='crest', legend=False, ax=ax)
                ax.set_title(f"Category Volume Breakdown: {col_target}", fontsize=13, fontweight='bold', pad=12, color='#1e3a8a')
                ax.set_xlabel(col_target, fontsize=10, fontweight='bold')
                for p in bars.patches:
                    h = p.get_height()
                    if h > 0:
                        ax.annotate(f'{int(h):,}', (p.get_x() + p.get_width()/2., h), ha='center', va='bottom', fontsize=8.5, fontweight='bold')

        else:
            # Fallback if no specific column matched: Use Department and Transaction_Amount_USD / Priority_Level dynamically
            if 'Department' in df.columns and 'Transaction_Amount_USD' in df.columns:
                dept_rev = df.groupby('Department')['Transaction_Amount_USD'].sum().reset_index().sort_values('Transaction_Amount_USD', ascending=False)
                bars = sns.barplot(data=dept_rev, x='Department', y='Transaction_Amount_USD', hue='Department', palette='viridis', legend=False, ax=ax)
                ax.set_title("Department Total Transaction Volume ($ USD)", fontsize=13, fontweight='bold', pad=12, color='#1e3a8a')
            elif 'Priority_Level' in df.columns:
                p_df = df['Priority_Level'].value_counts().reset_index()
                p_df.columns = ['Priority_Level', 'Count']
                sns.barplot(data=p_df, x='Priority_Level', y='Count', hue='Priority_Level', palette='rocket', legend=False, ax=ax)
                ax.set_title("Operational Ticket Volume by Priority Level", fontsize=13, fontweight='bold', pad=12, color='#1e3a8a')

        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()

        # Save physical output image
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(dash_path, dpi=300, bbox_inches='tight')
        plt.close('all')

        plot_description = f"Columns matched: {', '.join(matched_cols)}" if matched_cols else "Operational dimension view"
        return f"[Visualization Engine Output] Executive chart generated for: *\"{input_str}\"* ({plot_description}) and rendered below."
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