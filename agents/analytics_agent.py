import os
import sys
import re
import pandas as pd
from google.adk.agents import LlmAgent

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.config import get_dataset_path
except ImportError:
    def get_dataset_path():
        candidates = ["data/cleaned_enterprise_data_final.csv", "cleaned_enterprise_data.csv", "data/cleaned_enterprise_data.csv"]
        for c in candidates:
            if os.path.exists(c): return c
        return "cleaned_enterprise_data.csv"

def execute_pandas_analysis(query_or_code: str) -> str:
    """
    Dynamically analyzes the enterprise CSV dataset ('df') using pandas.
    Handles both direct Python pandas code snippets and natural language business queries.
    """
    try:
        df = pd.read_csv(get_dataset_path())
        input_str = str(query_or_code).strip()
        
        # 1. If raw Python code containing pandas syntax is passed
        if any(kw in input_str for kw in ['df[', 'groupby(', 'mean()', 'sum()', 'value_counts()', 'analysis_result =']):
            local_vars = {'df': df, 'pd': pd, 'analysis_result': None}
            exec(input_str, {}, local_vars)
            result = local_vars.get('analysis_result')
            if result is not None:
                return f"[Analytics Engine Output] {str(result)}"

        # 2. Dynamic Query Intent Processing Engine
        query_lower = input_str.lower()
        cols = df.columns.tolist()
        results_lines = [f"[Analytics Engine Output] Quantitative analysis for: *\"{input_str}\"*"]

        # Intent A: SLA / Breaches
        if any(w in query_lower for w in ['sla', 'breach', 'breached', 'compliance', 'ticket']):
            if 'SLA_Breached' in df.columns:
                total_records = len(df)
                total_breaches = int(df['SLA_Breached'].sum()) if df['SLA_Breached'].dtype in ['int64', 'float64', 'bool'] else int((df['SLA_Breached'] == True).sum())
                breach_pct = (total_breaches / total_records) * 100
                results_lines.append(f"• **Overall SLA Breach Rate**: `{breach_pct:.2f}%` ({total_breaches:,} breaches out of {total_records:,} queries)")

                if 'Priority_Level' in df.columns:
                    results_lines.append("• **SLA Breaches by Priority Level**:")
                    prio_gb = df.groupby('Priority_Level')['SLA_Breached'].agg(['count', 'sum'])
                    for prio, row in prio_gb.iterrows():
                        pct = (row['sum'] / row['count']) * 100
                        results_lines.append(f"  - **{prio}**: `{pct:.1f}%` breach rate ({int(row['sum']):,} / {int(row['count']):,})")

                if 'Department' in df.columns:
                    results_lines.append("• **SLA Breaches by Department**:")
                    dept_gb = df.groupby('Department')['SLA_Breached'].agg(['count', 'sum'])
                    for dept, row in dept_gb.head(4).iterrows():
                        pct = (row['sum'] / row['count']) * 100
                        results_lines.append(f"  - **{dept}**: `{pct:.1f}%` breach rate ({int(row['sum']):,} / {int(row['count']):,})")
                return "\n".join(results_lines)

        # Intent B: Department / Revenue / Transaction Amount
        if any(w in query_lower for w in ['department', 'revenue', 'transaction', 'amount', 'usd', 'financial', 'cost']):
            amt_col = 'Transaction_Amount_USD' if 'Transaction_Amount_USD' in df.columns else ('Revenue_Impact_USD' if 'Revenue_Impact_USD' in df.columns else None)
            if amt_col:
                total_val = df[amt_col].sum()
                avg_val = df[amt_col].mean()
                max_val = df[amt_col].max()
                min_val = df[amt_col].min()
                results_lines.append(f"• **{amt_col} Overview**: Total = `${total_val:,.2f}`, Average = `${avg_val:,.2f}`, Range = `${min_val:,.2f}` - `${max_val:,.2f}`")

                if 'Department' in df.columns:
                    results_lines.append("• **Department Financial Breakdown**:")
                    dept_amt = df.groupby('Department')[amt_col].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)
                    for dept, row in dept_amt.iterrows():
                        results_lines.append(f"  - **{dept}**: Total `${row['sum']:,.2f}` | Avg `${row['mean']:,.2f}` | Volume: {int(row['count']):,}")
                return "\n".join(results_lines)

        # Intent C: Processing Duration / Complexity
        if any(w in query_lower for w in ['processing', 'duration', 'time', 'seconds', 'complexity', 'speed']):
            if 'Processing_Time_Sec' in df.columns:
                avg_sec = df['Processing_Time_Sec'].mean()
                med_sec = df['Processing_Time_Sec'].median()
                max_sec = df['Processing_Time_Sec'].max()
                results_lines.append(f"• **Processing Duration Metrics**: Mean = `{avg_sec:.2f}s`, Median = `{med_sec:.2f}s`, Max = `{max_sec:.2f}s`")

                if 'Query_Complexity_Score' in df.columns:
                    results_lines.append("• **Duration by Complexity Tier**:")
                    comp_gb = df.groupby('Query_Complexity_Score')['Processing_Time_Sec'].mean().head(5)
                    for score, sec in comp_gb.items():
                        results_lines.append(f"  - **Complexity Score {score}**: Avg `{sec:.2f}s` duration")
                return "\n".join(results_lines)

        # Intent D: Human Review / Automation Rate
        if any(w in query_lower for w in ['human', 'review', 'automation', 'risk', 'manual', 'flag']):
            if 'Requires_Human_Review' in df.columns:
                rev_counts = df['Requires_Human_Review'].value_counts()
                true_cnt = rev_counts.get(True, rev_counts.get(1, 0))
                pct = (true_cnt / len(df)) * 100
                results_lines.append(f"• **Human Review Requirement Rate**: `{pct:.2f}%` ({true_cnt:,} queries flagged out of {len(df):,})")
                return "\n".join(results_lines)

        # Fallback Column Inspection Engine
        words = [w for w in re.findall(r'\w+', query_lower) if len(w) > 2]
        matching_cols = [c for c in cols if any(w in c.lower() for w in words)]

        if matching_cols:
            for col in matching_cols[:3]:
                if pd.api.types.is_numeric_dtype(df[col]):
                    results_lines.append(f"• **{col} Statistics**: Sum = `{df[col].sum():,.2f}`, Mean = `{df[col].mean():,.2f}`, Std = `{df[col].std():,.2f}`")
                else:
                    top_vals = df[col].value_counts().head(4).to_dict()
                    val_str = ", ".join([f"{k}: {v:,}" for k, v in top_vals.items()])
                    results_lines.append(f"• **{col} Top Categories**: {val_str}")
            return "\n".join(results_lines)

        # General Operational Summary
        avg_amt = df['Transaction_Amount_USD'].mean() if 'Transaction_Amount_USD' in df.columns else 0.0
        return (
            f"[Analytics Engine Output] Query: *\"{input_str}\"*\n"
            f"• Total Records Evaluated: `{len(df):,}`\n"
            f"• Dimensions Available: `{len(cols)}` ({', '.join(cols[:5])}...)\n"
            f"• Baseline Amount Mean: `${avg_amt:,.2f} USD`"
        )

    except Exception as e:
        return f"[Analytics Engine Output] Error executing pandas calculation: {str(e)}"

analytics_agent = LlmAgent(
    name="AnalyticsAgent",
    model="gemini-3.5-flash",
    description="Business Analysis & KPI Quantitative Specialist",
    instruction="""You are the Analytics Agent (Quantitative Engine). Your role is to compute exact numeric aggregations from the enterprise CSV dataset using 'execute_pandas_analysis'. Always prefix your analytical findings with '[Analytics Engine Output]'.""",
    tools=[execute_pandas_analysis]
)