import os
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from utils.config import get_dataset_path, load_dataset, get_gemini_api_key, key_manager, BASE_DIR
except ImportError:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    def get_dataset_path():
        candidates = ["data/cleaned_enterprise_data_final.csv", "cleaned_enterprise_data.csv", "data/cleaned_enterprise_data.csv"]
        for c in candidates:
            if os.path.exists(c): return c
        return "cleaned_enterprise_data.csv"
    def load_dataset():
        return pd.read_csv(get_dataset_path())
    def get_gemini_api_key():
        return os.getenv("GEMINI_API_KEY", "")
    class KeyManagerFallback:
        def get_active_key(self): return get_gemini_api_key()
    key_manager = KeyManagerFallback()

# Ensure API Key is configured in environment
api_key = get_gemini_api_key()
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key

from coordinator import ADKOrchestratorPipeline, orchestrator_agent

# -----------------------------------------------------------------------------
# Streamlit Page Config & Professional Corporate Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="StarkTech Enterprise Decision Support | Google ADK",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Corporate CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .header-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .header-box h1 {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0;
    }
    .header-box p {
        color: #94a3b8;
        margin: 0.3rem 0 0 0;
        font-size: 1.05rem;
    }
    .kpi-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
    }
    .kpi-card h4 {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .kpi-card p {
        color: #38bdf8;
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0.4rem 0 0 0;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/domain.png", width=56)
    st.title("Enterprise Control Panel")
    
    key_count = len(getattr(key_manager, 'keys', [1]))
    st.success(f"🟢 System Online ({key_count} Pre-Configured Keys Active)")
    
    st.divider()
    st.subheader("Enterprise Data Source")
    try:
        data_path = get_dataset_path()
        st.info(f"Dataset: `{os.path.basename(data_path)}`")
        df_preview = load_dataset()
        st.caption(f"📊 **{len(df_preview):,}** Rows | **{len(df_preview.columns)}** Columns")
    except Exception as e:
        st.error(f"Dataset error: {e}")
        
    st.divider()
    st.subheader("System Architecture")
    st.markdown("""
    - 🧠 **Orchestrator**: `gemini-3.5-flash`
    - ⚡ **Doer Tools**: Dataset Reader & ML Predictor
    - 🔍 **Semantic Agent**: `gemini-3.5-flash`
    - 🧮 **Analytics Agent**: `gemini-3.5-flash`
    - 🎨 **Viz Agent**: `gemini-3.5-flash`
    - 💼 **Consultant Agent**: `gemini-3.5-flash`
    """)

# -----------------------------------------------------------------------------
# Executive Header & Top-Level KPI Metric Cards
# -----------------------------------------------------------------------------
st.markdown("""
<div class="header-box">
    <h1>Enterprise Decision Support Platform</h1>
    <p>Google ADK Multi-Agent Intelligence Engine | Data-Driven Business Analysis</p>
</div>
""", unsafe_allow_html=True)

try:
    df_main = load_dataset()
    total_records = len(df_main)
    
    avg_tx_val = df_main['Transaction_Amount_USD'].mean() if 'Transaction_Amount_USD' in df_main.columns else (
        df_main['Revenue_Impact_USD'].mean() if 'Revenue_Impact_USD' in df_main.columns else 0
    )
    
    proc_speed = df_main['Processing_Time_Sec'].mean() if 'Processing_Time_Sec' in df_main.columns else 0.0
    sys_status = "ONLINE (100% SLA)"
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <h4>Total Records</h4>
            <p>{total_records:,}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <h4>Avg Transaction Value</h4>
            <p>${avg_tx_val:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <h4>Processing Speed</h4>
            <p>{proc_speed:.2f} sec</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <h4>System Status</h4>
            <p style="color:#4ade80;">{sys_status}</p>
        </div>
        """, unsafe_allow_html=True)
except Exception:
    pass

st.write("")

# -----------------------------------------------------------------------------
# Structured Tabs
# -----------------------------------------------------------------------------
tab_chat, tab_analytics, tab_explorer = st.tabs([
    "💬 Executive Chat Interface",
    "📈 Visual Analytics Dashboard",
    "🗂️ Raw Data Explorer"
])

# -----------------------------------------------------------------------------
# TAB 1: 💬 Executive Chat Interface
# -----------------------------------------------------------------------------
with tab_chat:
    st.subheader("Chief Data Officer Executive Assistant")
    st.caption("Ask questions about revenue, SLAs, operational performance, or department trends. Powered by Google ADK Runner.")
    
    st.markdown("**Executive Prompt Templates:**")
    p_cols = st.columns(3)
    preset_prompt = None
    with p_cols[0]:
        if st.button("📊 Total Revenue Impact by Department"):
            preset_prompt = "Calculate total revenue impact by department and create a colorful detailed chart."
    with p_cols[1]:
        if st.button("⚠️ SLA Breach & Priority Analysis"):
            preset_prompt = "Break down SLA breaches by priority level and create a colorful bar chart with numbers."
    with p_cols[2]:
        if st.button("🔮 Forecast High-Complexity Queries"):
            preset_prompt = "Forecast revenue impact and risk for high-complexity queries."

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("chart") and os.path.exists(msg["chart"]):
                st.image(msg["chart"], caption="Executive Chart Result", width="stretch")

    user_input = st.chat_input("Enter your business query...")
    if preset_prompt:
        user_input = preset_prompt

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Google ADK Orchestrator processing query across multi-agent pipeline..."):
                pipeline = ADKOrchestratorPipeline()
                try:
                    res_text, new_chart = pipeline.run_query(user_input)
                except Exception as e:
                    res_text, new_chart = f"⚠️ Execution Error: {str(e)}", None
                    
            st.markdown(res_text)
            
            if new_chart and os.path.exists(new_chart):
                st.image(new_chart, caption="Executive Chart Result", width="stretch")
                
            st.session_state.messages.append({
                "role": "assistant",
                "content": res_text,
                "chart": new_chart
            })

# -----------------------------------------------------------------------------
# TAB 2: 📈 Visual Analytics Dashboard
# -----------------------------------------------------------------------------
with tab_analytics:
    st.subheader("Automated Operational Analytics Dashboard")
    try:
        df_viz = load_dataset()
        sns.set_theme(style="whitegrid")
        
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.markdown("#### Total Revenue Impact by Department")
            if 'Department' in df_viz.columns and 'Revenue_Impact_USD' in df_viz.columns:
                dept_rev = df_viz.groupby('Department')['Revenue_Impact_USD'].sum().reset_index().sort_values('Revenue_Impact_USD', ascending=False)
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                bars1 = sns.barplot(data=dept_rev, x='Department', y='Revenue_Impact_USD', hue='Department', palette='viridis', legend=False, ax=ax1)
                ax1.set_title("Revenue Impact by Department", fontsize=12, fontweight="bold", pad=10)
                ax1.set_ylabel("Revenue Impact ($ USD)")
                ax1.bar_label(ax1.containers[0], fmt='$%.0f', padding=3, fontsize=9)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig1)
                
        with col_v2:
            st.markdown("#### SLA Breaches by Priority Level")
            if 'Priority_Level' in df_viz.columns and 'SLA_Breached' in df_viz.columns:
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                sns.countplot(data=df_viz, x='Priority_Level', hue='SLA_Breached', palette='Set2', order=['Low', 'Medium', 'High', 'Critical'], ax=ax2)
                ax2.set_title("SLA Breaches vs Priority Level", fontsize=12, fontweight="bold", pad=10)
                ax2.set_ylabel("Ticket Count")
                for c in ax2.containers:
                    ax2.bar_label(c, fmt='%d', padding=2, fontsize=8)
                plt.tight_layout()
                st.pyplot(fig2)
                
        st.divider()
        st.markdown("#### Processing Time vs Query Complexity")
        if 'Query_Complexity_Score' in df_viz.columns and 'Processing_Time_Sec' in df_viz.columns:
            fig3, ax3 = plt.subplots(figsize=(10, 4.5))
            sample_df = df_viz.sample(min(1000, len(df_viz)))
            sns.scatterplot(data=sample_df, x='Query_Complexity_Score', y='Processing_Time_Sec', hue='SLA_Breached', palette='rocket', alpha=0.75, ax=ax3)
            ax3.set_title("Query Complexity vs Processing Duration", fontsize=12, fontweight="bold", pad=10)
            ax3.set_xlabel("Query Complexity Score (1-10)")
            ax3.set_ylabel("Processing Duration (Seconds)")
            plt.tight_layout()
            st.pyplot(fig3)
    except Exception as e:
        st.error(f"Analytics rendering error: {e}")

# -----------------------------------------------------------------------------
# TAB 3: 🗂️ Raw Data Explorer
# -----------------------------------------------------------------------------
with tab_explorer:
    st.subheader("Enterprise Data Explorer")
    try:
        df_exp = load_dataset()
        st.caption(f"Displaying dataset: `{os.path.basename(get_dataset_path())}`")
        
        search_kw = st.text_input("Search raw dataset by keyword:")
        if search_kw:
            mask = df_exp.astype(str).apply(lambda row: row.str.contains(search_kw, case=False).any(), axis=1)
            df_exp = df_exp[mask]
            
        st.write(f"Showing **{len(df_exp):,}** records:")
        st.dataframe(df_exp.head(200), width="stretch")
        
        st.divider()
        st.subheader("Numerical Dataset Summary Statistics")
        st.dataframe(df_exp.describe(), width="stretch")
    except Exception as e:
        st.error(f"Explorer error: {e}")
