import os
import sys
import subprocess

def launch_dashboard():
    """Launches the Streamlit dashboard using python -m streamlit run app.py."""
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    print("==================================================================")
    print(" 🧠 LAUNCHING STARKTECH MULTI-AGENT EXECUTIVE DASHBOARD 🧠")
    print("==================================================================")
    print(f"Target App: {app_path}")
    print("Running: python -m streamlit run app.py\n")
    
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nExecutive Dashboard shutdown cleanly.")

def launch_cli(workflow_mode: str = "Orchestrator"):
    """Launches the command-line interactive session supporting mode-based execution mapping."""
    from coordinator import ADKOrchestratorPipeline, orchestrator_agent
    from utils.config import get_gemini_api_key

    print("==================================================================")
    print(" 🧠 STARKTECH MULTI-AGENT AI BUSINESS ANALYST (CLI MODE) 🧠")
    print("==================================================================")
    print(f"Active Workflow Mode: {workflow_mode}")
    print("Orchestrator Model: gemini-3.5-flash")
    print("Specialist Agents:  gemini-3.5-flash (Semantic, Analytics, Viz, Consultant)")
    print("==================================================================")
    print("Type your business query below (or 'exit' / 'quit' to stop).\n")
    
    api_key = get_gemini_api_key()
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key

    pipeline = ADKOrchestratorPipeline(agent=orchestrator_agent)
    
    while True:
        try:
            user_query = input(f"\n[{workflow_mode}] Business User > ").strip()
            if not user_query:
                continue
                
            if user_query.lower() in ['exit', 'quit']:
                print("\nShutting down AI Data Team session. Goodbye!")
                break
                
            print(f"\n⏳ Processing query via '{workflow_mode}' agent pipeline...")
            res_text, chart = pipeline.run_query(user_query, workflow_mode=workflow_mode)
            
            print("\n------------------------------------------------------------------")
            print(f"📊 AGENT RESPONSE:\n{res_text}")
            if chart:
                print(f"[Generated Chart Saved]: {chart}")
            print("------------------------------------------------------------------")
            
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting cleanly.")
            break
        except Exception as e:
            print(f"\n⚠️ [System Recovery] Terminal caught error: {str(e)}")
            print("Attempting session recovery... Chat session remains active.\n")

def main():
    mode = "Orchestrator"
    if "--semantic" in sys.argv: mode = "Semantic Validation"
    elif "--analytics" in sys.argv or "--eda" in sys.argv: mode = "Operational EDA"
    elif "--viz" in sys.argv: mode = "Visualization Generation"
    elif "--predict" in sys.argv: mode = "Predictive ML Risk"
    elif "--consultant" in sys.argv: mode = "Executive Summary"

    if "--cli" in sys.argv or mode != "Orchestrator":
        launch_cli(workflow_mode=mode)
    else:
        launch_dashboard()

if __name__ == "__main__":
    main()
