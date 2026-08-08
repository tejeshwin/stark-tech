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

def launch_cli():
    """Launches the command-line interactive session."""
    from coordinator import ADKOrchestratorPipeline, orchestrator_agent
    from utils.config import get_gemini_api_key

    print("==================================================================")
    print(" 🧠 STARKTECH MULTI-AGENT AI BUSINESS ANALYST (CLI MODE) 🧠")
    print("==================================================================")
    print("Orchestrator Model: gemini-1.5-pro")
    print("Specialist Agents:  gemini-1.5-flash (Semantic, Analytics, Viz, Consultant)")
    print("Deterministic Doers: Dataset Schema Reader & ML Predictor Tool")
    print("==================================================================")
    print("Type your business query below (or 'exit' / 'quit' to stop).\n")
    
    api_key = get_gemini_api_key()
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    else:
        user_key = input("Enter GEMINI_API_KEY (press Enter to skip if set in env): ").strip()
        if user_key:
            os.environ["GEMINI_API_KEY"] = user_key

    pipeline = ADKOrchestratorPipeline(agent=orchestrator_agent)
    
    while True:
        try:
            user_query = input("\nBusiness User > ").strip()
            if not user_query:
                continue
                
            if user_query.lower() in ['exit', 'quit']:
                print("\nShutting down AI Data Team session. Goodbye!")
                break
                
            print("\n⏳ Chief Data Officer orchestrating specialist agents...")
            response = pipeline.run_query(user_query)
            
            print("\n------------------------------------------------------------------")
            print(f"📊 CHIEF DATA OFFICER RESPONSE:\n{response}")
            print("------------------------------------------------------------------")
            
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting cleanly.")
            break
        except Exception as e:
            print(f"\n⚠️ [System Recovery] Terminal caught error: {str(e)}")
            print("Attempting session recovery... Chat session remains active.\n")

def main():
    if "--cli" in sys.argv:
        launch_cli()
    else:
        launch_dashboard()

if __name__ == "__main__":
    main()
