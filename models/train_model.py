import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_dataset, BASE_DIR

def train_and_save_ml_models():
    """
    Trains static machine learning models on the enterprise dataset
    and saves them into the models/ directory as .pkl files.
    """
    print("Loading dataset for ML model pre-training...")
    df = load_dataset()
    
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Train Revenue Impact Predictor (RandomForestRegressor)
    feature_cols = ['Query_Complexity_Score', 'Input_Tokens', 'Output_Tokens', 'Processing_Time_Sec']
    target_col = 'Revenue_Impact_USD'
    
    # Fill missing values if any
    X = df[feature_cols].copy().fillna(df[feature_cols].median())
    y = df[target_col].copy().fillna(df[target_col].median())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_size=0.2, random_state=42) if hasattr(train_test_split, 'test_size') else train_test_split(X, y, test_size=0.2, random_state=42)
    
    revenue_model = RandomForestRegressor(n_estimators=50, random_state=42)
    revenue_model.fit(X_train, y_train)
    
    revenue_model_path = os.path.join(models_dir, "revenue_forecast_model.pkl")
    with open(revenue_model_path, "wb") as f:
        pickle.dump({
            "model": revenue_model,
            "feature_names": feature_cols,
            "target_name": target_col,
            "r2_score": float(revenue_model.score(X_test, y_test))
        }, f)
    print(f"Revenue Model saved to: {revenue_model_path}")
    
    # 2. Train SLA Breach / Anomaly Risk Classifier (RandomForestClassifier)
    if 'SLA_Breached' in df.columns:
        y_sla = df['SLA_Breached'].astype(int)
        sla_model = RandomForestClassifier(n_estimators=50, random_state=42)
        sla_model.fit(X, y_sla)
        
        sla_model_path = os.path.join(models_dir, "sla_breach_model.pkl")
        with open(sla_model_path, "wb") as f:
            pickle.dump({
                "model": sla_model,
                "feature_names": feature_cols,
                "target_name": "SLA_Breached"
            }, f)
        print(f"SLA Breach Model saved to: {sla_model_path}")
        
    print("All static ML models trained and pickled successfully!")

if __name__ == "__main__":
    train_and_save_ml_models()
