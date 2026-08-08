import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

warnings.filterwarnings('ignore')

# Append root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.config import get_dataset_path, BASE_DIR
except ImportError:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def get_dataset_path():
        return os.path.join(BASE_DIR, "data", "cleaned_enterprise_data_final.csv")

class HumanReviewPredictorTrainer:
    """
    Production-ready Machine Learning module for preprocessing enterprise data,
    training a Random Forest Classifier to predict human review requirements,
    evaluating performance, displaying feature importances, and saving artifacts.
    """
    def __init__(self, data_path: str = None):
        self.data_path = data_path or get_dataset_path()
        self.target_col = 'Requires_Human_Review'
        self.model_filename = 'random_forest.pkl'
        self.models_dir = os.path.join(BASE_DIR, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_path = os.path.join(self.models_dir, self.model_filename)
        
        self.pipeline = None
        self.feature_names = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_preprocess_data(self, test_size: float = 0.2, sample_size: int = 50000, random_state: int = 42):
        """
        Loads CSV dataset, drops high-cardinality/ID columns, builds preprocessing pipeline,
        and splits into train and test sets.
        """
        print(f"[DATA] Loading enterprise dataset from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Check target column
        if self.target_col not in df.columns:
            cols_lower = {c.lower(): c for c in df.columns}
            if 'human_review' in cols_lower:
                self.target_col = cols_lower['human_review']
            elif 'requires_human_review' in cols_lower:
                self.target_col = cols_lower['requires_human_review']
            else:
                raise KeyError(f"Target column '{self.target_col}' not found in dataset.")

        print(f"[TARGET] Target Column identified: '{self.target_col}'")
        
        # Sample dataset for fast, optimal training if dataset is very large
        if len(df) > sample_size:
            print(f"[SAMPLING] Sub-sampling {sample_size:,} records from {len(df):,} total for efficient training...")
            df = df.sample(n=sample_size, random_state=random_state)

        # Convert target to binary integer (0 or 1)
        y = df[self.target_col].astype(int)
        
        # Drop identifiers and free-text columns
        drop_cols = [
            self.target_col, 'Record_ID', 'Transaction_Date', 'Timestamp',
            'Assigned_Agent', 'Employee_ID_Handler', 'Report_Summary_Text'
        ]
        feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        
        # Separate numerical and categorical columns
        num_cols = feature_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        cat_cols = feature_df.select_dtypes(include=['object', 'category', 'bool', 'str']).columns.tolist()
        
        print(f"[FEATURES] Features Identified: {len(num_cols)} Numerical, {len(cat_cols)} Categorical")

        # Define preprocessing transformers
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_cols),
                ('cat', cat_transformer, cat_cols)
            ]
        )

        # Build full pipeline with Random Forest Classifier
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=150,
                max_depth=16,
                min_samples_split=5,
                random_state=random_state,
                n_jobs=-1
            ))
        ])

        # Train/Test Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            feature_df, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"[SPLIT] Data Split: Train={self.X_train.shape[0]:,} samples, Test={self.X_test.shape[0]:,} samples")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self):
        """
        Trains the Random Forest model and extracts transformed feature names.
        """
        print("\n[TRAINING] Training Random Forest Classifier Pipeline...")
        self.pipeline.fit(self.X_train, self.y_train)
        print("[SUCCESS] Random Forest Model Training Complete!")

        # Retrieve feature names after OneHotEncoding
        preprocessor = self.pipeline.named_steps['preprocessor']
        num_cols = preprocessor.transformers_[0][2]
        cat_cols = preprocessor.transformers_[1][2]
        onehot_encoder = preprocessor.transformers_[1][1].named_steps['onehot']
        
        cat_encoded_cols = list(onehot_encoder.get_feature_names_out(cat_cols))
        self.feature_names = num_cols + cat_encoded_cols

    def evaluate_model(self) -> dict:
        """
        Evaluates the model on test set using standard classification metrics.
        """
        print("\n==================================================================")
        print(" RANDOM FOREST MODEL EVALUATION METRICS ")
        print("==================================================================")
        
        y_pred = self.pipeline.predict(self.X_test)
        y_prob = self.pipeline.predict_proba(self.X_test)[:, 1]

        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred)
        rec = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_prob)

        metrics = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }

        for metric_name, val in metrics.items():
            print(f"  * {metric_name:<12}: {val:.4f} ({val*100:.2f}%)")

        print("\n[REPORT] Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['No Review', 'Human Review Required']))

        print("\n[MATRIX] Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        return metrics

    def display_feature_importance(self, top_n: int = 15):
        """
        Displays and plots the top N feature importances.
        """
        rf_model = self.pipeline.named_steps['classifier']
        importances = rf_model.feature_importances_
        
        feature_imp_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

        print(f"\n==================================================================")
        print(f" TOP {top_n} MOST IMPORTANT FEATURES ")
        print("==================================================================")
        for idx, row in feature_imp_df.head(top_n).iterrows():
            print(f"  {idx+1:2d}. {row['Feature']:<40} : {row['Importance']:.5f}")

        # Save plot
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        top_df = feature_imp_df.head(top_n)
        ax = sns.barplot(data=top_df, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
        ax.set_title(f"Top {top_n} Feature Importances for Human Review Prediction", fontsize=13, fontweight='bold')
        ax.set_xlabel("Relative Importance Score")
        plt.tight_layout()
        
        plot_save_path = os.path.join(self.models_dir, "feature_importance.png")
        plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"\n[PLOT] Feature Importance Plot saved to: {plot_save_path}")

        return feature_imp_df

    def save_model(self):
        """
        Saves the trained pipeline and metadata as random_forest.pkl.
        """
        artifact = {
            'pipeline': self.pipeline,
            'target_col': self.target_col,
            'feature_names': self.feature_names
        }
        joblib.dump(artifact, self.model_path)
        print(f"\n[SAVE] Trained Model & Preprocessing Pipeline saved to:\n  -> {self.model_path}")
        return self.model_path

def main():
    trainer = HumanReviewPredictorTrainer()
    trainer.load_and_preprocess_data(test_size=0.2, sample_size=40000)
    trainer.train_model()
    trainer.evaluate_model()
    trainer.display_feature_importance(top_n=15)
    trainer.save_model()

if __name__ == '__main__':
    main()
