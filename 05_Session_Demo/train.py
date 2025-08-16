import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import joblib
import os
import json
from pathlib import Path
from datetime import datetime

# --- MLflow imports ---
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LoanApprovalModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_sample_data(self):
        logger.info("Creating sample loan approval dataset")
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'age': np.random.normal(35, 10, n_samples),
            'income': np.random.normal(50000, 20000, n_samples),
            'loan_amount': np.random.normal(200000, 80000, n_samples),
            'credit_score': np.random.normal(650, 100, n_samples),
            'employment_years': np.random.normal(5, 3, n_samples),
            'debt_to_income': np.random.uniform(0.1, 0.6, n_samples),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
            'property_type': np.random.choice(['Apartment', 'House', 'Condo'], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        df['age'] = df['age'].clip(18, 80)
        df['income'] = df['income'].clip(20000, 150000)
        df['loan_amount'] = df['loan_amount'].clip(50000, 500000)
        df['credit_score'] = df['credit_score'].clip(300, 850)
        df['employment_years'] = df['employment_years'].clip(0, 40)
        
        approval_prob = (
            (df['income'] / 100000) * 0.3 +
            (df['credit_score'] / 850) * 0.4 +
            (1 - df['debt_to_income']) * 0.2 +
            (df['employment_years'] / 40) * 0.1
        )
        
        df['loan_approved'] = (approval_prob > 0.5).astype(int)
        
        return df
    
    def preprocess_data(self, df):
        logger.info("Starting data preprocessing")
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Missing values: {df.isnull().sum().sum()}")
        
        df_processed = df.copy()
        
        categorical_columns = ['education', 'marital_status', 'property_type']
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        numerical_columns = ['age', 'income', 'loan_amount', 'credit_score', 'employment_years', 'debt_to_income']
        Q1 = df_processed[numerical_columns].quantile(0.25)
        Q3 = df_processed[numerical_columns].quantile(0.75)
        IQR = Q3 - Q1
        
        for col in numerical_columns:
            lower_bound = Q1[col] - 1.5 * IQR[col]
            upper_bound = Q3[col] + 1.5 * IQR[col]
            df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
        
        logger.info("Data preprocessing completed")
        return df_processed
    
    def train_model(self, df):
        logger.info("Starting model training")
        
        X = df.drop('loan_approved', axis=1)
        y = df['loan_approved']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model training completed. Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        return X_test, y_test, y_pred, accuracy
    
    def save_model(self, model_dir="models"):
        logger.info("Saving model artifacts")
        os.makedirs(model_dir, exist_ok=True)

        # Primary artifacts
        with open(f"{model_dir}/loan_model.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        joblib.dump(self.model, f"{model_dir}/loan_model.joblib")
        with open(f"{model_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(f"{model_dir}/label_encoders.pkl", 'wb') as f:
            pickle.dump(self.label_encoders, f)

        # Also write a copy for the workflow release asset (models/model.pkl)
        with open("model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        logger.info(f"Model artifacts saved to {model_dir}/ and model.pkl in CWD")
    
    def load_model(self, model_dir="models"):
        logger.info("Loading model artifacts")
        
        with open(f"{model_dir}/loan_model.pkl", 'rb') as f:
            self.model = pickle.load(f)
        with open(f"{model_dir}/scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        with open(f"{model_dir}/label_encoders.pkl", 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        logger.info("Model artifacts loaded successfully")

def write_reports(y_test, y_pred, accuracy, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cls_report_dict = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    metrics_json = {
        "accuracy": float(accuracy),
        "precision_weighted": float(cls_report_dict["weighted avg"]["precision"]),
        "recall_weighted": float(cls_report_dict["weighted avg"]["recall"]),
        "f1_weighted": float(cls_report_dict["weighted avg"]["f1-score"]),
        "class_0": cls_report_dict.get("0", {}),
        "class_1": cls_report_dict.get("1", {}),
        "confusion_matrix": cm.tolist()
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2))

    # Also save confusion matrix as CSV for convenience
    pd.DataFrame(cm, index=["Actual_0","Actual_1"], columns=["Pred_0","Pred_1"]).to_csv(out_dir / "confusion_matrix.csv")

    cls_report_text = classification_report(y_test, y_pred)
    md = []
    md.append("# Loan Model – Evaluation Report")
    md.append("")
    md.append(f"- **Accuracy**: `{accuracy:.4f}`")
    md.append("")
    md.append("## Classification Report")
    md.append("")
    md.append("```text")
    md.append(cls_report_text)
    md.append("```")
    md.append("")
    md.append("## Confusion Matrix")
    md.append("")
    md.append("```text")
    md.append(str(cm))
    md.append("```")
    (out_dir / "report.md").write_text("\n".join(md))
    logger.info(f"Wrote report to {out_dir}/report.md and metrics to {out_dir}/metrics.json")

def main():
    logger.info("Starting Loan Approval Model Training Pipeline")

    # --- MLflow local tracking setup ---
    # Store runs under repo-local model-demo/mlruns/ for easy artifact upload by CI
    tracking_dir = Path("mlruns")
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{tracking_dir.as_posix()}")
    mlflow.set_experiment("loan-approval")

    model = LoanApprovalModel()
    
    df = model.create_sample_data()
    logger.info(f"Sample data created with {len(df)} records")
    
    df_processed = model.preprocess_data(df)
    X_test, y_test, y_pred, accuracy = model.train_model(df_processed)

    # --- Reports for CI summary/artifacts ---
    reports_dir = Path("reports")
    write_reports(y_test, y_pred, accuracy, reports_dir)

    # --- Log to MLflow ---
    run_name = f"rf-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        # Params
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", model.model.n_estimators)
        mlflow.log_param("random_state", model.model.random_state)
        mlflow.log_param("scaler", "StandardScaler")
        # Metrics
        cls_report_dict = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("precision_weighted", float(cls_report_dict["weighted avg"]["precision"]))
        mlflow.log_metric("recall_weighted", float(cls_report_dict["weighted avg"]["recall"]))
        mlflow.log_metric("f1_weighted", float(cls_report_dict["weighted avg"]["f1-score"]))
        # Artifacts (reports and confusion matrix)
        mlflow.log_artifact(reports_dir / "report.md", artifact_path="reports")
        mlflow.log_artifact(reports_dir / "metrics.json", artifact_path="reports")
        if (reports_dir / "confusion_matrix.csv").exists():
            mlflow.log_artifact(reports_dir / "confusion_matrix.csv", artifact_path="reports")
        # Model
        mlflow.sklearn.log_model(sk_model=model.model, artifact_path="model")

        run_id = run.info.run_id
        artifact_uri = mlflow.get_artifact_uri()
        # Save a small text file so CI can echo in the Summary
        (reports_dir / "mlflow_run.txt").write_text(
            f"- **MLflow run_id**: `{run_id}`\n"
            f"- **Artifact URI**: `{artifact_uri}`\n"
        )
        logger.info(f"MLflow run complete: run_id={run_id}, artifact_uri={artifact_uri}")

    # Persist model files (for release + artifacts)
    model.save_model()

    logger.info("Pipeline completed successfully")
    
    sample_prediction = model.model.predict(model.scaler.transform(X_test[:5]))
    logger.info(f"Sample predictions: {sample_prediction}")

if __name__ == "__main__":
    main()
