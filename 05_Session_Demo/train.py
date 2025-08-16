import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import joblib
import os

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
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
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
        
        with open(f"{model_dir}/loan_model.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        
        joblib.dump(self.model, f"{model_dir}/loan_model.joblib")
        
        with open(f"{model_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(f"{model_dir}/label_encoders.pkl", 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        logger.info(f"Model artifacts saved to {model_dir}/")
    
    def load_model(self, model_dir="models"):
        logger.info("Loading model artifacts")
        
        with open(f"{model_dir}/loan_model.pkl", 'rb') as f:
            self.model = pickle.load(f)
        
        with open(f"{model_dir}/scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(f"{model_dir}/label_encoders.pkl", 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        logger.info("Model artifacts loaded successfully")

def main():
    logger.info("Starting Loan Approval Model Training Pipeline")
    
    model = LoanApprovalModel()
    
    df = model.create_sample_data()
    logger.info(f"Sample data created with {len(df)} records")
    
    df_processed = model.preprocess_data(df)
    
    X_test, y_test, y_pred, accuracy = model.train_model(df_processed)
    
    model.save_model()
    
    logger.info("Pipeline completed successfully")
    
    sample_prediction = model.model.predict(model.scaler.transform(X_test[:5]))
    logger.info(f"Sample predictions: {sample_prediction}")

if __name__ == "__main__":
    main()