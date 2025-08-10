import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def generate_sample_data(n_samples=1000):
    np.random.seed(42)
    
    data = []
    for _ in range(n_samples):
        income = np.random.normal(50000, 20000)
        income = max(20000, income)
        
        credit_score = np.random.normal(650, 100)
        credit_score = np.clip(credit_score, 300, 850)
        
        employment_years = np.random.poisson(3)
        employment_years = max(0, employment_years)
        
        loan_amount = np.random.normal(100000, 50000)
        loan_amount = max(10000, loan_amount)
        
        debt_to_income = loan_amount / income
        
        approval_prob = 0.1
        if income >= 40000:
            approval_prob += 0.3
        if credit_score >= 600:
            approval_prob += 0.4
        if credit_score >= 700:
            approval_prob += 0.2
        if debt_to_income <= 4:
            approval_prob += 0.2
        if employment_years >= 2:
            approval_prob += 0.1
        
        approved = np.random.random() < approval_prob
        
        data.append({
            'income': income,
            'credit_score': int(credit_score),
            'loan_amount': loan_amount,
            'employment_years': employment_years,
            'approved': int(approved)
        })
    
    return pd.DataFrame(data)

def train_model():
    print("Generating sample data...")
    df = generate_sample_data(1000)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Approval rate: {df['approved'].mean():.2%}")
    
    X = df[['income', 'credit_score', 'loan_amount', 'employment_years']]
    y = df['approved']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training model...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/loan_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("Model and scaler saved to models/ directory")

if __name__ == "__main__":
    train_model()