from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

model = None
scaler = None

def load_model():
    global model, scaler
    try:
        model = joblib.load('models/loan_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        print("Model and scaler loaded successfully")
    except FileNotFoundError:
        print("Model files not found. Using rule-based approach.")
        model = None
        scaler = None

@app.on_event("startup")
async def startup_event():
    load_model()

class LoanRequest(BaseModel):
    income: float
    credit_score: int
    loan_amount: float
    employment_years: int

class LoanResponse(BaseModel):
    approved: bool
    message: str
    confidence: float

def predict_loan_approval(loan_data: LoanRequest) -> LoanResponse:
    if model is not None and scaler is not None:
        features = np.array([[
            loan_data.income,
            loan_data.credit_score,
            loan_data.loan_amount,
            loan_data.employment_years
        ]])
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled)[0].max()
        
        approved = bool(prediction)
        message = "Loan approved" if approved else "Loan rejected"
        
        return LoanResponse(
            approved=approved,
            message=message,
            confidence=float(confidence)
        )
    else:
        raise Exception("Model not loaded. Please train the model first by running: python train_model.py")

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/predict-loan", response_model=LoanResponse)
def predict_loan(loan_request: LoanRequest):
    return predict_loan_approval(loan_request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)