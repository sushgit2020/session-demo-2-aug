import streamlit as st
import joblib
import numpy as np
import os

@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/loan_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

def predict_loan(model, scaler, income, credit_score, loan_amount, employment_years):
    if model is None or scaler is None:
        return None, None, "Model not found. Please train the model first."
    
    features = np.array([[income, credit_score, loan_amount, employment_years]])
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)[0]
    confidence = model.predict_proba(features_scaled)[0].max()
    
    return bool(prediction), float(confidence), None

def main():
    st.set_page_config(
        page_title="Loan Approval Predictor",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 Loan Approval Predictor")
    st.markdown("Enter your financial details to check loan approval probability")
    
    model, scaler = load_model()
    
    if model is None:
        st.error("⚠️ Model not found! Please train the model first by running: `python train_model.py`")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Personal Information")
        
        income = st.number_input(
            "Annual Income ($)",
            min_value=10000,
            max_value=500000,
            value=50000,
            step=1000,
            help="Your total annual income"
        )
        
        credit_score = st.slider(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=650,
            help="Your credit score (300-850)"
        )
        
        loan_amount = st.number_input(
            "Loan Amount ($)",
            min_value=1000,
            max_value=1000000,
            value=100000,
            step=1000,
            help="Amount you want to borrow"
        )
        
        employment_years = st.selectbox(
            "Years of Employment",
            options=list(range(0, 21)),
            index=2,
            help="Number of years in current employment"
        )
    
    with col2:
        st.subheader("📊 Loan Analysis")
        
        debt_to_income = loan_amount / income
        st.metric("Debt-to-Income Ratio", f"{debt_to_income:.2f}")
        
        if debt_to_income <= 3:
            st.success("✅ Good debt-to-income ratio")
        elif debt_to_income <= 5:
            st.warning("⚠️ Moderate debt-to-income ratio")
        else:
            st.error("❌ High debt-to-income ratio")
        
        if credit_score >= 700:
            st.success("✅ Excellent credit score")
        elif credit_score >= 600:
            st.info("ℹ️ Good credit score")
        else:
            st.warning("⚠️ Fair credit score")
        
        if employment_years >= 2:
            st.success("✅ Stable employment history")
        else:
            st.warning("⚠️ Limited employment history")
    
    st.divider()
    
    if st.button("🔍 Check Loan Approval", type="primary", use_container_width=True):
        with st.spinner("Analyzing your application..."):
            approved, confidence, error = predict_loan(
                model, scaler, income, credit_score, loan_amount, employment_years
            )
            
            if error:
                st.error(error)
            else:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if approved:
                        st.success("🎉 Loan Approved!")
                    else:
                        st.error("❌ Loan Rejected")
                
                with col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                
                with col3:
                    if confidence >= 0.8:
                        st.success("High Confidence")
                    elif confidence >= 0.6:
                        st.warning("Moderate Confidence")
                    else:
                        st.error("Low Confidence")
                
                st.divider()
                
                st.subheader("💡 Recommendations")
                
                if not approved:
                    recommendations = []
                    
                    if credit_score < 700:
                        recommendations.append("• Improve your credit score by paying bills on time and reducing debt")
                    
                    if debt_to_income > 3:
                        recommendations.append("• Consider a smaller loan amount or increase your income")
                    
                    if employment_years < 2:
                        recommendations.append("• Build a longer employment history for better approval chances")
                    
                    if income < 40000:
                        recommendations.append("• Consider increasing your income before applying")
                    
                    if recommendations:
                        for rec in recommendations:
                            st.write(rec)
                    else:
                        st.write("• Your application is borderline. Consider waiting and reapplying later.")
                else:
                    st.write("🎊 Congratulations! Your loan application looks good.")
                    if confidence < 0.8:
                        st.write("• Consider improving your financial profile for even better terms.")

if __name__ == "__main__":
    main()