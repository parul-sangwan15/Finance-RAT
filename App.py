import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load pre-trained credit risk model
@st.cache
def load_model():
    return LogisticRegression()

# Function to assess credit risk
def assess_credit_risk(features):
    model = load_model()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    risk_probability = model.predict_proba(scaled_features)[:, 1]
    return risk_probability

def main():
    st.title("Irish Credit Risk Assessment Tool")
    
    # User input for credit risk assessment
    st.header("Enter Borrower Information")
    income = st.number_input("Income (€)", min_value=0)
    expenses = st.number_input("Expenses (€)", min_value=0)
    assets = st.number_input("Total Assets (€)", min_value=0)
    liabilities = st.number_input("Total Liabilities (€)", min_value=0)
    credit_history = st.selectbox("Credit History", ["Good", "Fair", "Poor"])
    loan_amount = st.number_input("Loan Amount (€)", min_value=0)
    loan_purpose = st.text_input("Loan Purpose")
    
    # Convert credit history to numerical representation
    credit_history_map = {"Good": 2, "Fair": 1, "Poor": 0}
    
    # Predict credit risk
    if st.button("Assess Credit Risk"):
        features = pd.DataFrame({
            "Income": [income],
            "Expenses": [expenses],
            "Assets": [assets],
            "Liabilities": [liabilities],
            "Credit_History": [credit_history_map[credit_history]],
            "Loan_Amount": [loan_amount]
        })
        risk_probability = assess_credit_risk(features)
        st.write("Credit Risk Probability:", risk_probability)

if __name__ == "__main__":
    main()
