import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load pre-trained credit risk model
@st.cache_data
def load_model():
    model = LogisticRegression()
    # Assuming you have a dataset named 'data' containing features and target
    data = pd.read_csv("your_data.csv")  # Load your data here
    X = data.drop(columns=["target_column"])
    y = data["target_column"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    return model

# Function to assess credit risk
def assess_credit_risk(model, features):
    scaler = StandardScaler()
    scaled_features = scaler.transform(features)
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
    
    # Load model
    model = load_model()
    
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
        risk_probability = assess_credit_risk(model, features)
        st.write("Credit Risk Probability:", risk_probability)

if __name__ == "__main__":
    main()
