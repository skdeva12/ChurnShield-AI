
import pandas as pd
import joblib

# Load the saved Random Forest pipeline
model = joblib.load("data/random_forest_model.pkl")

# Example customer
customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 6,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 90.0,
    "TotalCharges": 540.0
}])

# Predict churn probability
probability = model.predict_proba(customer)[0, 1]

# Business-selected threshold
threshold = 0.35

print(f"Churn probability: {probability:.1%}")

if probability >= threshold:
    print("Prediction: HIGHER CHURN RISK")
else:
    print("Prediction: LOWER CHURN RISK")
