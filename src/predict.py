import joblib
import pandas as pd

from preprocessing import clean_data, create_features


# Load trained model
MODEL_PATH = "models/logistic_churn_model.pkl"

model = joblib.load(MODEL_PATH)


# Create a new customer
customer = pd.DataFrame([{
    "gender": "Male",
    "SeniorCitizen": 1,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.50,
    "TotalCharges": 191.00
}])


# Apply the same preprocessing and feature engineering
customer = clean_data(customer)
customer = create_features(customer)


# Make prediction
prediction = model.predict(customer)[0]

probability = model.predict_proba(customer)[0][1]


# Display result
if prediction == 1:
    result = "Likely to Churn"
else:
    result = "Likely to Stay"


print("Prediction:", result)
print("Churn Probability:", round(probability * 100, 2), "%")