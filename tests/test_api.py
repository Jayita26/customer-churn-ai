import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "Logistic Regression"
    assert data["task"] == "Customer Churn Prediction"


def test_prediction():
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 351.75
    }

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "churn_probability" in data
    assert "churn_probability_percent" in data
    assert "risk_level" in data

    assert data["risk_level"] in [
    "Low",
    "Medium",
    "High"
]

    assert data["prediction"] in [
        "Likely to Churn",
        "Likely to Stay"
    ]

    assert 0 <= data["churn_probability"] <= 1

    assert 0 <= data["churn_probability_percent"] <= 100

def test_negative_tenure_rejected():

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": -5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 351.75
    }

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 422


def test_negative_monthly_charges_rejected():

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": -70.35,
        "TotalCharges": 351.75
    }

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 422

def test_invalid_senior_citizen_rejected():

    customer = {
        "gender": "Female",
        "SeniorCitizen": 2,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 351.75
    }

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 422

def test_risk_level_is_valid():

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 351.75
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    data = response.json()

    probability = data["churn_probability_percent"]
    risk = data["risk_level"]

    if probability < 30:
        assert risk == "Low"
    elif probability <= 70:
        assert risk == "Medium"
    else:
        assert risk == "High"


def test_risk_level_present_in_prediction():

    customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 20.00,
        "TotalCharges": 1200.00
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    data = response.json()

    assert "risk_level" in data
    assert data["risk_level"] in ["Low", "Medium", "High"]

def test_feature_importance():

    response = client.get("/feature-importance")

    assert response.status_code == 200

    data = response.json()

    assert "features" in data

    assert len(data["features"]) == 10

    for feature in data["features"]:

        assert "feature" in feature
        assert "coefficient" in feature
        assert "absolute_importance" in feature

        assert isinstance(
            feature["coefficient"],
            float
        )

        assert isinstance(
            feature["absolute_importance"],
            float
        )

def test_prediction_error_handling(monkeypatch):

    def mock_predict(*args, **kwargs):
        raise Exception("Test prediction failure")

    monkeypatch.setattr(
        "app.main.model.predict",
        mock_predict
    )

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 351.75
    }

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 500

    data = response.json()

    assert data["detail"] == (
        "Prediction failed due to an internal server error."
    )