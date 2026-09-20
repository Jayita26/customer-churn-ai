import sys
import os
import logging

import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Import preprocessing functions
# --------------------------------------------------

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.preprocessing import clean_data, create_features


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Machine Learning API for Customer Churn Prediction",
    version="1.1.0"
)


STATIC_DIR = os.path.join(
    os.path.dirname(__file__),
    "static"
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


# --------------------------------------------------
# Load ML model
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "logistic_churn_model.pkl"
)

model = joblib.load(MODEL_PATH)

logger.info("Machine learning model loaded successfully")


# --------------------------------------------------
# Input data model
# --------------------------------------------------

class CustomerData(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int

    PhoneService: str
    MultipleLines: str
    InternetService: str

    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str

    StreamingTV: str
    StreamingMovies: str

    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    MonthlyCharges: float
    TotalCharges: float


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/", include_in_schema=False)
def home():

    logger.info("Frontend accessed")

    return FileResponse(
        os.path.join(STATIC_DIR, "index.html")
    )


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health_check():

    logger.info("Health check requested")

    return {
        "status": "healthy",
        "model_loaded": True
    }


# --------------------------------------------------
# Model information
# --------------------------------------------------

@app.get("/model-info")
def model_info():

    return {
        "model": "Logistic Regression",
        "task": "Customer Churn Prediction",
        "target": "Churn",
        "output": "Churn probability"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_churn(customer: CustomerData):

    logger.info("Prediction request received")

    customer_dict = customer.model_dump()

    customer_df = pd.DataFrame([customer_dict])

    customer_df = clean_data(customer_df)

    customer_df = create_features(customer_df)

    prediction = model.predict(customer_df)[0]

    probability = model.predict_proba(customer_df)[0][1]

    if prediction == 1:
        result = "Likely to Churn"
    else:
        result = "Likely to Stay"

    logger.info(
        f"Prediction completed: {result}, "
        f"probability={probability:.4f}"
    )

    return {
        "prediction": result,
        "churn_probability": round(
            float(probability),
            4
        ),
        "churn_probability_percent": round(
            float(probability) * 100,
            2
        )
    }