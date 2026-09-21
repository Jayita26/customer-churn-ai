import sys
import os
import logging
import time
import uuid

import joblib
import pandas as pd

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

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

@app.middleware("http")
async def request_logging_middleware(request, call_next):

    request_id = str(uuid.uuid4())[:8]

    start_time = time.time()

    logger.info(
        f"Request started | "
        f"id={request_id} | "
        f"method={request.method} | "
        f"path={request.url.path}"
    )

    response = await call_next(request)

    duration = time.time() - start_time

    logger.info(
        f"Request completed | "
        f"id={request_id} | "
        f"status={response.status_code} | "
        f"time={duration:.4f}s"
    )

    response.headers["X-Request-ID"] = request_id

    return response

api_v1 = APIRouter(
    prefix="/api/v1"
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

def get_feature_importance():

    preprocessor = model.named_steps["preprocessor"]

    classifier = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    coefficients = classifier.coef_[0]

    importance = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "absolute_importance": abs(coefficients)
    })

    importance = importance.sort_values(
        by="absolute_importance",
        ascending=False
    )

    return importance


# --------------------------------------------------
# Input data model
# --------------------------------------------------



class CustomerData(BaseModel):

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)

    Partner: str
    Dependents: str

    tenure: int = Field(ge=0)

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

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

class PredictionResponse(BaseModel):

    prediction: str
    churn_probability: float
    churn_probability_percent: float
    risk_level: str

class HealthResponse(BaseModel):

    status: str
    model_loaded: bool

class ModelInfoResponse(BaseModel):

    model: str
    task: str
    target: str
    output: str

class FeatureImportanceItem(BaseModel):

    feature: str
    coefficient: float
    absolute_importance: float


class FeatureImportanceResponse(BaseModel):

    features: list[FeatureImportanceItem]

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
@api_v1.get(
    "/health",
    response_model=HealthResponse
)
@app.get(
    "/health",
    response_model=HealthResponse
)
def health_check():

    logger.info("Health check requested")

    return {
        "status": "healthy",
        "model_loaded": True
    }


# --------------------------------------------------
# Model information
# --------------------------------------------------

@api_v1.get(
    "/model-info",
    response_model=ModelInfoResponse
)
@app.get(
    "/model-info",
    response_model=ModelInfoResponse
)
def model_info():
    

    return {
        "model": "Logistic Regression",
        "task": "Customer Churn Prediction",
        "target": "Churn",
        "output": "Churn probability"
    }


@api_v1.get(
    "/feature-importance",
    response_model=FeatureImportanceResponse
)
@app.get(
    "/feature-importance",
    response_model=FeatureImportanceResponse
)
def feature_importance():

    logger.info("Feature importance requested")

    importance = get_feature_importance()

    top_features = importance.head(10)

    return {
        "features": top_features[
            ["feature", "coefficient", "absolute_importance"]
        ].to_dict(orient="records")
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@api_v1.post(
    "/predict",
    response_model=PredictionResponse
)
@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_churn(customer: CustomerData):

    try:

        logger.info("Prediction request received")

        customer_dict = customer.model_dump()

        customer_df = pd.DataFrame([customer_dict])

        customer_df = clean_data(customer_df)
        customer_df = create_features(customer_df)

        prediction = model.predict(customer_df)[0]

        probability = model.predict_proba(customer_df)[0][1]

        probability_percent = float(probability) * 100

        if prediction == 1:
            result = "Likely to Churn"
        else:
            result = "Likely to Stay"

        if probability_percent < 30:
            risk_level = "Low"
        elif probability_percent <= 70:
            risk_level = "Medium"
        else:
            risk_level = "High"

        logger.info(
            f"Prediction completed: {result}, "
            f"probability={probability:.4f}, "
            f"risk={risk_level}"
        )

        return {
            "prediction": result,
            "churn_probability": round(
                float(probability),
                4
            ),
            "churn_probability_percent": round(
                probability_percent,
                2
            ),
            "risk_level": risk_level
        }

    except Exception as e:

        logger.exception(
            "Prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal server error."
        )
app.include_router(api_v1)
app.include_router(api_v1)