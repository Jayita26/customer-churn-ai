# Customer Churn Prediction

## Project Overview

This project is an end-to-end machine learning system that predicts whether a customer is likely to churn.

The project demonstrates a complete machine learning and deployment workflow:

**Dataset → EDA → Feature Engineering → Model Training → Evaluation → Prediction → FastAPI → Testing → Docker → Web Frontend → CI/CD**

The system provides a REST API and a web-based interface for making customer churn predictions.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Pydantic
* Joblib
* Pytest
* HTML
* CSS
* JavaScript
* Docker
* Git & GitHub
* GitHub Actions

## Machine Learning

Two classification models were trained:

1. Logistic Regression
2. Random Forest

The **Logistic Regression model** is used by the production prediction API.

The API returns:

* Churn prediction
* Churn probability
* Churn probability percentage
* Risk level

## Data Processing

The preprocessing pipeline performs:

* Missing-value handling
* Data type conversion
* Feature engineering
* Categorical encoding using `OneHotEncoder`
* Train-test splitting
* Consistent preprocessing during prediction

## Feature Engineering

The following additional features were created:

* Average monthly spend
* Total number of services
* New customer indicator
* Internet service indicator
* Customer support indicator

## API

The application is built using **FastAPI**.

### API Version 1

The current versioned API is available under:

```text
/api/v1
```

### Health Check

```text
GET /api/v1/health
```

Checks whether the API and machine learning model are running.

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Model Information

```text
GET /api/v1/model-info
```

Returns information about the machine learning model and prediction task.

### Feature Importance

```text
GET /api/v1/feature-importance
```

Returns the top model features and their coefficients.

### Prediction

```text
POST /api/v1/predict
```

Accepts customer information and returns a churn prediction.

Example response:

```json
{
  "prediction": "Likely to Churn",
  "churn_probability": 0.5547,
  "churn_probability_percent": 55.47,
  "risk_level": "Medium"
}
```

### API Documentation

FastAPI automatically provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

## Risk Classification

The API converts the predicted churn probability into a project-defined risk level:

| Churn Probability | Risk Level |
| ----------------- | ---------- |
| Below 30%         | Low        |
| 30%–70%           | Medium     |
| Above 70%         | High       |

These thresholds are application-level rules used for this project.

## Input Validation

The API uses **Pydantic** to validate incoming customer data.

Examples of validation rules include:

* `SeniorCitizen` must be 0 or 1
* `tenure` cannot be negative
* `MonthlyCharges` cannot be negative
* `TotalCharges` cannot be negative

Invalid requests are rejected with an appropriate HTTP validation response.

## Error Handling

The prediction endpoint includes server-side error handling.

If an unexpected error occurs during prediction, the API returns an HTTP 500 response with a generic error message instead of exposing internal implementation details.

## Automated Testing

The project uses **Pytest** for automated testing.

The test suite covers:

* API home endpoint
* Health endpoint
* Model information endpoint
* Prediction endpoint
* Input validation
* Risk-level validation
* Feature importance
* Prediction error handling
* Data loading
* Data cleaning
* Feature engineering

Current test result:

```text
14 passed
```

## Continuous Integration

GitHub Actions is used to automatically run the test suite whenever changes are pushed to the `main` branch or submitted through a pull request.

The CI workflow:

1. Checks out the repository
2. Sets up Python 3.11
3. Installs project dependencies
4. Runs `pytest`

## Web Frontend

A simple web interface was developed using:

* HTML
* CSS
* JavaScript

Users can enter customer information through a form and receive a churn prediction without manually creating a JSON request.

## Docker

The FastAPI application is containerized using Docker.

### Build the Docker Image

```bash
docker build -t customer-churn-api .
```

### Run the Container

```bash
docker run -d -p 8000:8000 --name customer-churn-container customer-churn-api
```

The application is then available at:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## Project Structure

```text
customer-churn-ai/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app/
│   ├── main.py
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── script.js
│
├── data/
│   └── customer_churn.csv
│
├── models/
│   ├── evaluation_results.csv
│   └── logistic_churn_model.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── tests/
│   ├── test_api.py
│   └── test_preprocessing.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

The Random Forest model is generated locally but excluded from GitHub because of its large file size.

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Jayita26/customer-churn-ai.git
```

### 2. Open the project

```bash
cd customer-churn-ai
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API

```bash
uvicorn app.main:app --reload
```

### 5. Open the application

```text
http://127.0.0.1:8000
```

### 6. Open API documentation

```text
http://127.0.0.1:8000/docs
```

### 7. Run automated tests

```bash
pytest -v
```

## Docker Deployment

Alternatively, run the complete application using Docker:

```bash
docker build -t customer-churn-api .
```

```bash
docker run -d -p 8000:8000 --name customer-churn-container customer-churn-api
```

Then open:

```text
http://127.0.0.1:8000
```

## Project Highlights

This project demonstrates practical experience with:

* Machine learning model development
* Data preprocessing
* Feature engineering
* REST API development
* API input validation
* API versioning
* Model explainability
* Error handling
* Automated testing
* Continuous integration
* Docker containerization
* Web application integration
* Git and GitHub workflow

## Future Improvements

Possible future improvements include:

* Authentication and authorization
* Database integration
* Cloud deployment
* Advanced model monitoring
* Model versioning
* Improved model performance
* Production monitoring
* More comprehensive CI/CD
* LLM or RAG-based customer support integration

## Author

**Jayita Maiti**

M.Sc. Data Science
