# Customer Churn Prediction

## Project Overview

This project builds an end-to-end machine learning system to predict whether a customer is likely to churn.

The project covers the complete workflow:

Dataset → EDA → Feature Engineering → Model Training → Evaluation → Prediction → FastAPI → Docker → Web Frontend

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Pydantic
- Joblib
- HTML
- CSS
- JavaScript
- Docker
- Git & GitHub

## Machine Learning

Two classification models were trained:

1. Logistic Regression
2. Random Forest

The Logistic Regression model is used by the prediction API.

## Features

The system performs:

- Data cleaning
- Exploratory Data Analysis
- Feature engineering
- Categorical encoding using OneHotEncoder
- Train-test splitting
- Machine learning model training
- Model evaluation
- Churn probability prediction
- REST API development
- Docker containerization
- Web-based prediction interface

## Feature Engineering

Additional features were created:

- Average monthly spend
- Total number of services
- New customer indicator
- Internet service indicator
- Customer support indicator

## API Endpoints

### Home

`GET /`

Returns information about the application.

### Health Check

`GET /health`

Checks whether the API and model are running.

### Model Information

`GET /model-info`

Returns information about the machine learning model.

### Prediction

`POST /predict`

Accepts customer information and returns:

- Prediction
- Churn probability
- Churn probability percentage

## Web Frontend

A simple web interface was created using:

- HTML
- CSS
- JavaScript

Users can enter customer information through a form and receive the churn prediction without manually entering JSON.

## Docker

The FastAPI application is containerized using Docker.

The application can be run using:

`docker run -d -p 8000:8000 --name customer-churn-container customer-churn-api`

The API is then available at:

`http://127.0.0.1:8000`

## Project Structure

customer-churn-ai/

├── app/

│   ├── main.py

│   └── static/

│       ├── index.html

│       ├── style.css

│       └── script.js

├── data/

│   └── customer_churn.csv

├── models/

│   ├── evaluation_results.csv

│   └── logistic_churn_model.pkl

├── notebooks/

│   ├── 01_eda.ipynb

│   └── 02_model_evaluation.ipynb

├── src/

│   ├── preprocessing.py

│   ├── train.py

│   └── predict.py

├── .dockerignore

├── .gitignore

├── Dockerfile

├── README.md

└── requirements.txt

The Random Forest model is generated locally but excluded from GitHub because of its large file size.

## How to Run

### 1. Clone the repository

`git clone https://github.com/Jayita26/customer-churn-ai.git`

### 2. Open the project

`cd customer-churn-ai`

### 3. Install dependencies

`pip install -r requirements.txt`

### 4. Run the API

`uvicorn app.main:app --reload`

### 5. Open the application

`http://127.0.0.1:8000`

### 6. Open API documentation

`http://127.0.0.1:8000/docs`

## Future Improvements

- Add authentication
- Add database integration
- Add cloud deployment
- Add monitoring and logging
- Add automated testing
- Improve model performance
- Add CI/CD pipeline

## Author

Jayita Maiti

M.Sc. Data Science