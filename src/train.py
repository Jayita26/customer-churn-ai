import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from preprocessing import (
    load_data,
    clean_data,
    create_features
)

DATA_PATH = "C:/Users/Jayita/Documents/customer-churn-ai/data/customer_churn.csv"

df = load_data(DATA_PATH)

print("Original shape:", df.shape)

df = clean_data(df)

df = create_features(df)

df = df.drop("customerID", axis=1)

X = df.drop("Churn", axis=1)

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

stratify=y

categorical_columns = X.select_dtypes(
    include=["object"]
).columns

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns

print("Categorical columns:")
print(list(categorical_columns))

print("\nNumerical columns:")
print(list(numerical_columns))

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)

print("Training Logistic Regression...")

logistic_pipeline.fit(
    X_train,
    y_train
)

logistic_predictions = logistic_pipeline.predict(X_test)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

print(
    "Logistic Regression Accuracy:",
    logistic_accuracy
)

random_forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

print("Training Random Forest...")

random_forest_pipeline.fit(
    X_train,
    y_train
)

rf_predictions = random_forest_pipeline.predict(
    X_test
)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print(
    "Random Forest Accuracy:",
    rf_accuracy
)

print("Logistic Regression:", logistic_accuracy)
print("Random Forest:", rf_accuracy)

os.makedirs("models", exist_ok=True)

joblib.dump(
    logistic_pipeline,
    "models/logistic_churn_model.pkl"
)

joblib.dump(
    random_forest_pipeline,
    "models/random_forest_churn_model.pkl"
)

loaded_model = joblib.load(
    "models/logistic_churn_model.pkl"
)

test_prediction = loaded_model.predict(
    X_test.iloc[:5]
)

print("Predictions:", test_prediction)

probabilities = loaded_model.predict_proba(
    X_test.iloc[:5]
)

print(probabilities)

probabilities[:, 1]

sample = X_test.iloc[[0]]

prediction = loaded_model.predict(sample)[0]

probability = loaded_model.predict_proba(sample)[0][1]

print("Prediction:", prediction)
print("Churn probability:", probability)

