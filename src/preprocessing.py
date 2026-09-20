import pandas as pd
import numpy as np


def load_data(file_path):
    """
    Load the customer churn dataset.
    """
    return pd.read_csv(file_path)


def clean_data(df):
    """
    Clean raw customer churn data.
    """

    df = df.copy()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Replace missing TotalCharges with 0
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    return df


def create_features(df):
    """
    Create engineered features.
    """

    df = df.copy()

    # Average historical monthly spending
    df["average_monthly_spend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    # Count services with explicit Yes
    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["total_services"] = (
        df[service_columns] == "Yes"
    ).sum(axis=1)

    # New customer indicator
    df["is_new_customer"] = (
        df["tenure"] <= 6
    ).astype(int)

    # Internet indicator
    df["has_internet"] = (
        df["InternetService"] != "No"
    ).astype(int)

    # Support indicator
    df["has_support"] = (
        (df["OnlineSecurity"] == "Yes") |
        (df["TechSupport"] == "Yes")
    ).astype(int)

    return df