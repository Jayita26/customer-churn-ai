import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.preprocessing import (
    load_data,
    clean_data,
    create_features
)


def test_load_data():
    df = load_data("data/customer_churn.csv")

    assert df is not None
    assert len(df) > 0


def test_clean_data():
    df = load_data("data/customer_churn.csv")

    cleaned_df = clean_data(df)

    assert cleaned_df["TotalCharges"].dtype != "object"


def test_create_features():
    df = load_data("data/customer_churn.csv")

    df = clean_data(df)
    df = create_features(df)

    assert "average_monthly_spend" in df.columns
    assert "total_services" in df.columns
    assert "is_new_customer" in df.columns
    assert "has_internet" in df.columns
    assert "has_support" in df.columns