"""Data preparation and model training for the telecom churn project."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "Churn"
REQUIRED_COLUMNS = {"tenure", "MonthlyCharges", "TotalCharges", TARGET}


def load_data(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Load a Telco CSV and normalize columns used by the project."""
    frame = pd.read_csv(source)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"CSV is missing required columns: {missing_names}")
    return prepare_data(frame)


def prepare_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy without mutating the caller's DataFrame."""
    cleaned = frame.copy()
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(0)
    cleaned[TARGET] = cleaned[TARGET].astype(str).str.strip().str.title()
    cleaned = cleaned[cleaned[TARGET].isin(["Yes", "No"])].copy()
    return cleaned


def build_model(frame: pd.DataFrame) -> Pipeline:
    """Fit and return a leakage-resistant random forest pipeline."""
    cleaned = prepare_data(frame)
    features = cleaned.drop(columns=[TARGET])
    target = cleaned[TARGET]
    categorical = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = features.select_dtypes(exclude=["object", "category", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
        ]
    )
    model.fit(features, target)
    return model


def predict_customer(model: Pipeline, customer: pd.DataFrame) -> pd.DataFrame:
    """Return churn labels and probabilities for one or more customers."""
    probabilities = model.predict_proba(customer)
    classes = list(model.classes_)
    churn_index = classes.index("Yes") if "Yes" in classes else 0
    return pd.DataFrame(
        {
            "Prediction": model.predict(customer),
            "Churn probability": probabilities[:, churn_index],
        }
    )
