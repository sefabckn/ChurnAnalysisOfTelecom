import pandas as pd

from src.churn_analysis.pipeline import build_model, load_data, predict_customer


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenure": [1, 24, 3, 48],
            "MonthlyCharges": [90, 60, 95, 40],
            "TotalCharges": ["90", "1440", " ", "1920"],
            "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
            "InternetService": ["Fiber optic", "DSL", "Fiber optic", "DSL"],
            "Churn": ["Yes", "No", "Yes", "No"],
        }
    )


def test_load_data_normalizes_total_charges():
    cleaned = load_data(__import__("io").StringIO(sample_data().to_csv(index=False)))
    assert cleaned["TotalCharges"].isna().sum() == 0
    assert cleaned.loc[2, "TotalCharges"] == 0


def test_model_scores_customer():
    frame = sample_data()
    model = build_model(frame)
    scored = predict_customer(model, frame.drop(columns=["Churn"]).iloc[:1])
    assert scored.loc[0, "Prediction"] in {"Yes", "No"}
    assert 0 <= scored.loc[0, "Churn probability"] <= 1