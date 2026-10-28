from datetime import datetime

import pandas as pd
import pytest

from patient_risk.model import train_model
from patient_risk.risk import assign_risk_tier, explain_prediction, score_patient


def sample_data():
    return pd.DataFrame(
        {
            "age": [45, 78, 62, 81, 39, 70, 55, 86],
            "prior_admissions_12m": [0, 3, 1, 4, 0, 2, 1, 5],
            "sodium": [140.0, 128.0, 137.0, None, 142.0, 133.0, None, 126.0],
            "readmitted_30d": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, "low"),
        (0.2999, "low"),
        (0.30, "medium"),
        (0.5999, "medium"),
        (0.60, "high"),
        (1.0, "high"),
    ],
)
def test_assigns_risk_tiers(probability, expected):
    assert assign_risk_tier(probability) == expected


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), True, "0.5"])
def test_rejects_invalid_probabilities(probability):
    with pytest.raises(ValueError):
        assign_risk_tier(probability)


def test_explanation_returns_at_most_requested_number_of_factors():
    frame = sample_data()
    model = train_model(frame)
    row = frame.drop(columns=["readmitted_30d"]).iloc[[1]]

    factors = explain_prediction(model, row, top_n=2)

    assert len(factors) <= 2
    assert all(isinstance(factor, str) and factor for factor in factors)


def test_explanation_rejects_wrong_feature_columns():
    model = train_model(sample_data())
    row = pd.DataFrame({"age": [78], "sodium": [128.0]})

    with pytest.raises(ValueError, match="match the training columns"):
        explain_prediction(model, row)


def test_score_patient_returns_score_and_explanation():
    frame = sample_data()
    model = train_model(frame)
    row = frame.drop(columns=["readmitted_30d"]).iloc[[1]]
    prediction_time = datetime(2026, 1, 15, 10, 0)

    score, explanation = score_patient(
        model, "patient-1", prediction_time, row, top_n=3
    )

    assert score.patient_id == "patient-1"
    assert score.prediction_time == prediction_time
    assert 0 <= score.probability <= 1
    assert score.tier in {"low", "medium", "high"}
    assert explanation.patient_id == "patient-1"
    assert len(explanation.top_factors) <= 3


def test_score_patient_rejects_empty_patient_id():
    frame = sample_data()
    model = train_model(frame)
    row = frame.drop(columns=["readmitted_30d"]).iloc[[1]]

    with pytest.raises(ValueError, match="Patient ID"):
        score_patient(model, " ", datetime(2026, 1, 15), row)


def test_score_patient_rejects_multiple_rows():
    frame = sample_data()
    model = train_model(frame)
    rows = frame.drop(columns=["readmitted_30d"]).iloc[:2]

    with pytest.raises(ValueError, match="one-row"):
        score_patient(model, "patient-1", datetime(2026, 1, 15), rows)
