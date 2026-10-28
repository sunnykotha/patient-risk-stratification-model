import pandas as pd

from patient_risk.feature_preparation import prepare_feature_set
from patient_risk.models import FeatureSet
from patient_risk.model import train_model
from patient_risk.risk import score_patient


def test_training_to_patient_risk_score_end_to_end():
    training = pd.DataFrame({
        "age": [25, 35, 45, 55, 65, 75, 85, 90],
        "prior_admissions_12m": [0, 0, 1, 1, 2, 3, 4, 5],
        "sodium": [140, 139, 138, 137, 135, 130, 128, 125],
        "readmitted_30d": [0, 0, 0, 0, 1, 1, 1, 1],
    })

    model = train_model(training)
    medians = training.drop(columns=["readmitted_30d"]).median()

    patient_features = FeatureSet(
        patient_id="patient-001",
        prediction_time=pd.Timestamp("2026-10-28T10:00:00"),
        values={
            "age": 78.0,
            "prior_admissions_12m": 3.0,
            "sodium": 128.0,
        },
        missing_indicators={
            "age": False,
            "prior_admissions_12m": False,
            "sodium": False,
        },
    )

    prepared = prepare_feature_set(patient_features, medians)

    # Supply the original features expected by the trained model.
    feature_columns = list(model.named_steps["imputer"].feature_names_in_)
    feature_row = prepared.reindex(columns=feature_columns)

    score, explanation = score_patient(
        model=model,
        patient_id="patient-001",
        prediction_time=pd.Timestamp("2026-10-28T10:00:00"),
        feature_row=feature_row,
    )

    assert 0.0 <= score.probability <= 1.0
    assert score.tier in {"low", "medium", "high"}
    assert score.patient_id == "patient-001"
    assert explanation.patient_id == "patient-001"
    assert len(explanation.top_factors) > 0
