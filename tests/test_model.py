import pandas as pd
import pytest

from patient_risk.model import train_model


def training_data():
    return pd.DataFrame(
        {
            "age": [45, 78, 62, 81, 39, 70, 55, 86],
            "prior_admissions_12m": [0, 3, 1, 4, 0, 2, 1, 5],
            "sodium": [140.0, 128.0, 137.0, None, 142.0, 133.0, None, 126.0],
            "readmitted_30d": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def test_trains_pipeline_and_predicts_probabilities():
    model = train_model(training_data())
    features = training_data().drop(columns=["readmitted_30d"])

    probabilities = model.predict_proba(features)

    assert probabilities.shape == (8, 2)
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()
    assert (abs(probabilities.sum(axis=1) - 1) < 1e-9).all()


def test_pipeline_imputes_missing_feature_values():
    model = train_model(training_data())
    features = pd.DataFrame(
        {"age": [78], "prior_admissions_12m": [3], "sodium": [None]}
    )

    probabilities = model.predict_proba(features)

    assert probabilities.shape == (1, 2)


def test_rejects_empty_training_data():
    with pytest.raises(ValueError, match="must not be empty"):
        train_model(pd.DataFrame())


def test_rejects_missing_target_column():
    frame = training_data().drop(columns=["readmitted_30d"])

    with pytest.raises(ValueError, match="Missing target column"):
        train_model(frame)


def test_rejects_missing_target_labels():
    frame = training_data()
    frame.loc[0, "readmitted_30d"] = None

    with pytest.raises(ValueError, match="must not be missing"):
        train_model(frame)


def test_rejects_non_binary_target_labels():
    frame = training_data()
    frame.loc[0, "readmitted_30d"] = 2

    with pytest.raises(ValueError, match="must be binary"):
        train_model(frame)


def test_rejects_training_data_with_only_one_class():
    frame = training_data()
    frame["readmitted_30d"] = 0

    with pytest.raises(ValueError, match="both target classes"):
        train_model(frame)


def test_rejects_non_numeric_features():
    frame = training_data()
    frame["unit_name"] = "mmol/L"

    with pytest.raises(ValueError, match="must be numeric"):
        train_model(frame)


def test_rejects_all_missing_feature_column():
    frame = training_data()
    frame["weight"] = None

    with pytest.raises(ValueError, match="all-missing features"):
        train_model(frame)


def test_rejects_training_data_without_features():
    frame = pd.DataFrame({"readmitted_30d": [0, 1]})

    with pytest.raises(ValueError, match="feature columns"):
        train_model(frame)
