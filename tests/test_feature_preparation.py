from datetime import datetime

import pandas as pd
import pytest

from patient_risk.feature_preparation import prepare_feature_set
from patient_risk.imputation import fit_cohort_medians
from patient_risk.models import FeatureSet


def test_feature_set_is_imputed_using_training_medians():
    training = pd.DataFrame(
        {
            "age": [50.0, 70.0, 90.0],
            "sodium": [130.0, 140.0, 150.0],
        }
    )
    medians = fit_cohort_medians(training)
    feature_set = FeatureSet(
        patient_id="p1",
        prediction_time=datetime(2026, 1, 1),
        values={"age": 78.0, "sodium": 0.0},
        missing_indicators={"age": False, "sodium": True},
    )

    prepared = prepare_feature_set(feature_set, medians)

    assert prepared.loc[0, "age"] == 78.0
    assert prepared.loc[0, "sodium"] == 140.0
    assert prepared.loc[0, "age_missing"] == 0
    assert prepared.loc[0, "sodium_missing"] == 1


def test_missing_training_feature_is_filled_from_training_median():
    training = pd.DataFrame({"age": [50.0, 70.0, 90.0], "sodium": [130.0, 140.0, 150.0]})
    medians = fit_cohort_medians(training)
    feature_set = FeatureSet(
        patient_id="p1",
        prediction_time=datetime(2026, 1, 1),
        values={"age": 78.0},
        missing_indicators={"age": False},
    )

    prepared = prepare_feature_set(feature_set, medians)

    assert prepared.loc[0, "sodium"] == 140.0
    assert prepared.loc[0, "sodium_missing"] == 1


def test_preparation_does_not_modify_feature_set():
    training = pd.DataFrame({"age": [50.0, 70.0, 90.0]})
    medians = fit_cohort_medians(training)
    feature_set = FeatureSet(
        patient_id="p1",
        prediction_time=datetime(2026, 1, 1),
        values={"age": 78.0},
        missing_indicators={"age": False},
    )
    original_values = dict(feature_set.values)
    original_flags = dict(feature_set.missing_indicators)

    prepare_feature_set(feature_set, medians)

    assert feature_set.values == original_values
    assert feature_set.missing_indicators == original_flags


def test_preparation_rejects_non_feature_set_input():
    with pytest.raises(ValueError, match="FeatureSet"):
        prepare_feature_set({"age": 78}, pd.Series({"age": 70.0}))

def test_indicator_columns_in_medians_do_not_create_duplicate_indicators():
    training = pd.DataFrame({
        "age": [50.0, 70.0, 90.0],
        "sodium": [130.0, 140.0, 150.0],
        "age_missing": [0, 0, 0],
        "sodium_missing": [0, 0, 0],
    })
    medians = fit_cohort_medians(training)
    feature_set = FeatureSet(
        patient_id="p1",
        prediction_time=datetime(2026, 1, 1),
        values={"age": 78.0, "sodium": 0.0},
        missing_indicators={"age": False, "sodium": True},
    )

    prepared = prepare_feature_set(feature_set, medians)

    assert list(prepared.columns) == [
        "age", "sodium", "age_missing", "sodium_missing"
    ]
    assert prepared.loc[0, "age"] == 78.0
    assert prepared.loc[0, "sodium"] == 140.0
    assert prepared.loc[0, "age_missing"] == 0
    assert prepared.loc[0, "sodium_missing"] == 1
