import pandas as pd
import pytest

from patient_risk.imputation import (
    fit_cohort_medians,
    transform_with_medians,
)


def test_imputes_missing_value_with_training_median():
    training = pd.DataFrame({"age": [60.0, 80.0, 70.0]})
    medians = fit_cohort_medians(training)

    result = transform_with_medians(
        pd.DataFrame({"age": [None, 90.0]}), medians
    )

    assert result["age"].tolist() == [70.0, 90.0]
    assert result["age_missing"].tolist() == [1, 0]


def test_all_features_get_missing_indicators():
    training = pd.DataFrame({
        "age": [60.0, 80.0],
        "sodium": [130.0, 140.0],
    })
    medians = fit_cohort_medians(training)
    result = transform_with_medians(
        pd.DataFrame({"age": [None], "sodium": [None]}), medians
    )

    assert result.loc[0, "age_missing"] == 1
    assert result.loc[0, "sodium_missing"] == 1


def test_test_data_does_not_change_training_median():
    training = pd.DataFrame({"age": [60.0, 80.0]})
    medians = fit_cohort_medians(training)

    test_frame = pd.DataFrame({"age": [10000.0, None]})
    result = transform_with_medians(test_frame, medians)

    assert medians["age"] == 70.0
    assert result.loc[1, "age"] == 70.0
    assert result.loc[0, "age"] == 10000.0


def test_rejects_empty_training_data():
    with pytest.raises(ValueError, match="must not be empty"):
        fit_cohort_medians(pd.DataFrame({"age": []}))


def test_rejects_all_missing_training_column():
    with pytest.raises(ValueError, match="all-missing"):
        fit_cohort_medians(pd.DataFrame({"age": [None, None]}))


def test_rejects_missing_expected_feature():
    medians = pd.Series({"age": 70.0})
    with pytest.raises(ValueError, match="missing training features"):
        transform_with_medians(pd.DataFrame({"weight": [80.0]}), medians)


def test_original_input_is_not_modified():
    training = pd.DataFrame({"age": [60.0, 80.0]})
    medians = fit_cohort_medians(training)
    source = pd.DataFrame({"age": [None]})

    transform_with_medians(source, medians)

    assert pd.isna(source.loc[0, "age"])