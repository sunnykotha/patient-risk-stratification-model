import pandas as pd
import pytest

from patient_risk.evaluation import evaluate_model


def evaluation_data():
    return pd.DataFrame(
        {
            "age": [
                45, 78, 62, 81, 39, 70, 55, 86,
                49, 75, 67, 83, 42, 72, 58, 89,
                51, 77, 64, 80, 36, 69, 60, 91,
            ],
            "prior_admissions_12m": [
                0, 3, 1, 4, 0, 2, 1, 5,
                0, 3, 1, 4, 0, 2, 1, 5,
                0, 3, 1, 4, 0, 2, 1, 5,
            ],
            "sodium": [
                140, 128, 137, None, 142, 133, None, 126,
                139, 129, 136, 127, 143, 132, 138, 125,
                141, 130, 135, None, 144, 134, 139, 124,
            ],
            "readmitted_30d": [
                0, 1, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1, 0, 1,
            ],
        }
    )


def test_evaluation_returns_held_out_metrics_and_calibration():
    result = evaluate_model(evaluation_data())

    assert 0 <= result["brier_score"] <= 1
    assert 0 <= result["roc_auc"] <= 1
    assert result["test_size"] == 6
    assert isinstance(result["calibration"], list)
    assert result["calibration"]
    assert all(row["count"] > 0 for row in result["calibration"])
    assert all(
        0 <= row["mean_predicted_probability"] <= 1
        and 0 <= row["observed_readmission_rate"] <= 1
        for row in result["calibration"]
    )


def test_evaluation_rejects_missing_target():
    frame = evaluation_data().drop(columns=["readmitted_30d"])

    with pytest.raises(ValueError, match="Missing target"):
        evaluate_model(frame)


def test_evaluation_rejects_invalid_target_values():
    frame = evaluation_data()
    frame.loc[0, "readmitted_30d"] = 3

    with pytest.raises(ValueError, match="binary"):
        evaluate_model(frame)


def test_evaluation_rejects_only_one_target_class():
    frame = evaluation_data()
    frame["readmitted_30d"] = 0

    with pytest.raises(ValueError, match="both target classes"):
        evaluate_model(frame)


def test_evaluation_rejects_invalid_test_size():
    with pytest.raises(ValueError, match="test_size"):
        evaluate_model(evaluation_data(), test_size=1)


def test_evaluation_rejects_invalid_bin_count():
    with pytest.raises(ValueError, match="calibration_bins"):
        evaluate_model(evaluation_data(), calibration_bins=0)


def test_evaluation_rejects_non_dataframe_input():
    with pytest.raises(ValueError, match="pandas DataFrame"):
        evaluate_model([{"age": 60, "readmitted_30d": 1}])


def test_evaluation_does_not_mutate_input_data():
    frame = evaluation_data()
    original = frame.copy(deep=True)

    evaluate_model(frame)

    pd.testing.assert_frame_equal(frame, original)

def test_calibration_counts_cover_all_held_out_predictions():
    result = evaluate_model(evaluation_data(), calibration_bins=4)

    assert sum(row["count"] for row in result["calibration"]) == result["test_size"]
    for row in result["calibration"]:
        assert row["count"] > 0
        assert 0.0 <= row["mean_predicted_probability"] <= 1.0
        assert 0.0 <= row["observed_readmission_rate"] <= 1.0


def test_calibration_reports_predicted_and_observed_rates_separately():
    result = evaluate_model(evaluation_data())

    for row in result["calibration"]:
        assert "mean_predicted_probability" in row
        assert "observed_readmission_rate" in row
        assert row["mean_predicted_probability"] != row["observed_readmission_rate"] or (
            0.0 <= row["observed_readmission_rate"] <= 1.0
        )
