from pathlib import Path

import pandas as pd

from patient_risk.evaluation import evaluate_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_readmissions.csv"


def test_synthetic_dataset_has_valid_rows_features_and_both_classes():
    data = pd.read_csv(DATA_PATH)

    assert len(data) == 240
    assert "readmitted_30d" in data.columns
    assert data["readmitted_30d"].isin([0, 1]).all()
    assert data["readmitted_30d"].nunique() == 2
    assert 0 < data["readmitted_30d"].mean() < 0.5
    assert data.drop(columns="readmitted_30d").select_dtypes(
        include="number"
    ).shape[1] == len(data.columns) - 1
    assert data.drop(columns="readmitted_30d").isna().any().any()


def test_model_evaluates_on_the_synthetic_dataset():
    data = pd.read_csv(DATA_PATH)

    result = evaluate_model(
        data,
        target_column="readmitted_30d",
        test_size=0.25,
        random_state=42,
        calibration_bins=5,
    )

    assert result["test_size"] == 60
    assert 0.0 <= result["brier_score"] <= 1.0
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert sum(row["count"] for row in result["calibration"]) == 60
    assert all(
        0.0 <= row["mean_predicted_probability"] <= 1.0
        and 0.0 <= row["observed_readmission_rate"] <= 1.0
        for row in result["calibration"]
    )


def test_synthetic_dataset_is_reproducible():
    from scripts.generate_synthetic_data import generate_dataset

    first = generate_dataset()
    second = generate_dataset()

    pd.testing.assert_frame_equal(first, second)
