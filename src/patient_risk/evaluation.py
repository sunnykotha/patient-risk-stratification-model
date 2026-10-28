import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .model import train_model


def evaluate_model(
    data,
    target_column="readmitted_30d",
    test_size=0.25,
    random_state=42,
    calibration_bins=5,
):
    """Evaluate a readmission pipeline on a held-out test set."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("Data must not be empty")
    if target_column not in data.columns:
        raise ValueError(f"Missing target column: {target_column}")
    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)):
        raise ValueError("test_size must be numeric")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1")
    if (
        isinstance(calibration_bins, bool)
        or not isinstance(calibration_bins, int)
        or calibration_bins < 1
    ):
        raise ValueError("calibration_bins must be a positive integer")

    features = data.drop(columns=[target_column])
    target = data[target_column]

    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("Target labels must be non-missing binary values: 0 or 1")
    if target.nunique() != 2:
        raise ValueError("Evaluation data must contain both target classes")

    class_counts = target.value_counts()
    if class_counts.min() < 2:
        raise ValueError("Each target class needs at least two examples")

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target.astype(int),
            test_size=test_size,
            random_state=random_state,
            stratify=target.astype(int),
        )
    except ValueError as exc:
        raise ValueError(f"Could not create a stratified holdout: {exc}") from exc

    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Training and test partitions must each contain both classes")

    model = train_model(
        x_train.assign(**{target_column: y_train})
    )
    probabilities = model.predict_proba(x_test)[:, 1]

    brier = float(brier_score_loss(y_test, probabilities))
    auc = float(roc_auc_score(y_test, probabilities))

    result_rows = []
    edges = [i / calibration_bins for i in range(calibration_bins + 1)]
    groups = pd.cut(
        probabilities,
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    )

    calibration_frame = pd.DataFrame(
        {
            "probability": probabilities,
            "actual": y_test.to_numpy(),
            "bin": groups,
        }
    )

    for interval, group in calibration_frame.groupby("bin", observed=True):
        result_rows.append(
            {
                "bin": str(interval),
                "count": int(len(group)),
                "mean_predicted_probability": float(group["probability"].mean()),
                "observed_readmission_rate": float(group["actual"].mean()),
            }
        )

    return {
        "model": model,
        "brier_score": brier,
        "roc_auc": auc,
        "test_size": int(len(y_test)),
        "calibration": result_rows,
    }
