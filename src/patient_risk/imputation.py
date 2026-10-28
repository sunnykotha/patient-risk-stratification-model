import pandas as pd


def fit_cohort_medians(training_frame: pd.DataFrame) -> pd.Series:
    """Learn numeric feature medians from training data only."""
    if training_frame.empty:
        raise ValueError("Training data must not be empty")

    all_missing = [
        column for column in training_frame.columns
        if training_frame[column].isna().all()
    ]
    if all_missing:
        raise ValueError(
            f"Cannot impute all-missing training columns: {all_missing}"
        )

    numeric = training_frame.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("Training data must contain numeric features")

    if len(numeric.columns) != len(training_frame.columns):
        raise ValueError("Training data must contain only numeric features")

    medians = numeric.median()
    return medians


def transform_with_medians(
    frame: pd.DataFrame, medians: pd.Series
) -> pd.DataFrame:
    """Impute with training medians and add a flag for every numeric column."""
    result = frame.copy()
    numeric_columns = list(medians.index)

    missing_columns = set(numeric_columns) - set(result.columns)
    if missing_columns:
        raise ValueError(
            f"Input is missing training features: {sorted(missing_columns)}"
        )

    for column in numeric_columns:
        result[f"{column}_missing"] = result[column].isna().astype(int)
        result[column] = result[column].fillna(medians[column])

    return result