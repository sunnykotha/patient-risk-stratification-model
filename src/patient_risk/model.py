import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_model(training_frame, target_column="readmitted_30d"):
    """Fit a readmission model using training data only."""
    if not isinstance(training_frame, pd.DataFrame):
        raise ValueError("Training data must be a pandas DataFrame")

    if training_frame.empty:
        raise ValueError("Training data must not be empty")

    if target_column not in training_frame.columns:
        raise ValueError(f"Missing target column: {target_column}")

    features = training_frame.drop(columns=[target_column])
    target = training_frame[target_column]

    if features.empty:
        raise ValueError("Training data must contain feature columns")

    if features.isna().all().any():
        columns = features.columns[features.isna().all()].tolist()
        raise ValueError(f"Cannot train with all-missing features: {columns}")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise ValueError("Training features must be numeric")

    if target.isna().any():
        raise ValueError("Target labels must not be missing")

    if not target.isin([0, 1]).all():
        raise ValueError("Target labels must be binary: 0 or 1")

    if target.nunique() != 2:
        raise ValueError("Training data must contain both target classes: 0 and 1")

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )
    pipeline.fit(features, target.astype(int))
    return pipeline
