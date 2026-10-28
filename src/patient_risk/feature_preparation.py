import pandas as pd

from .models import FeatureSet
from .imputation import transform_with_medians


def prepare_feature_set(feature_set: FeatureSet, medians: pd.Series) -> pd.DataFrame:
    """Prepare one FeatureSet using training medians fitted on original features."""
    if not isinstance(feature_set, FeatureSet):
        raise ValueError("feature_set must be a FeatureSet")

    # Missing-indicator columns are output features, not features to impute again.
    feature_names = [
        name for name in medians.index
        if not str(name).endswith("_missing")
    ]
    feature_medians = medians.loc[feature_names]

    row_values = dict(feature_set.values)
    missing_flags = dict(feature_set.missing_indicators)

    for name, is_missing in missing_flags.items():
        if is_missing:
            row_values[name] = float("nan")

    for name in feature_names:
        if name not in row_values:
            row_values[name] = float("nan")
            missing_flags[name] = True
        else:
            missing_flags.setdefault(name, False)

    row = pd.DataFrame([row_values])
    row = row[feature_names]

    # Imputation also generates one missing indicator per original feature.
    return transform_with_medians(row, feature_medians)
