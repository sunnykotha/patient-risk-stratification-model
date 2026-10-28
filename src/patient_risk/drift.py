import pandas as pd


def monitor_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    mean_shift_threshold: float = 0.20,
    missing_rate_shift_threshold: float = 0.10,
) -> dict:
    """Compare numeric feature means and missing rates across two batches."""
    if not isinstance(reference, pd.DataFrame) or not isinstance(current, pd.DataFrame):
        raise ValueError("Reference and current data must be pandas DataFrames")
    if reference.empty or current.empty:
        raise ValueError("Reference and current data must not be empty")
    if list(reference.columns) != list(current.columns):
        raise ValueError("Reference and current columns must match in name and order")
    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in reference.dtypes
    ) or not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in current.dtypes
    ):
        raise ValueError("Drift monitoring requires numeric features")
    for name, value in (
        ("mean_shift_threshold", mean_shift_threshold),
        ("missing_rate_shift_threshold", missing_rate_shift_threshold),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    report = {}
    for column in reference.columns:
        ref_values = reference[column]
        cur_values = current[column]

        ref_missing = float(ref_values.isna().mean())
        cur_missing = float(cur_values.isna().mean())
        missing_shift = abs(cur_missing - ref_missing)

        ref_mean = ref_values.mean()
        cur_mean = cur_values.mean()

        if pd.isna(ref_mean) or pd.isna(cur_mean):
            relative_mean_shift = None
            mean_shifted = False
        else:
            scale = max(abs(float(ref_mean)), 1.0)
            relative_mean_shift = abs(float(cur_mean) - float(ref_mean)) / scale
            mean_shifted = relative_mean_shift > mean_shift_threshold

        report[column] = {
            "reference_missing_rate": ref_missing,
            "current_missing_rate": cur_missing,
            "missing_rate_shift": missing_shift,
            "reference_mean": (
                None if pd.isna(ref_mean) else float(ref_mean)
            ),
            "current_mean": (
                None if pd.isna(cur_mean) else float(cur_mean)
            ),
            "relative_mean_shift": relative_mean_shift,
            "drift_detected": (
                mean_shifted
                or missing_shift > missing_rate_shift_threshold
            ),
        }

    return {
        "drift_detected": any(
            item["drift_detected"] for item in report.values()
        ),
        "features": report,
    }
