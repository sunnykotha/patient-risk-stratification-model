import re

import pandas as pd

from .models import Explanation, RiskScore


def assign_risk_tier(probability):
    """Map a predicted probability to an initial risk tier."""
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise ValueError("Probability must be numeric")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be between 0 and 1")

    if probability < 0.30:
        return "low"
    if probability < 0.60:
        return "medium"
    return "high"


def _factor_label(feature_name):
    if feature_name.startswith("missingindicator_"):
        name = feature_name[len("missingindicator_"):]
        return f"Missing {name.replace('_', ' ')}"
    if feature_name.endswith("_missing"):
        name = feature_name[:-len("_missing")]
        return f"Missing {name.replace('_', ' ')}"
    return feature_name.replace("_", " ")


def explain_prediction(model, feature_row, top_n=3):
    """Rank feature contributions relative to the training-data baseline."""
    if not isinstance(feature_row, pd.DataFrame) or len(feature_row) != 1:
        raise ValueError("Feature input must be a one-row pandas DataFrame")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")

    try:
        imputer = model.named_steps["imputer"]
        classifier = model.named_steps["classifier"]
    except (AttributeError, KeyError):
        raise ValueError("Model must be a fitted imputer/classifier pipeline") from None

    if not hasattr(classifier, "coef_"):
        raise ValueError("Model must be fitted before explaining predictions")

    expected_columns = list(imputer.feature_names_in_)
    if list(feature_row.columns) != expected_columns:
        raise ValueError("Feature columns must match the training columns and order")

    transformed = imputer.transform(feature_row)[0]
    feature_names = imputer.get_feature_names_out()
    medians = list(imputer.statistics_)

    # Compare observed/imputed values against training medians.
    # Missingness indicators use zero as the baseline.
    baseline = medians + [0.0] * (len(feature_names) - len(medians))
    coefficients = classifier.coef_[0]

    contributions = []
    for name, value, base, coefficient in zip(
        feature_names, transformed, baseline, coefficients
    ):
        contribution = float(coefficient * (value - base))
        if contribution != 0.0:
            contributions.append((name, contribution))

    contributions.sort(key=lambda item: abs(item[1]), reverse=True)
    return tuple(
        _factor_label(name)
        for name, _ in contributions[:top_n]
    )


def score_patient(model, patient_id, prediction_time, feature_row, top_n=3):
    """Return a RiskScore and its leading model-based explanations."""
    if not isinstance(patient_id, str) or not patient_id.strip():
        raise ValueError("Patient ID must not be empty")

    if not isinstance(feature_row, pd.DataFrame) or len(feature_row) != 1:
        raise ValueError("Feature input must be a one-row pandas DataFrame")

    try:
        probability = float(model.predict_proba(feature_row)[0][1])
    except (AttributeError, IndexError, ValueError) as exc:
        raise ValueError("Could not produce a readmission probability") from exc

    tier = assign_risk_tier(probability)
    factors = explain_prediction(model, feature_row, top_n)

    score = RiskScore(
        patient_id=patient_id,
        prediction_time=prediction_time,
        probability=probability,
        tier=tier,
    )
    explanation = Explanation(
        patient_id=patient_id,
        top_factors=factors,
    )
    return score, explanation
