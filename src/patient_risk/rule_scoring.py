def score_risk_points(age, prior_admissions_12m, sodium, weight_missing):
    """Calculate the task's rule-based example score and missing-weight flag."""
    if isinstance(age, bool) or not isinstance(age, (int, float)):
        raise ValueError("Age must be numeric")
    if not 0 <= age <= 120:
        raise ValueError("Age must be between 0 and 120")

    if (
        isinstance(prior_admissions_12m, bool)
        or not isinstance(prior_admissions_12m, int)
        or prior_admissions_12m < 0
    ):
        raise ValueError("Prior admissions must be a non-negative integer")

    if sodium is not None:
        if isinstance(sodium, bool) or not isinstance(sodium, (int, float)):
            raise ValueError("Sodium must be numeric or None")
        if not 100 <= sodium <= 200:
            raise ValueError("Sodium must be within the supported range")

    if not isinstance(weight_missing, bool):
        raise ValueError("weight_missing must be a boolean")

    factors = {}
    factors["age"] = 2 if age >= 75 else 0
    factors["prior_admissions_12m"] = (
        4 if prior_admissions_12m >= 3 else 0
    )
    factors["sodium"] = 2 if sodium is not None and sodium < 135 else 0

    total_points = sum(factors.values())
    tier = "high" if total_points >= 7 else (
        "medium" if total_points >= 4 else "low"
    )

    return {
        "total_points": total_points,
        "tier": tier,
        "factors": factors,
        "missing_indicators": {"weight_missing": weight_missing},
    }
