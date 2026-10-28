import pytest

from patient_risk.rule_scoring import score_risk_points


def test_task_worked_example_scores_eight_and_high_risk():
    result = score_risk_points(
        age=78,
        prior_admissions_12m=3,
        sodium=128,
        weight_missing=True,
    )

    assert result["factors"] == {
        "age": 2,
        "prior_admissions_12m": 4,
        "sodium": 2,
    }
    assert result["total_points"] == 8
    assert result["tier"] == "high"
    assert result["missing_indicators"]["weight_missing"] is True


def test_normal_sodium_does_not_add_low_sodium_points():
    result = score_risk_points(78, 3, 140, True)

    assert result["factors"]["sodium"] == 0
    assert result["total_points"] == 6


def test_missing_sodium_does_not_add_low_sodium_points():
    result = score_risk_points(78, 3, None, False)

    assert result["factors"]["sodium"] == 0
    assert result["total_points"] == 6


def test_lower_risk_example():
    result = score_risk_points(45, 0, 140, False)

    assert result["total_points"] == 0
    assert result["tier"] == "low"


@pytest.mark.parametrize("age", [-1, 121, True, "78"])
def test_rejects_invalid_age(age):
    with pytest.raises(ValueError, match="Age"):
        score_risk_points(age, 0, 140, False)


@pytest.mark.parametrize("admissions", [-1, 1.5, True])
def test_rejects_invalid_prior_admission_count(admissions):
    with pytest.raises(ValueError, match="Prior admissions"):
        score_risk_points(78, admissions, 128, False)


@pytest.mark.parametrize("sodium", [99, 201, float("nan"), True, "128"])
def test_rejects_invalid_sodium(sodium):
    with pytest.raises(ValueError, match="Sodium"):
        score_risk_points(78, 3, sodium, False)


def test_rejects_non_boolean_missing_weight_flag():
    with pytest.raises(ValueError, match="weight_missing"):
        score_risk_points(78, 3, 128, 1)
