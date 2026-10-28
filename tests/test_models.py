from datetime import datetime, timedelta, timezone

import pytest

from patient_risk.models import (
    Admission,
    Diagnosis,
    Explanation,
    FeatureSet,
    LabResult,
    Patient,
    RiskScore,
    VitalSign,
)


NOW = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)


def test_patient_accepts_valid_age():
    assert Patient("P001", 78).age == 78


@pytest.mark.parametrize("age", [-1, 121, True, 78.5])
def test_patient_rejects_invalid_age(age):
    with pytest.raises(ValueError):
        Patient("P001", age)


def test_admission_accepts_valid_dates():
    admission = Admission("A001", "P001", NOW, NOW + timedelta(days=3))
    assert admission.discharged_at > admission.admitted_at


def test_admission_rejects_discharge_before_admission():
    with pytest.raises(ValueError):
        Admission("A001", "P001", NOW, NOW - timedelta(hours=1))


def test_admission_allows_unknown_discharge():
    assert Admission("A001", "P001", NOW).discharged_at is None


def test_lab_result_accepts_missing_value():
    result = LabResult("L001", "P001", "sodium", None, "mmol/L", NOW)
    assert result.value is None


def test_lab_result_rejects_infinite_value():
    with pytest.raises(ValueError):
        LabResult("L001", "P001", "sodium", float("inf"), "mmol/L", NOW)


def test_vital_sign_accepts_measurement():
    vital = VitalSign("V001", "P001", "heart_rate", 80, "bpm", NOW)
    assert vital.value == 80


def test_vital_sign_rejects_nan():
    with pytest.raises(ValueError):
        VitalSign("V001", "P001", "heart_rate", float("nan"), "bpm", NOW)


def test_diagnosis_requires_code_and_description():
    assert Diagnosis("I10", "Hypertension").diagnosis_code == "I10"
    with pytest.raises(ValueError):
        Diagnosis("", "Hypertension")


def test_feature_set_requires_indicator_for_each_feature():
    features = FeatureSet(
        "P001", NOW,
        {"age": 78.0, "sodium": 128.0},
        {"age": False, "sodium": False},
    )
    assert features.values["sodium"] == 128.0


def test_feature_set_rejects_missing_indicator_mismatch():
    with pytest.raises(ValueError):
        FeatureSet("P001", NOW, {"age": 78.0}, {})


def test_risk_score_accepts_probability_and_tier():
    score = RiskScore("P001", NOW, 0.8, "high")
    assert score.probability == 0.8
    assert score.tier == "high"


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_risk_score_rejects_invalid_probability(probability):
    with pytest.raises(ValueError):
        RiskScore("P001", NOW, probability, "high")


def test_risk_score_rejects_unknown_tier():
    with pytest.raises(ValueError):
        RiskScore("P001", NOW, 0.5, "critical")


def test_explanation_keeps_top_factors():
    explanation = Explanation("P001", ("prior admissions", "low sodium"))
    assert len(explanation.top_factors) == 2


def test_explanation_rejects_invalid_factor():
    with pytest.raises(ValueError):
        Explanation("P001", ("prior admissions", ""))
