from datetime import datetime, timedelta, timezone

import pytest

from patient_risk.features import build_features
from patient_risk.models import Admission, LabResult, Patient, VitalSign


T = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
PATIENT = Patient("P001", 78)


def build(admissions=(), labs=(), vitals=(), **kwargs):
    return build_features(
        PATIENT, list(admissions), list(labs), list(vitals), T, **kwargs
    )


def test_basic_features_and_comorbidity():
    result = build(comorbidity_index=4)
    assert result.values["age"] == 78
    assert result.values["prior_admissions_12m"] == 0
    assert result.values["comorbidity_index"] == 4


def test_counts_completed_admissions_within_previous_12_months():
    admissions = [
        Admission("A1", "P001", T - timedelta(days=30),
                  T - timedelta(days=25)),
        Admission("A2", "P001", T - timedelta(days=200),
                  T - timedelta(days=195)),
    ]
    result = build(admissions=admissions)
    assert result.values["prior_admissions_12m"] == 2


def test_excludes_admissions_older_than_12_months():
    old = Admission(
        "OLD", "P001", T - timedelta(days=366), T - timedelta(days=360)
    )
    result = build(admissions=[old])
    assert result.values["prior_admissions_12m"] == 0


def test_excludes_current_admission_from_prior_count():
    current = Admission("CURRENT", "P001", T - timedelta(days=3), T)
    result = build(
        admissions=[current], current_admission_id="CURRENT"
    )
    assert result.values["prior_admissions_12m"] == 0
    assert result.values["length_of_stay_days"] == 3


def test_uses_latest_lab_available_at_prediction_time():
    labs = [
        LabResult("L1", "P001", "sodium", 135, "mmol/L", T - timedelta(days=1)),
        LabResult("L2", "P001", "sodium", 128, "mmol/L", T),
        LabResult("L3", "P001", "sodium", 140, "mmol/L", T + timedelta(minutes=1)),
    ]
    result = build(labs=labs)
    assert result.values["last_sodium"] == 128


def test_future_lab_is_not_used():
    future = LabResult(
        "L1", "P001", "hemoglobin", 15, "g/dL", T + timedelta(seconds=1)
    )
    result = build(labs=[future])
    assert result.missing_indicators["last_hemoglobin"] is True


def test_vital_trend_uses_only_previous_24_hours():
    vitals = [
        VitalSign("V1", "P001", "heart_rate", 70, "bpm",
                  T - timedelta(hours=12)),
        VitalSign("V2", "P001", "heart_rate", 80, "bpm", T),
        VitalSign("V3", "P001", "heart_rate", 200, "bpm",
                  T + timedelta(minutes=1)),
        VitalSign("V4", "P001", "heart_rate", 50, "bpm",
                  T - timedelta(hours=25)),
    ]
    result = build(vitals=vitals)
    assert result.values["heart_rate_trend_24h"] == 10


def test_single_vital_measurement_has_zero_trend():
    vital = VitalSign("V1", "P001", "weight", 70, "kg", T)
    result = build(vitals=[vital])
    assert result.values["weight_trend_24h"] == 0


def test_missing_labs_are_flagged():
    result = build()
    assert result.missing_indicators["last_sodium"] is True
    assert result.missing_indicators["last_hemoglobin"] is True


def test_future_admission_is_excluded():
    future = Admission(
        "A1", "P001", T + timedelta(days=1), T + timedelta(days=3)
    )
    result = build(admissions=[future])
    assert result.values["prior_admissions_12m"] == 0


def test_incompatible_timezones_are_rejected():
    naive_time = datetime(2026, 1, 10, 12, 0)
    admission = Admission("A1", "P001", naive_time, naive_time)
    with pytest.raises(ValueError, match="timezones"):
        build(admissions=[admission])


@pytest.mark.parametrize("index", [-1, float("nan"), float("inf")])
def test_invalid_comorbidity_index_is_rejected(index):
    with pytest.raises(ValueError):
        build(comorbidity_index=index)

def test_vital_trend_is_missing_with_only_one_measurement():
    from datetime import datetime

    from patient_risk.features import build_features
    from patient_risk.models import Patient, VitalSign

    prediction_time = datetime(2026, 1, 10, 12)
    patient = Patient("p1", 78)
    vitals = [
        VitalSign(
            measurement_id="v1",
            patient_id="p1",
            name="heart_rate",
            value=80.0,
            unit="bpm",
            observed_at=datetime(2026, 1, 10, 8),
        )
    ]

    features = build_features(
        patient=patient,
        admissions=[],
        lab_results=[],
        vital_signs=vitals,
        prediction_time=prediction_time,
    )

    assert "heart_rate_trend_24h" in features.values
    assert features.missing_indicators["heart_rate_trend_24h"] is True


def test_missing_weight_measurements_create_missing_trend_feature():
    result = build()

    assert "weight_trend_24h" in result.values
    assert result.values["weight_trend_24h"] == 0.0
    assert result.missing_indicators["weight_trend_24h"] is True
