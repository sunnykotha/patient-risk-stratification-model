from datetime import datetime, timezone

import pytest

from patient_risk.cleaning import clean_lab_result, clean_vital_sign
from patient_risk.models import LabResult, VitalSign


T = datetime(2026, 1, 1, tzinfo=timezone.utc)


def lab(name, value, unit):
    return LabResult("L1", "P1", name, value, unit, T)


def vital(name, value, unit):
    return VitalSign("V1", "P1", name, value, unit, T)


def test_normalizes_sodium_unit():
    result = clean_lab_result(lab("sodium", 128, "MEQ/L"))
    assert result.value == 128
    assert result.unit == "mmol/L"


def test_normalizes_hemoglobin_g_per_litre():
    result = clean_lab_result(lab("hemoglobin", 120, "g/L"))
    assert result.value == pytest.approx(12)
    assert result.unit == "g/dL"


def test_impossible_heart_rate_becomes_missing():
    result = clean_vital_sign(vital("heart_rate", 900, "bpm"))
    assert result.value is None


def test_valid_heart_rate_is_preserved():
    result = clean_vital_sign(vital("heart_rate", 80, "bpm"))
    assert result.value == 80


def test_normalizes_weight_from_pounds():
    result = clean_vital_sign(vital("weight", 220, "lb"))
    assert result.value == pytest.approx(99.7903214)
    assert result.unit == "kg"


def test_non_finite_lab_value_is_rejected_at_entity_boundary():
    with pytest.raises(ValueError, match="finite"):
        lab("sodium", float("nan"), "mmol/L")


def test_unknown_lab_unit_is_rejected():
    with pytest.raises(ValueError, match="Unsupported unit"):
        clean_lab_result(lab("sodium", 128, "unknown"))


def test_unknown_measurement_is_rejected():
    with pytest.raises(ValueError, match="Unsupported measurement"):
        clean_vital_sign(vital("unknown", 10, "unit"))