from dataclasses import replace
from math import isfinite

from .models import LabResult, VitalSign


# Plausibility bounds are data-quality checks, not diagnostic reference ranges.
LAB_RULES = {
    "sodium": (
        100.0, 200.0,
        {"mmol/l": ("mmol/L", 1.0), "meq/l": ("mmol/L", 1.0)},
    ),
    "hemoglobin": (
        2.0, 25.0,
        {"g/dl": ("g/dL", 1.0), "g/l": ("g/dL", 0.1)},
    ),
}

VITAL_RULES = {
    "heart_rate": (
        20.0, 250.0,
        {"bpm": ("bpm", 1.0), "beats/min": ("bpm", 1.0)},
    ),
    "weight": (
        1.0, 500.0,
        {"kg": ("kg", 1.0), "lb": ("kg", 0.45359237),
         "lbs": ("kg", 0.45359237)},
    ),
    "temperature_c": (
        25.0, 45.0,
        {"c": ("C", 1.0), "°c": ("C", 1.0)},
    ),
    "temperature_f": (
        77.0, 113.0,
        {"f": ("F", 1.0), "°f": ("F", 1.0)},
    ),
    "systolic_bp": (50.0, 300.0, {"mmhg": ("mmHg", 1.0)}),
    "diastolic_bp": (20.0, 200.0, {"mmhg": ("mmHg", 1.0)}),
    "respiratory_rate": (
        4.0, 70.0,
        {"breaths/min": ("breaths/min", 1.0),
         "breaths/minute": ("breaths/min", 1.0)},
    ),
    "oxygen_saturation": (
        50.0, 100.0, {"%": ("%", 1.0), "percent": ("%", 1.0)},
    ),
}


def _clean(record, name, rules):
    key = name.strip().lower()
    if key not in rules:
        raise ValueError(f"Unsupported measurement: {name}")

    minimum, maximum, units = rules[key]
    unit_key = record.unit.strip().lower()

    # Normalize common degree-symbol encoding variants defensively.
    unit_key = unit_key.replace("º", "°").replace("˚", "°")
    if unit_key not in units:
        raise ValueError(f"Unsupported unit {record.unit!r} for {name}")

    normalized_unit, multiplier = units[unit_key]
    value = record.value

    if value is not None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Measurement must be numeric or None")
        if not isfinite(value):
            value = None
        else:
            value = float(value) * multiplier
            if not minimum <= value <= maximum:
                value = None

    return replace(record, value=value, unit=normalized_unit)


def clean_lab_result(result: LabResult) -> LabResult:
    """Normalize a supported lab unit and null implausible values."""
    return _clean(result, result.name, LAB_RULES)


def clean_vital_sign(vital: VitalSign) -> VitalSign:
    """Normalize a supported vital unit and null implausible values."""
    return _clean(vital, vital.name, VITAL_RULES)
