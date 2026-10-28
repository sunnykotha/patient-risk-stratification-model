from dataclasses import dataclass
from datetime import datetime
from math import isfinite


def _require_datetime(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime")


def _require_same_timezone(
    first: datetime, second: datetime, first_name: str, second_name: str
) -> None:
    if (first.utcoffset() is None) != (second.utcoffset() is None):
        raise ValueError(
            f"{first_name} and {second_name} must use compatible timezones"
        )


@dataclass(frozen=True)
class Patient:
    patient_id: str
    age: int

    def __post_init__(self):
        if not self.patient_id.strip():
            raise ValueError("patient_id must not be empty")
        if isinstance(self.age, bool) or not isinstance(self.age, int):
            raise ValueError("age must be an integer")
        if not 0 <= self.age <= 120:
            raise ValueError("age must be between 0 and 120")


@dataclass(frozen=True)
class Admission:
    admission_id: str
    patient_id: str
    admitted_at: datetime
    discharged_at: datetime | None = None

    def __post_init__(self):
        if not self.admission_id.strip() or not self.patient_id.strip():
            raise ValueError("admission_id and patient_id must not be empty")

        _require_datetime(self.admitted_at, "admitted_at")

        if self.discharged_at is not None:
            _require_datetime(self.discharged_at, "discharged_at")
            _require_same_timezone(
                self.admitted_at, self.discharged_at,
                "admitted_at", "discharged_at"
            )
            if self.discharged_at < self.admitted_at:
                raise ValueError("discharged_at cannot precede admitted_at")


@dataclass(frozen=True)
class LabResult:
    result_id: str
    patient_id: str
    name: str
    value: float | None
    unit: str
    observed_at: datetime

    def __post_init__(self):
        if not all(
            isinstance(value, str) and value.strip()
            for value in (self.result_id, self.patient_id, self.name, self.unit)
        ):
            raise ValueError("Lab result identifiers, name, and unit are required")
        _require_datetime(self.observed_at, "observed_at")
        if self.value is not None:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise ValueError("lab value must be numeric or None")
            if not isfinite(self.value):
                raise ValueError("lab value must be finite or None")


@dataclass(frozen=True)
class VitalSign:
    measurement_id: str
    patient_id: str
    name: str
    value: float | None
    unit: str
    observed_at: datetime

    def __post_init__(self):
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                self.measurement_id, self.patient_id, self.name, self.unit
            )
        ):
            raise ValueError("Vital sign identifiers, name, and unit are required")
        _require_datetime(self.observed_at, "observed_at")
        if self.value is not None:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise ValueError("vital value must be numeric or None")
            if not isfinite(self.value):
                raise ValueError("vital value must be finite or None")


@dataclass(frozen=True)
class Diagnosis:
    diagnosis_code: str
    description: str

    def __post_init__(self):
        if not self.diagnosis_code.strip() or not self.description.strip():
            raise ValueError("diagnosis code and description must not be empty")


@dataclass(frozen=True)
class FeatureSet:
    patient_id: str
    prediction_time: datetime
    values: dict[str, float]
    missing_indicators: dict[str, bool]

    def __post_init__(self):
        if not self.patient_id.strip():
            raise ValueError("patient_id must not be empty")
        _require_datetime(self.prediction_time, "prediction_time")

        if set(self.values) != set(self.missing_indicators):
            raise ValueError("Each feature must have a missing indicator")

        for name, value in self.values.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("feature names must not be empty")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("feature values must be numeric")
            if not isfinite(value):
                raise ValueError("feature values must be finite")

        for name, flag in self.missing_indicators.items():
            if not isinstance(flag, bool):
                raise ValueError(f"Missing indicator for {name} must be boolean")


@dataclass(frozen=True)
class RiskScore:
    patient_id: str
    prediction_time: datetime
    probability: float
    tier: str

    def __post_init__(self):
        if not self.patient_id.strip():
            raise ValueError("patient_id must not be empty")
        _require_datetime(self.prediction_time, "prediction_time")

        if isinstance(self.probability, bool) or not isinstance(
            self.probability, (int, float)
        ):
            raise ValueError("probability must be numeric")
        if not isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("probability must be between 0 and 1")
        if self.tier not in {"low", "medium", "high"}:
            raise ValueError("tier must be low, medium, or high")


@dataclass(frozen=True)
class Explanation:
    patient_id: str
    top_factors: tuple[str, ...]

    def __post_init__(self):
        if not self.patient_id.strip():
            raise ValueError("patient_id must not be empty")
        if not isinstance(self.top_factors, tuple):
            raise ValueError("top_factors must be a tuple")
        if any(not isinstance(factor, str) or not factor.strip()
               for factor in self.top_factors):
            raise ValueError("Each contributing factor must be a non-empty string")
