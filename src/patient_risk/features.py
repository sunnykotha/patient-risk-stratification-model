from datetime import timedelta
from math import isfinite

from .cleaning import clean_lab_result, clean_vital_sign, LAB_RULES, VITAL_RULES
from .models import Admission, FeatureSet, LabResult, Patient, VitalSign


def _validate_time(value, prediction_time, field_name):
    if (value.utcoffset() is None) != (prediction_time.utcoffset() is None):
        raise ValueError(
            f"{field_name} and prediction_time must use compatible timezones"
        )


def build_features(
    patient: Patient,
    admissions: list[Admission],
    lab_results: list[LabResult],
    vital_signs: list[VitalSign],
    prediction_time,
    comorbidity_index: float = 0.0,
    current_admission_id: str | None = None,
) -> FeatureSet:
    """Build features using only records available by prediction_time.

    Missing numeric values use temporary zero placeholders with missing flags.
    Cohort-median imputation is implemented in the next pipeline step.
    """

    if not hasattr(prediction_time, "tzinfo"):
        raise ValueError("prediction_time must be a datetime")

    if isinstance(comorbidity_index, bool) or not isinstance(
        comorbidity_index, (int, float)
    ):
        raise ValueError("comorbidity_index must be numeric")

    if not isfinite(comorbidity_index) or comorbidity_index < 0:
        raise ValueError("comorbidity_index must be finite and non-negative")

    for admission in admissions:
        if admission.patient_id == patient.patient_id:
            _validate_time(
                admission.admitted_at, prediction_time, "admitted_at"
            )
            if admission.discharged_at is not None:
                _validate_time(
                    admission.discharged_at, prediction_time, "discharged_at"
                )

    for result in lab_results:
        if result.patient_id == patient.patient_id:
            _validate_time(
                result.observed_at, prediction_time, "lab observed_at"
            )

    for vital in vital_signs:
        if vital.patient_id == patient.patient_id:
            _validate_time(
                vital.observed_at, prediction_time, "vital observed_at"
            )

    patient_admissions = [
        admission
        for admission in admissions
        if admission.patient_id == patient.patient_id
        and admission.admitted_at <= prediction_time
        and (
            admission.discharged_at is None
            or admission.discharged_at <= prediction_time
        )
    ]

    window_start = prediction_time - timedelta(days=365)

    prior_admissions = [
        admission
        for admission in patient_admissions
        if admission.admission_id != current_admission_id
        and window_start <= admission.admitted_at < prediction_time
        and admission.discharged_at is not None
    ]

    features = {
        "age": float(patient.age),
        "prior_admissions_12m": float(len(prior_admissions)),
        "comorbidity_index": float(comorbidity_index),
    }

    missing = {name: False for name in features}

    current_admission = next(
        (
            admission
            for admission in patient_admissions
            if admission.admission_id == current_admission_id
        ),
        None,
    )

    if current_admission is None:
        features["length_of_stay_days"] = 0.0
        missing["length_of_stay_days"] = True
    else:
        features["length_of_stay_days"] = (
            prediction_time - current_admission.admitted_at
        ).total_seconds() / 86400
        missing["length_of_stay_days"] = False

    patient_labs = [
        clean_lab_result(result)
        for result in lab_results
        if result.patient_id == patient.patient_id
        and result.observed_at <= prediction_time
        and result.name.strip().lower() in LAB_RULES
        and result.name.strip().lower() in {"sodium", "hemoglobin"}
    ]

    for lab_name in ("sodium", "hemoglobin"):
        candidates = [
            result for result in patient_labs
            if result.name.strip().lower() == lab_name and result.value is not None
        ]
        candidates.sort(key=lambda result: result.observed_at)

        feature_name = f"last_{lab_name}"
        if candidates:
            features[feature_name] = float(candidates[-1].value)
            missing[feature_name] = False
        else:
            features[feature_name] = 0.0
            missing[feature_name] = True

    patient_vitals = [
        (
            clean_vital_sign(vital)
            if vital.name.strip().lower() in VITAL_RULES
            else vital
        )
        for vital in vital_signs
        if vital.patient_id == patient.patient_id
        and prediction_time - timedelta(hours=24)
        <= vital.observed_at <= prediction_time
        and vital.value is not None
    ]
    patient_vitals = [
        vital for vital in patient_vitals if vital.value is not None
    ]

    vital_names = sorted({vital.name.strip().lower() for vital in patient_vitals})

    for vital_name in vital_names:
        measurements = sorted(
            (
                vital for vital in patient_vitals
                if vital.name.strip().lower() == vital_name
            ),
            key=lambda vital: vital.observed_at,
        )

        feature_name = f"{vital_name}_trend_24h"
        if len(measurements) >= 2:
            features[feature_name] = (
                float(measurements[-1].value - measurements[0].value)
            )
            missing[feature_name] = False
        else:
            features[feature_name] = 0.0
            missing[feature_name] = True

    # Keep the weight-trend feature present even when recent weight
    # measurements are unavailable, so preprocessing can retain its
    # missing-value indicator.
    if "weight_trend_24h" not in features:
        features["weight_trend_24h"] = 0.0
        missing["weight_trend_24h"] = True

    return FeatureSet(
        patient_id=patient.patient_id,
        prediction_time=prediction_time,
        values=features,
        missing_indicators=missing,
    )
