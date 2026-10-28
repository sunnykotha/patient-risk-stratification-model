from datetime import datetime, timedelta

from patient_risk.features import build_features
from patient_risk.models import Admission, LabResult, Patient, VitalSign


def test_future_lab_results_do_not_change_features():
    prediction_time = datetime(2026, 1, 10, 12)
    patient = Patient("p1", 78)
    earlier = LabResult(
        "l1", "p1", "sodium", 128.0, "mmol/L",
        prediction_time - timedelta(hours=2),
    )
    future = LabResult(
        "l2", "p1", "sodium", 145.0, "mmol/L",
        prediction_time + timedelta(hours=2),
    )

    before = build_features(
        patient, [], [earlier], [], prediction_time
    )
    after = build_features(
        patient, [], [earlier, future], [], prediction_time
    )

    assert before.values == after.values
    assert before.missing_indicators == after.missing_indicators


def test_future_vital_signs_do_not_change_features():
    prediction_time = datetime(2026, 1, 10, 12)
    patient = Patient("p1", 78)
    earlier = VitalSign(
        "v1", "p1", "heart_rate", 80.0, "bpm",
        prediction_time - timedelta(hours=2),
    )
    future = VitalSign(
        "v2", "p1", "heart_rate", 140.0, "bpm",
        prediction_time + timedelta(hours=1),
    )

    before = build_features(
        patient, [], [], [earlier], prediction_time
    )
    after = build_features(
        patient, [], [], [earlier, future], prediction_time
    )

    assert before.values == after.values
    assert before.missing_indicators == after.missing_indicators


def test_future_admissions_do_not_change_prior_admission_count():
    prediction_time = datetime(2026, 1, 10, 12)
    patient = Patient("p1", 78)
    prior = Admission(
        "a1", "p1",
        prediction_time - timedelta(days=30),
        prediction_time - timedelta(days=25),
    )
    future = Admission(
        "a2", "p1",
        prediction_time + timedelta(days=2),
        prediction_time + timedelta(days=5),
    )

    before = build_features(patient, [prior], [], [], prediction_time)
    after = build_features(patient, [prior, future], [], [], prediction_time)

    assert before.values["prior_admissions_12m"] == 1
    assert after.values["prior_admissions_12m"] == 1


def test_other_patient_records_do_not_change_features():
    prediction_time = datetime(2026, 1, 10, 12)
    patient = Patient("p1", 78)
    own_lab = LabResult(
        "l1", "p1", "sodium", 128.0, "mmol/L",
        prediction_time - timedelta(hours=2),
    )
    other_lab = LabResult(
        "l2", "p2", "sodium", 145.0, "mmol/L",
        prediction_time - timedelta(hours=1),
    )

    before = build_features(patient, [], [own_lab], [], prediction_time)
    after = build_features(
        patient, [], [own_lab, other_lab], [], prediction_time
    )

    assert before.values == after.values
    assert before.missing_indicators == after.missing_indicators


def test_feature_builder_does_not_mutate_input_lists():
    prediction_time = datetime(2026, 1, 10, 12)
    patient = Patient("p1", 78)
    admissions = [
        Admission(
            "a1", "p1",
            prediction_time - timedelta(days=30),
            prediction_time - timedelta(days=25),
        )
    ]
    labs = [
        LabResult(
            "l1", "p1", "sodium", 128.0, "mmol/L",
            prediction_time - timedelta(hours=2),
        )
    ]
    vitals = [
        VitalSign(
            "v1", "p1", "heart_rate", 80.0, "bpm",
            prediction_time - timedelta(hours=1),
        )
    ]
    originals = (admissions.copy(), labs.copy(), vitals.copy())

    build_features(patient, admissions, labs, vitals, prediction_time)

    assert admissions == originals[0]
    assert labs == originals[1]
    assert vitals == originals[2]
