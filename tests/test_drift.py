import pandas as pd
import pytest

from patient_risk.drift import monitor_feature_drift


def test_no_drift_for_identical_batches():
    frame = pd.DataFrame({"age": [50, 60, 70], "sodium": [135, 138, 140]})
    result = monitor_feature_drift(frame, frame.copy())
    assert result["drift_detected"] is False


def test_detects_mean_shift():
    reference = pd.DataFrame({"age": [50, 60, 70]})
    current = pd.DataFrame({"age": [90, 95, 100]})
    result = monitor_feature_drift(reference, current)
    assert result["drift_detected"] is True
    assert result["features"]["age"]["drift_detected"] is True


def test_detects_missing_rate_shift():
    reference = pd.DataFrame({"sodium": [135, 138, 140, 137]})
    current = pd.DataFrame({"sodium": [None, None, 140, 137]})
    result = monitor_feature_drift(reference, current)
    assert result["features"]["sodium"]["missing_rate_shift"] == 0.5
    assert result["drift_detected"] is True


@pytest.mark.parametrize(
    "reference,current",
    [
        (pd.DataFrame(), pd.DataFrame({"age": [60]})),
        (pd.DataFrame({"age": [60]}), pd.DataFrame({"weight": [70]})),
        (pd.DataFrame({"age": ["old"]}), pd.DataFrame({"age": ["new"]})),
    ],
)
def test_rejects_invalid_batches(reference, current):
    with pytest.raises(ValueError):
        monitor_feature_drift(reference, current)


def test_rejects_invalid_threshold():
    frame = pd.DataFrame({"age": [60, 70]})
    with pytest.raises(ValueError, match="mean_shift_threshold"):
        monitor_feature_drift(frame, frame, mean_shift_threshold=2)
