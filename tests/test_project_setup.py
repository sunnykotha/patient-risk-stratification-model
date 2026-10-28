import pandas as pd
import sklearn

import patient_risk


def test_required_dependencies_are_available():
    assert pd.__version__
    assert sklearn.__version__


def test_patient_risk_package_is_importable():
    assert patient_risk.__doc__ is not None