# Patient Risk Stratification Model Pipeline

## Purpose

Estimate a patient's probability of 30-day readmission and provide contributing factors to help care managers prioritize follow-up.

This project is an educational implementation. It is not a clinically validated decision-support system.

## Requirements

- Python 3.13 or another Python version supported by the dependencies in `requirements.txt`
- PowerShell on Windows

## Setup

Open PowerShell in the project directory and run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the virtual environment already exists, skip the first command.

## Generate synthetic data

Run:

```powershell
.\.venv\Scripts\python.exe .\scripts\generate_synthetic_data.py
```

The generated dataset is saved to:

`data/synthetic_readmissions.csv`

The data is simulated for software development and testing. It does not represent real patients or establish clinical performance.

## Run tests

Run the complete test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

## Project structure

- `src/patient_risk/models.py` — core data models
- `src/patient_risk/features.py` — feature engineering using time-bounded clinical records
- `src/patient_risk/cleaning.py` — validation and cleaning of clinical measurements
- `src/patient_risk/imputation.py` — median imputation and missing indicators
- `src/patient_risk/feature_preparation.py` — prepares features for model input
- `src/patient_risk/model.py` — logistic-regression training pipeline
- `src/patient_risk/risk.py` — risk tiers and coefficient-based explanations
- `src/patient_risk/rule_scoring.py` — points-based worked example
- `src/patient_risk/evaluation.py` — held-out evaluation and calibration reporting
- `src/patient_risk/drift.py` — feature drift monitoring
- `tests/` — unit, integration, leakage, and synthetic-data tests
- `scripts/generate_synthetic_data.py` — reproducible synthetic dataset generator

## Important limitations

- Calibration reporting compares predicted probabilities with observed outcomes; it does not by itself guarantee calibrated probabilities.
- Risk-tier thresholds are implementation choices and have not been clinically validated.
- Synthetic test results do not establish real-world predictive accuracy, safety, or clinical utility.
- Any real clinical use requires appropriate data governance, external validation, clinical review, and ongoing monitoring.

## Feature Engineering and Prediction-Time Rules

Features are built using records available at the requested prediction time.

- **Age:** patient's age.
- **Prior admissions:** completed admissions within the preceding 12 months, excluding the current admission from the count.
- **Comorbidity index:** supplied to the feature builder as an input.
- **Length of stay:** elapsed time from admission to prediction time when the current admission is available.
- **Laboratory results:** the latest eligible sodium and hemoglobin results observed no later than prediction time.
- **Vital-sign trends:** the difference between the last and first available measurements of each vital type during the preceding 24 hours. At least two measurements are required for a trend.

Missing features are represented using missing indicators. Features that cannot be calculated are not treated as reliable clinical measurements.

The feature-building process filters records by patient and prediction time. The leakage tests check that future records do not influence historical predictions.

## Data Cleaning and Missing Values

The cleaning layer validates supported measurement names, units, numeric values, and plausible ranges. Supported unit conversions are applied where configured. Invalid or implausible measurements are represented as missing rather than accepted as valid observations.

The imputation layer uses numeric medians and missing-value indicators. Medians should be fitted on training data and then reused for validation, testing, and future predictions to avoid information leakage.

The configured ranges are data-quality checks; they are not a substitute for clinical reference ranges or professional interpretation.

## Model Training and Risk Interpretation

The predictive pipeline uses logistic regression to estimate the probability of 30-day readmission. The configured model uses class weighting to help address an imbalanced outcome distribution.

The model's probability-based risk tiers are:

- **Low:** probability below 0.30.
- **Medium:** probability from 0.30 up to, but not including, 0.60.
- **High:** probability of 0.60 or greater.

Explanations use model coefficient contributions relative to a median-feature baseline. These contributions describe the fitted model's behavior; they do not establish causation or provide a clinical diagnosis.

## Separate Points-Based Worked Example

The project also contains a rule-based scoring example. This is separate from the logistic-regression probability and its risk tiers.

For the supplied example:

| Factor | Points |
|---|---:|
| Age 78 | +2 |
| Three prior admissions | +4 |
| Sodium of 128 | +2 |
| Missing recent weight | Flag only; no points |
| **Total** | **8** |

The rule-based score is classified as **high risk** at 7 points or more. This example demonstrates the specified rules; it is not a validated clinical scoring instrument.

## Evaluation and Calibration

The evaluation module assesses predictions on a held-out test set and reports:

- Brier score.
- ROC AUC.
- Calibration-bin summaries comparing mean predicted probability with observed readmission frequency.

These outputs help assess predictive performance and calibration. Reporting calibration does not itself correct poorly calibrated probabilities. Results from synthetic data must not be interpreted as evidence of real-world clinical performance.

## Drift Monitoring

The drift module compares numeric feature distributions and missing-value rates between datasets. It can help identify changes that warrant investigation.

This is a basic monitoring signal. It does not independently establish outcome drift, detect every form of data-quality problem, or prove that predictive performance has deteriorated.

## Test Coverage

The automated test suite covers data cleaning, feature engineering, time-based leakage safeguards, imputation, model behavior, risk interpretation, evaluation, drift monitoring, and synthetic-data generation.

Run the full suite using:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

A passing test suite confirms the tested software behavior. It does not establish clinical validity, patient safety, or suitability for production use.

## Scope and Limitations

The comorbidity index is supplied as a feature-builder input. Do not assume that the pipeline automatically calculates it from diagnosis records unless that calculation is implemented separately.

Before real-world use, the model would require representative clinical data, appropriate external validation, review of calibration and thresholds, data governance, and clinical oversight.
