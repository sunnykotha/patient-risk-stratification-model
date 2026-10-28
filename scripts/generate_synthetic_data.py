from pathlib import Path

import numpy as np
import pandas as pd


def generate_dataset(number_of_patients=240, random_state=42):
    """Generate reproducible synthetic data for software testing only."""
    if number_of_patients < 40:
        raise ValueError("number_of_patients must be at least 40")

    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 96, number_of_patients).astype(float)
    prior_admissions = np.clip(
        rng.poisson(0.9, number_of_patients), 0, 8
    ).astype(float)
    comorbidity = np.clip(
        rng.poisson(2.0, number_of_patients), 0, 10
    ).astype(float)
    length_of_stay = np.clip(
        rng.gamma(2.0, 2.5, number_of_patients), 1, 25
    )
    sodium = np.clip(
        rng.normal(139, 4, number_of_patients), 120, 150
    )
    hemoglobin = np.clip(
        rng.normal(12.5, 2.0, number_of_patients), 6, 18
    )
    heart_rate_trend = np.clip(
        rng.normal(0, 8, number_of_patients), -30, 35
    )
    systolic_bp_trend = np.clip(
        rng.normal(0, 10, number_of_patients), -40, 40
    )
    weight_missing = rng.binomial(1, 0.25, number_of_patients)

    # Simulated missing measurements.
    sodium[rng.random(number_of_patients) < 0.12] = np.nan
    hemoglobin[rng.random(number_of_patients) < 0.10] = np.nan
    heart_rate_trend[rng.random(number_of_patients) < 0.08] = np.nan
    systolic_bp_trend[rng.random(number_of_patients) < 0.08] = np.nan

    # Generate outcomes from underlying risk factors. Missing lab values
    # are temporarily replaced with typical values only for label generation.
    sodium_for_label = np.where(np.isnan(sodium), 139, sodium)
    hemoglobin_for_label = np.where(np.isnan(hemoglobin), 12.5, hemoglobin)
    heart_rate_for_label = np.where(
        np.isnan(heart_rate_trend), 0, heart_rate_trend
    )
    bp_for_label = np.where(np.isnan(systolic_bp_trend), 0, systolic_bp_trend)

    log_odds = (
        -2.65
        + 0.025 * (age - 60)
        + 0.32 * prior_admissions
        + 0.22 * comorbidity
        + 0.075 * (length_of_stay - 4)
        + 0.055 * (135 - sodium_for_label)
        + 0.12 * (12 - hemoglobin_for_label)
        + 0.025 * np.maximum(heart_rate_for_label, 0)
        + 0.008 * np.maximum(-bp_for_label, 0)
        + 0.25 * weight_missing
    )
    probabilities = 1 / (1 + np.exp(-log_odds))
    readmitted = rng.binomial(1, probabilities)

    # Ensure enough examples of both classes for stratified testing.
    if readmitted.sum() < 2:
        readmitted[np.argsort(probabilities)[-2:]] = 1
    if (readmitted == 0).sum() < 2:
        readmitted[np.argsort(probabilities)[:2]] = 0

    return pd.DataFrame({
        "age": age,
        "prior_admissions_12m": prior_admissions,
        "comorbidity_index": comorbidity,
        "length_of_stay_days": length_of_stay,
        "sodium": sodium,
        "hemoglobin": hemoglobin,
        "heart_rate_trend": heart_rate_trend,
        "systolic_bp_trend": systolic_bp_trend,
        "weight_missing": weight_missing.astype(float),
        "readmitted_30d": readmitted.astype(int),
    })


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / "synthetic_readmissions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset()
    dataset.to_csv(output_path, index=False)

    positives = int(dataset["readmitted_30d"].sum())
    print(f"Generated: {output_path}")
    print(f"Rows: {len(dataset)}")
    print(f"Readmitted within 30 days: {positives}")
    print(f"Not readmitted: {len(dataset) - positives}")
    print(f"Missing feature values: {int(dataset.isna().sum().sum())}")
