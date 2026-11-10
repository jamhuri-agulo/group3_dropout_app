"""
Builds a SYNTHETIC stand-in for the real model artifacts, with exactly the same
schema and pipeline layout as the notebook's final model. It exists so the app
and tests can run before the real model_artifacts/ folder is copied in.

The data is random. The scores it produces mean nothing, and the app shows a
red warning banner whenever this model is loaded.

    python build_demo_artifacts.py            # writes model_artifacts_demo/
    python build_demo_artifacts.py --out DIR
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Same 30 input columns, same order, same grouping as the notebook (semester-2 columns excluded)
CATEGORICAL = ["Marital Status", "Application mode", "Course", "Previous qualification", "Nacionality",
               "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"]
BINARY = ["Daytime/evening attendance", "Displaced", "Educational special needs", "Debtor",
          "Tuition fees up to date", "Gender", "Scholarship holder", "International"]
FEATURES = [
    "Marital Status", "Application mode", "Application order", "Course", "Daytime/evening attendance",
    "Previous qualification", "Previous qualification (grade)", "Nacionality", "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation", "Admission grade", "Displaced",
    "Educational special needs", "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder",
    "Age at enrollment", "International", "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)", "Unemployment rate", "Inflation rate", "GDP",
]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL + BINARY]
SEM2 = [f"Curricular units 2nd sem ({s})" for s in
        ("credited", "enrolled", "evaluations", "approved", "grade", "without evaluations")]

CODES = {
    "Marital Status": [1, 2, 3, 4, 5, 6],
    "Application mode": [1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57],
    "Course": [33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991],
    "Previous qualification": [1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43],
    "Nacionality": [1, 2, 6, 11, 13, 14, 17, 21, 22, 24, 25, 26, 32, 41, 62, 100, 101, 103, 105, 108, 109],
    "Mother's qualification": [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 18, 19, 22, 26, 27, 29, 30, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
    "Father's qualification": [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 22, 25, 26, 27, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
    "Mother's occupation": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 122, 123, 125, 131, 132, 134, 141, 143, 144, 151, 152, 153, 171, 173, 175, 191, 192, 193, 194],
    "Father's occupation": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 101, 102, 103, 112, 114, 121, 122, 123, 124, 131, 132, 134, 135, 141, 143, 144, 151, 152, 153, 154, 161, 163, 171, 172, 174, 175, 181, 182, 183, 192, 193, 194, 195],
}


def make_preprocessor():  # identical to the notebook
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("bin", "passthrough", BINARY),
    ])


def synthetic_students(n: int, rng: np.random.Generator) -> pd.DataFrame:
    d = {c: rng.choice(CODES[c], n) for c in CATEGORICAL}
    enrolled = rng.integers(0, 9, n)
    d.update({
        "Application order": rng.integers(0, 10, n),
        "Previous qualification (grade)": rng.normal(132, 13, n).clip(95, 190),
        "Admission grade": rng.normal(127, 14, n).clip(95, 190),
        "Age at enrollment": (rng.gamma(2.0, 4.5, n) + 17).round(),
        "Curricular units 1st sem (credited)": rng.poisson(0.7, n),
        "Curricular units 1st sem (enrolled)": enrolled,
        "Curricular units 1st sem (evaluations)": enrolled + rng.integers(0, 4, n),
        "Curricular units 1st sem (approved)": rng.binomial(enrolled, rng.beta(4, 2, n)),
        "Curricular units 1st sem (grade)": rng.normal(10.6, 4.0, n).clip(0, 18.9),
        "Curricular units 1st sem (without evaluations)": rng.poisson(0.14, n),
        "Unemployment rate": rng.normal(11.6, 2.6, n).clip(7.6, 16.2),
        "Inflation rate": rng.normal(1.2, 1.4, n).clip(-0.8, 3.7),
        "GDP": rng.normal(0.0, 2.3, n).clip(-4.1, 3.5),
        "Daytime/evening attendance": rng.binomial(1, 0.89, n),
        "Displaced": rng.binomial(1, 0.55, n),
        "Educational special needs": rng.binomial(1, 0.01, n),
        "Debtor": rng.binomial(1, 0.11, n),
        "Tuition fees up to date": rng.binomial(1, 0.88, n),
        "Gender": rng.binomial(1, 0.35, n),
        "Scholarship holder": rng.binomial(1, 0.25, n),
        "International": rng.binomial(1, 0.02, n),
    })
    return pd.DataFrame(d)[FEATURES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "model_artifacts_demo"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    X = synthetic_students(2500, rng)
    rate = X["Curricular units 1st sem (approved)"] / X["Curricular units 1st sem (enrolled)"].clip(lower=1)
    logit = (0.5 - 3.0 * rate - 1.5 * X["Tuition fees up to date"] + 1.2 * X["Debtor"]
             + 0.05 * (X["Age at enrollment"] - 22) + 1.0)
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    pipe = Pipeline([("prep", make_preprocessor()),
                     ("model", LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"))])
    pipe.fit(X, y)

    prob = pipe.predict_proba(X)[:, 1]
    prec, rec, thr = precision_recall_curve(y, prob)
    ok = rec[:-1] >= 0.80
    threshold = float(thr[np.where(ok)[0][np.argmax(prec[:-1][ok])]])

    prep = pipe.named_steps["prep"]
    means = prep.transform(X).mean(axis=0)
    metadata = {
        "version": "v1", "synthetic": True,
        "saved": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_type": "SYNTHETIC DEMO - LogisticRegression (C=0.3, class_weight=balanced) in a scikit-learn Pipeline",
        "task": "binary_classification (1 = Dropout)",
        "prediction_point": "end of semester 1",
        "input_features": FEATURES, "threshold": threshold,
        "training_samples": len(X), "excluded_columns": SEM2,
        "sklearn_version": sklearn.__version__,
        "feature_means": {n: round(float(m), 6) for n, m in zip(prep.get_feature_names_out(), means)},
    }
    joblib.dump(pipe, out / "dropout_pipeline_v1.joblib")
    (out / "metadata_v1.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote synthetic demo model to {out} (threshold {threshold:.3f})")


if __name__ == "__main__":
    main()
