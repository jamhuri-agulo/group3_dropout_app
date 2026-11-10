"""Run with: python -m pytest -q"""
import subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predictor import ArtifactError, DropoutPredictor, InputError

@pytest.fixture(scope="session")
def predictor(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    subprocess.run([sys.executable, str(ROOT / "build_demo_artifacts.py"), "--out", str(out)], check=True)
    return DropoutPredictor(out)

@pytest.fixture()
def students():
    return pd.read_csv(ROOT / "sample_students.csv")

def test_scores_are_valid_and_flag_matches_threshold(predictor, students):
    r = predictor.predict(students).results
    assert len(r) == len(students)
    assert r["risk_score"].between(0, 1).all()
    assert (r["flagged"] == (r["risk_score"] >= predictor.threshold - 1e-3)).all()
    assert set(r["predicted_class"]) <= {"Dropout", "Not Dropout"}
    assert list(r["Student ID"]) == list(students["Student ID"])

def test_semester2_and_target_columns_are_dropped_not_used(predictor, students):
    leaky = students.copy()
    leaky["Curricular units 2nd sem (approved)"] = 6
    leaky["Target"] = "Graduate"
    a = predictor.predict(students); b = predictor.predict(leaky)
    assert np.allclose(a.results["risk_score"], b.results["risk_score"])
    assert any("not available at the end of semester 1" in n for n in b.notes)

def test_missing_column_is_rejected(predictor, students):
    with pytest.raises(InputError, match="Debtor"):
        predictor.predict(students.drop(columns=["Debtor"]))

def test_non_numeric_and_non_binary_values_are_rejected(predictor, students):
    bad = students.copy(); bad["Admission grade"] = bad["Admission grade"].astype(object)
    bad.loc[0, "Admission grade"] = "high"
    with pytest.raises(InputError, match="row 1: Admission grade"): predictor.predict(bad)
    bad = students.copy(); bad.loc[1, "Gender"] = 2
    with pytest.raises(InputError, match="Gender"): predictor.predict(bad)

def test_unseen_category_is_noted_but_scored(predictor, students):
    odd = students.copy(); odd.loc[0, "Course"] = 123456
    s = predictor.predict(odd)
    assert any("Course" in n and "not" in n for n in s.notes)
    assert np.isfinite(s.results["risk_score"]).all()

def test_headers_with_stray_whitespace_are_tolerated(predictor, students):
    messy = students.rename(columns={"Daytime/evening attendance": "Daytime/evening attendance\t"})
    assert len(predictor.predict(messy).results) == len(students)

def test_explanations_add_up_to_the_model_logit(predictor, students):
    X, _, _ = predictor.prepare(students); contrib = predictor.contributions(X)
    logit = predictor.pipeline.decision_function(X)
    assert np.allclose(contrib.sum(axis=1) + predictor.base_logit, logit, atol=1e-6)
    assert set(contrib.columns) == set(predictor.features)

def test_top_factors_have_the_right_signs(predictor, students):
    X, _, _ = predictor.prepare(students); contrib = predictor.contributions(X)
    up, down = predictor.top_factors(contrib.iloc[0], X.iloc[0])
    assert (up["contribution"] > 0).all() and (down["contribution"] < 0).all()

def test_missing_artifacts_give_a_helpful_error(tmp_path):
    with pytest.raises(ArtifactError, match="model_artifacts"): DropoutPredictor(tmp_path)
