"""Run with:  python -m pytest tests -q   (uses a synthetic model in a temp folder: tests API behaviour only)."""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    subprocess.run([sys.executable, str(ROOT / "build_demo_artifacts.py"), "--out", str(out)], check=True)
    os.environ["DROPOUT_ARTIFACTS"] = str(out)
    os.environ["ALLOW_SYNTHETIC"] = "1"
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


@pytest.fixture()
def row():
    r = pd.read_csv(ROOT / "sample_students.csv").iloc[0].drop("Student ID")
    return {k: float(v) for k, v in r.items()}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy" and len(body["model_sha256"]) == 64


def test_schema_lists_30_fields(client):
    assert len(client.get("/schema").json()["features"]) == 30


def test_predict_ok(client, row):
    r = client.post("/predict", json={"student_id": "T-1", "features": row})
    assert r.status_code == 200
    b = r.json()
    assert b["student_id"] == "T-1" and 0 <= b["risk_score"] <= 1
    assert b["predicted_class"] in ("Dropout", "Not Dropout")
    assert b["flagged"] == (b["predicted_class"] == "Dropout")


def test_missing_field_rejected(client, row):
    row.pop("Debtor")
    r = client.post("/predict", json={"features": row})
    assert r.status_code == 422 and "Debtor" in r.text


def test_non_numeric_value_rejected(client, row):
    row["Admission grade"] = "high"
    assert client.post("/predict", json={"features": row}).status_code == 422


def test_null_value_rejected(client, row):
    row["Admission grade"] = None
    assert client.post("/predict", json={"features": row}).status_code == 422


def test_binary_field_must_be_0_or_1(client, row):
    row["Gender"] = 2
    r = client.post("/predict", json={"features": row})
    assert r.status_code == 422 and "Gender" in r.text


def test_unseen_category_scored_with_note(client, row):
    row["Course"] = 123456
    r = client.post("/predict", json={"features": row})
    assert r.status_code == 200 and any("Course" in n for n in r.json()["notes"])


def test_extreme_value_warns_but_scores(client, row):
    row["Curricular units 1st sem (without evaluations)"] = 500
    r = client.post("/predict", json={"features": row})
    assert r.status_code == 200 and any("far outside" in n for n in r.json()["notes"])


def test_infinity_rejected(client, row):
    row["Admission grade"] = 1e999   # parsed by JSON as inf by some clients
    r = client.post("/predict", content='{"features": %s}' % str(row).replace("'", '"').replace("inf", "Infinity"),
                    headers={"Content-Type": "application/json"})
    assert r.status_code in (422, 400)


def test_batch(client, row):
    r = client.post("/predict/batch", json={"students": [{"student_id": "a", "features": row},
                                                          {"student_id": "b", "features": row}]})
    assert r.status_code == 200
    b = r.json()
    assert b["count"] == 2 and [p["student_id"] for p in b["predictions"]] == ["a", "b"]


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"students": []}).status_code == 422


def test_oversized_batch_rejected(client, row):
    r = client.post("/predict/batch", json={"students": [{"features": row}] * 1001})
    assert r.status_code == 422


def test_malformed_json_rejected(client):
    r = client.post("/predict", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422
