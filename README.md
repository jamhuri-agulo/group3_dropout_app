# Group 3: Student Dropout Risk Prototype (FUSE-AI-201, Module 5)

Predicts, at the **end of semester one**, whether a student is likely to drop out (binary: Dropout vs Graduate/Enrolled).
Decision support only. Do not use scores to penalise students or make automated decisions.

## Data
UCI "Predict Students' Dropout and Academic Success" (dataset 697): https://archive.ics.uci.edu/dataset/697
The six second-semester columns are excluded to prevent leakage. Download the dataset yourself to rerun the notebook.

## Files
| File | Purpose |
|---|---|
| `Group3_Module5_Implementation_Notebook.ipynb` | Data loading, split, baselines, model comparison, tuning, final model, evaluation, saving artifacts |
| `model_artifacts/` | Saved pipeline (`dropout_pipeline_v1.joblib`) and `metadata_v1.json` from the notebook |
| `predictor.py` | Loads the artifacts, validates input, scores students |
| `app.py` | Streamlit interface (single student and batch CSV) |
| `sample_students.csv` | Five fictional students for testing |
| `build_demo_artifacts.py` | Builds a SYNTHETIC demo model (random data) if the real artifacts are missing; the app shows a warning banner when it is used |
| `test_predictor.py` | Automated tests (run on the synthetic demo model; they check application behaviour, not accuracy) |
| `requirements.txt` | Python packages (pinned to the versions the model was trained with) |

## Install
```
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Run
1. Place the notebook's saved model and metadata in `model_artifacts/`.
2. `python -m streamlit run app.py`, then open http://localhost:8501
3. **Single student** tab: fill the form and press *Estimate dropout risk*.
   **Batch CSV** tab: upload a CSV with the 30 input columns (see `sample_students.csv`), press *Score uploaded students*, then download the results.
4. Command line: `python predictor.py --csv sample_students.csv --explain`
5. Tests: `python -m pytest -q`

## Results (held-out test set, threshold 0.519)
Dropout recall 0.83, precision 0.79, F1 0.81, ROC-AUC 0.922. Recall is lower for some groups (for example scholarship holders); see the report.

## Limitations
One Portuguese institution; associations not causes; scores are not certainties. Input validation for missing, malformed and unseen values was not fully tested before submission.
