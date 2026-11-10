# Group 3 — Student Dropout Risk Prototype

## Run locally

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Build synthetic demo artifacts (for interface/testing only):

   ```bash
   python build_demo_artifacts.py
   ```

4. Launch the application:

   ```bash
   streamlit run app.py
   ```

5. Optional command-line scoring:

   ```bash
   python predictor.py --csv sample_students.csv --explain
   ```

## Use the real trained model

Copy the trained model's `dropout_pipeline_v1.joblib` and `metadata_v1.json` into `model_artifacts/`, or set `DROPOUT_ARTIFACTS` to the directory containing them. The metadata and pipeline must match the expected schema. Restart Streamlit after changing artifacts.

## Important limitation

The synthetic demo model is trained on generated random data. Its predictions are not meaningful estimates of actual student dropout risk. It exists to exercise the interface and tests. Use the group's real trained model for model evaluation, and do not use the prototype to make punitive or automated decisions about students.
