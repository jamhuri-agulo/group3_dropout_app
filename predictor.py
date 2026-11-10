"""
Prediction core for the early dropout-risk prototype (Group 3, Module 5).

Wraps the scikit-learn pipeline + metadata saved by the modelling notebook and
adds what an application needs on top of it:

  * input validation (all 30 semester-1 fields present, sane types/values)
  * a leakage guard (semester-2 columns and the Target column are dropped)
  * a risk score, a flag at the validation-tuned threshold, and per-student
    "what pushed the score up or down" explanations (logistic regression only)
  * a small command-line interface for batch scoring

Usage (CLI):
    python predictor.py --csv sample_students.csv --explain
    python predictor.py --csv my_students.csv --out scored.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = HERE / "model_artifacts"        # real model exported from Colab
DEMO_ARTIFACTS = HERE / "model_artifacts_demo"      # synthetic stand-in (see build_demo_artifacts.py)
VERSION = "v1"


class InputError(ValueError):
    """Student data cannot be scored (missing column, bad value, ...)."""


class ArtifactError(FileNotFoundError):
    """Model files are missing or cannot be loaded."""


@dataclass
class Scored:
    results: pd.DataFrame   # identifier columns + risk_score, flagged, predicted_class
    features: pd.DataFrame  # the cleaned model inputs, same row order
    notes: list             # human-readable warnings about the input


def find_artifact_dir(explicit=None) -> Path:
    """explicit path > $DROPOUT_ARTIFACTS > model_artifacts/ > model_artifacts_demo/."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("DROPOUT_ARTIFACTS")
    if env:
        return Path(env)
    for d in (DEFAULT_ARTIFACTS, DEMO_ARTIFACTS):
        if (d / f"metadata_{VERSION}.json").exists():
            return d
    return DEFAULT_ARTIFACTS


class DropoutPredictor:
    def __init__(self, artifact_dir=None, version: str = VERSION):
        self.artifact_dir = find_artifact_dir(artifact_dir)
        model_path = self.artifact_dir / f"dropout_pipeline_{version}.joblib"
        meta_path = self.artifact_dir / f"metadata_{version}.json"
        missing = [p.name for p in (model_path, meta_path) if not p.exists()]
        if missing:
            raise ArtifactError(
                f"Missing {', '.join(missing)} in '{self.artifact_dir}'. Copy the "
                "model_artifacts folder produced by the notebook next to this file "
                "(or run build_demo_artifacts.py for a synthetic demo model)."
            )
        self.meta = json.loads(meta_path.read_text())
        try:
            self.pipeline = joblib.load(model_path)
        except Exception as exc:  # version mismatch is the usual cause
            raise ArtifactError(
                f"Could not load {model_path.name}: {exc}. Usually this means a different "
                "scikit-learn version than the one used for training "
                f"({self.meta.get('sklearn_version', 'not recorded in metadata')})."
            ) from exc

        self.threshold = float(self.meta["threshold"])
        self.features = list(self.meta["input_features"])
        self.excluded = list(self.meta.get("excluded_columns", []))
        self.is_synthetic = bool(self.meta.get("synthetic", False))
        self._read_schema()

        self.load_warnings = []
        trained_with = self.meta.get("sklearn_version")
        if trained_with:
            import sklearn
            if trained_with.split(".")[:2] != sklearn.__version__.split(".")[:2]:
                self.load_warnings.append(
                    f"Model trained with scikit-learn {trained_with}, running {sklearn.__version__}."
                )

    # ------------------------------------------------------------------ schema
    def _read_schema(self):
        steps = self.pipeline.named_steps
        if "prep" not in steps or "model" not in steps:
            raise ArtifactError("Pipeline must have steps named 'prep' and 'model'.")
        prep, model = steps["prep"], steps["model"]
        groups = {name: list(cols) for name, _, cols in prep.transformers_ if name in ("num", "cat", "bin")}
        self.numeric = groups.get("num", [])
        self.categorical = groups.get("cat", [])
        self.binary = groups.get("bin", [])
        if set(self.features) != set(self.numeric + self.categorical + self.binary):
            raise ArtifactError("Metadata input_features do not match the pipeline's columns.")

        ohe = prep.named_transformers_["cat"]
        self.categories = {c: cats.tolist() for c, cats in zip(self.categorical, ohe.categories_)}
        scaler = prep.named_transformers_["num"]
        self.num_mean = dict(zip(self.numeric, scaler.mean_))
        self.num_std = dict(zip(self.numeric, scaler.scale_))

        # Explanation setup (needs a linear model)
        self.can_explain = hasattr(model, "coef_")
        names = prep.get_feature_names_out()
        fm = self.meta.get("feature_means")
        self.explanations_centered = bool(fm) and all(n in fm for n in names)
        self._means = np.array([fm[n] for n in names]) if self.explanations_centered else np.zeros(len(names))
        self._owners = [self._owner(n) for n in names]
        self.base_logit = (
            float(model.intercept_[0] + model.coef_[0] @ self._means) if self.can_explain else None
        )

    def _owner(self, out_name: str) -> str:
        """Map an output column such as 'cat__Course_171' back to its input field."""
        prefix, rest = out_name.split("__", 1)
        if prefix != "cat":
            return rest
        return max((c for c in self.categorical if rest.startswith(c + "_")), key=len)

    # -------------------------------------------------------------- validation
    def prepare(self, df: pd.DataFrame):
        """Validate/clean raw input. Returns (X, extra_columns, notes)."""
        if df is None or len(df) == 0:
            raise InputError("No student rows to score.")
        notes = []
        df = df.copy().reset_index(drop=True)
        df.columns = [str(c).strip() for c in df.columns]

        drop = [c for c in df.columns if c in self.excluded or c == "Target"]
        if drop:
            notes.append(
                "Ignored columns that are not available at the end of semester 1: " + ", ".join(drop)
            )
            df = df.drop(columns=drop)

        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise InputError("Missing required column(s): " + ", ".join(missing))
        extras = df[[c for c in df.columns if c not in self.features]]

        X = df[self.features].apply(pd.to_numeric, errors="coerce")
        bad = X.isna() | np.isinf(X)   # inf would crash the scaler, so treat it as invalid
        if bad.any().any():
            cells = [f"row {i + 1}: {c}" for c in X.columns[bad.any()] for i in X.index[bad[c]]]
            raise InputError(_clip("Missing or non-numeric value(s) at " + "; ".join(cells), cells))

        for c in self.binary:
            wrong = X.index[~X[c].isin([0, 1])]
            if len(wrong):
                raise InputError(f"'{c}' must be 0 or 1 (row(s) {_rows(wrong)}).")

        for c in self.categorical:
            unseen = sorted(set(X[c].unique()) - set(self.categories[c]))
            if unseen:
                notes.append(
                    f"{c}: code(s) {[int(u) if float(u).is_integer() else u for u in unseen]} were not "
                    "seen in training and are treated as 'unknown' (they add nothing to the score)."
                )

        for c in self.numeric:
            sd = self.num_std[c]
            if sd:
                far = X.index[((X[c] - self.num_mean[c]).abs() / sd) > 5]
                if len(far):
                    notes.append(f"{c}: value(s) far outside the training range (row(s) {_rows(far)}) - check for typos.")
        return X, extras, notes

    # ----------------------------------------------------------------- scoring
    def predict(self, df: pd.DataFrame) -> Scored:
        X, extras, notes = self.prepare(df)
        prob = self.pipeline.predict_proba(X)[:, 1]
        out = extras.copy()
        out["risk_score"] = np.round(prob, 3)
        out["flagged"] = prob >= self.threshold
        out["predicted_class"] = np.where(out["flagged"], "Dropout", "Not Dropout")
        return Scored(out, X, notes)

    # ------------------------------------------------------------ explanations
    def contributions(self, X: pd.DataFrame):
        """Per-student, per-field contribution to the log-odds, relative to the
        average training student. Sums (plus base_logit) to the model's logit."""
        if not self.can_explain:
            return None
        prep, model = self.pipeline.named_steps["prep"], self.pipeline.named_steps["model"]
        raw = (prep.transform(X) - self._means) * model.coef_[0]
        by_field = pd.DataFrame(raw, columns=self._owners, index=X.index).T.groupby(level=0).sum().T
        return by_field

    def top_factors(self, contrib_row: pd.Series, x_row: pd.Series, k: int = 5, min_effect: float = 0.05):
        """Return (raising, lowering) DataFrames with feature / value / contribution."""
        s = contrib_row.sort_values()
        up, down = s[s > min_effect].tail(k)[::-1], s[s < -min_effect].head(k)
        make = lambda ser: pd.DataFrame(
            {"feature": ser.index, "value": [x_row[f] for f in ser.index], "contribution": ser.values}
        )
        return make(up), make(down)

    def main_risk_factors(self, X: pd.DataFrame, k: int = 3) -> list:
        contrib = self.contributions(X)
        if contrib is None:
            return [""] * len(X)
        out = []
        for i in range(len(X)):
            up, _ = self.top_factors(contrib.iloc[i], X.iloc[i], k)
            out.append("; ".join(f"{r.feature} = {r.value:g}" for r in up.itertuples()))
        return out


# --------------------------------------------------------------------- helpers
def _rows(idx) -> str:
    shown = [str(i + 1) for i in list(idx)[:5]]
    return ", ".join(shown) + (" ..." if len(idx) > 5 else "")


def _clip(msg: str, cells: list, limit: int = 8) -> str:
    if len(cells) <= limit:
        return msg
    return "Missing or non-numeric value(s) at " + "; ".join(cells[:limit]) + f"; ... and {len(cells) - limit} more."


# ------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score students for end-of-semester-1 dropout risk.")
    ap.add_argument("--csv", required=True, help="CSV with the 30 semester-1 columns (any extra ID columns are kept)")
    ap.add_argument("--out", help="write the scored table to this CSV")
    ap.add_argument("--artifacts", help="folder with the .joblib and metadata .json files")
    ap.add_argument("--explain", action="store_true", help="add each student's top risk-raising factors")
    args = ap.parse_args(argv)

    try:
        predictor = DropoutPredictor(args.artifacts)
        df = pd.read_csv(args.csv, sep=None, engine="python")  # sniffs ',' vs ';'
        scored = predictor.predict(df)
    except (ArtifactError, InputError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    res = scored.results
    if args.explain:
        res["main_risk_factors"] = predictor.main_risk_factors(scored.features)
    res = res.sort_values("risk_score", ascending=False)

    if predictor.is_synthetic:
        print("WARNING: synthetic demo model - scores are NOT meaningful.\n", file=sys.stderr)
    for n in scored.notes + predictor.load_warnings:
        print(f"NOTE: {n}", file=sys.stderr)
    print(f"Flag threshold: {predictor.threshold:.3f}  |  flagged {int(res['flagged'].sum())} of {len(res)}\n")
    print(res.to_string(index=False))
    if args.out:
        res.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
