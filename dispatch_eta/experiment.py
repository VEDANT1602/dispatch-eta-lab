"""One predeclared CPU experiment: no tuning or refitting on calibration/test."""

from pathlib import Path
import csv
import hashlib
import json
import platform

import numpy as np
import xgboost

from .data import FEATURES, generate, matrix, package, split, targets
from .evaluate import audit
from .intervals import bounds, calibrate


ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = {"objective": "reg:squarederror", "n_estimators": 180, "max_depth": 4,
              "learning_rate": 0.06, "min_child_weight": 10, "reg_lambda": 3.0,
              "subsample": 1.0, "colsample_bytree": 1.0, "tree_method": "hist",
              "max_bin": 128, "n_jobs": 1, "device": "cpu", "random_state": 73}


class LinearBaseline:
    """OLS with the same known-at-dispatch features; no identity/target fields."""
    def fit(self, x, y):
        self.coefficients = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
        return self

    def predict(self, x):
        return np.maximum(0, np.column_stack([np.ones(len(x)), x]) @ self.coefficients)


def fit_models(training):
    x, y = matrix(training), targets(training)
    baseline = LinearBaseline().fit(x, y)
    candidate = xgboost.XGBRegressor(**PARAMETERS)
    candidate.fit(x, y)
    return {"linear": baseline, "xgboost": candidate}


def predict(model, events):
    return np.maximum(0, model.predict(matrix(events)).astype(float))


def source_hashes():
    paths = sorted((ROOT / "dispatch_eta").glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    paths += [ROOT / "requirements.txt"]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def write_predictions(events, predictions, path):
    common = ["event_id", "dispatch_minute", "day", "route_id", "courier_id", "zone", "rush", "shift", "eta_minutes"]
    fields = common + [name + "_" + field for name in predictions for field in ("prediction", "lower", "upper")]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, event in enumerate(events):
            row = {key: event[key] for key in common}
            for name, arrays in predictions.items():
                for field, values in zip(("prediction", "lower", "upper"), arrays):
                    row[name + "_" + field] = float(values[index])
            writer.writerow(row)


def run(output, seed=73, events_per_day=180, alpha=0.1):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    events = generate(seed, events_per_day)
    groups = split(events)
    models = fit_models(groups["train"])
    results, predictions, calibration = {}, {}, {}
    for name, model in models.items():
        cal = calibrate(targets(groups["calibration"]), predict(model, groups["calibration"]), alpha)
        point = predict(model, groups["test"])
        lower, upper = bounds(point, cal)
        results[name] = audit(groups["test"], targets(groups["test"]), point, lower, upper)
        predictions[name] = point, lower, upper
        calibration[name] = cal.as_dict()
    packed = package(events, output / "data/deliveries.csv.gz")
    metrics = {
        "schema_version": 1, "scope": "original synthetic delivery simulation; no company/customer data or production claim",
        "dataset": {**packed, "seed": seed, "events_per_day": events_per_day, "days": 36},
        "counts": {**{key: len(value) for key, value in groups.items()}, "embargo_excluded": len(events) - sum(map(len, groups.values()))},
        "protocol": {"train_days": "[0,18)", "calibration_days": "[20,24)", "test_days": "[26,36)",
                     "training_fit_as_of_day": 20, "calibration_as_of_day": 26,
                     "mechanism_shift_day": 31, "identity_disjoint": ["route_id", "courier_id"],
                     "features": list(FEATURES), "features_available": "before dispatch",
                     "model_selection": "fixed parameters; no search, early stopping, or calibration/test fitting",
                     "interval_method": "absolute-residual split conformal with ceil((n+1)*(1-alpha)) order statistic",
                     "coverage_claim": "exchangeability is required for the usual marginal guarantee; time drift violates it",
                     "policy_loss": {"late_minute_weight": 4, "slack_minute_weight": 1, "units": "illustrative simulation units, not money"}},
        "parameters": PARAMETERS, "calibration": calibration, "test": results,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "xgboost": xgboost.__version__,
                    "platform": platform.system(), "device": "cpu", "xgboost_threads": 1},
        "source_sha256": source_hashes(),
    }
    write_predictions(groups["test"], predictions, output / "predictions.csv")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    from .report import render
    render(output)
    return metrics
