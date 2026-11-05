"""Duration, interval and promise metrics with explicit denominators."""

import numpy as np
import math


def promise_policy(actual, promise, late_weight=4.0, slack_weight=1.0):
    actual, promise = np.asarray(actual, dtype=float), np.asarray(promise, dtype=float)
    if actual.ndim != 1 or actual.shape != promise.shape:
        raise ValueError("matching one-dimensional arrays required")
    if not np.isfinite(actual).all():
        raise ValueError("finite observations required")
    if not math.isfinite(late_weight) or not math.isfinite(slack_weight) or late_weight < 0 or slack_weight < 0:
        raise ValueError("nonnegative policy weights required")
    n = len(actual)
    if not n or not np.isfinite(promise).all():
        return {"n": n, "status": "no_rows" if not n else "unbounded_promise",
                "missed_promises": None, "miss_rate": None, "mean_late_minutes": None,
                "mean_slack_minutes": None, "mean_loss_units": None}
    late, slack = np.maximum(actual - promise, 0), np.maximum(promise - actual, 0)
    return {"n": n, "status": "ok", "missed_promises": int((late > 0).sum()),
            "miss_rate": float((late > 0).mean()), "mean_late_minutes": float(late.mean()),
            "mean_slack_minutes": float(slack.mean()),
            "mean_loss_units": float((late_weight * late + slack_weight * slack).mean())}


def metrics(actual, prediction, lower, upper):
    arrays = [np.asarray(a, dtype=float) for a in (actual, prediction, lower, upper)]
    actual, prediction, lower, upper = arrays
    if any(a.ndim != 1 or a.shape != actual.shape for a in arrays):
        raise ValueError("matching one-dimensional arrays required")
    if np.isnan(np.concatenate(arrays)).any() or not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError("finite observations/predictions and non-NaN bounds required")
    if (lower > upper).any():
        raise ValueError("invalid interval")
    n = len(actual)
    error = prediction - actual
    covered = (actual >= lower) & (actual <= upper)
    finite = np.isfinite(upper - lower).all()
    return {
        "n": n, "mae_minutes": float(np.abs(error).mean()) if n else None,
        "rmse_minutes": float(np.sqrt((error ** 2).mean())) if n else None,
        "mean_signed_error_minutes": float(error.mean()) if n else None,
        "underestimate_over_10_minutes_count": int((error < -10).sum()),
        "underestimate_over_10_minutes_rate": float((error < -10).mean()) if n else None,
        "covered": int(covered.sum()), "coverage": float(covered.mean()) if n else None,
        "lower_misses": int((actual < lower).sum()), "upper_misses": int((actual > upper).sum()),
        "mean_interval_width_minutes": float((upper - lower).mean()) if n and finite else None,
        "interval_status": ("finite" if finite else "unbounded") if n else "not_applicable",
        "point_promise": promise_policy(actual, prediction), "upper_promise": promise_policy(actual, upper),
    }


def slice_masks(events):
    return {
        "all": np.ones(len(events), dtype=bool),
        "before_shift": np.asarray([not e["shift"] for e in events]),
        "mechanism_shift": np.asarray([bool(e["shift"]) for e in events]),
        "rush_hour": np.asarray([bool(e["rush"]) for e in events]),
        "off_peak": np.asarray([not e["rush"] for e in events]),
        **{"zone_" + zone: np.asarray([e["zone"] == zone for e in events]) for zone in ("core", "suburban", "hills")},
        **{"shift_zone_" + zone: np.asarray([bool(e["shift"]) and e["zone"] == zone for e in events]) for zone in ("core", "suburban", "hills")},
    }


def audit(events, actual, prediction, lower, upper):
    return {name: metrics(actual[mask], prediction[mask], lower[mask], upper[mask])
            for name, mask in slice_masks(events).items()}
