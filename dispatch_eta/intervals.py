"""Split conformal: finite-sample residual order statistic, without test labels."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Calibration:
    n: int
    alpha: float
    rank: int
    radius: float | None

    def as_dict(self):
        return {"n": self.n, "alpha": self.alpha, "rank": self.rank, "radius_minutes": self.radius,
                "status": "finite" if self.radius is not None else "unbounded_small_sample"}


def calibrate(y, predictions, alpha=0.1):
    y, predictions = np.asarray(y, dtype=float), np.asarray(predictions, dtype=float)
    if y.ndim != 1 or predictions.shape != y.shape or not len(y):
        raise ValueError("nonempty matching one-dimensional arrays required")
    if not np.isfinite(y).all() or not np.isfinite(predictions).all() or not 0 < alpha < 1:
        raise ValueError("finite values and alpha in (0,1) required")
    n = len(y)
    rank = math.ceil((n + 1) * (1 - alpha))
    radius = float(np.sort(np.abs(y - predictions))[rank - 1]) if rank <= n else None
    return Calibration(n, alpha, rank, radius)


def bounds(predictions, calibration):
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 1 or not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("finite nonnegative predictions required")
    if calibration.radius is None:
        return np.zeros(len(predictions)), np.full(len(predictions), np.inf)
    return np.maximum(0, predictions - calibration.radius), predictions + calibration.radius
