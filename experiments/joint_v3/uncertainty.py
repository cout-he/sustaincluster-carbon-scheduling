"""Small, deterministic uncertainty operations with explicit carbon units."""
import numpy as np


def robust_gain(gains, kappa=1.0):
    """Return mean, sample standard deviation, and lower gain bound in kg CO2e."""
    values = np.asarray(gains, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError('Carbon gain scenarios must be a nonempty finite vector')
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError('kappa must be nonnegative and finite')
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return mean, std, mean - float(kappa) * std
