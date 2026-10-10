"""Classical Croston with fixed alpha, applied only to historical candidates."""
import numpy as np


def is_intermittent(values, config):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    c = config['intermittent']
    if len(x) < c['minimum_observations']:
        return False
    positive = (x > 0).sum()
    adi = len(x) / positive if positive else np.inf
    return bool(adi >= c['adi_threshold'] or (x == 0).mean() >= c['zero_fraction_threshold'])


def croston(values, alpha=0.1):
    """Initialize size at first positive and interval at its 1-based position.
    Only positive arrivals update states. Missing days cannot mean zero arrivals.
    Returns the constant expected daily rate, not a prediction interval.
    """
    x = np.asarray(values, dtype=float)
    if not 0 < alpha <= 1 or np.any(x < 0):
        raise ValueError('Invalid Croston parameters or demand')
    if not len(x) or not np.isfinite(x).all():
        return np.nan
    arrivals = np.flatnonzero(x > 0)
    if not len(arrivals):
        return 0.0
    first = arrivals[0]
    size, interval, previous = x[first], float(first + 1), first
    for index in arrivals[1:]:
        size = alpha * x[index] + (1 - alpha) * size
        interval = alpha * (index - previous) + (1 - alpha) * interval
        previous = index
    return float(size / interval)
