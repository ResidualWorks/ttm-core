import numpy as np

from .validation import validate_trajectory


def geometric_median(T, eps=1e-6, max_iter=100):
    """Compute geometric median using Weiszfeld's algorithm."""
    T = validate_trajectory(T)

    if len(T) == 1:
        return T[0].copy()

    y = np.mean(T, axis=0)

    for _ in range(max_iter):
        dist = np.linalg.norm(T - y, axis=1)
        dist = np.maximum(dist, eps)

        weights = 1.0 / dist
        y_next = np.sum(T * weights[:, None], axis=0) / np.sum(weights)

        if np.linalg.norm(y_next - y) < eps:
            return y_next

        y = y_next

    return y
