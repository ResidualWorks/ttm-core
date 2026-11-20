import numpy as np

from .median import geometric_median
from .validation import validate_trajectory


def convergence_index(T):
    """Compute convergence index toward the geometric median."""
    T = validate_trajectory(T)

    if len(T) == 1:
        return 1.0

    m = geometric_median(T)
    distances = np.linalg.norm(T - m, axis=1)
    return 1.0 / (1.0 + np.mean(distances))
