import numpy as np

from .validation import validate_trajectory


def trajectory_length(T):
    """Compute total path length of a trajectory."""
    T = validate_trajectory(T)

    if len(T) < 2:
        return 0.0

    diffs = np.diff(T, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))
