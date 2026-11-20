import numpy as np

from .exceptions import DimensionMismatchError
from .validation import validate_trajectory


def trajectory_distance(A, B):
    """Compute mean pointwise Euclidean distance between two trajectories."""
    A = validate_trajectory(A)
    B = validate_trajectory(B)

    if A.shape[1] != B.shape[1]:
        raise DimensionMismatchError("Dimension mismatch between trajectories.")

    n = min(len(A), len(B))
    return float(np.mean(np.linalg.norm(A[:n] - B[:n], axis=1)))
