import numpy as np

from .exceptions import DegenerateVectorError
from .validation import validate_trajectory


def normalize_to_sphere(T):
    """Project points onto the unit sphere."""
    T = validate_trajectory(T)

    norms = np.linalg.norm(T, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise DegenerateVectorError("Zero-length vector encountered.")

    return T / norms
