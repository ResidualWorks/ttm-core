import numpy as np

from .exceptions import (
    EmptyTrajectoryError,
    InvalidTrajectoryError,
)


def validate_trajectory(T):
    """Validate and convert input to a 2D finite trajectory array."""
    T = np.asarray(T, dtype=float)

    if T.ndim != 2:
        raise InvalidTrajectoryError("Trajectory must be 2D (N, D).")

    if len(T) == 0:
        raise EmptyTrajectoryError("Trajectory is empty.")

    if not np.isfinite(T).all():
        raise InvalidTrajectoryError("Trajectory contains NaN or Inf.")

    return T
