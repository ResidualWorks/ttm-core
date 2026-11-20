import numpy as np

from .exceptions import DimensionMismatchError
from .validation import validate_trajectory


def residual(observed, template):
    """Extract residual between observed trajectory and template."""
    observed = validate_trajectory(observed)
    template = validate_trajectory(template)

    if observed.shape[1] != template.shape[1]:
        raise DimensionMismatchError("Dimension mismatch between observed and template.")

    n = min(len(observed), len(template))
    return observed[:n] - template[:n]
