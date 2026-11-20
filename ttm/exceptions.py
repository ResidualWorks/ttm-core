class TTMError(Exception):
    """Base exception for TTM."""
    pass


class InvalidTrajectoryError(TTMError):
    """Trajectory validation failed."""
    pass


class EmptyTrajectoryError(TTMError):
    """Trajectory is empty."""
    pass


class DimensionMismatchError(TTMError):
    """Trajectory dimensions do not match."""
    pass


class DegenerateVectorError(TTMError):
    """Zero-length or degenerate vector encountered."""
    pass


class EmptyLibraryError(TTMError):
    """Topology library is empty."""
    pass
