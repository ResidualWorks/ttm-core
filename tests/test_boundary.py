import numpy as np
import pytest

from ttm.convergence import convergence_index
from ttm.distance import trajectory_distance
from ttm.exceptions import (
    DegenerateVectorError,
    DimensionMismatchError,
    EmptyLibraryError,
    EmptyTrajectoryError,
    InvalidTrajectoryError,
)
from ttm.median import geometric_median
from ttm.projection import normalize_to_sphere
from ttm.topology import match_topology
from ttm.trajectory import trajectory_length
from ttm.validation import validate_trajectory


def test_empty_trajectory():
    with pytest.raises(EmptyTrajectoryError):
        validate_trajectory(np.empty((0, 2)))


def test_nan_input():
    with pytest.raises(InvalidTrajectoryError):
        validate_trajectory(np.array([[1.0, np.nan]]))


def test_inf_input():
    with pytest.raises(InvalidTrajectoryError):
        validate_trajectory(np.array([[1.0, np.inf]]))


def test_single_point_length():
    T = np.array([[1.0, 2.0]])
    assert trajectory_length(T) == 0.0


def test_zero_vector_projection():
    T = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(DegenerateVectorError):
        normalize_to_sphere(T)


def test_dimension_mismatch():
    A = np.zeros((5, 2))
    B = np.zeros((5, 3))
    with pytest.raises(DimensionMismatchError):
        trajectory_distance(A, B)


def test_duplicate_points_median():
    T = np.array([[5.0, 7.0]] * 20)
    m = geometric_median(T)
    assert np.allclose(m, [5.0, 7.0])


def test_single_point_convergence():
    T = np.array([[1.0, 1.0]])
    c = convergence_index(T)
    assert np.isclose(c, 1.0)


def test_empty_library():
    T = np.array([[1.0, 2.0]])
    with pytest.raises(EmptyLibraryError):
        match_topology(T, {})


def test_large_values():
    T = np.array([[1e12, 0.0], [0.0, 1e12]])
    value = trajectory_length(T)
    assert np.isfinite(value)


def test_tiny_values():
    T = np.array([[1e-12, 0.0, 0.0]])
    result = normalize_to_sphere(T)
    assert np.allclose(result, [[1.0, 0.0, 0.0]])
