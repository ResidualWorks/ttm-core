import numpy as np

from ttm import (
    convergence_index,
    geometric_median,
    match_topology,
    normalize_to_sphere,
    residual,
    trajectory_distance,
    trajectory_length,
)


def test_projection_unit_norm():
    T = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    result = normalize_to_sphere(T)
    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0)


def test_trajectory_length_simple():
    T = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    assert trajectory_length(T) == 2.0


def test_distance_identity():
    T = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert np.isclose(trajectory_distance(T, T), 0.0)


def test_residual_self_zero():
    T = np.array([[1.0, 2.0], [3.0, 4.0]])
    R = residual(T, T)
    assert np.allclose(R, 0.0)


def test_match_identity():
    orbit = np.array([[0.0, 0.0], [1.0, 0.0]])
    library = {"a": orbit, "b": np.array([[10.0, 10.0], [11.0, 11.0]])}
    result = match_topology(orbit, library)
    assert result.name == "a"
    assert 0.0 <= result.score <= 1.0


def test_convergence_dense_vs_sparse():
    dense = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    sparse = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    assert convergence_index(dense) > convergence_index(sparse)


def test_geometric_median_single():
    T = np.array([[5.0, 4.0]])
    m = geometric_median(T)
    assert np.allclose(m, [5.0, 4.0])
