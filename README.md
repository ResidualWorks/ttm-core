# Topological Trajectory Model (TTM)

TTM is a trajectory-oriented framework for representing observations as motion through a state space.

Instead of asking:

```text
What is this?
```

TTM asks:

```text
How does it move?
```

The framework focuses on trajectory geometry, topology matching, and residual extraction.

---

## Philosophy

Many apparently different phenomena share similar trajectories.

Examples:

- epidemics
- market bubbles
- panic buying
- opinion cascades
- moral panics

TTM treats observations as trajectories rather than labels.

Reasoning follows:

```text
Observe
 ↓
Match
 ↓
Residual
 ↓
Reason
```

Classification is cheap.

Topology matching is cheap.

Residual reasoning is expensive.

---

## Scope

TTM is an observation framework.

TTM provides:

- state-space projection
- trajectory analysis
- geometric median computation
- convergence measurement
- topology matching
- residual extraction

TTM does NOT provide:

- risk scoring
- trust estimation
- intervention policies
- governance logic
- decision thresholds
- topology weighting schemes

These decisions are intentionally left to users and downstream applications.

TTM focuses on representing and comparing trajectories.
Interpretation and policy are outside the scope of this project.

---

## Public API

```python
normalize_to_sphere()

trajectory_length()

geometric_median()

convergence_index()

trajectory_distance()

match_topology()

residual()
```

---

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

---

## Quick Start

```python
import numpy as np
from ttm import normalize_to_sphere, trajectory_length, residual

T = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])

projected = normalize_to_sphere(T)
length = trajectory_length(projected)
```

---

## Examples

`examples/dunning_kruger_orbit.py` requires matplotlib, which is not a dependency of TTM:

```bash
pip install matplotlib
```

---

## License

MIT License
