"""
Simplified Dunning-Kruger orbit visualization.

This is a geometric demonstration only.
No risk scoring, no policy, no intervention.
"""

import numpy as np
import matplotlib.pyplot as plt

from ttm import normalize_to_sphere, trajectory_length, geometric_median, residual


def generate_dunning_kruger_orbit(n=40):
    """Generate a simple closed orbit approximating Dunning-Kruger dynamics."""
    t = np.linspace(0, 2 * np.pi, n)
    # Overconfidence -> collapse -> recovery -> caution
    x = np.sin(t) * (1 + 0.3 * np.sin(2 * t))
    y = np.cos(t) * (1 + 0.2 * np.cos(3 * t))
    z = 0.3 * np.sin(2 * t)
    return np.stack([x, y, z], axis=1)


def main():
    raw = generate_dunning_kruger_orbit()
    projected = normalize_to_sphere(raw)

    length = trajectory_length(projected)
    median = geometric_median(projected)
    res = residual(projected, np.mean(projected, axis=0, keepdims=True).repeat(len(projected), axis=0))

    print(f"Trajectory length: {length:.4f}")
    print(f"Geometric median: {median}")
    print(f"Residual mean norm: {np.mean(np.linalg.norm(res, axis=1)):.4f}")

    # Simple plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(projected[:, 0], projected[:, 1], projected[:, 2], "b-", linewidth=1.5)
    ax.scatter(*median, c="r", s=40, label="median")
    ax.set_title("Dunning-Kruger Orbit (projected)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("dunning_kruger_orbit.png", dpi=120)
    print("Saved: dunning_kruger_orbit.png")


if __name__ == "__main__":
    main()
