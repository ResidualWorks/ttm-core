from dataclasses import dataclass

from .distance import trajectory_distance
from .exceptions import EmptyLibraryError


@dataclass(frozen=True)
class MatchResult:
    name: str
    score: float


def match_topology(trajectory, library):
    """Match a trajectory against a library of known topologies."""
    if not library:
        raise EmptyLibraryError("Topology library is empty.")

    best_name = None
    best_score = -1.0

    for name, ref in library.items():
        d = trajectory_distance(trajectory, ref)
        score = 1.0 / (1.0 + d)

        if score > best_score:
            best_score = score
            best_name = name

    return MatchResult(best_name, best_score)
