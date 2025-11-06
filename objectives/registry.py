from __future__ import annotations

from typing import Callable, Dict, Tuple
import numpy as np

from objectives.rosenbrock import rosenbrock, rosenbrock_grad
from objectives.sphere import sphere, sphere_grad
from objectives.himmelblau import himmelblau, himmelblau_grad
from objectives.rastrigin import rastrigin, rastrigin_grad


ObjectiveFn = Callable[[np.ndarray], float]
GradFn = Callable[[np.ndarray], np.ndarray]


OBJECTIVES: Dict[str, dict] = {
    "rosenbrock": {
        "fn": rosenbrock,
        "grad": rosenbrock_grad,
        "bounds": ((-2.0, 2.0), (-1.0, 3.0)),
        "start": [-1.5, 2.0],
        "global_mins": [[1.0, 1.0]],
    },
    "sphere": {
        "fn": sphere,
        "grad": sphere_grad,
        "bounds": ((-2.5, 2.5), (-2.5, 2.5)),
        "start": [2.0, -1.5],
        "global_mins": [[0.0, 0.0]],
    },
    "himmelblau": {
        "fn": himmelblau,
        "grad": himmelblau_grad,
        "bounds": ((-6.0, 6.0), (-6.0, 6.0)),
        "start": [-3.0, 3.0],
        "global_mins": [
            [3.0, 2.0],
            [-2.805118, 3.131312],
            [-3.779310, -3.283186],
            [3.584428, -1.848126],
        ],
    },
    "rastrigin": {
        "fn": rastrigin,
        "grad": rastrigin_grad,
        "bounds": ((-5.12, 5.12), (-5.12, 5.12)),
        "start": [3.5, -3.5],
        "global_mins": [[0.0, 0.0]],
    },
}


def get_objective(name: str) -> Tuple[ObjectiveFn, GradFn, tuple, list, list]:
    key = name.lower()
    if key not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {name}")
    item = OBJECTIVES[key]
    return item["fn"], item["grad"], item["bounds"], item["start"], item.get("global_mins", [])


