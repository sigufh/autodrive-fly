from __future__ import annotations

import numpy as np
from scipy import sparse


def iter_propagation(
    adjacency: sparse.csr_matrix,
    initial: np.ndarray,
    *,
    steps: int = 6,
    leak: float = 0.35,
    recurrent_gain: float = 1.0,
    nonlinear: bool = False,
) -> np.ndarray:
    """Propagate scalar state over every canonical MaleCNS node.

    The linear default exactly matches transfer-basis compilation. A bounded
    tanh variant is available for later nonlinear dynamics experiments.
    """
    if initial.shape != (adjacency.shape[0],):
        raise ValueError(f"expected state shape {(adjacency.shape[0],)}, got {initial.shape}")
    if steps < 0 or not 0 <= leak <= 1:
        raise ValueError("steps must be non-negative and leak must be in [0, 1]")
    state = initial.astype(np.float32, copy=True)
    yield state
    for _ in range(steps):
        recurrent = adjacency @ state * recurrent_gain
        if nonlinear:
            recurrent = np.tanh(recurrent)
        state = (1.0 - leak) * state + leak * recurrent
        yield state


def propagate(adjacency, initial, **kwargs) -> np.ndarray:
    return np.stack(list(iter_propagation(adjacency, initial, **kwargs)))
