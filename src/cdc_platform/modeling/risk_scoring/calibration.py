from __future__ import annotations
import numpy as np

def platt_scale(probs: np.ndarray, labels: np.ndarray, iters: int = 2000, lr: float = 0.05) -> tuple[float, float]:
    """
    Learn A,B so calibrated = sigmoid(A*logit(p)+B).
    """
    eps = 1e-6
    p = np.clip(probs, eps, 1 - eps)
    logit = np.log(p / (1 - p))

    A, B = 1.0, 0.0
    y = labels.astype(float)

    def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

    for _ in range(iters):
        z = A * logit + B
        s = sigmoid(z)
        dA = np.mean((s - y) * logit)
        dB = np.mean(s - y)
        A -= lr * dA
        B -= lr * dB

    return float(A), float(B)
