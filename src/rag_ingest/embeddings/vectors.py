"""Vector helpers shared by the embedders (numpy is imported only here and below)."""
import numpy as np


def normalise_rows(vectors: np.ndarray) -> np.ndarray:
    """float32 rows scaled to length 1 (a zero row stays zero)."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
