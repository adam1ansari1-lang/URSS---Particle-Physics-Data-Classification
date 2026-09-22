"""
kernel_cache.py
Week 4, Task 2: reusable quantum-kernel caching, generalising the pattern
already validated in week3_experiments.py so it works for any circuit
config (qubits, layers, entangler, angle_max) rather than one hardcoded
setup.

Same idea as Week 3: hash a config dict -> short filename -> .npy on disk.
If that exact config was ever computed before, load it instead of
re-running the (expensive) quantum simulation.

One extension beyond Week 3: the config also includes a hash of the
ACTUAL data array (not just a seed/n_samples pair). Week 3's dataset was
generated inline in that same script, so "dataset_seed + n_samples" was
enough to identify it. Here, get_or_build_kernel_matrix just receives
whatever X the caller passes in -- if the toy generator's parameters
change, or Week 5 swaps in real data, a seed number alone wouldn't catch
that the underlying points are different. Hashing the array itself does.
"""

import hashlib
import json
import os

import numpy as np

from quantum_kernels import exact_kernel_overlap

CACHE_DIR = "results/kernel_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def entangler_name(entangler):
    """
    Turn an entangler function (or None) into a stable string for the
    cache key, so adding a new entangler in Task 5 never requires
    remembering to update a name by hand -- the function's own __name__
    IS the cache-key name.
    """
    return "none" if entangler is None else entangler.__name__


def hash_array(X):
    """Short, stable fingerprint of a data array's actual contents."""
    return hashlib.sha256(np.asarray(X).tobytes()).hexdigest()[:16]


def make_cache_key(config):
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def build_exact_kernel_matrix(X, n_qubits, layers, entangler):
    """Full N x N Gram matrix over one array of angle-scaled points."""
    N = len(X)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):  # symmetric: only compute the upper triangle
            val = exact_kernel_overlap(X[i], X[j], n_qubits, layers=layers, entangler=entangler)
            K[i, j] = val
            K[j, i] = val
    return K


def get_or_build_kernel_matrix(X_angles, n_qubits, layers, entangler, extra_config=None):
    """
    X_angles: the FULL array of angle-scaled points (train+test together),
        matching the Week 3 pattern of building one matrix and slicing it
        with np.ix_ afterward, rather than separate train/test matrices.
    extra_config: dict of anything else that affects the kernel values and
        must therefore be part of the cache key (e.g. {"angle_max": 1.5,
        "split_seed": 3}). n_qubits/layers/entangler are always included
        automatically.

    Returns the full N x N kernel matrix.
    """
    config = {
        "dataset_hash": hash_array(X_angles),
        "n_qubits": n_qubits,
        "layers": layers,
        "entangler": entangler_name(entangler),
    }
    if extra_config:
        config.update(extra_config)

    key = make_cache_key(config)
    path = os.path.join(CACHE_DIR, f"kernel_{key}.npy")

    if os.path.exists(path):
        print(f"Cache HIT  -- {path}")
        return np.load(path)

    print(f"Cache MISS -- building fresh matrix for config={config}")
    K = build_exact_kernel_matrix(X_angles, n_qubits, layers, entangler)
    np.save(path, K)
    return K
