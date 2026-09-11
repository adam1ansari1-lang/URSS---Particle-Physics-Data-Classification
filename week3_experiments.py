# ============ TASK 12 + TASK 14 (real dataset) ============
# Build the quantum kernel matrix on the ACTUAL Week 1 toy dataset,
# cache it, and feed it into a precomputed-kernel SVM -- properly
# scaled and split per seed, with no data leakage.

import numpy as np
import hashlib   # turns settings into a short unique filename
import json       # turns settings into one stable, sortable string
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from quantum_kernels import exact_kernel_overlap, basic_entangler   # Tasks 2/4/10
from feature_scaling import scale_to_range                           # Task 8
from data import generate_toy_dataset                                 # Week 1's real generator

CACHE_DIR = "results/kernel_cache"
os.makedirs(CACHE_DIR, exist_ok=True)   # make the folder if it doesn't exist yet


def make_cache_key(config):
    # sort_keys=True: the SAME settings always give the SAME key
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def build_exact_kernel_matrix(X, n_qubits, layers, entangler):
    N = len(X)
    K = np.zeros((N, N))   # empty grid to fill in
    for i in range(N):
        for j in range(N):
            K[i, j] = exact_kernel_overlap(X[i], X[j], n_qubits, layers=layers, entangler=entangler)
    return K


def get_or_build_kernel_matrix(X, config, n_qubits, layers, entangler):
    key = make_cache_key(config)
    path = os.path.join(CACHE_DIR, f"kernel_{key}.npy")

    if os.path.exists(path):
        print(f"Cache HIT -- loading {path}")
        return np.load(path)

    print(f"Cache MISS -- building a fresh matrix for config={config}")
    K = build_exact_kernel_matrix(X, n_qubits, layers, entangler)
    np.save(path, K)   # save so next run with the SAME config is instant
    return K


def evaluate_precomputed_kernel_svm(K, y, train_idx, test_idx, C=1.0):
    K_train = K[np.ix_(train_idx, train_idx)]   # similarities: train vs train
    K_test = K[np.ix_(test_idx, train_idx)]     # similarities: test vs train

    y_train, y_test = y[train_idx], y[test_idx]

    svm = SVC(kernel="precomputed", C=C)
    svm.fit(K_train, y_train)

    scores = svm.decision_function(K_test)
    return roc_auc_score(y_test, scores)


# --- Real toy dataset, matching Week 1's setup ---
DATASET_SEED = 0
N_SAMPLES = 200   # the real project size

X_raw, y = generate_toy_dataset(
    n_samples=N_SAMPLES, n_dim=2, signal_fraction=0.35,
    bg_sigma=1.0, shell_radius=2.0, shell_width=0.3, random_state=DATASET_SEED
)

aucs = []
for seed in range(5):   # 5 seeds, per the project's ground rules -- mean +/- std, not one number
    idx = np.arange(len(X_raw))
    train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=seed, stratify=y)

    # scaler fit on TRAINING points ONLY -- avoids leaking test info into
    # the scaling range, matching the Week 1/2 convention exactly
    train_min = X_raw[train_idx].min(axis=0)
    train_max = X_raw[train_idx].max(axis=0)
    X_scaled = scale_to_range(X_raw, train_min, train_max, target_min=0, target_max=np.pi)

    # a DIFFERENT split each seed means a DIFFERENT scaler, so the cache
    # key must include split_seed too, or seeds would collide
    config = {
        "dataset_seed": DATASET_SEED,
        "n_samples": N_SAMPLES,
        "split_seed": seed,
        "n_qubits": 2,
        "layers": 2,
        "entangler": "basic_cnot",
    }
    K = get_or_build_kernel_matrix(X_scaled.tolist(), config, n_qubits=2, layers=2, entangler=basic_entangler)

    auc = evaluate_precomputed_kernel_svm(K, y, train_idx, test_idx)
    aucs.append(auc)
    print(f"seed={seed}  AUC={auc:.3f}")

aucs = np.array(aucs)
print(f"\nQuantum kernel SVM AUC: {aucs.mean():.3f} +/- {aucs.std():.3f}")
print("Compare against Week 1 baselines:")
print("  Logistic Regression: 0.500 +/- 0.044")
print("  RBF SVM:              0.845 +/- 0.038")
