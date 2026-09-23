"""
run_task5_sweep.py
Task 5/6: sweep quantum feature-map structure (entanglement, depth,
qubit count) and record test/train AUC + kernel diagnostics per config.

angle_max and C stay fixed (Task 3's tuning is a SEPARATE thing) --
this sweep is about circuit STRUCTURE, not about re-tuning per config.
"""

import csv
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

from data import generate_toy_dataset
from feature_scaling import scale_to_range
from kernel_cache import get_or_build_kernel_matrix
from quantum_kernels import chain_entangler

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SEEDS = 5
ANGLE_MAX = np.pi  # fixed representative setting -- not re-tuned per config
C_FIXED = 10.0      # fixed representative C -- ditto

# Separation calibrated so n_dim=2 reproduces the familiar shell_radius=2.0,
# then scaled as sqrt(n_dim) so separation stays constant across
# dimensions. Without this, the qubit-count sweep would re-run Week 1's
# difficulty-vs-dimension curve instead of isolating the kernel's behaviour.
BG_SIGMA = 1.0
SEPARATION = 2.0 / (BG_SIGMA * np.sqrt(2))


def shell_radius_for(n_dim):
    return SEPARATION * BG_SIGMA * np.sqrt(n_dim)


def run_one_config(n_dim, n_qubits, layers, entangler, seeds=range(N_SEEDS)):
    shell_radius = shell_radius_for(n_dim)
    X_raw, y = generate_toy_dataset(
        n_samples=200, n_dim=n_dim, signal_fraction=0.35,
        bg_sigma=BG_SIGMA, shell_radius=shell_radius, shell_width=0.3 * shell_radius,
        random_state=0,
    )

    test_aucs, train_aucs = [], []
    representative = {}

    for seed in seeds:
        idx = np.arange(len(X_raw))
        train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=seed, stratify=y)
        X_train_raw, y_train = X_raw[train_idx], y[train_idx]
        y_test = y[test_idx]

        train_min = X_train_raw.min(axis=0)
        train_max = X_train_raw.max(axis=0)
        X_all_angles = scale_to_range(X_raw, train_min, train_max, target_min=0, target_max=ANGLE_MAX)

        K_full = get_or_build_kernel_matrix(
            X_all_angles, n_qubits, layers, entangler,
            extra_config={"angle_max": ANGLE_MAX, "n_dim": n_dim},
        )
        K_train = K_full[np.ix_(train_idx, train_idx)]
        K_test = K_full[np.ix_(test_idx, train_idx)]

        svm = SVC(kernel="precomputed", C=C_FIXED)
        svm.fit(K_train, y_train)

        test_aucs.append(roc_auc_score(y_test, svm.decision_function(K_test)))
        train_aucs.append(roc_auc_score(y_train, svm.decision_function(K_train)))

        if seed == seeds[0]:
            off_diag = K_train[~np.eye(len(K_train), dtype=bool)]
            representative = {
                "off_diag_mean": off_diag.mean(),
                "off_diag_std": off_diag.std(),
                "eigenvalues": np.linalg.eigvalsh(K_train),
            }

    return {
        "n_dim": n_dim,
        "n_qubits": n_qubits,
        "layers": layers,
        "entangler": "none" if entangler is None else entangler.__name__,
        "test_auc_mean": np.mean(test_aucs),
        "test_auc_std": np.std(test_aucs),
        "train_auc_mean": np.mean(train_aucs),
        "train_auc_std": np.std(train_aucs),
        "off_diag_mean": representative["off_diag_mean"],
        "off_diag_std": representative["off_diag_std"],
    }


def write_csv(rows, filename):
    path = os.path.join(RESULTS_DIR, filename)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def sweep_depth():
    print("\n=== Depth sweep (n_qubits=2, entangler=chain_entangler) ===")
    rows = []
    for layers in [1, 2, 3, 4]:
        r = run_one_config(n_dim=2, n_qubits=2, layers=layers, entangler=chain_entangler)
        rows.append(r)
        print(f"layers={layers}  test_auc={r['test_auc_mean']:.3f}+/-{r['test_auc_std']:.3f}  "
              f"train_auc={r['train_auc_mean']:.3f}  off_diag_mean={r['off_diag_mean']:.3f}")
    write_csv(rows, "task5_depth_sweep.csv")


def sweep_entanglement():
    print("\n=== Entanglement sweep (n_qubits=2, layers=3) ===")
    rows = []
    for entangler in [None, chain_entangler]:
        r = run_one_config(n_dim=2, n_qubits=2, layers=3, entangler=entangler)
        rows.append(r)
        print(f"entangler={r['entangler']}  test_auc={r['test_auc_mean']:.3f}+/-{r['test_auc_std']:.3f}  "
              f"train_auc={r['train_auc_mean']:.3f}  off_diag_mean={r['off_diag_mean']:.3f}")
    write_csv(rows, "task5_entanglement_sweep.csv")


def sweep_qubit_count():
    print("\n=== Qubit-count sweep (layers=2, entangler=chain_entangler) ===")
    rows = []
    for n_dim in [2, 3, 4, 6, 8]:
        r = run_one_config(n_dim=n_dim, n_qubits=n_dim, layers=2, entangler=chain_entangler)
        rows.append(r)
        print(f"n_qubits={n_dim}  test_auc={r['test_auc_mean']:.3f}+/-{r['test_auc_std']:.3f}  "
              f"train_auc={r['train_auc_mean']:.3f}  off_diag_mean={r['off_diag_mean']:.3f}")
    write_csv(rows, "task5_qubit_count_sweep.csv")


if __name__ == "__main__":
    sweep_depth()
    sweep_entanglement()
    sweep_qubit_count()
