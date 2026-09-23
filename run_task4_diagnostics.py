"""
run_task4_diagnostics.py
Task 4: run diagnostics (diagonal/off-diagonal distributions, mean and
spread, eigenvalue spectrum, structure) on RBF, polynomial, and quantum
kernels for the SAME representative train split, so they are comparable.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel

from data import generate_toy_dataset
from kernels import polynomial_kernel
from feature_scaling import scale_to_range
from kernel_cache import get_or_build_kernel_matrix
from diagnostics import diagnose_kernel_matrix

# Fixed representative settings -- diagnostics compare KERNEL TYPES here,
# not a hyperparameter sweep (that's Task 5).
SEED = 0
N_QUBITS = 2
QUANTUM_LAYERS = 1
QUANTUM_ENTANGLER = None
ANGLE_MAX = np.pi
POLY_DEGREE = 2


def main():
    X_raw, y = generate_toy_dataset(
        n_samples=200, n_dim=2, signal_fraction=0.35,
        bg_sigma=1.0, shell_radius=2.0, shell_width=0.3, random_state=0,
    )

    idx = np.arange(len(X_raw))
    train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=y)
    X_train_raw, y_train = X_raw[train_idx], y[train_idx]

    scaler = StandardScaler().fit(X_train_raw)
    X_train_std = scaler.transform(X_train_raw)

    K_rbf = rbf_kernel(X_train_std, X_train_std, gamma=None)
    K_poly = polynomial_kernel(X_train_std, X_train_std, p=POLY_DEGREE)

    train_min = X_train_raw.min(axis=0)
    train_max = X_train_raw.max(axis=0)
    X_all_angles = scale_to_range(X_raw, train_min, train_max, target_min=0, target_max=ANGLE_MAX)
    K_full_q = get_or_build_kernel_matrix(
        X_all_angles, N_QUBITS, QUANTUM_LAYERS, QUANTUM_ENTANGLER,
        extra_config={"angle_max": ANGLE_MAX},
    )
    K_quantum = K_full_q[np.ix_(train_idx, train_idx)]

    results = {}
    results["RBF"] = diagnose_kernel_matrix(K_rbf, y_train, "RBF kernel (Task 4)")
    results["Polynomial"] = diagnose_kernel_matrix(K_poly, y_train, "Polynomial kernel (Task 4)")
    results["Quantum"] = diagnose_kernel_matrix(K_quantum, y_train, "Quantum kernel (Task 4)")

    print("\n=== Comparison summary ===")
    print(f"{'Kernel':12s} {'diag mean':>10s} {'off-diag mean':>15s} {'off-diag std':>13s}")
    for name, r in results.items():
        print(f"{name:12s} {r['diag_mean']:>10.4f} {r['off_diag_mean']:>15.4f} {r['off_diag_std']:>13.4f}")


if __name__ == "__main__":
    main()
