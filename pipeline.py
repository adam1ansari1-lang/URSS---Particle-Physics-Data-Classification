"""
pipeline.py
Runs all four models (LogReg, RBF, Polynomial, Quantum SVM) on the same
split/scaling/metric/CV so AUC differences reflect the model, not setup.
"""

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

from kernels import tune_logistic_regression, tune_rbf_svm, polynomial_kernel
from feature_scaling import scale_to_range
from kernel_cache import get_or_build_kernel_matrix


def tune_precomputed_svm(K_train, y_train, cv=5, C_grid=None):
    # Same C grid/CV style as tune_rbf_svm, just for a precomputed kernel.
    if C_grid is None:
        C_grid = np.logspace(-2, 3, 11)

    grid = GridSearchCV(
        SVC(kernel="precomputed"),
        {"C": C_grid},
        cv=cv,
        scoring="roc_auc",
    )
    grid.fit(K_train, y_train)
    return grid.best_estimator_, grid.best_params_


def build_polynomial_kernel_matrices(X_train, X_test, p=2):
    # Hand-derived kernel(x,y) = (x . y)^p, not sklearn's built-in 'poly'.
    K_train = polynomial_kernel(X_train, X_train, p=p)
    K_test = polynomial_kernel(X_test, X_train, p=p)
    return K_train, K_test


def build_quantum_kernel_matrices(
    X_all_angles, train_idx, test_idx, n_qubits, layers=1, entangler=None, angle_max=np.pi
):
    # X_all_angles covers the WHOLE dataset -- one cached matrix, sliced
    # into train/test blocks below, rather than two separate builds.
    K_full = get_or_build_kernel_matrix(
        X_all_angles, n_qubits, layers, entangler,
        extra_config={"angle_max": angle_max},
    )
    K_train = K_full[np.ix_(train_idx, train_idx)]
    K_test = K_full[np.ix_(test_idx, train_idx)]
    return K_train, K_test


def run_seed(
    X_raw, y, seed, n_qubits,
    test_size=0.3,
    poly_degree=2,
    quantum_layers=1,
    quantum_entangler=None,
    angle_max=np.pi,
):
    # One split, reused by every model below -- the "equal footing" rule.
    idx = np.arange(len(X_raw))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y
    )
    X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    results = {}

    # Scaler fit on TRAIN only -- avoids leaking test info into the scale.
    scaler = StandardScaler().fit(X_train_raw)
    X_train_std = scaler.transform(X_train_raw)
    X_test_std = scaler.transform(X_test_raw)

    # --- Logistic Regression: direct sklearn fit, no kernel involved ---
    logreg = tune_logistic_regression(X_train_std, y_train, random_state=seed)
    results["logreg"] = roc_auc_score(y_test, logreg.decision_function(X_test_std))

    # --- RBF SVM: direct sklearn fit, C and gamma tuned internally ---
    rbf = tune_rbf_svm(X_train_std, y_train, random_state=seed)
    results["rbf"] = roc_auc_score(y_test, rbf.decision_function(X_test_std))

    # --- Polynomial SVM: precomputed kernel, C tuned by CV ---
    K_train_poly, K_test_poly = build_polynomial_kernel_matrices(
        X_train_std, X_test_std, p=poly_degree
    )
    poly_svm, poly_params = tune_precomputed_svm(K_train_poly, y_train)
    results["poly"] = roc_auc_score(y_test, poly_svm.decision_function(K_test_poly))

    # --- Quantum SVM: angles need their own scaler (bounded, periodic) ---
    train_min = X_train_raw.min(axis=0)
    train_max = X_train_raw.max(axis=0)
    # Applied to the FULL dataset (not just train) so we get one N x N
    # matrix to cache, then slice into train/test blocks below.
    X_all_angles = scale_to_range(X_raw, train_min, train_max, target_min=0, target_max=angle_max)

    K_train_q, K_test_q = build_quantum_kernel_matrices(
        X_all_angles, train_idx, test_idx,
        n_qubits=n_qubits, layers=quantum_layers, entangler=quantum_entangler,
        angle_max=angle_max,
    )
    quantum_svm, quantum_params = tune_precomputed_svm(K_train_q, y_train)
    results["quantum"] = roc_auc_score(y_test, quantum_svm.decision_function(K_test_q))

    return results


if __name__ == "__main__":
    # Quick manual check: 5 seeds on the Week 1 toy dataset.
    from data import generate_toy_dataset

    X_raw, y = generate_toy_dataset(
        n_samples=200, n_dim=2, signal_fraction=0.35,
        bg_sigma=1.0, shell_radius=2.0, shell_width=0.3, random_state=0,
    )

    all_results = {name: [] for name in ["logreg", "rbf", "poly", "quantum"]}

    for seed in range(5):
        seed_results = run_seed(X_raw, y, seed, n_qubits=2)
        for name, auc in seed_results.items():
            all_results[name].append(auc)
        print(f"seed={seed}  " + "  ".join(f"{k}={v:.3f}" for k, v in seed_results.items()))

    print("\n=== Summary (mean +/- std over 5 seeds) ===")
    for name, aucs in all_results.items():
        aucs = np.array(aucs)
        print(f"{name:8s}: {aucs.mean():.3f} +/- {aucs.std():.3f}")
