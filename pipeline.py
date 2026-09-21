"""
pipeline.py
Week 4, Task 1: one shared pipeline evaluating all four models
(Logistic Regression, RBF SVM, Polynomial SVM, Quantum SVM) on the
SAME train/test split, with the SAME "fit preprocessing on train only"
rule, the SAME evaluation metric, and the SAME style of cross-validation.

Two families of model share one evaluation:
  - "Direct" models (Logistic Regression, RBF SVM) call sklearn's normal
    .fit(X, y) on standardised features.
  - "Kernel" models (Polynomial, Quantum) build an explicit N x N Gram
    matrix from TRAIN data, cross-validate C on that matrix via
    kernel="precomputed", then score on a TEST x TRAIN block.

Quantum kernel matrices here are built directly (exact_kernel_overlap,
looped). Task 2 swaps this for the cached version from week3_experiments.py
so the pipeline does not recompute the same quantum kernel over and over --
kept uncached here so Task 1's logic is easy to read and verify on its own.
"""

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

from kernels import tune_logistic_regression, tune_rbf_svm, polynomial_kernel
from quantum_kernels import exact_kernel_overlap
from feature_scaling import scale_to_range


def tune_precomputed_svm(K_train, y_train, cv=5, C_grid=None):
    """
    Cross-validate C for an SVM given a precomputed train-train kernel
    matrix. Works for ANY kernel (polynomial, quantum, ...) as long as
    K_train is the full N_train x N_train Gram matrix: GridSearchCV
    re-indexes both rows and columns per fold for a precomputed kernel,
    so this is a genuine cross-validation, not a shortcut -- exactly the
    "equal footing" style already used for tune_rbf_svm.
    """
    if C_grid is None:
        C_grid = np.logspace(-2, 3, 11)  # same span as tune_rbf_svm's C grid

    grid = GridSearchCV(
        SVC(kernel="precomputed"),
        {"C": C_grid},
        cv=cv,
        scoring="roc_auc",
    )
    grid.fit(K_train, y_train)
    return grid.best_estimator_, grid.best_params_


def build_polynomial_kernel_matrices(X_train, X_test, p=2):
    """
    Explicit train-train and test-train Gram matrices using the
    hand-derived polynomial_kernel formula (kernel(x,y) = (x . y)^p),
    consistent with using hand-derived kernels over sklearn's built-in
    kernel='poly'.
    """
    K_train = polynomial_kernel(X_train, X_train, p=p)
    K_test = polynomial_kernel(X_test, X_train, p=p)
    return K_train, K_test


def build_quantum_kernel_matrices(X_train, X_test, n_qubits, layers=1, entangler=None):
    """
    Explicit train-train and test-train Gram matrices using the exact
    (statevector) quantum kernel validated in Week 3.
    """
    n_train, n_test = len(X_train), len(X_test)

    K_train = np.zeros((n_train, n_train))
    for i in range(n_train):
        for j in range(n_train):
            K_train[i, j] = exact_kernel_overlap(
                X_train[i], X_train[j], n_qubits, layers=layers, entangler=entangler
            )

    K_test = np.zeros((n_test, n_train))
    for i in range(n_test):
        for j in range(n_train):
            K_test[i, j] = exact_kernel_overlap(
                X_test[i], X_train[j], n_qubits, layers=layers, entangler=entangler
            )

    return K_train, K_test


def run_seed(
    X_raw, y, seed, n_qubits,
    test_size=0.3,
    poly_degree=2,
    quantum_layers=1,
    quantum_entangler=None,
    angle_max=np.pi,
):
    """
    Run all four models on ONE seed's train/test split.
    Returns {"logreg": auc, "rbf": auc, "poly": auc, "quantum": auc}.

    Shared across every model: this split, "fit-on-train-only" for
    whatever preprocessing that model needs, roc_auc as the metric,
    and GridSearchCV with the same cv/scoring settings.
    """
    idx = np.arange(len(X_raw))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y
    )
    X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    results = {}

    # --- Direct models: standardised features, scaler fit on train only ---
    scaler = StandardScaler().fit(X_train_raw)
    X_train_std = scaler.transform(X_train_raw)
    X_test_std = scaler.transform(X_test_raw)

    logreg = tune_logistic_regression(X_train_std, y_train, random_state=seed)
    results["logreg"] = roc_auc_score(y_test, logreg.decision_function(X_test_std))

    rbf = tune_rbf_svm(X_train_std, y_train, random_state=seed)
    results["rbf"] = roc_auc_score(y_test, rbf.decision_function(X_test_std))

    # --- Polynomial SVM: precomputed kernel, C tuned by CV ---
    K_train_poly, K_test_poly = build_polynomial_kernel_matrices(
        X_train_std, X_test_std, p=poly_degree
    )
    poly_svm, poly_params = tune_precomputed_svm(K_train_poly, y_train)
    results["poly"] = roc_auc_score(y_test, poly_svm.decision_function(K_test_poly))

    # --- Quantum SVM: angle-scaled features, precomputed kernel, C tuned by CV ---
    train_min = X_train_raw.min(axis=0)
    train_max = X_train_raw.max(axis=0)
    X_train_angles = scale_to_range(X_train_raw, train_min, train_max, target_min=0, target_max=angle_max)
    X_test_angles = scale_to_range(X_test_raw, train_min, train_max, target_min=0, target_max=angle_max)

    K_train_q, K_test_q = build_quantum_kernel_matrices(
        X_train_angles, X_test_angles,
        n_qubits=n_qubits, layers=quantum_layers, entangler=quantum_entangler,
    )
    quantum_svm, quantum_params = tune_precomputed_svm(K_train_q, y_train)
    results["quantum"] = roc_auc_score(y_test, quantum_svm.decision_function(K_test_q))

    return results


if __name__ == "__main__":
    # Quick manual check: run all 4 models over 5 seeds on the Week 1 toy dataset.
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
