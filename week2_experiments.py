"""
week2_experiments.py
Runs Week 2 Tasks 5-7: precomputed-kernel SVM check, RBF vs polynomial
kernel comparison (repeated over multiple seeds), and Gram matrix
diagnostics.

Tasks 1-4 live as reusable functions in kernels.py; this script uses them
on the real Week 1 toy dataset.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

from data import generate_toy_dataset
from kernels import quadratic_feature_map, polynomial_kernel, verify_kernel_trick
from diagnostics import diagnose_kernel_matrix

RANDOM_STATE = 0
SEEDS = [0, 1, 2, 3, 4]  # repeat Task 6 over 5 seeds, per the review comment


def evaluate_kernels_one_seed(X, y, seed):
    """
    Run the Task 6 comparison (RBF vs. our own hand-built polynomial
    kernel) for one train/test split.

    Returns a dict of results, including the fitted objects for that
    split - the caller reuses these for the ROC plot and Task 7
    diagnostics when seed == RANDOM_STATE, so nothing is re-fit or
    re-scaled with a different train/test split than what's being reported.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    scaler = StandardScaler().fit(X_train)  # fit on train only, no leakage
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- RBF ---
    # No probability=True: roc_auc_score/roc_curve work fine with
    # decision_function scores, and probability=True triggers an
    # expensive extra internal calibration step we don't need here.
    
    rbf_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10]}
    rbf_search = GridSearchCV(
        SVC(kernel='rbf'), rbf_grid, scoring='roc_auc', cv=5
    )
    rbf_search.fit(X_train_s, y_train)
    auc_rbf = roc_auc_score(y_test, rbf_search.decision_function(X_test_s))

    # --- Polynomial: OUR hand-built homogeneous kernel K(x,y) = (x.y)^2 ---
    
    # scikit-learn's kernel='poly' is actually (gamma*x.y + coef0)^degree -
    # a DIFFERENT kernel to the one built and verified in Tasks 1-4.
    # kernel='precomputed' with our own polynomial_kernel() keeps this
    # consistent with the rest of the week. Our kernel has no gamma/coef0
    # knobs by construction, so only C is tunable here.
    K_train = polynomial_kernel(X_train_s, X_train_s, p=2)
    poly_grid = {'C': [0.1, 1, 10, 100]}
    poly_search = GridSearchCV(
        SVC(kernel='precomputed'), poly_grid, scoring='roc_auc', cv=5
    )
    poly_search.fit(K_train, y_train)
    K_test = polynomial_kernel(X_test_s, X_train_s, p=2)  # shape (n_test, n_train)
    auc_poly = roc_auc_score(y_test, poly_search.decision_function(K_test))

    return {
        "seed": seed,
        "auc_rbf": auc_rbf,
        "auc_poly": auc_poly,
        "rbf_best_params": rbf_search.best_params_,
        "poly_best_params": poly_search.best_params_,
        "rbf_search": rbf_search,
        "poly_search": poly_search,
        "X_train_s": X_train_s,
        "X_test_s": X_test_s,
        "y_train": y_train,
        "y_test": y_test,
        "K_train": K_train,
        "K_test": K_test,
    }


def main():
    os.makedirs("plots", exist_ok=True)  # avoid crashing if plots/ doesn't exist yet

    # Small dataset (N<=100) so kernel matrices stay easy to visualise
    X, y = generate_toy_dataset(n_samples=100, n_dim=2, random_state=RANDOM_STATE)
    print("Dataset shape:", X.shape, " Class counts:", np.bincount(y.astype(int)))

    # Tasks 1-2 sanity check: should print [1, 2.8284..., 4]
    phi_check = quadratic_feature_map(np.array([[1, 2]]))
    print("\nphi(1,2) =", phi_check)

    # Task 4: explicit feature map vs direct kernel formula should agree
    # to machine precision. assert (instead of just printing) so this
    # fails loudly and automatically if it's ever violated.
    K_explicit, K_direct, max_diff = verify_kernel_trick(X, p=2)
    print(f"\nTask 4 - max |Method A - Method B| = {max_diff:.2e}")
    assert max_diff < 1e-10, (
        f"Kernel trick verification failed: max difference {max_diff:.2e} "
        "is larger than expected machine precision (should be roughly "
        "1e-14 to 1e-16; anything near 1e-6 or above signals a real bug)."
    )

    # Task 5: linear SVM on explicit features vs precomputed-kernel SVM
    # should give (near) identical decision functions
    C_value = 1.0
    phi_X = quadratic_feature_map(X)

    svm_explicit = SVC(kernel='linear', C=C_value)
    svm_explicit.fit(phi_X, y)
    decision_explicit = svm_explicit.decision_function(phi_X)

    svm_precomputed = SVC(kernel='precomputed', C=C_value)
    svm_precomputed.fit(K_direct, y)
    decision_precomputed = svm_precomputed.decision_function(K_direct)

    max_decision_diff = np.max(np.abs(decision_explicit - decision_precomputed))
    agree_fraction = np.mean(
        svm_explicit.predict(phi_X) == svm_precomputed.predict(K_direct)
    )
    print(f"\nTask 5 - max decision function difference: {max_decision_diff:.2e}")
    print(f"Task 5 - fraction of matching predictions:  {agree_fraction:.3f}")

    # Task 6: RBF vs our own polynomial kernel, repeated over 5 seeds so
    # the reported result is a reproducible mean +/- std, not one lucky split.
    
    print("\nTask 6 - RBF vs Polynomial comparison (5 seeds):")
    results = [evaluate_kernels_one_seed(X, y, seed) for seed in SEEDS]

    aucs_rbf = np.array([r["auc_rbf"] for r in results])
    aucs_poly = np.array([r["auc_poly"] for r in results])
    print(f"  RBF:  AUC = {aucs_rbf.mean():.3f} +/- {aucs_rbf.std():.3f}")
    print(f"  Poly: AUC = {aucs_poly.mean():.3f} +/- {aucs_poly.std():.3f}")
    for r in results:
        print(
            f"    seed {r['seed']}: RBF {r['auc_rbf']:.3f} (params {r['rbf_best_params']}), "
            f"Poly {r['auc_poly']:.3f} (params {r['poly_best_params']})"
        )

    # Use the primary seed's already-fitted models for the ROC plot and
    # Task 7, rather than re-fitting anything with a different split.
    primary = next(r for r in results if r["seed"] == RANDOM_STATE)

    plt.figure(figsize=(5, 5))
    fpr, tpr, _ = roc_curve(primary["y_test"], primary["rbf_search"].decision_function(primary["X_test_s"]))
    plt.plot(fpr, tpr, label="RBF")
    fpr, tpr, _ = roc_curve(primary["y_test"], primary["poly_search"].decision_function(primary["K_test"]))
    plt.plot(fpr, tpr, label="Poly (our kernel)")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title(f'RBF vs Polynomial Kernel SVM (seed={RANDOM_STATE})')
    plt.savefig('plots/rbf_vs_poly_roc.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Task 7: Gram matrix diagnostics, on the TRAINING kernel matrix from
    # the primary seed (the worksheet asks for the "training Gram matrix"
    # specifically), using RBF's actual best gamma from Task 6 - not an
    # arbitrary hardcoded value, so we diagnose the kernel we really use.
    # X_train_s here was already scaled using a scaler fit on the training
    # split only, so there is no leakage.
    
    best_gamma = primary["rbf_search"].best_params_["gamma"]
    print(f"\nTask 7 - using best RBF gamma from Task 6: {best_gamma}")

    K_poly_train = primary["K_train"]

    X_train_s = primary["X_train_s"]
    
    # RBF kernel by hand: exp(-gamma * ||x - x'||^2) for every pair.
    # Broadcasting trick: X_train_s[:, None, :] has shape (n,1,2),
    # X_train_s[None, :, :] has shape (1,n,2); subtracting broadcasts
    # both size-1 axes up to n, giving every pairwise difference vector
    # x_i - x_j at once, in one (n, n, 2) array - no loop needed.
    
    diffs = X_train_s[:, None, :] - X_train_s[None, :, :]
    K_rbf_train = np.exp(-best_gamma * np.sum(diffs**2, axis=2))

    print()
    diagnose_kernel_matrix(K_poly_train, primary["y_train"], "Polynomial kernel")
    diagnose_kernel_matrix(K_rbf_train, primary["y_train"], "RBF kernel")


if __name__ == "__main__":
    main()
