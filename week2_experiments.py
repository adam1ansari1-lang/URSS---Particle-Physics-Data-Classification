"""
week2_experiments.py
Runs Week 2 Tasks 5-7: precomputed-kernel SVM check, RBF vs polynomial
kernel comparison, and Gram matrix diagnostics.

Tasks 1-4 live as reusable functions in kernels.py; this script uses them
on the real Week 1 toy dataset.
"""

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
np.random.seed(RANDOM_STATE)


def main():
    # Small dataset (N<=100) so kernel matrices stay easy to visualise
    X, y = generate_toy_dataset(n_samples=100, n_dim=2, random_state=RANDOM_STATE)
    print("Dataset shape:", X.shape, " Class counts:", np.bincount(y.astype(int)))

    # Tasks 1-2 sanity check: should print [1, 2.8284..., 4]
    phi_check = quadratic_feature_map(np.array([[1, 2]]))
    print("\nphi(1,2) =", phi_check)

    # Task 4: explicit feature map vs direct kernel formula should agree
    # to machine precision (~1e-14 to 1e-16)
    K_explicit, K_direct, max_diff = verify_kernel_trick(X, p=2)
    print(f"\nTask 4 - max |Method A - Method B| = {max_diff:.2e}")

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

    # Task 6: RBF vs polynomial kernel, same splits/CV/scaler for both
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler().fit(X_train)  # fit on train only, no leakage
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    rbf_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10]}
    rbf_search = GridSearchCV(
        SVC(kernel='rbf', probability=True), rbf_grid, scoring='roc_auc', cv=5
    )
    rbf_search.fit(X_train_s, y_train)

    poly_grid = {'C': [0.1, 1, 10, 100], 'coef0': [0, 0.5, 1]}
    poly_search = GridSearchCV(
        SVC(kernel='poly', degree=2, probability=True), poly_grid,
        scoring='roc_auc', cv=5,
    )
    poly_search.fit(X_train_s, y_train)

    print("\nTask 6 - RBF vs Polynomial comparison:")
    for name, model in [('RBF', rbf_search), ('Poly (deg 2)', poly_search)]:
        auc_test = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
        print(f"  {name}: best params = {model.best_params_}, test AUC = {auc_test:.3f}")

    # ROC curves for both models, on the same axes
    plt.figure(figsize=(5, 5))
    for name, model in [('RBF', rbf_search), ('Poly (deg 2)', poly_search)]:
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_s)[:, 1])
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('RBF vs Polynomial Kernel SVM')
    plt.savefig('plots/rbf_vs_poly_roc.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Task 7: Gram matrix diagnostics for both kernels
    X_scaled = StandardScaler().fit_transform(X)
    K_poly_full = polynomial_kernel(X_scaled, X_scaled, p=2)

    # RBF kernel by hand: exp(-gamma * ||x - x'||^2) for every pair
    diffs = X_scaled[:, None, :] - X_scaled[None, :, :]
    K_rbf_full = np.exp(-0.5 * np.sum(diffs**2, axis=2))  # gamma=0.5

    print()
    diagnose_kernel_matrix(K_poly_full, y, "Polynomial kernel")
    diagnose_kernel_matrix(K_rbf_full, y, "RBF kernel")


if __name__ == "__main__":
    main()
