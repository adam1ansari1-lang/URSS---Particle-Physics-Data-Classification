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
    
    assert max_diff < 1e-10, f"Kernel trick check failed: {max_diff:.2e}" #RAISES ERROR IF DIFFERENCE IS BIG & "the difference IS small" -> True when things are good

    # Task 5: linear SVM on explicit features vs precomputed-kernel SVM
    # should give (near) identical decision functions
    C_value = 1.0
    phi_X = quadratic_feature_map(X)

    svm_explicit = SVC(kernel='linear', C=C_value)  # FINDS phi for each feature map vigorously
    svm_explicit.fit(phi_X, y)
    decision_explicit = svm_explicit.decision_function(phi_X)

    svm_precomputed = SVC(kernel='precomputed', C=C_value)  # DOESN'T find phi, does the kernel trick
    svm_precomputed.fit(K_direct, y)
    decision_precomputed = svm_precomputed.decision_function(K_direct)

    max_decision_diff = np.max(np.abs(decision_explicit - decision_precomputed))
    agree_fraction = np.mean(
        svm_explicit.predict(phi_X) == svm_precomputed.predict(K_direct)
        # we get an array of predicted labels for each event for each method
        # (1,0,0,1), and we see if the methods align (True, True, False, ...),
        # then take the mean of that, e.g. 98/100 = 98%
    )
    print(f"\nTask 5 - max decision function difference: {max_decision_diff:.2e}")
    print(f"Task 5 - fraction of matching predictions:  {agree_fraction:.3f}")
    # GET AN EXTREMELY SMALL NUMBER - EXPECTED SINCE IT IS THE SAME THING -
    # THIS IS THE ERROR IN THE COMPUTER

    # Task 6: RBF vs polynomial kernel, same splits/CV/scaler for both -
    # TWO DIFFERENT KERNELS
    #
    # RBF:        K(x,y) = exp(-gamma * ||x - x'||^2)   -- similarity based on distance only
    # Polynomial: K(x,y) = (gamma * x.y + coef0)^degree  -- similarity based on dot product

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

    best_gamma = rbf_search.best_params_["gamma"]

    # Task 7: Gram matrix diagnostics for both kernels
    X_scaled = StandardScaler().fit_transform(X)
    K_poly_full = polynomial_kernel(X_scaled, X_scaled, p=2)

    # RBF kernel by hand: exp(-gamma * ||x - x'||^2) for every pair
    #
    # It computes every pairwise difference vector x_i - x_j between all
    # events in X_scaled, all at once, with no loop.
    #   X_scaled[:, None, :] reshapes Training set - (N, 2), (n events and 2 features each) -> (N, 1, 2) - new axis in the middle
    #   X_scaled[None, :, :] reshapes (N, 2) -> (1, N, 2) - new axis at the front
    # Subtracting the two triggers numpy's broadcasting: any axis of size 1
    # automatically stretches to match the other array's size along that axis.
    # So (N,1,2) - (1,N,2) -> both size-1 axes stretch to N, giving shape (N,N,2).
 
    # Entry diffs[i, j, :] is the 2-number vector x_i - x_j - every pairwise, diff is the array holding it
    # difference computed in one operation, not just one pair.
    
    diffs = X_scaled[:, None, :] - X_scaled[None, :, :]
    K_rbf_full = np.exp(-best_gamma * np.sum(diffs**2, axis=2))  # We now use the best gamma

    #THEN WE GET THE DIFFFFERENCE SQUARED , we need that for the rbf kernel formula 

    #Without diffs we would need to write a nested loop over every pair of events by hand, very long 

    
    diagnose_kernel_matrix(K_poly_full, y, "Polynomial kernel")
    diagnose_kernel_matrix(K_rbf_full, y, "RBF kernel")


if __name__ == "__main__":
    main()
