"""
kernels.py
Classical model construction for Week 1.
Week 2: explicit feature maps and kernel functions added below.
(Quantum kernel code will be added here in a later week.)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np


def tune_logistic_regression(X_train, y_train, cv=5, random_state=0):
    """
    Tune Logistic Regression's regularisation strength C via cross-validation.
    Returns the best fitted estimator.

    CV is cross validation: split the training data into 5 equal chunks
    ("folds"). Train on 4 of them, test on the 1 left out, 
    repeat 5 times so each fold gets a turn being the held-out one, 
    then average the 5 scores. Reduces risl of a picking a hyperparameter
    that got lucky on one particular split
    """
    param_grid = {"C": np.logspace(-3, 3, 13)} #Defined C to try 13 values from 10^-3 to 10^3, evenly spaced
    #because log scale C matters over orders of magnitude, not linear steps
    grid = GridSearchCV(
        LogisticRegression(max_iter=5000), #Automatic search for every value of C
        param_grid, #Dictionary listing hyperparameter values we want GridSearch
        cv=cv,
        scoring="roc_auc",
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_


def tune_rbf_svm(X_train, y_train, cv=5, random_state=0, fast=False):
    """
    Tune RBF SVM's C and gamma via cross-validation.
    Both C and gamma are searched, matching the 'equal footing' ground rule:
    a two-parameter search, comparable to what we'll later give the quantum kernel.

    fast=True uses a coarser grid - fine for quick exploratory sweeps,
    not for the final reported numbers (use fast=False, the default, for those).
    """
    if fast:
        param_grid = {"C": np.logspace(-1, 2, 4), "gamma": np.logspace(-2, 1, 4)} 
        #if fast=True, use a small grid: only 4 values each for C and gamma
        #(4×4 = 16 combinations total) — quicker but coarser, meant for exploratory runs
        #like the dimensionality sweep
    else:
        param_grid = {
            "C": np.logspace(-2, 3, 11),
            "gamma": np.logspace(-3, 2, 11),
        }
    grid = GridSearchCV(
        SVC(kernel="rbf", probability=True),
        param_grid,
        cv=cv,
        scoring="roc_auc",
    )
    grid.fit(X_train, y_train) #runs the whole search: trains + cross-validates a model for every (C, gamma)
    #pair, tracks the best-scoring combination
    return grid.best_estimator_


# ---------------------------------------------------------------------------
# Week 2: explicit feature maps and direct kernel functions
# ---------------------------------------------------------------------------

def quadratic_feature_map(X):
    """
    Explicit quadratic feature map, built by hand (Task 1):
        phi(x) = (x1^2, sqrt(2)*x1*x2, x2^2)

    X : ndarray, shape (n_samples, 2)
    Returns : ndarray, shape (n_samples, 3)

    Sanity check from the worksheet: phi(1,2) = (1, 2*sqrt(2), 4)

    DONE IN EXPERIMENTS WEEK 2
    """
    x1 = X[:, 0] #pulls out column 0 (every row's 𝑥1 value) as one array
    x2 = X[:, 1] #pulls out column 1 (every row's x2 value) as one array
    #then three new arrays get computed
    return np.column_stack([x1**2, np.sqrt(2) * x1 * x2, x2**2]) 




def polynomial_kernel(X1, X2, p=2):
    """
    Direct polynomial kernel (Tasks 2-3): K(x,y) = (x . y)^p,
    vectorised over every pair of events in X1 and X2.

    X1 : ndarray, shape (n1, n_dim)
    X2 : ndarray, shape (n2, n_dim)
    Returns : ndarray, shape (n1, n2)
    """
    return (X1 @ X2.T) ** p
    
# X2.T flips X2 to shape (n_dim, n2) — now its rows are dimensions instead of events. That makes X1 @ X2.T valid: (n1, n_dim) @ (n_dim, n2) → (n1, n2).

def verify_kernel_trick(X, p=2):
    """
    Task 4: build the training Gram matrix two independent ways and compare.
      Method A: explicit feature map, then dot products  -> phi(X) . phi(X)^T
      Method B: direct kernel formula                     -> K(X, X)

    Only valid for p=2, since quadratic_feature_map is the degree-2 map.
    Returns (K_explicit, K_direct, max_abs_difference).
    Expect max_abs_difference ~ 1e-14 to 1e-16 (machine precision).
    A much larger difference (e.g. ~1e-6) signals a real bug, not rounding.
    """
    phi_X = quadratic_feature_map(X)
    K_explicit = phi_X @ phi_X.T
    K_direct = polynomial_kernel(X, X, p=p)
    max_diff = np.max(np.abs(K_explicit - K_direct))
    return K_explicit, K_direct, max_diff
