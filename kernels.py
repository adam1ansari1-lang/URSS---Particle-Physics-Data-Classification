"""
kernels.py
Classical model construction for Week 1.
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
    """
    param_grid = {"C": np.logspace(-3, 3, 13)}
    grid = GridSearchCV(
        LogisticRegression(max_iter=5000),
        param_grid,
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
    grid.fit(X_train, y_train)
    return grid.best_estimator_
