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
