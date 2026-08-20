"""
experiments.py
Week 1: train/tune classical baselines over 5 random seeds,
report ROC/AUC as mean +/- std, and save a combined ROC plot.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score

from data import generate_toy_dataset
from kernels import tune_logistic_regression, tune_rbf_svm


def run_seed(seed, n_dim=2):
    # 1. Fresh toy dataset for this seed
    X, y = generate_toy_dataset(n_samples=500, n_dim=n_dim, random_state=seed)

    # 2. Split BEFORE scaling, so the scaler never sees test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed #test size =0.3 means 30% data in test, 70 in training
    )

    # 3. Standardise using training-set statistics only
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4. Tune + fit both models
    logreg = tune_logistic_regression(X_train_s, y_train, random_state=seed)
    svm = tune_rbf_svm(X_train_s, y_train, random_state=seed)

<<<<<<< HEAD
    results = {}
    for name, model in [("LogReg", logreg), ("RBF SVM", svm)]:
        y_score = model.predict_proba(X_test_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        results[name] = {"fpr": fpr, "tpr": tpr, "auc": auc}

    # Record the winning hyperparameters GridSearchCV picked for this seed,
    # so we have a permanent record of "what did tuning actually choose",
    # not just the final AUC.
    results["LogReg"]["params"] = {"C": float(logreg.get_params()["C"])}
    results["RBF SVM"]["params"] = {
        "C": float(svm.get_params()["C"]),
        "gamma": float(svm.get_params()["gamma"]),
    }
=======
    results = {} #empty dictionary to store results for both models
    for name, model in [("LogReg", logreg), ("RBF SVM", svm)]:      #loops twice, once with LogReg and then SVM
        y_score = model.predict_proba(X_test_s)[:, 1]       #returns 2 columns, 0 and 1 for BR and Sig, grabs just the probability of signal column
        fpr, tpr, _ = roc_curve(y_test, y_score)       #false postivie rate and TPR, @many thresholds
                                                      #the _ throws away the third returned value (the threshold value itself)
        auc = roc_auc_score(y_test, y_score)         #AUC number from same true labels and predicted probabilities
        results[name] = {"fpr": fpr, "tpr": tpr, "auc": auc} #hands back the dictionary
>>>>>>> 8d86c6be25a6b1931432b422ac7a8893cfa2e080
    return results


def main(n_dim=2, n_seeds=5):
    all_results = {"LogReg": [], "RBF SVM": []}
    for seed in range(n_seeds):
        res = run_seed(seed, n_dim=n_dim)  #runs entire pipeline, data gen, split, scale, tune & evaluation
        for name in all_results:
            all_results[name].append(res[name])
        print(f"seed {seed}: LogReg AUC={res['LogReg']['auc']:.3f}  "
              f"RBF SVM AUC={res['RBF SVM']['auc']:.3f}")

    # Summary: mean +/- std AUC per model
    print("\nSummary over", n_seeds, "seeds:")
    for name, runs in all_results.items():
        aucs = np.array([r["auc"] for r in runs])
        print(f"  {name}: AUC = {aucs.mean():.3f} +/- {aucs.std():.3f}")

    # Log raw per-seed results (AUC + chosen hyperparameters) to results/,
    # per the project convention: results/ holds raw numbers, plots/ holds figures.
    os.makedirs("results", exist_ok=True)
    csv_path = "results/week1_aucs.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_dim", "seed", "model", "auc", "best_params"])
        for name, runs in all_results.items():
            for i, r in enumerate(runs):
                writer.writerow([n_dim, i, name, f"{r['auc']:.4f}", r["params"]])
    print(f"Saved {csv_path}")

    # Plot ROC curves for all seeds, one panel per model
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (name, runs) in zip(axes, all_results.items()):
        for i, r in enumerate(runs):      #i is the seed number and r is the data point itself
            ax.plot(r["fpr"], r["tpr"], alpha=0.5, label=f"seed {i} (AUC={r['auc']:.2f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
        ax.set_title(name)
        ax.set_xlabel("False Positive Rate")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("True Positive Rate")
    plt.suptitle(f"ROC curves, {n_dim}D toy dataset, {n_seeds} seeds")
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/roc_curves.png", dpi=150)
    print("Saved plots/roc_curves.png")


if __name__ == "__main__":
    main()
