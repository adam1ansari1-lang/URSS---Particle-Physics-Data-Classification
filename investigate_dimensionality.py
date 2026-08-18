"""
Investigation: how does dimensionality affect problem difficulty,
if the shell radius is kept fixed?

Two things to look at:
1. RBF SVM AUC vs number of dimensions (fixed shell_radius=2.0)
2. Radial distribution of the BACKGROUND alone, for different dimensions
   (this shows why higher dimensions behave differently: in high-D,
   an isotropic Gaussian's mass concentrates in a thin shell too -
   "concentration of measure")
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from data import generate_toy_dataset
from kernels import tune_rbf_svm

dims_to_test = [2, 3, 5, 8, 12, 20]
n_seeds = 3

# --- Part 1: AUC vs dimension, shell_radius fixed ---
mean_aucs, std_aucs = [], []
for d in dims_to_test:
    aucs = []
    for seed in range(n_seeds):
        X, y = generate_toy_dataset(n_samples=500, n_dim=d, shell_radius=2.0, random_state=seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=seed
        )
        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
        svm = tune_rbf_svm(X_train_s, y_train, random_state=seed, fast=True)
        y_score = svm.predict_proba(X_test_s)[:, 1]
        aucs.append(roc_auc_score(y_test, y_score))
    mean_aucs.append(np.mean(aucs))
    std_aucs.append(np.std(aucs))
    print(f"dim={d}: AUC = {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")

# --- Part 2: radial distribution of background alone, fixed bg_sigma=1.0 ---
plt.figure(figsize=(11, 5))

plt.subplot(1, 2, 1)
plt.errorbar(dims_to_test, mean_aucs, yerr=std_aucs, marker="o", capsize=3)
plt.axhline(0.5, color="k", linestyle="--", alpha=0.4, label="chance")
plt.xlabel("Number of dimensions")
plt.ylabel("RBF SVM AUC")
plt.title("Fixed shell radius = 2.0")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
rng = np.random.default_rng(0)
for d in dims_to_test:
    bg = rng.normal(0, 1, size=(5000, d))
    r = np.linalg.norm(bg, axis=1)
    plt.hist(r, bins=50, density=True, histtype="step", label=f"d={d}")
plt.axvline(2.0, color="k", linestyle="--", alpha=0.5, label="shell radius=2.0")
plt.xlabel("radius r")
plt.ylabel("density")
plt.title("Background radial distribution vs dimension")
plt.legend(fontsize=7)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("dimensionality_investigation.png", dpi=150)
print("Saved dimensionality_investigation.png")
