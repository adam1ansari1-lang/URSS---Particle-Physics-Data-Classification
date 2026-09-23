"""
run_task6_plots.py
Task 6: simple plots from the Task 5 sweep CSVs -- test/train AUC and
off-diagonal magnitude, vs depth, entanglement, and qubit count.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_vs_x(df, x_col, x_label, title, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.errorbar(df[x_col], df["test_auc_mean"], yerr=df["test_auc_std"],
                 marker='o', capsize=4, label="test AUC")
    ax1.plot(df[x_col], df["train_auc_mean"], marker='s', linestyle='--', label="train AUC")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("AUC")
    ax1.set_title(f"{title}: AUC")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.errorbar(df[x_col], df["off_diag_mean"], yerr=df["off_diag_std"],
                 marker='o', color='tab:green', capsize=4)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("off-diagonal mean")
    ax2.set_title(f"{title}: kernel concentration")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_entanglement_bars(df, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    x = range(len(df))
    labels = df["entangler"]

    ax1.bar([i - 0.15 for i in x], df["test_auc_mean"], width=0.3, yerr=df["test_auc_std"],
            capsize=4, label="test AUC")
    ax1.bar([i + 0.15 for i in x], df["train_auc_mean"], width=0.3, label="train AUC")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("AUC")
    ax1.set_title("Entanglement: AUC")
    ax1.legend()

    ax2.bar(x, df["off_diag_mean"], yerr=df["off_diag_std"], capsize=4, color='tab:green')
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("off-diagonal mean")
    ax2.set_title("Entanglement: kernel concentration")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    depth_df = pd.read_csv(os.path.join(RESULTS_DIR, "task5_depth_sweep.csv"))
    plot_vs_x(depth_df, "layers", "circuit depth (layers)", "Depth sweep", "task6_depth_sweep.png")

    qubit_df = pd.read_csv(os.path.join(RESULTS_DIR, "task5_qubit_count_sweep.csv"))
    plot_vs_x(qubit_df, "n_qubits", "qubit count", "Qubit-count sweep", "task6_qubit_count_sweep.png")

    entangle_df = pd.read_csv(os.path.join(RESULTS_DIR, "task5_entanglement_sweep.csv"))
    plot_entanglement_bars(entangle_df, "task6_entanglement_sweep.png")
