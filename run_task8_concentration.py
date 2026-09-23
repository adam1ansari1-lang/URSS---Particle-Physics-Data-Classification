"""
run_task8_concentration.py
Task 8: plot off-diagonal kernel magnitude vs depth and vs qubit count,
on one figure, to see whether increasing expressiveness pushes the
kernel toward K = I (off-diagonal -> 0).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

depth_df = pd.read_csv(os.path.join(RESULTS_DIR, "task5_depth_sweep.csv"))
qubit_df = pd.read_csv(os.path.join(RESULTS_DIR, "task5_qubit_count_sweep.csv"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.errorbar(depth_df["layers"], depth_df["off_diag_mean"], yerr=depth_df["off_diag_std"],
             marker='o', capsize=4, color='tab:green')
ax1.axhline(0, color='gray', linestyle=':', linewidth=1, label="K = I limit")
ax1.set_xlabel("circuit depth (layers)")
ax1.set_ylabel("off-diagonal mean")
ax1.set_title("Concentration vs depth")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.errorbar(qubit_df["n_qubits"], qubit_df["off_diag_mean"], yerr=qubit_df["off_diag_std"],
             marker='o', capsize=4, color='tab:purple')
ax2.axhline(0, color='gray', linestyle=':', linewidth=1, label="K = I limit")
ax2.set_xlabel("qubit count")
ax2.set_ylabel("off-diagonal mean")
ax2.set_title("Concentration vs qubit count")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
path = os.path.join(PLOTS_DIR, "task8_concentration.png")
plt.savefig(path, dpi=150)
plt.close(fig)
print(f"Wrote {path}")
