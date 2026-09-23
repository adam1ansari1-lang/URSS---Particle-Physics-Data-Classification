"""
run_week4.py
Single entry point for Week 4. Running this twice (cache-cold, then
cache-warm) should give identical numbers -- so anyone can regenerate
results from a clean checkout without hunting through scripts.
"""

import csv
import os

import numpy as np

from data import generate_toy_dataset
from pipeline import run_seed

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SEEDS = 5  # ground rule: >=5 seeds, mean +/- std, never a bare number


def run_model_comparison():
    # Task 1/2: all 4 models, N_SEEDS seeds, raw per-seed AUCs to CSV.
    X_raw, y = generate_toy_dataset(
        n_samples=200, n_dim=2, signal_fraction=0.35,
        bg_sigma=1.0, shell_radius=2.0, shell_width=0.3, random_state=0,
    )

    rows = []
    for seed in range(N_SEEDS):
        seed_results = run_seed(X_raw, y, seed, n_qubits=2)
        rows.append({"seed": seed, **seed_results})
        print(f"seed={seed}  " + "  ".join(f"{k}={v:.3f}" for k, v in seed_results.items()))

    out_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "logreg", "rbf", "poly", "quantum"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")

    print(f"\n=== Summary (mean +/- std over {N_SEEDS} seeds) ===")
    for name in ["logreg", "rbf", "poly", "quantum"]:
        aucs = np.array([r[name] for r in rows])
        print(f"{name:8s}: {aucs.mean():.3f} +/- {aucs.std():.3f}")


def run_quantum_tuning_comparison():
    # Task 3: same comparison, but quantum now cross-validates angle_max
    # AND C (tune_angle_max=True) instead of a fixed angle_max.
    X_raw, y = generate_toy_dataset(
        n_samples=200, n_dim=2, signal_fraction=0.35,
        bg_sigma=1.0, shell_radius=2.0, shell_width=0.3, random_state=0,
    )

    rows = []
    for seed in range(N_SEEDS):
        print(f"\n--- seed={seed} ---")
        seed_results = run_seed(X_raw, y, seed, n_qubits=2, tune_angle_max=True)
        rows.append({"seed": seed, **seed_results})
        print(
            f"seed={seed}  logreg={seed_results['logreg']:.3f}  rbf={seed_results['rbf']:.3f}  "
            f"poly={seed_results['poly']:.3f}  quantum={seed_results['quantum']:.3f}  "
            f"(angle_max={seed_results['quantum_angle_max']:.3f}, C={seed_results['quantum_C']:.3g})"
        )

    out_path = os.path.join(RESULTS_DIR, "task3_quantum_tuning.csv")
    fieldnames = ["seed", "logreg", "rbf", "poly", "quantum", "quantum_angle_max", "quantum_C"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_path}")

    print(f"\n=== Summary (mean +/- std over {N_SEEDS} seeds) ===")
    for name in ["logreg", "rbf", "poly", "quantum"]:
        aucs = np.array([r[name] for r in rows])
        print(f"{name:8s}: {aucs.mean():.3f} +/- {aucs.std():.3f}")

    winning_angles = [r["quantum_angle_max"] for r in rows]
    print(f"\nWinning angle_max per seed: {winning_angles}")


if __name__ == "__main__":
    print("=== Week 4: model comparison ===")
    run_model_comparison()

    print("\n=== Week 4 Task 3: fair quantum tuning (angle_max + C) ===")
    run_quantum_tuning_comparison()
