"""
Visualise the 2D toy dataset: background cloud vs signal ring.
Run this from the project root; it saves a plot into plots/.
"""

import matplotlib.pyplot as plt
from data import generate_toy_dataset

X, y = generate_toy_dataset(n_samples=500, n_dim=2, random_state=0)

plt.figure(figsize=(6, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], s=15, alpha=0.6, label="Background", color="steelblue")
plt.scatter(X[y == 1, 0], X[y == 1, 1], s=15, alpha=0.7, label="Signal", color="crimson")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Physical feature space")
plt.legend()
plt.axis("equal")
plt.grid(alpha=0.3)
plt.tight_layout()
import os
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/toy_dataset_2d.png", dpi=150)
print("Saved plots/toy_dataset_2d.png")
