"""
diagnostics.py
Checks whether a kernel matrix is "healthy": symmetry, PSD, eigenvalue
spectrum, and diagonal/off-diagonal structure.
"""

import numpy as np
import matplotlib.pyplot as plt


def diagnose_kernel_matrix(K, y, title, save_path=None, psd_tolerance=-1e-8):
    """
    Run all diagnostics on a kernel matrix K, print the results, save a
    plot (heatmap, eigenvalue spectrum, diagonal/off-diagonal histograms).
    Returns a dict of stats so kernels can be compared programmatically.
    """

    # Symmetry: K(x_i,x_j) should equal K(x_j,x_i).
    symmetry_error = np.max(np.abs(K - K.T))

    sort_order = np.argsort(y)
    K_sorted = K[sort_order][:, sort_order]

    eigenvalues = np.linalg.eigvalsh(K)
    smallest_eigenvalue = eigenvalues[0]
    largest_eigenvalue = eigenvalues[-1]
    is_psd = smallest_eigenvalue >= psd_tolerance

    # Diagonal = self-similarity (~1 for a normalized/quantum kernel).
    # Off-diagonal = similarity between DIFFERENT events -- the real
    # signal. K -> I shows up as off-diagonal collapsing toward 0.
    diag_values = np.diag(K)
    off_diag_mask = ~np.eye(K.shape[0], dtype=bool)
    off_diag_values = K[off_diag_mask]

    diag_mean, diag_std = diag_values.mean(), diag_values.std()
    off_diag_mean, off_diag_std = off_diag_values.mean(), off_diag_values.std()

    print(f"--- {title} ---")
    print(f"Max symmetry error:     {symmetry_error:.2e}")
    print(f"Smallest eigenvalue:    {smallest_eigenvalue:.2e}")
    print(f"Largest eigenvalue:     {largest_eigenvalue:.2e}")
    print(f"Is PSD?                 {is_psd}")
    print(f"Diagonal:      mean={diag_mean:.4f}  std={diag_std:.4f}")
    print(f"Off-diagonal:  mean={off_diag_mean:.4f}  std={off_diag_std:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    heatmap_ax, spectrum_ax, hist_ax = axes

    heatmap_plot = heatmap_ax.imshow(K_sorted, cmap='viridis')
    heatmap_ax.set_title(f'{title} - Gram matrix (sorted by class)')
    plt.colorbar(heatmap_plot, ax=heatmap_ax)

    spectrum_ax.plot(eigenvalues[::-1], 'o-', markersize=3)
    spectrum_ax.set_title(f'{title} - eigenvalue spectrum')
    spectrum_ax.set_xlabel('rank (0 = largest eigenvalue)')
    spectrum_ax.set_ylabel('eigenvalue')

    bins = np.linspace(min(off_diag_values.min(), diag_values.min()),
                        max(off_diag_values.max(), diag_values.max()), 40)
    hist_ax.hist(off_diag_values, bins=bins, alpha=0.6, label='off-diagonal', color='tab:blue')
    hist_ax.hist(diag_values, bins=bins, alpha=0.6, label='diagonal', color='tab:orange')
    hist_ax.set_title(f'{title} - diagonal vs off-diagonal')
    hist_ax.set_xlabel('kernel value')
    hist_ax.set_ylabel('count')
    hist_ax.legend()

    plt.tight_layout()

    if save_path is None:
        safe_title = title.replace(' ', '_').lower()
        save_path = f"plots/{safe_title}_diagnostics.png"

    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    return {
        "eigenvalues": eigenvalues,
        "symmetry_error": symmetry_error,
        "is_psd": is_psd,
        "diag_values": diag_values,
        "off_diag_values": off_diag_values,
        "diag_mean": diag_mean,
        "diag_std": diag_std,
        "off_diag_mean": off_diag_mean,
        "off_diag_std": off_diag_std,
    }
