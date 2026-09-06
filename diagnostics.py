"""
diagnostics.py
Week 2: checks that tell us whether a kernel matrix is "healthy".

A kernel matrix (Gram matrix) is a table of similarity scores - one row
and one column per event. This file checks that table four ways.
"""

import numpy as np
import matplotlib.pyplot as plt


def diagnose_kernel_matrix(K, y, title, save_path=None, psd_tolerance=-1e-8):
    """
    Run all four Week 2 diagnostics on a kernel matrix K, print the results,
    and save a plot of the heatmap and eigenvalue spectrum.

    K : the kernel/Gram matrix, shape (N, N)
    y : class labels, shape (N,) - 0 = background, 1 = signal
    title : short name for this kernel, used in prints and the plot filename
    save_path : where to save the plot (default built from title)
    psd_tolerance : how negative the smallest eigenvalue can be before we
                    call it a real problem rather than floating-point noise

    Returns the eigenvalues of K.
    """

    # Symmetry: K(x_i,x_j) should equal K(x_j,x_i), so K should equal K.T.
    # Expect this near zero (~1e-16); a bigger number means a real bug.
    symmetry_error = np.max(np.abs(K - K.T))

    # Sort rows/columns by class so background and signal each form a
    # contiguous block in the heatmap (reveals the [A,B;B^T,C] structure).
    sort_order = np.argsort(y)
    K_sorted = K[sort_order][:, sort_order]

    # Eigenvalues (eigvalsh is used since K is symmetric - faster & more
    # accurate than the general eig). Returned smallest to largest.
    eigenvalues = np.linalg.eigvalsh(K)
    smallest_eigenvalue = eigenvalues[0]
    largest_eigenvalue = eigenvalues[-1]

    # PSD check: a valid kernel matrix shouldn't have a clearly negative
    # eigenvalue. Tiny negative numbers are just rounding error, so we only
    # fail this if it's more negative than psd_tolerance.
    is_psd = smallest_eigenvalue >= psd_tolerance

    print(f"--- {title} ---")
    print(f"Max symmetry error:  {symmetry_error:.2e}")
    print(f"Smallest eigenvalue: {smallest_eigenvalue:.2e}")
    print(f"Largest eigenvalue:  {largest_eigenvalue:.2e}")
    print(f"Is PSD?              {is_psd}")

    # Left plot: heatmap (look for the four class blocks).
    # Right plot: eigenvalues, largest first (a few tall then flat = good;
    # all similar height = bad - see notes).
    fig, (heatmap_ax, spectrum_ax) = plt.subplots(1, 2, figsize=(11, 4))

    heatmap_plot = heatmap_ax.imshow(K_sorted, cmap='viridis')
    heatmap_ax.set_title(f'{title} - Gram matrix (sorted by class)')
    plt.colorbar(heatmap_plot, ax=heatmap_ax)

    spectrum_ax.plot(eigenvalues[::-1], 'o-', markersize=3)
    spectrum_ax.set_title(f'{title} - eigenvalue spectrum')
    spectrum_ax.set_xlabel('rank (0 = largest eigenvalue)')
    spectrum_ax.set_ylabel('eigenvalue')

    plt.tight_layout()

    if save_path is None:
        safe_title = title.replace(' ', '_').lower()
        save_path = f"plots/{safe_title}_diagnostics.png"

    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    return eigenvalues
