"""
data.py
Toy dataset generator for HEP-style signal/background classification,
and (later, Week 5) the real HEP data loader.

Both must return the SAME shape of output: (X, y)
  X : ndarray, shape (n_samples, n_dim)   - the feature vectors
  y : ndarray, shape (n_samples,)         - labels, 0 = background, 1 = signal

Keeping that interface fixed now means swapping toy -> real data later
is a one-line change.
"""

import numpy as np


def generate_toy_dataset(
    n_samples=500,
    n_dim=2,
    signal_fraction=0.35,
    bg_sigma=1.0, #BR STD
    shell_radius=2.0,
    shell_width=0.3,
    random_state=None,
):
    """
    Generate a toy HEP-style dataset.

    Background: isotropic Gaussian cloud around the origin.
    Signal: points on a thin spherical shell of radius `shell_radius`,
            standing in for a resonance at fixed invariant mass.

    Parameters
    ----------
    n_samples : total number of events (signal + background)
    n_dim : number of features (= number of qubits later)
    signal_fraction : fraction of events that are signal (~0.35)
    bg_sigma : std dev of each background feature
    shell_radius : mean radius of the signal shell
    shell_width : std dev of the signal radius (shell thickness)
    random_state : int or None, for reproducibility

    Returns
    -------
    X : ndarray (n_samples, n_dim)
    y : ndarray (n_samples,)  0 = background, 1 = signal
    """
    rng = np.random.default_rng(random_state)

    n_signal = int(round(n_samples * signal_fraction))
    n_background = n_samples - n_signal

    # 1. Background: each feature ~ N(0, bg_sigma^2), independently
    X_background = rng.normal(loc=0.0, scale=bg_sigma, size=(n_background, n_dim))

    # 2. Signal direction: draw a Gaussian vector, then normalise to unit length.
    #    Gaussian components -> uniform spread over the sphere in any dimension.
    raw_directions = rng.normal(loc=0.0, scale=1.0, size=(n_signal, n_dim))
    norms = np.linalg.norm(raw_directions, axis=1, keepdims=True)
    unit_directions = raw_directions / norms

    # 3 & 4. Signal radius: draw from a narrow Gaussian around shell_radius,
    #        giving the shell a small width (finite resonance width).
    radii = rng.normal(loc=shell_radius, scale=shell_width, size=(n_signal, 1)) 
    #lets us have n_signal rows, a column vector and random radius per signal event

    X_signal = unit_directions * radii

    # 5. Combine and label
    X = np.vstack([X_background, X_signal]) #stacks the 2 point arrays on top of eachother, into one array X. BR rows first then signal rows
    y = np.concatenate([np.zeros(n_background), np.ones(n_signal)])
    #Array of 0s per BR event and 1s per signal, concatenate joins those 2 arrays into one array y, 
    #in the same ofer as X so y[i] correctly labels x[i]

    # Shuffle so signal/background aren't grouped in order
    shuffle_idx = rng.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    #This shuffle function re orders the rows of x and y so that they are in random order - good for training

    return X, y


if __name__ == "__main__":    #makes sure this block only runs when we execute data.py directly 
    # Quick manual check
    X, y = generate_toy_dataset(n_samples=500, n_dim=2, random_state=0)
    print("X shape:", X.shape, "y shape:", y.shape)
    print("signal fraction:", y.mean())
