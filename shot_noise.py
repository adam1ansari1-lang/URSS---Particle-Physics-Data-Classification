# TASK 5

import pennylane as qml
import numpy as np
from quantum_kernels import feature_map, exact_kernel_overlap

n_qubits = 2

# FIX 1: don't set shots on the device (deprecated) -- keep device exact,
# attach shots per-call instead using qml.set_shots
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def _kernel_circuit(x1, x2, layers=1, entangler=None):
    """Compute-uncompute circuit, no shots baked in."""
    feature_map(x1, layers=layers, entangler=entangler)          # compute half
    qml.adjoint(feature_map)(x2, layers=layers, entangler=entangler)  # uncompute half
    return qml.probs(wires=range(n_qubits))


def quantum_kernel_with_shots(x1, x2, shots, layers=1, entangler=None, seed=None):
    """Run the circuit with a finite number of shots."""
    if seed is not None:
        np.random.seed(seed)   # fixes the randomness for this one call -> reproducible

    shot_circuit = qml.set_shots(_kernel_circuit, shots=shots)   # attach shots at call time
    probs = shot_circuit(x1, x2, layers=layers, entangler=entangler)
    return probs[0]   # P(00) = kernel value


# --- Investigate shot noise for one fixed pair of events ---
x1 = [0.5, 1.2]
x2 = [2.0, 0.3]

k_exact = exact_kernel_overlap(x1, x2, n_qubits)   # TASK 4 METHOD -- ground truth
print(f"Exact kernel value (Task 4 method): {k_exact:.6f}\n")

shot_counts = [100, 200, 500, 1000, 2000, 5000]

# FIX 2: a single run's error is one random sample, not a measurement of
# spread -- repeat many times per shot count and use the EMPIRICAL std,
# so we can actually compare it to the predicted sigma_K formula.
N_REPEATS = 100

for n_shots in shot_counts:
    estimates = []
    for repeat in range(N_REPEATS):
        estimate = quantum_kernel_with_shots(x1, x2, shots=n_shots, seed=repeat)  # new seed each time -> independent draw
        estimates.append(estimate)

    estimates = np.array(estimates)
    empirical_mean = estimates.mean()     # average of the 100 estimates
    empirical_std = estimates.std()       # actual measured spread

    predicted_sigma = np.sqrt(k_exact * (1 - k_exact) / n_shots)   # theory: sigma_K ~ sqrt(K(1-K)/N)

    print(f"shots={n_shots:5d}  empirical_mean={empirical_mean:.6f}  "
          f"empirical_std={empirical_std:.6f}  predicted_sigma={predicted_sigma:.6f}")


# ============ TASK 6 ============

def exact_kernel_matrix(X, n_qubits):
    """Exact N x N kernel matrix (Task 4 method), for reference."""
    N = len(X)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = exact_kernel_overlap(X[i], X[j], n_qubits)
    return K


def finite_shot_kernel_matrix(X, n_qubits, shots, seed=None):
    """Finite-shot N x N kernel matrix. K[i,j] and K[j,i] are separate
    circuit runs with independent noise -- they are NOT guaranteed equal."""
    N = len(X)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            pair_seed = None if seed is None else seed * 10_000 + i * 100 + j  # unique seed per entry
            K[i, j] = quantum_kernel_with_shots(X[i], X[j], shots=shots, seed=pair_seed)
    return K


X_toy = [
    [0.5, 1.2],
    [2.0, 0.3],
    [0.1, 0.1],
    [3.0, 2.9],
]

print("\n=== Task 6: Finite-shot kernel matrix ===\n")

K_exact = exact_kernel_matrix(X_toy, n_qubits=2)
print("Exact kernel matrix:")
print(np.round(K_exact, 4))

exact_asymmetry = np.abs(K_exact - K_exact.T)   # should be ~0, exact method is symmetric by construction
print(f"Exact matrix max asymmetry: {exact_asymmetry.max():.2e}  (should be ~0, floating point only)")

exact_eigenvalues = np.linalg.eigvalsh(K_exact)   # eigvalsh assumes symmetric input -- fine here, it IS symmetric
print(f"Exact matrix smallest eigenvalue: {exact_eigenvalues.min():.6f}  (should be >= 0)\n")

# Shots method needed because a real quantum computer can't read the exact
# statevector -- only probabilities from repeated measurements.

for n_shots in [100, 200, 500, 1000, 2000, 5000]:
    K_shots = finite_shot_kernel_matrix(X_toy, n_qubits=2, shots=n_shots, seed=n_shots)

    # FIX 3: eigvalsh silently assumes symmetry (only reads the lower
    # triangle) -- measure the asymmetry ourselves BEFORE relying on that.
    asymmetry = np.abs(K_shots - K_shots.T)
    max_asym = asymmetry.max()     # worst single mismatched pair
    mean_asym = asymmetry.mean()   # average mismatch across the whole matrix

    # Only now symmetrise explicitly (average with own transpose) before
    # doing the eigenvalue/PSD check, instead of letting eigvalsh assume it.
    K_symmetrised = (K_shots + K_shots.T) / 2
    eigenvalues = np.linalg.eigvalsh(K_symmetrised)
    min_eigenvalue = eigenvalues.min()
    is_psd = min_eigenvalue >= -1e-10   # tiny tolerance for floating-point noise

    avg_diff = np.mean(np.abs(K_shots - K_exact))   # how far the whole matrix is from truth

    print(f"shots={n_shots:5d}  avg|diff|={avg_diff:.6f}  "
          f"max_asym={max_asym:.6f}  mean_asym={mean_asym:.6f}  "
          f"min_eig(symmetrised)={min_eigenvalue:+.6f}  PSD={is_psd}")
