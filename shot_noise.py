#TASK 5

import pennylane as qml
import numpy as np
from quantum_kernels import feature_map, exact_kernel_overlap


def quantum_kernel_with_shots(x1, x2, n_qubits, shots, layers=1, entangler=None):
    """Same compute-uncompute circuit as before, but run with a finite number of shots."""
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    @qml.qnode(dev)
    def circuit():
        feature_map(x1, layers=layers, entangler=entangler)
        qml.adjoint(feature_map)(x2, layers=layers, entangler=entangler)
        return qml.probs(wires=range(n_qubits))

    probs = circuit()
    return probs[0]


# --- Investigate shot noise for one fixed pair of events ---

n_qubits = 2
x1 = [0.5, 1.2]
x2 = [2.0, 0.3]

k_exact = exact_kernel_overlap(x1, x2, n_qubits) #TASK 4 METHOD
print(f"Exact kernel value (Task 4 method): {k_exact:.6f}\n")

shot_counts = [100, 200, 500, 1000, 2000, 5000]

for n_shots in shot_counts:
    k_estimate = quantum_kernel_with_shots(x1, x2, n_qubits, shots=n_shots)
    error = abs(k_estimate - k_exact)

    # Predicted statistical uncertainty from counting statistics
    predicted_sigma = np.sqrt(k_exact * (1 - k_exact) / n_shots)

    print(f"shots={n_shots:5d}  estimate={k_estimate:.6f}  "
          f"error={error:.6f}  predicted_sigma={predicted_sigma:.6f}")


# ============ TASK 6 ============

def exact_kernel_matrix(X, n_qubits):
    """Build the exact N x N kernel matrix using Task 4's method."""
    N = len(X)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = exact_kernel_overlap(X[i], X[j], n_qubits)
    return K


def finite_shot_kernel_matrix(X, n_qubits, shots):
    """Build the N x N kernel matrix using finite-shot measurements."""
    N = len(X)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = quantum_kernel_with_shots(X[i], X[j], n_qubits, shots=shots)
    return K


# --- Small toy dataset, 4 EVENTS THEREFORE 4x4 MATRIX kept deliberately tiny for debugging ---
X_toy = [
    [0.5, 1.2],
    [2.0, 0.3],
    [0.1, 0.1],
    [3.0, 2.9],
]

print("\n=== Task 6: Finite-shot kernel matrix ===\n")

K_exact = exact_kernel_matrix(X_toy, n_qubits=2) #TASK 4 METHOD
print("Exact kernel matrix:")
print(np.round(K_exact, 4))
print()

exact_eigenvalues = np.linalg.eigvalsh(K_exact) #EIGENVALUE CHECK
print(f"Exact matrix smallest eigenvalue: {exact_eigenvalues.min():.6f}  (should be >= 0)\n")

for n_shots in [100, 200, 500, 1000, 2000, 5000]:
    K_shots = finite_shot_kernel_matrix(X_toy, n_qubits=2, shots=n_shots)


#WE MUST DO SHOTS METHOD BECAUSE EXACT METHOD IS NOT REALISTICALLY POSSIBLE IN A REAL QUANTUM COMPUTER, WE CANNOT GET THE EXACT STATEVECTOR, WE CAN ONLY GET PROBABILITIES FROM MEASUREMENTS.

    # Average absolute difference across all entries, as a single summary number
    avg_diff = np.mean(np.abs(K_shots - K_exact))

    eigenvalues = np.linalg.eigvalsh(K_shots)
    min_eigenvalue = eigenvalues.min()
    is_psd = min_eigenvalue >= -1e-10   # allow for tiny floating-point noise

    print(f"shots={n_shots:5d}  avg |difference|={avg_diff:.6f}  "
         f"smallest eigenvalue={min_eigenvalue:+.6f}  PSD={is_psd}")