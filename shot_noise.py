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

k_exact = exact_kernel_overlap(x1, x2, n_qubits)
print(f"Exact kernel value (Task 4 method): {k_exact:.6f}\n")

shot_counts = [100, 200, 500, 1000, 2000, 5000]

for n_shots in shot_counts:
    k_estimate = quantum_kernel_with_shots(x1, x2, n_qubits, shots=n_shots)
    error = abs(k_estimate - k_exact)

    # Predicted statistical uncertainty from counting statistics
    predicted_sigma = np.sqrt(k_exact * (1 - k_exact) / n_shots)

    print(f"shots={n_shots:5d}  estimate={k_estimate:.6f}  "
          f"error={error:.6f}  predicted_sigma={predicted_sigma:.6f}")