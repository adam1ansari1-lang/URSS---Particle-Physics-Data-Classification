# ============ TASK 11 ============
import pennylane as qml
from quantum_kernels import feature_map, basic_entangler

n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)


def kernel_value(x1, x2, layers):
    @qml.qnode(dev)
    def circuit():
        feature_map(x1, layers=layers, entangler=basic_entangler)
        qml.adjoint(feature_map)(x2, layers=layers, entangler=basic_entangler)
        return qml.probs(wires=range(n_qubits))
    return circuit()[0]


def test_depth_changes_kernel():
    x1 = [0.5, 1.2]
    x2 = [2.0, 0.3]

    k_shallow = kernel_value(x1, x2, layers=1)
    k_deep = kernel_value(x1, x2, layers=3)
    difference = abs(k_shallow - k_deep)

    assert difference > 1e-6, (
        f"layers=1 and layers=3 gave nearly identical kernels "
        f"(difference={difference:.2e})."
    )
    print(f"PASS: depth changed the kernel (shallow={k_shallow:.6f}, "
          f"deep={k_deep:.6f}, diff={difference:.2e})")


test_depth_changes_kernel()


# ============ TASK 13 ============
import numpy as np
from quantum_kernels import exact_kernel_overlap

n_qubits_t13 = 2
dev_t13 = qml.device("default.qubit", wires=n_qubits_t13)


@qml.qnode(dev_t13)
def pennylane_circuit(x1, x2):
    qml.templates.AngleEmbedding(x1, wires=range(n_qubits_t13), rotation="X")
    qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=range(n_qubits_t13), rotation="X")
    return qml.probs(wires=range(n_qubits_t13))


def test_matches_pennylane_builtin():
    x1 = [0.5, 1.2]
    x2 = [2.0, 0.3]

    k_mine = exact_kernel_overlap(x1, x2, n_qubits_t13)
    k_pennylane = pennylane_circuit(x1, x2)[0]

    diff = abs(k_mine - k_pennylane)
    print(f"My kernel:        {k_mine:.10f}")
    print(f"PennyLane kernel: {k_pennylane:.10f}")
    print(f"Difference:       {diff:.2e}")

    assert diff < 1e-8, "Mismatch with PennyLane's own built-in embedding!"
    print("PASS: matches PennyLane's built-in embedding kernel.")


test_matches_pennylane_builtin()
