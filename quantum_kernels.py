import pennylane as qml
import numpy as np


def feature_map(x, layers=1, entangler=None):
    n_qubits = len(x)
    for layer in range(layers):
        for i in range(n_qubits):
            qml.RX(x[i], wires=i)
        if entangler is not None and layer < layers - 1:
            entangler(wires=range(n_qubits))


def quantum_kernel_circuit(x1, x2, n_qubits, layers=1, entangler=None):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        feature_map(x1, layers=layers, entangler=entangler)
        qml.adjoint(feature_map)(x2, layers=layers, entangler=entangler)
        return qml.probs(wires=range(n_qubits))

    result = circuit()
    return result[0]


def exact_kernel_overlap(x1, x2, n_qubits, layers=1, entangler=None):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def state_circuit(x):
        feature_map(x, layers=layers, entangler=entangler)
        return qml.state()

    psi1 = state_circuit(x1)
    psi2 = state_circuit(x2)
    overlap = np.vdot(psi2, psi1)
    return np.abs(overlap) ** 2


def basic_entangler(wires):
    qml.CNOT(wires=[wires[0], wires[1]])


def trailing_unitary_A(wires):
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RZ(1.234, wires=wires[1])


def trailing_unitary_B(wires):
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(0.777, wires=wires[0])
    qml.Hadamard(wires=wires[1])


def kernel_with_trailing_V(x1, x2, trailing_V, n_qubits=2):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        feature_map(x1, layers=1)
        if trailing_V is not None:
            trailing_V(wires=range(n_qubits))
        if trailing_V is not None:
            qml.adjoint(trailing_V)(wires=range(n_qubits))
        qml.adjoint(feature_map)(x2, layers=1)
        return qml.probs(wires=range(n_qubits))

    return circuit()[0]


def state_with_layers(x, layers, n_qubits=2):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        feature_map(x, layers=layers, entangler=basic_entangler)
        return qml.state()

    return circuit()


# Everything below only runs when THIS file is executed directly
# (python quantum_kernels.py), NOT when other files import functions
# from it (e.g. "from quantum_kernels import feature_map"). This is
# what stops every other script that imports this file from being
# flooded with Task 1/3/4/9/10 demo output.
if __name__ == "__main__":
    # --- Task 1 demo ---
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def demo_circuit(x):
        qml.RX(x, wires=0)
        return qml.state()

    print(demo_circuit(0.5))

    # --- Task 4 validation ---
    n_qubits = 2
    test_pairs = [
        ([0.5, 1.2], [0.5, 1.2]),
        ([0.5, 1.2], [2.0, 0.3]),
        ([0.1, 0.1], [3.0, 2.9]),
    ]
    for x1, x2 in test_pairs:
        k_circuit = quantum_kernel_circuit(x1, x2, n_qubits)
        k_exact = exact_kernel_overlap(x1, x2, n_qubits)
        diff = abs(k_circuit - k_exact)
        print(f"x1={x1}, x2={x2}  compute-uncompute={k_circuit:.10f}  exact={k_exact:.10f}  diff={diff:.2e}")
        assert diff < 1e-8

    # --- Task 9 demo ---
    x1_t9, x2_t9 = [0.5, 1.2], [2.0, 0.3]
    k_no_V = kernel_with_trailing_V(x1_t9, x2_t9, trailing_V=None)
    k_with_A = kernel_with_trailing_V(x1_t9, x2_t9, trailing_V=trailing_unitary_A)
    k_with_B = kernel_with_trailing_V(x1_t9, x2_t9, trailing_V=trailing_unitary_B)
    print(f"\nTask 9: no_V={k_no_V:.10f}  with_A={k_with_A:.10f}  with_B={k_with_B:.10f}")
    max_diff_t9 = max(abs(k_no_V - k_with_A), abs(k_no_V - k_with_B))
    assert max_diff_t9 < 1e-8
    print("PASS: trailing V cancels, as predicted.")

    # --- Task 10 demo ---
    x_t10 = [0.5, 1.2]
    state_1layer = state_with_layers(x_t10, layers=1)
    state_3layers = state_with_layers(x_t10, layers=3)
    print(f"\nTask 10: layers=1 state={state_1layer}")
    print(f"Task 10: layers=3 state={state_3layers}")

    dev_plain = qml.device("default.qubit", wires=2)

    @qml.qnode(dev_plain)
    def plain_embedding_circuit():
        feature_map(x_t10, layers=1)
        return qml.state()

    state_plain = plain_embedding_circuit()
    diff_t10 = np.max(np.abs(state_1layer - state_plain))
    assert diff_t10 < 1e-10
    print(f"PASS: layers=1 matches plain embedding, diff={diff_t10:.2e}")
