#---------TASKS 1-4 and 9-10----------

#TASK 1: Import the PennyLane library and build simple quantum circuit that rotates a qubit by an angle x and returns the statevector.

import pennylane as qml
import numpy as np   

# A device is the simulator that runs the circuit.
# "default.qubit" is PennyLane's exact simulator, wires=2 means 2 qubits.
dev = qml.device("default.qubit", wires=2)

# The decorator turns this plain function into an actual runnable circuit,
# bound to the device above.
@qml.qnode(dev)
def circuit(x):
    qml.RX(x, wires=0)   # rotate qubit 0 by angle x
    return qml.state()   # return the exact statevector (simulator-only "cheat")

print(circuit(0.5))

#----------------------------------------
#TASK 2 - 
#----------------------------------------

# Built feature_map(x, layers, entangler) — 
# one RX rotation per feature/qubit, with layers=1 
# reducing to plain angle embedding (raw E(x) ) and no entangler
# (verified on the toy dataset's 2-feature events).

def feature_map(x, layers=1, entangler=None):
    """
    Quantum feature map U(x) that encodes a classical event x into qubits.

    Parameters
    ----------
    x : array-like
        The classical event, e.g. (x1, x2) for a 2-feature dataset.
        One qubit is used per feature.
    layers : int
        Number of embedding blocks. layers=1 means just one embedding,
        no entangling gates at all (the simplest possible case).
    entangler : callable or None
        A function that applies entangling gates across the qubits.
        Ignored when layers=1, since there's nothing to entangle between.
    """
    n_qubits = len(x) #how many features are in x

    for layer in range(layers):
        # Embedding step: one RX rotation per feature, one qubit per feature.
        # This is E(x)
        for i in range(n_qubits):
            qml.RX(x[i], wires=i) #here we apply features value as an angle
        
        #eg x[0] rotates qubit 0, x[1] rotates qubit 1, etc. 

        # Entangling step: only applied *between* embeddings, never after
        # the very last one - ONLY if entangler is provided. This is the "entangling" part of the feature map.
        if entangler is not None and layer < layers - 1:
            entangler(wires=range(n_qubits))


# ============ TASK 3 ============

def quantum_kernel_circuit(x1, x2, n_qubits, layers=1, entangler=None):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        feature_map(x1, layers=layers, entangler=entangler)   # compute - rotates qubits based on x1
        qml.adjoint(feature_map)(x2, layers=layers, entangler=entangler)  # uncompute - literally opposite of the x1 qubit rotation
        return qml.probs(wires=range(n_qubits))

    result = circuit()
    return result[0]


x1 = [0.5, 1.2] #means rotate qubit 0 by 0.5 and qubit 1 by 1.2
x2 = [0.5, 1.2]

probs = quantum_kernel_circuit(x1, x2, n_qubits=2, layers=1)
print("Full probability vector:", probs)
print("Kernel value K(x1, x2) =", probs) #measures qubit 00


#=========== TASK 4 ============

def exact_kernel_overlap(x1, x2, n_qubits, layers=1, entangler=None):

    """Way B: exact statevector overlap, computed directly in Python."""

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)    #dev is device btw, qnode is a decorator that turns a function into a quantum circuit
    def state_circuit(x):
        feature_map(x, layers=layers, entangler=entangler)
        return qml.state()

    psi1 = state_circuit(x1)   # |ψ(x1)⟩
    psi2 = state_circuit(x2)   # |ψ(x2)⟩

    overlap = np.vdot(psi2, psi1)   # <ψ(x2)|ψ(x1)>, np.vdot conjugates the first argument , bra-ket rule, always conjugate left vector
    return np.abs(overlap) ** 2     # |<ψ(x2)|ψ(x1)>|^2


# --- Compare Way A (circuit) vs Way B (exact overlap) ---
n_qubits = 2
test_pairs = [
    ([0.5, 1.2], [0.5, 1.2]),   # identical -> expect ~1
    ([0.5, 1.2], [2.0, 0.3]),   # different -> expect something in (0,1)
    ([0.1, 0.1], [3.0, 2.9]),   # quite different -> expect small value
]

for x1, x2 in test_pairs:
    k_circuit = quantum_kernel_circuit(x1, x2, n_qubits)
    k_exact = exact_kernel_overlap(x1, x2, n_qubits)
    diff = abs(k_circuit - k_exact)

    print(f"x1={x1}, x2={x2}")
    print(f"  compute-uncompute: {k_circuit:.10f}")
    print(f"  exact overlap:     {k_exact:.10f}")
    print(f"  difference:        {diff:.2e}")
    print()

    assert diff < 1e-8, f"Mismatch too large: {diff}"



# ============ TASK 9 ============
# Proves that a fixed unitary V, applied only AFTER the data embedding,
# cancels out of the compute-uncompute kernel completely -- no matter
# what V actually is. This is why entanglement must be interleaved
# BETWEEN embeddings (Task 10) rather than tacked on at the end.

def trailing_unitary_A(wires):
    """One arbitrary, FIXED (x-independent) entangling block."""
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RZ(1.234, wires=wires[1])


def trailing_unitary_B(wires):
    """A DIFFERENT fixed entangling block -- shows the cancellation
    isn't a fluke of one particular choice of V."""
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(0.777, wires=wires[0])
    qml.Hadamard(wires=wires[1])


def kernel_with_trailing_V(x1, x2, trailing_V, n_qubits=2):
    """Compute-uncompute circuit with a fixed V tacked on after each
    embedding, in BOTH the compute and uncompute halves."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        feature_map(x1, layers=1)                                # compute: E(x1)
        if trailing_V is not None:
            trailing_V(wires=range(n_qubits))                     # compute: V

        if trailing_V is not None:
            qml.adjoint(trailing_V)(wires=range(n_qubits))         # uncompute: V^dagger (undo most recent first)
        qml.adjoint(feature_map)(x2, layers=1)                     # uncompute: E(x2)^dagger

        return qml.probs(wires=range(n_qubits))

    return circuit()[0]


x1_t9 = [0.5, 1.2]
x2_t9 = [2.0, 0.3]

k_no_V   = kernel_with_trailing_V(x1_t9, x2_t9, trailing_V=None)
k_with_A = kernel_with_trailing_V(x1_t9, x2_t9, trailing_V=trailing_unitary_A)
k_with_B = kernel_with_trailing_V(x1_t9, x2_t9, trailing_V=trailing_unitary_B)

print("\n=== Task 9: Feature-map cancellation ===")
print(f"Kernel with NO trailing V:   {k_no_V:.10f}")
print(f"Kernel with trailing V = A:  {k_with_A:.10f}")
print(f"Kernel with trailing V = B:  {k_with_B:.10f}")

max_diff_t9 = max(abs(k_no_V - k_with_A), abs(k_no_V - k_with_B))
print(f"Largest difference between any two: {max_diff_t9:.2e}")

assert max_diff_t9 < 1e-8, "V should have cancelled completely -- it did not!"
print("PASS: the trailing unitary V has no effect on the kernel, as predicted.")



# ============ TASK 10 ============
# Data re-uploading: interleave a real entangler BETWEEN repeated
# embeddings, using the SAME feature_map function from Task 2 

#DIFFERENCE between feature map with one layer and task 2 (no entangler at all) = 0

def basic_entangler(wires):
    """The simplest fixed entangling layer W: one CNOT. Never depends
    on the data x -- same gate every time, for every event."""
    qml.CNOT(wires=[wires[0], wires[1]])


def state_with_layers(x, layers, n_qubits=2):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        feature_map(x, layers=layers, entangler=basic_entangler)
        return qml.state()

    return circuit()


x_t10 = [0.5, 1.2] #x for task 10 demonstration

print("\n=== Task 10: Data re-uploading ===")

# Sanity check: layers=1 must exactly match the plain embedding from
# Tasks 2-8, since feature_map's entangler check ("layer < layers - 1")
# never fires when layers=1 
state_1layer = state_with_layers(x_t10, layers=100)
print("layers=1 state:", state_1layer)

# deeper circuit: entangler fires twice (between embeddings
# 1-2 and 2-3), but never after the third and final embedding.
state_3layers = state_with_layers(x_t10, layers=3)
print("layers=3 state:", state_3layers)

# compare layers=1 here against a plain no-entangler embedding.
dev_plain = qml.device("default.qubit", wires=2)

@qml.qnode(dev_plain)
def plain_embedding_circuit():
    feature_map(x_t10, layers=1)   # no entangler at all, matches Task 2
    return qml.state()

state_plain = plain_embedding_circuit()
diff_t10 = np.max(np.abs(state_1layer - state_plain))
print(f"Max difference vs plain Task 2 embedding: {diff_t10:.2e}")

assert diff_t10 < 1e-10, "layers=1 should reduce EXACTLY to the plain embedding!"
print("PASS: layers=1 reduces exactly to the plain angle-embedding case.")