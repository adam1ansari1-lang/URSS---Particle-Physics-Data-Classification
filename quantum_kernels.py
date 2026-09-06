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