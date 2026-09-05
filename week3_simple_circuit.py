import pennylane as qml

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
