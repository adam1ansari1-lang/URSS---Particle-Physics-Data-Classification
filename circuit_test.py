import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import os

# ---- Block A: set up the simulator ----
# "default.qubit" = PennyLane's built-in simulator (not real hardware)
# wires=2 means: give us a 2-qubit register
dev = qml.device("default.qubit", wires=2)

# ---- Block B: define the circuit ----
@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)       # put qubit 0 into superposition
    qml.CNOT(wires=[0, 1])      # entangle qubit 0 with qubit 1
    return qml.state()          # return the full statevector

# ---- Block C: run it and inspect the result ----
result = circuit()
print("Statevector:", result)

# Compute probabilities from the statevector.
# Each entry of `result` is a complex amplitude; probability = |amplitude|^2
probabilities = np.abs(result) ** 2

basis_labels = ["|00>", "|01>", "|10>", "|11>"]

# ---- Save a plot to plots/ ----
os.makedirs("plots", exist_ok=True)  # create the folder if it doesn't already exist

plt.figure(figsize=(5, 4))
plt.bar(basis_labels, probabilities, color="steelblue")
plt.ylabel("Probability")
plt.title("Two-qubit Bell state — basis state probabilities")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("plots/bell_state_check.png")
print("Plot saved to plots/bell_state_check.png")