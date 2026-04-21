import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def build(n_qubits: int, graph=None, p: int = 1, gamma: float = np.pi / 4, beta: float = np.pi / 4, **kwargs) -> QuantumCircuit:
    if graph is None:
        # Default: ring graph (MaxCut on a cycle)
        graph = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    for layer in range(p):
        # Cost unitary: exp(-i * gamma * C) where C is the MaxCut cost operator
        for (u, v) in graph:
            qc.rzz(2 * gamma, u, v)
        # Mixer unitary: exp(-i * beta * B) where B = sum of X_i
        for i in range(n_qubits):
            qc.rx(2 * beta, i)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc
