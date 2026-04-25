"""
Grover's Search circuit
"""

import numpy as np
from qiskit import QuantumCircuit


def build(n_qubits: int, marked_state: str = None, **kwargs) -> QuantumCircuit:
    if marked_state is None:
        marked_state = '1' * n_qubits
    assert len(marked_state) == n_qubits, "marked_state length must equal n_qubits"

    n_iterations = max(1, round(np.pi / 4 * np.sqrt(2 ** n_qubits)))
    qc = QuantumCircuit(n_qubits, n_qubits)

    qc.h(range(n_qubits))
    for _ in range(n_iterations):
        _oracle(qc, n_qubits, marked_state)
        _diffusion(qc, n_qubits)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def _oracle(qc: QuantumCircuit, n_qubits: int, marked_state: str):
    # Flip qubits where marked_state bit is '0' so the all-ones state is the target
    for i, bit in enumerate(reversed(marked_state)):
        if bit == '0':
            qc.x(i)

    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)

    for i, bit in enumerate(reversed(marked_state)):
        if bit == '0':
            qc.x(i)


def _diffusion(qc: QuantumCircuit, n_qubits: int):
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))

    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)

    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
