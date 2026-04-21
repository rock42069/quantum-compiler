import numpy as np
from qiskit import QuantumCircuit


def build(n_qubits: int, **kwargs) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    _qft_rotations(qc, n_qubits)
    _swap_registers(qc, n_qubits)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def _qft_rotations(qc: QuantumCircuit, n: int):
    for i in range(n):
        qc.h(i)
        for j in range(i + 1, n):
            qc.cp(np.pi / (2 ** (j - i)), i, j)


def _swap_registers(qc: QuantumCircuit, n: int):
    for i in range(n // 2):
        qc.swap(i, n - i - 1)
