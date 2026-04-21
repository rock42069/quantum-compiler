from qiskit import QuantumCircuit


def build(n_qubits: int, secret_string: str = None, **kwargs) -> QuantumCircuit:
    if secret_string is None:
        secret_string = '1' * n_qubits
    assert len(secret_string) == n_qubits, "secret_string length must equal n_qubits"

    # n_qubits data qubits + 1 ancilla qubit for the oracle
    qc = QuantumCircuit(n_qubits + 1, n_qubits)

    # Ancilla starts in |-> = H|1>
    qc.x(n_qubits)
    qc.h(n_qubits)

    qc.h(range(n_qubits))

    # Oracle: CNOT from qubit i to ancilla for each '1' in secret_string
    for i, bit in enumerate(reversed(secret_string)):
        if bit == '1':
            qc.cx(i, n_qubits)

    qc.h(range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc
