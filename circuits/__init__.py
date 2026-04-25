"""Circuit registry: maps algorithm names to their build() functions."""

from . import grover, qft

ALGORITHMS = {
    'grover': grover,
    'qft': qft,
}

def build_circuit(algorithm: str, n_qubits: int, **kwargs):
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: {list(ALGORITHMS)}")
    return ALGORITHMS[algorithm].build(n_qubits, **kwargs)
