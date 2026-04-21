from . import grover, qft, bv, qaoa

ALGORITHMS = {
    'grover': grover,
    'qft': qft,
    'bv': bv,
    'qaoa': qaoa,
}

def build_circuit(algorithm: str, n_qubits: int, **kwargs):
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: {list(ALGORITHMS)}")
    return ALGORITHMS[algorithm].build(n_qubits, **kwargs)
