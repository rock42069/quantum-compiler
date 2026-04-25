"""
Backend selection for experiments.

FakeNairobiV2  — 7 qubits, real IBM noise data, heavy-hex topology
FakeSherbrooke — 127 qubits, Eagle r3 topology
"""

from qiskit_ibm_runtime.fake_provider import FakeNairobiV2, FakeSherbrooke
from qiskit.transpiler import CouplingMap


NAIROBI_MAX_QUBITS = 7
SHERBROOKE_MAX_QUBITS = 127


def get_backend(name: str = 'nairobi', n_qubits: int = None): 
    if name == 'nairobi':
        if n_qubits is not None and n_qubits > NAIROBI_MAX_QUBITS:
            return FakeSherbrooke()
        return FakeNairobiV2()
    if name == 'sherbrooke':
        return FakeSherbrooke()

    # Auto-select by qubit count
    if n_qubits is not None and n_qubits <= NAIROBI_MAX_QUBITS:
        return FakeNairobiV2()
    return FakeSherbrooke()


def linear_coupling_map(n: int) -> CouplingMap:
    """n-qubit linear chain: 0–1–2–…–(n-1)."""
    return CouplingMap.from_line(n)


def grid_coupling_map(rows: int, cols: int) -> CouplingMap:
    """rows × cols grid coupling map."""
    return CouplingMap.from_grid(rows, cols)


def ring_coupling_map(n: int) -> CouplingMap:
    """n-qubit ring: 0–1–2–…–(n-1)–0."""
    edges = [(i, (i + 1) % n) for i in range(n)]
    edges += [(b, a) for a, b in edges]   # bidirectional
    return CouplingMap(couplinglist=edges)
