"""
Primary metric is TVD (Total Variation Distance) between the ideal statevector
distribution and the noisy simulation output. 

Secondary metrics: gate_count, cx_count, depth (structural complexity).
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel


SHOTS = 8192        # minimum for stable probability estimates
SIM_QUBIT_LIMIT = 20  # circuits wider than this skip statevector simulation


def tvd(ideal_probs: dict, noisy_probs: dict) -> float:
    """
    Total Variation Distance between two probability distributions.

    TVD = 0.5 * sum |P_ideal(s) - P_noisy(s)| over all basis states.
    Range: [0, 1]. TVD=0 means perfect compilation; TVD=1 means no overlap.
    """
    all_states = set(ideal_probs) | set(noisy_probs)
    return 0.5 * sum(
        abs(ideal_probs.get(s, 0.0) - noisy_probs.get(s, 0.0))
        for s in all_states
    )


def gate_count(circuit: QuantumCircuit) -> int:
    """Total number of gate operations (excludes measurements and barriers)."""
    return circuit.size(lambda inst: inst.operation.name not in ('measure', 'barrier', 'reset'))


def cx_count(circuit: QuantumCircuit) -> int:
    """Number of two-qubit CX / CNOT gates."""
    return circuit.count_ops().get('cx', 0)


def depth(circuit: QuantumCircuit) -> int:
    """Circuit depth (critical path length)."""
    return circuit.depth()


def _get_counts(circuit: QuantumCircuit, shots: int, noise_model=None, seed: int = 42) -> dict:
    """Run *circuit* on AerSimulator and return normalised probability dict."""
    if noise_model is not None:
        sim = AerSimulator(noise_model=noise_model)
    else:
        sim = AerSimulator(method='statevector')

    result = sim.run(circuit, shots=shots, seed_simulator=seed).result()
    raw = result.get_counts()
    total = sum(raw.values())
    return {state: count / total for state, count in raw.items()}


def compute_all_metrics(original: QuantumCircuit, compiled: QuantumCircuit,
                        backend, seed: int = 42) -> dict:
    """
    Compute structural and fidelity metrics for a compiled circuit.

    For circuits with more than SIM_QUBIT_LIMIT qubits the statevector
    simulation is skipped (2^n memory is infeasible) and tvd/probs are None.

    Returns a flat dict suitable for JSON serialisation.
    """
    struct = {
        'gate_count': gate_count(compiled),
        'cx_count': cx_count(compiled),
        'depth': depth(compiled),
    }

    if original.num_qubits > SIM_QUBIT_LIMIT:
        return {**struct, 'tvd': None, 'ideal_probs': {}, 'noisy_probs': {}}

    noise_model = NoiseModel.from_backend(backend)
    ideal_probs = _get_counts(original, shots=SHOTS, seed=seed)
    noisy_probs = _get_counts(compiled, shots=SHOTS, noise_model=noise_model, seed=seed)

    return {
        **struct,
        'tvd': tvd(ideal_probs, noisy_probs),
        'ideal_probs': ideal_probs,
        'noisy_probs': noisy_probs,
    }


def compute_tvd_vs_noise(original: QuantumCircuit, compiled: QuantumCircuit,
                         backend, scales=(0.5, 1.0, 2.0), seed: int = 42) -> dict:
    """
    Sweep noise scale and compute TVD at each level (for Graph 3c).

    Returns {scale: tvd_value}.
    """
    from hardware.noise_models import scaled_noise_model

    ideal_probs = _get_counts(original, shots=SHOTS, seed=seed)
    results = {}
    for scale in scales:
        nm = scaled_noise_model(backend, scale=scale)
        noisy_probs = _get_counts(compiled, shots=SHOTS, noise_model=nm, seed=seed)
        results[scale] = tvd(ideal_probs, noisy_probs)
    return results
