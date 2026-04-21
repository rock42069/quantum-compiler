"""
Noise model helpers for the TVD-vs-noise-scale experiment (Graph 3c).

get_noise_model()       — extract the full noise model from a backend
scaled_noise_model()    — scale all error rates by a constant factor

Scaling works by multiplying every gate/readout error probability by `scale`,
which lets us sweep across 0.5×, 1×, 2× noise to observe TVD vs noise plots.
"""

from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import depolarizing_error, thermal_relaxation_error


def get_noise_model(backend) -> NoiseModel:
    """Extract the standard noise model from a BackendV2."""
    return NoiseModel.from_backend(backend)


def scaled_noise_model(backend, scale: float = 1.0) -> NoiseModel:
    """
    Build a noise model from *backend* with all error rates multiplied by *scale*.

    scale < 1.0 → quieter hardware (best case)
    scale = 1.0 → real device noise
    scale > 1.0 → noisier hardware (worst case)

    We rebuild each QuantumError by scaling its depolarizing parameter.
    Probabilities are clamped to [0, 1] to keep the model physical.
    """
    base_model = NoiseModel.from_backend(backend)
    if scale == 1.0:
        return base_model

    scaled = NoiseModel(basis_gates=base_model.basis_gates)

    for instruction, qargs_dict in base_model._local_quantum_errors.items():
        for qargs, error in qargs_dict.items():
            scaled_error = _scale_error(error, scale)
            scaled.add_quantum_error(scaled_error, instruction, list(qargs))

    for instruction, error in base_model._default_quantum_errors.items():
        scaled.add_all_qubit_quantum_error(_scale_error(error, scale), instruction)

    # Readout errors
    for qargs, error in base_model._local_readout_errors.items():
        probs = error.probabilities
        new_probs = [min(p * scale, 1.0) for p in probs]
        # Renormalise rows so each sums to 1
        from qiskit_aer.noise.errors import ReadoutError
        n = len(qargs)
        import numpy as np
        matrix = np.array(new_probs).reshape(2**n, 2**n)
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        scaled.add_readout_error(ReadoutError(matrix), list(qargs))

    return scaled


def _scale_error(error, scale: float):
    """Scale a QuantumError's noise by multiplying its depolarizing parameter."""
    # Rebuild as depolarizing error with scaled total probability
    # error.to_dict() gives circuits + probabilities
    total_p = sum(
        p for p, _ in zip(error.probabilities, error.circuits) if _ is not None
    )
    # Use the first circuit's num_qubits as the gate size
    n_qubits = error.num_qubits
    new_p = min(total_p * scale, 1.0)
    return depolarizing_error(new_p, n_qubits)
