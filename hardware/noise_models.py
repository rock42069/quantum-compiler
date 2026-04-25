"""
Noise model helpers for the TVD-vs-noise-scale experiment (Graph 3c).

get_noise_model()       — extract the full noise model from a backend
scaled_noise_model()    — scale all error rates by a constant factor
"""

from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import depolarizing_error, thermal_relaxation_error


def get_noise_model(backend) -> NoiseModel:
    """Extract the standard noise model from a BackendV2."""
    return NoiseModel.from_backend(backend)


def scaled_noise_model(backend, scale: float = 1.0) -> NoiseModel:
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

    # Readout errors — error.probabilities is a 2D confusion matrix (2^n × 2^n)
    import numpy as np
    from qiskit_aer.noise.errors import ReadoutError
    for qargs, error in base_model._local_readout_errors.items():
        matrix = np.array(error.probabilities) * scale
        matrix = np.minimum(matrix, 1.0)
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
