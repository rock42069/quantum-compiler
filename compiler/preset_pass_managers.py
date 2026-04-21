"""
Preset PassManagers for optimization levels 0–3.

Mirrors the structure of qiskit/transpiler/preset_passmanagers/level{0,1,2,3}.py
but as a single dispatch function. Each level assembles a StagedPassManager with:

  init → layout → routing → translation → optimization → scheduling

We call generate_preset_pass_manager() for the correct Qiskit internals, then
wrap the result in our PassManager so experiments can swap out stages cleanly.

Level  Layout                    Routing        Optimization
0      TrivialLayout             BasicSwap       —
1      TrivialLayout→DenseLayout BasicSwap       Optimize1qGates
2      DenseLayout→VF2Layout     LookaheadSwap   CommutativeCancellation, CXCancellation
3      SabreLayout               SabreSwap       Full peephole + gate cancellation
"""

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from .pass_manager import PassManager


def get_pass_manager(optimization_level: int, backend, seed: int = 42) -> PassManager:
    """
    Return a PassManager for the given optimization level and backend.

    Args:
        optimization_level: 0 (minimal) through 3 (aggressive).
        backend: A Qiskit BackendV2 (e.g. FakeNairobiV2).
        seed: Seed for stochastic passes (SabreLayout, SabreSwap). Always
              seed for reproducibility — unseeded runs differ each invocation.

    Returns:
        PassManager wrapping a Qiskit StagedPassManager.
    """
    if optimization_level not in (0, 1, 2, 3):
        raise ValueError(f"optimization_level must be 0–3, got {optimization_level}")

    qiskit_pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        seed_transpiler=seed,
    )
    pm = PassManager.from_qiskit(qiskit_pm)
    pm._level = optimization_level
    pm._seed = seed
    pm._backend = backend
    return pm
