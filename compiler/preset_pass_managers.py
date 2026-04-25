"""
Preset PassManagers for optimization levels 0–5.

Levels 0–3 delegate directly to Qiskit's generate_preset_pass_manager().
Levels 4–5 clone Level 3 and layer on progressively more aggressive passes.

Level  Layout           Routing     Optimization
0      TrivialLayout    BasicSwap    —
1      Trivial→Dense    BasicSwap    Optimize1qGates
2      Dense→VF2        Lookahead    CommutativeCancellation, CXCancellation
3      SabreLayout      SabreSwap    Full peephole + UnitarySynthesis
4      SabreLayout      SabreSwap    Level 3 + TemplateOptimization + diagonal removal
5      SabreLayout      SabreSwap    Level 4 + HoareOptimizer (needs z3)
"""

from qiskit.transpiler import PassManager as _QiskitPassManager
from qiskit.transpiler.passes import (
    RemoveDiagonalGatesBeforeMeasure,
    TemplateOptimization,
)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from .pass_manager import PassManager


def get_pass_manager(optimization_level: int, backend, seed: int = 42) -> PassManager:
    """
    Return a PassManager for the given optimization level and backend.

    Args:
        optimization_level: 0 (minimal) through 5 (research-grade).
        backend: A Qiskit BackendV2 (e.g. FakeNairobiV2).
        seed: Seed for stochastic passes. Always seed for reproducibility.

    Returns:
        PassManager wrapping a Qiskit StagedPassManager.
    """
    if optimization_level not in range(6):
        raise ValueError(f"optimization_level must be 0–5, got {optimization_level}")

    if optimization_level <= 3:
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

    if optimization_level == 4:
        return _get_level4_pass_manager(backend, seed)
    return _get_level5_pass_manager(backend, seed)


# ── Level 4: Template Matching + Diagonal Removal ──────────────────────────

def _get_level4_pass_manager(backend, seed: int) -> PassManager:
    """
    Level 4 — Level 3 + TemplateOptimization + RemoveDiagonalGatesBeforeMeasure.
    """
    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend, seed_transpiler=seed
    )
    extra = _QiskitPassManager([
        TemplateOptimization(),
        RemoveDiagonalGatesBeforeMeasure(),
    ])
    pm.optimization = (pm.optimization or _QiskitPassManager()) + extra

    wrapped = PassManager.from_qiskit(pm)
    wrapped._level = 4
    wrapped._seed = seed
    wrapped._backend = backend
    return wrapped


# ── Level 5: Hoare Logic Gate Elimination ─────────────────────────────────

def _get_level5_pass_manager(backend, seed: int) -> PassManager:
    """
    Level 5 — Level 4 + HoareOptimizer.

    HoareOptimizer (arXiv:1810.00375) uses Hoare logic to prove, at each gate,
    what state the control qubit must be in. If a controlled gate is applied when
    the control is provably in state |0⟩ (or |1⟩), the gate is redundant and
    eliminated. Works especially well on oracles, ancilla-heavy circuits, and
    circuits with many controlled-X/Z gates.
    """
    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend, seed_transpiler=seed
    )
    extra_passes = [
        TemplateOptimization(),
        RemoveDiagonalGatesBeforeMeasure(),
    ]

    try:
        from qiskit.transpiler.passes import HoareOptimizer
        extra_passes.append(HoareOptimizer(size=3))
    except ImportError:
        pass  # z3 not installed; skip HoareOptimizer

    extra = _QiskitPassManager(extra_passes)
    pm.optimization = (pm.optimization or _QiskitPassManager()) + extra

    wrapped = PassManager.from_qiskit(pm)
    wrapped._level = 5
    wrapped._seed = seed
    wrapped._backend = backend
    return wrapped

