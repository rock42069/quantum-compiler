"""
Preset PassManagers for optimization levels 0–6.

Levels 0–3 delegate directly to Qiskit's generate_preset_pass_manager().
Levels 4–6 clone Level 3 and layer on progressively more aggressive passes.

Level  Layout           Routing     Optimization
0      TrivialLayout    BasicSwap    —
1      Trivial→Dense    BasicSwap    Optimize1qGates
2      Dense→VF2        Lookahead    CommutativeCancellation, CXCancellation
3      SabreLayout      SabreSwap    Full peephole + UnitarySynthesis
4      SabreLayout      SabreSwap    Level 3 + TemplateOptimization + diagonal removal
5      SabreLayout      SabreSwap    Level 4 + HoareOptimizer (needs z3)
6      SabreLayout      SabreSwap    Level 5 + ZX-calculus (needs pyzx)
"""

import copy

from qiskit.transpiler import PassManager as _QiskitPassManager
from qiskit.transpiler.passes import (
    RemoveDiagonalGatesBeforeMeasure,
    TemplateOptimization,
)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from .pass_manager import PassManager
from .pyzx_pass import PyZXOptimize


def get_pass_manager(optimization_level: int, backend, seed: int = 42) -> PassManager:
    """
    Return a PassManager for the given optimization level and backend.

    Args:
        optimization_level: 0 (minimal) through 6 (research-grade).
        backend: A Qiskit BackendV2 (e.g. FakeNairobiV2).
        seed: Seed for stochastic passes. Always seed for reproducibility.

    Returns:
        PassManager wrapping a Qiskit StagedPassManager.
    """
    if optimization_level not in range(7):
        raise ValueError(f"optimization_level must be 0–6, got {optimization_level}")

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
    if optimization_level == 5:
        return _get_level5_pass_manager(backend, seed)
    # level 6
    return _get_level6_pass_manager(backend, seed)


# ── Level 4: Template Matching + Diagonal Removal ──────────────────────────

def _get_level4_pass_manager(backend, seed: int) -> PassManager:
    """
    Level 4 — Level 3 + TemplateOptimization + RemoveDiagonalGatesBeforeMeasure.

    TemplateOptimization searches the circuit DAG for subgraphs that match
    known gate templates (e.g. 3-gate patterns that collapse to fewer gates)
    and replaces them. It catches reductions that CommutativeCancellation misses
    because it operates on larger multi-gate windows.

    RemoveDiagonalGatesBeforeMeasure strips Z/S/T/Rz gates immediately before
    measurement — they commute with the computational basis measurement and
    have zero effect on the output distribution.
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

    Requires the `z3-solver` package (pip install z3-solver). If z3 is not
    installed, this level silently degrades to Level 4.

    The `size` parameter limits the number of qubits in sub-circuits that
    HoareOptimizer fully analyses — larger values find more reductions but
    scale exponentially. size=3 is a practical default.
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


# ── Level 6: ZX-Calculus (PyZX) ───────────────────────────────────────────

def _get_level6_pass_manager(backend, seed: int) -> PassManager:
    """
    Level 6 — Level 5 + ZX-calculus optimization via PyZX.

    PyZX translates the circuit into a ZX-diagram and applies rewriting rules:
      - Spider fusion (merge adjacent spiders of the same colour)
      - Local complementation (simplify graph around a Clifford vertex)
      - Pivoting (remove pairs of connected Hadamard edges)

    Together these achieve state-of-the-art T-count reduction and typically
    reduce 2-qubit gate counts by an additional 15–20% beyond Level 5 on
    Clifford+T circuits. The effect is smaller on circuits already heavily
    optimized by Qiskit's synthesis passes.

    Requires: pip install pyzx
    Falls back to Level 5 behaviour if pyzx is not installed.

    Note: ZX-calculus optimization runs on the final translated circuit (after
    UnitarySynthesis has already decomposed to basis gates), so the PyZX pass
    is appended as a post-optimization step rather than integrated into the
    Qiskit stage pipeline.
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
        pass

    extra_passes.append(PyZXOptimize(simplify_strategy='full_reduce'))

    extra = _QiskitPassManager(extra_passes)
    pm.optimization = (pm.optimization or _QiskitPassManager()) + extra

    wrapped = PassManager.from_qiskit(pm)
    wrapped._level = 6
    wrapped._seed = seed
    wrapped._backend = backend
    return wrapped
