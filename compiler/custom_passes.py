"""
Custom PassManagers for experiments T1–T5.

Each experiment clones the relevant preset StagedPassManager and swaps out
exactly one stage, isolating the effect of that single change on TVD / gate
count / depth. This mirrors how Qiskit's StagedPassManager is designed to be
extended — stages are named attributes you can replace with a new PassManager.

T1  Level 3 but SabreSwap → LookaheadSwap       SWAP overhead: routing algo
T2  Level 3 but SabreLayout → VF2Layout          Noise-aware vs topology layout
T3  Level 2 but optimization stage cleared       Isolated impact of gate cancellation
T4  Level 2 but CommutationAnalysis prepended    Pre-routing gate reduction
T5  Level 3 run across 5 seeds                   Variance in stochastic pass quality

Note: StochasticSwap and NoiseAdaptiveLayout were removed in Qiskit 2.x.
LookaheadSwap and VF2Layout are their functional successors.
"""

import copy

from qiskit.transpiler import PassManager as _QiskitPassManager
from qiskit.transpiler.passes import (
    LookaheadSwap,
    VF2Layout,
    CommutationAnalysis,
)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from .pass_manager import PassManager


def _clone_staged_pm(optimization_level: int, backend, seed: int):
    """Generate a fresh StagedPassManager that we can mutate."""
    return generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        seed_transpiler=seed,
    )


def get_t1_pass_manager(backend, seed: int = 42) -> PassManager:
    """
    T1 — Replace SabreSwap with LookaheadSwap at Level 3.

    Measures: how much SWAP overhead does the routing algorithm introduce?
    LookaheadSwap is a greedy BFS router; SabreSwap uses decay heuristics.
    """
    pm = _clone_staged_pm(3, backend, seed)
    coupling_map = backend.coupling_map
    pm.routing = _QiskitPassManager([
        LookaheadSwap(coupling_map, search_depth=4, search_width=4),
    ])
    return PassManager.from_qiskit(pm)


def get_t2_pass_manager(backend, seed: int = 42) -> PassManager:
    """
    T2 — Replace SabreLayout with VF2Layout at Level 3.

    Measures: subgraph-isomorphism layout (VF2Layout scores candidate placements
    by gate error rates) vs iterative SABRE placement (SabreLayout).

    Implementation: use the Level 2 preset (which already uses VF2Layout with
    proper conditional fallback) but replace its routing stage with SabreSwap
    so that routing is identical to Level 3. This isolates the layout algorithm
    as the only variable — the same approach Qiskit uses internally.
    """
    from qiskit.transpiler.passes import SabreSwap

    pm = _clone_staged_pm(2, backend, seed)   # Level 2 uses VF2Layout
    coupling_map = backend.coupling_map
    pm.routing = _QiskitPassManager([
        SabreSwap(coupling_map, heuristic='decay', seed=seed),
    ])
    return PassManager.from_qiskit(pm)


def get_t3_pass_manager(backend, seed: int = 42) -> PassManager:
    """
    T3 — Disable gate cancellation passes at Level 2.

    Measures: the isolated contribution of CommutativeCancellation and
    CommutativeInverseCancellation to circuit depth and TVD.
    Removing them leaves all other Level 2 stages intact.
    """
    pm = _clone_staged_pm(2, backend, seed)
    pm.optimization = _QiskitPassManager()   # empty — no gate cancellation
    return PassManager.from_qiskit(pm)


def get_t4_pass_manager(backend, seed: int = 42) -> PassManager:
    """
    T4 — Prepend CommutationAnalysis before routing at Level 2.

    Measures: whether pre-computing commutativity relations before routing
    allows the router to find cheaper SWAP paths. The analysis results are
    written into property_set and can be read by passes in later stages.
    """
    pm = _clone_staged_pm(2, backend, seed)
    existing_routing = pm.routing
    pm.routing = _QiskitPassManager([CommutationAnalysis()]) + (existing_routing or _QiskitPassManager())
    return PassManager.from_qiskit(pm)


def get_t5_pass_managers(backend, seeds=(0, 1, 2, 3, 4)) -> list:
    """
    T5 — Level 3 compiled across multiple seeds.

    Returns a list of PassManagers, one per seed. Run each on the same
    circuit and collect {gate_count, depth, TVD} to measure the variance
    introduced by SabreLayout's stochastic component.
    """
    return [PassManager.from_qiskit(_clone_staged_pm(3, backend, s)) for s in seeds]


# Convenience map for external callers
CUSTOM_EXPERIMENTS = {
    'T1': get_t1_pass_manager,
    'T2': get_t2_pass_manager,
    'T3': get_t3_pass_manager,
    'T4': get_t4_pass_manager,
}
