# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Analyze and tweak Qiskit's transpiler pipeline across optimization levels (0–3), test on multiple quantum algorithms, and visualize the impact of compilation quality on output fidelity and circuit structure.

The core thesis: **better compilation = sharper probability peaks at correct answer states and fewer noisy gates**.

---

## Setup

```bash
pip install qiskit>=1.0 qiskit-aer>=0.14 qiskit-ibm-runtime>=0.20 matplotlib numpy
pip install arline-benchmarks arline-quantum  # note: requires Python 3.8–3.10
```

## Common Commands

```bash
# Run a single experiment (transpile + simulate + plot)
python compiler/transpile_all.py --algorithm grover --n_qubits 4

# Generate all plots for a given algorithm
python plots/histogram.py --algorithm grover
python plots/scaling.py --algorithm grover

# Run ARLINE benchmark suite
python analysis/benchmark.py

# Inspect what passes a given optimization level uses
python -c "
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeNairobi
pm = generate_preset_pass_manager(optimization_level=2, backend=FakeNairobi())
print(pm.passes())
"
```

---

## Architecture

### Pipeline Flow

```
Circuit (circuits/) → transpile at levels 0–3 (compiler/) → simulate with noise (hardware/) → metrics + plots (analysis/, plots/)
```

Each stage is independent: circuits are defined once, compiled results are saved to disk, analysis reads saved circuits.

### Module Responsibilities

- **`circuits/`** — Parameterized circuit builders. Each file exposes a `build(n_qubits, **kwargs) -> QuantumCircuit` function. Grover takes `marked_state`; BV takes `secret_string`; QAOA takes a graph.
- **`hardware/`** — Backend and noise model definitions. `backends.py` centralizes FakeBackend selection (FakeNairobi ≤7 qubits, FakeSherbrooke ≤127). `noise_models.py` provides scaled noise models for metric (d).
- **`compiler/`** — `transpile_all.py` runs all 4 optimization levels for a given circuit+backend, collects structural metrics, and saves compiled circuits to disk. `custom_passes.py` defines the tweaked PassManagers (see experiments T1–T5 below).
- **`analysis/`** — `metrics.py` computes TVD, gate count, depth. `benchmark.py` wraps ARLINE.
- **`plots/`** — Standalone scripts that read saved results and produce figures.

### Key Design Decisions

**Error metric = TVD (Total Variation Distance)**
Not just gate count. TVD between the ideal statevector distribution and the noisy simulation output is the primary quality signal — it directly corresponds to how "peaked" vs "spread" the probability histogram looks.

```python
def tvd(ideal_probs, noisy_probs):
    all_states = set(ideal_probs) | set(noisy_probs)
    return 0.5 * sum(abs(ideal_probs.get(s,0) - noisy_probs.get(s,0)) for s in all_states)
```

**Always seed stochastic passes**
`seed_transpiler=42` and `seed_simulator=42` everywhere. SabreSwap and StochasticSwap are randomized — unseeded runs produce different results each time, making comparisons meaningless.

**Save compiled circuits to disk**
Rerunning transpilation is expensive at scale. Use `qiskit.qpy` to serialize:
```python
import qiskit.qpy as qpy
with open('results/grover_4q_level3.qpy', 'wb') as f:
    qpy.dump(compiled_circuit, f)
```

---

## Hardware Platform

```python
from qiskit_ibm_runtime.fake_provider import FakeNairobi  # 7 qubits, real IBM noise data
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke  # 127 qubits

# Manual coupling maps for controlled topology experiments
from qiskit.transpiler import CouplingMap
CouplingMap.from_line(5)    # linear chain
CouplingMap.from_grid(2, 3) # 2x3 grid
```

Basis gate set (modern IBM): `['cx', 'rz', 'sx', 'x']`

---

## Transpiler Pass Structure

| Level | Layout | Routing | Optimization |
|-------|--------|---------|--------------|
| 0 | TrivialLayout | BasicSwap | None |
| 1 | TrivialLayout → DenseLayout | BasicSwap | Optimize1qGates |
| 2 | DenseLayout → NoiseAdaptiveLayout | StochasticSwap | CommutativeCancellation, CXCancellation |
| 3 | SabreLayout | SabreSwap | Full peephole + gate cancellation |

Key source files in Qiskit: `qiskit/transpiler/preset_passmanagers/level{0,1,2,3}.py`, `passes/routing/sabre_swap.py`, `passes/layout/sabre_layout.py`

---

## Experiments (Transpiler Tweaks)

| ID | Change | What It Tests |
|---|---|---|
| T1 | Replace `SabreSwap` → `StochasticSwap` at Level 3 | SWAP overhead of routing algorithm |
| T2 | Replace `SabreLayout` → `NoiseAdaptiveLayout` | Noise-aware vs topology-aware mapping |
| T3 | Disable gate cancellation passes at Level 2 | Isolated impact of 1-qubit optimization |
| T4 | Insert `CommutationAnalysis` before routing | Pre-routing gate reduction |
| T5 | Vary `seed_transpiler` across 5 seeds, fixed level | Variance in stochastic pass quality |

---

## Target Graphs

**Graph 1 — Probability Histograms (Bad vs Good Compilation)**
Side-by-side bar charts of output state probabilities under noisy simulation. Bad = Level 0 (flat distribution), Good = Level 3 (sharp peak at correct answer state). One pair per algorithm.

**Graph 2 — Circuit Diagrams**
`circuit.draw('mpl')` for Level 0 vs Level 3 compiled circuits — visually shows gate count and depth reduction.

**Graph 3 — Metric Scaling Plots**
- Gate count / depth / TVD vs optimization level (per algorithm)
- CX count vs n_qubits (for each optimization level)
- TVD vs noise scale factor (0.5×, 1×, 2×)

---

## Algorithm Suite

| Algorithm | File | Key Parameter |
|---|---|---|
| Grover's Search | `circuits/grover.py` | `n_qubits`, `marked_state` (default: all-ones) |
| QFT | `circuits/qft.py` | `n_qubits` |
| Bernstein-Vazirani | `circuits/bv.py` | `secret_string` |
| QAOA | `circuits/qaoa.py` | graph, `p` (layers) |

All circuits expose `build(n_qubits, **kwargs) -> QuantumCircuit`.

---

## Simulation Recipe

```python
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

# Ideal baseline
ideal_sim = AerSimulator(method='statevector')
ideal_result = ideal_sim.run(circuit, shots=8192).result()

# Noisy run
noise_model = NoiseModel.from_backend(backend)
noisy_sim = AerSimulator(noise_model=noise_model)
noisy_result = noisy_sim.run(compiled_circuit, shots=8192).result()
```

Use 8192 shots minimum for stable probability estimates. Run each experiment 5× with different seeds and report mean ± std.

---

## ARLINE Integration

ARLINE's `BenchmarkPipeline` takes `CircuitConfig` + `HardwareConfig` + `CompilerConfig` (wrapping Qiskit transpile) and outputs gate count/depth in a structured format. Use it for standardized circuit instances and cross-compiler comparison. ARLINE has Grover, QFT, BV, and random circuits built in.
