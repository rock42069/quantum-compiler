# Quantum Compiler — Internals Explained

## 1. Backend Models

**What is a "backend"?**

A backend represents a physical quantum processor. It encodes three things that the transpiler needs:
- **Coupling map** — which pairs of qubits can run a two-qubit gate (e.g. CX). Not every qubit talks to every other; the topology is a sparse graph.
- **Basis gate set** — the native gates the hardware understands. IBM hardware uses `['cx', 'rz', 'sx', 'x']`. Every other gate in the circuit must be decomposed into these.
- **Calibration data** — per-qubit/per-gate error rates, T1/T2 coherence times, readout errors. This is what the noise model is built from.

**FakeNairobiV2** (`hardware/backends.py:10`)

A 7-qubit simulator that ships with real IBM calibration data taken from the physical Nairobi device. Its coupling map is a heavy-hex fragment (not a full graph — most qubits only connect to 2–3 neighbours). Used for experiments with ≤7 qubits because the noise data is realistic without requiring cloud access.

**FakeSherbrooke** (`hardware/backends.py:10`)

A 127-qubit Eagle r3 topology simulator. Same idea, larger device. Used when algorithms need more qubits (e.g. QAOA on large graphs).

**Coupling map helpers** (`hardware/backends.py:33–47`)

For controlled topology experiments the project also exposes manually constructed coupling maps:

```python
linear_coupling_map(n)       # 0–1–2–…–(n-1)  — worst case for routing
grid_coupling_map(rows, cols) # 2D grid
ring_coupling_map(n)          # circular, bidirectional
```

These let you test how routing algorithm performance changes with topology, independent of real device noise.

**Selection logic** (`hardware/backends.py:14–30`)

```python
get_backend('nairobi')   # → FakeNairobiV2
get_backend('sherbrooke') # → FakeSherbrooke
get_backend(n_qubits=5)  # auto-selects: ≤7 → Nairobi, else Sherbrooke
```

---

## 2. Noise Models

**Why noise models?**

Real quantum hardware has errors on every operation. A noise model attached to AerSimulator makes the classical simulation mimic this behaviour, so you can evaluate how compilation quality affects output fidelity without touching real hardware.

**Base noise model** (`hardware/noise_models.py:15–17`)

```python
NoiseModel.from_backend(backend)
```

Extracts three error types from the backend's calibration data:
- **Gate errors** — depolarizing channels applied after each gate. A depolarizing channel randomly applies X, Y, or Z with some probability, smearing the quantum state.
- **Readout errors** — probability that measuring |0⟩ returns "1" or vice versa. Encoded as a 2×2 confusion matrix per qubit.
- **Thermal relaxation** — T1 (energy decay) and T2 (dephasing) errors over gate duration. Longer gates on slower hardware accumulate more error.

**Scaled noise model** (`hardware/noise_models.py:20–57`)

Used for Graph 3c (TVD vs noise scale). Rebuilds the noise model with all error probabilities multiplied by `scale`:

```python
scaled_noise_model(backend, scale=0.5)  # quieter: test best-case
scaled_noise_model(backend, scale=1.0)  # real device
scaled_noise_model(backend, scale=2.0)  # noisier: test worst-case
```

Implementation detail (`hardware/noise_models.py:60–70`): each `QuantumError` is rebuilt as a new depolarizing error with `p_new = min(p_old * scale, 1.0)`. Readout error matrices are rescaled and re-normalised so each row still sums to 1.

**How noise interacts with compilation**

Fewer gates after compilation = fewer error opportunities. This is why TVD (the fidelity metric) drops as optimization level rises — the same algorithm runs in fewer noisy operations.

---

## 3. Optimization Levels 0–3

### Architecture: PassManager and Stages

Every optimization level is a **StagedPassManager** — an ordered pipeline of named stages:

```
init → layout → routing → translation → optimization → scheduling
```

Each stage is itself a `PassManager` (list of passes). Our wrapper (`compiler/pass_manager.py`) mirrors Qiskit's internal `PassManager` class:

- `AnalysisPass` — reads the DAG, writes results to `property_set`. Does not modify the circuit. (`pass_manager.py:54`)
- `TransformationPass` — reads the DAG and `property_set`, returns a modified DAG. (`pass_manager.py:65`)
- `PassManager.run()` — converts circuit → DAG, runs each pass in order, converts back. (`pass_manager.py:115–144`)

The actual preset levels are generated via `generate_preset_pass_manager()` and wrapped (`compiler/preset_pass_managers.py:40–48`).

---

### Level 0 — Bare minimum (`preset_pass_managers.py:24`)

**Goal:** Get the circuit running on hardware. No optimisation.

| Stage | Pass | What it does |
|---|---|---|
| Layout | `TrivialLayout` | Maps logical qubit 0 → physical qubit 0, 1 → 1, etc. Ignores connectivity and noise. |
| Routing | `BasicSwap` | Inserts SWAP gates wherever a CX can't execute. Uses a simple greedy scan. |
| Translation | `BasisTranslator` | Decomposes all gates to `{cx, rz, sx, x}`. |
| Optimization | — | None. |

**Result:** Circuits are valid but bloated. Many extra SWAPs, no gate cancellation, worst TVD.

---

### Level 1 — Light optimisation (`preset_pass_managers.py:24`)

**Goal:** Reduce single-qubit gate overhead with minimal compile time.

| Stage | Pass | What it does |
|---|---|---|
| Layout | `TrivialLayout` → `DenseLayout` | Falls back to DenseLayout (places qubits where the coupling is densest) if TrivialLayout produces too many SWAPs. |
| Routing | `BasicSwap` | Same greedy router as Level 0. |
| Optimization | `Optimize1qGates` | Merges and cancels consecutive single-qubit gates. E.g. `RZ(π)·RZ(π) = I` is removed. |

**Result:** Noticeably fewer single-qubit gates, similar two-qubit gate count to Level 0.

---

### Level 2 — Moderate optimisation (`preset_pass_managers.py:24`)

**Goal:** Better layout, smarter routing, and gate cancellation.

| Stage | Pass | What it does |
|---|---|---|
| Layout | `DenseLayout` → `VF2Layout` | VF2Layout uses subgraph isomorphism to find a qubit placement that minimises expected error (uses the backend's gate error rates). |
| Routing | `LookaheadSwap` | BFS-based router that looks several steps ahead to find cheaper SWAP paths than BasicSwap. |
| Optimization | `CommutativeCancellation`, `CXCancellation` | Detects gates that commute and cancels pairs. E.g. `CX·CX = I` on the same qubits is eliminated. |

**Result:** Significant reduction in CX count and depth vs Level 1. TVD improves substantially.

---

### Level 3 — Aggressive optimisation (`preset_pass_managers.py:24`)

**Goal:** Minimum gate count and depth, regardless of compile time.

| Stage | Pass | What it does |
|---|---|---|
| Layout | `SabreLayout` | Iterative, stochastic. Runs SABRE routing forward and backward several times to co-optimise layout and routing simultaneously. Finds better placements than VF2Layout for circuits with many two-qubit gates. |
| Routing | `SabreSwap` | Decay-heuristic router. Maintains per-qubit "decay" scores so recently used paths are deprioritised, spreading SWAPs more evenly across the coupling map. |
| Optimization | Full peephole: `Optimize1qGates`, `CommutativeCancellation`, `CXCancellation`, `ConsolidateBlocks`, `UnitarySynthesis` | Multi-pass optimisation. `ConsolidateBlocks` merges adjacent gates into unitary matrices; `UnitarySynthesis` re-synthesises them into the shortest possible basis-gate sequence. |

**Result:** Lowest gate count, depth, and TVD. SabreLayout+SabreSwap is stochastic — seeded at 42 for reproducibility (`preset_pass_managers.py:40`).

---

### Summary Table

| Level | Layout | Routing | Optimization | Compile Time |
|---|---|---|---|---|
| 0 | TrivialLayout | BasicSwap | None | Fastest |
| 1 | Trivial → Dense | BasicSwap | 1q gate merge | Fast |
| 2 | Dense → VF2 | LookaheadSwap | Commutative cancellation | Moderate |
| 3 | SabreLayout | SabreSwap | Full peephole + synthesis | Slowest |

---

## 4. Custom Experiments (T1–T5)

Defined in `compiler/custom_passes.py`. Each clones a preset StagedPassManager and swaps exactly one stage:

| Exp | File Location | Change | Measures |
|---|---|---|---|
| T1 | `custom_passes.py:41` | Level 3, SabreSwap → LookaheadSwap | SWAP overhead of routing algorithm |
| T2 | `custom_passes.py:56` | Level 3 routing, Level 2 layout (VF2Layout) | Noise-aware vs topology-aware mapping |
| T3 | `custom_passes.py:78` | Level 2, optimization stage cleared | Isolated impact of gate cancellation |
| T4 | `custom_passes.py:91` | Level 2, CommutationAnalysis prepended before routing | Pre-routing gate reduction |
| T5 | `custom_passes.py:105` | Level 3 across 5 seeds | Variance from SabreLayout stochasticity |

---

## 5. How It All Connects

```
circuits/grover.py          build(n_qubits) → QuantumCircuit
        ↓
compiler/preset_pass_managers.py   get_pass_manager(level, backend)
        ↓
compiler/pass_manager.py           PassManager.run(circuit) → compiled_circuit
        ↓
analysis/metrics.py                compute_all_metrics(original, compiled, backend)
    ├── AerSimulator (statevector)  → ideal_probs
    └── AerSimulator + NoiseModel   → noisy_probs → TVD
        ↓
plots/histogram.py, plots/scaling.py   visualise results
```

Compiled circuits are serialised to `results/<algo>_<n>q_level<l>.qpy` via `qiskit.qpy` so transpilation only runs once (`transpile_all.py:46–48`).
