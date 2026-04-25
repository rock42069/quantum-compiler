"""
Top-level runner: transpile a circuit at all optimization levels, simulate
under noise, collect metrics, and save compiled circuits to disk.

Usage:
    python compiler/transpile_all.py --algorithm grover --n_qubits 4
    python compiler/transpile_all.py --algorithm qft --n_qubits 4
"""

import argparse
import json
import os
import sys

import qiskit.qpy as qpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuits import build_circuit
from compiler.preset_pass_managers import get_pass_manager
from hardware.backends import get_backend
from analysis.metrics import compute_all_metrics

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def run_all(algorithm: str, n_qubits: int, backend_name: str = 'nairobi',
            seed: int = 42, max_level: int = 3, **circuit_kwargs) -> dict:
    """
    Transpile *algorithm* at levels 0–max_level on *backend*, simulate, collect metrics.

    Levels 0–3 use Qiskit presets. Levels 4–5 are custom extensions:
      4 — Level 3 + TemplateOptimization + diagonal gate removal
      5 — Level 4 + HoareOptimizer (needs z3-solver)

    Returns a dict keyed by level with compiled circuit + metrics.
    Compiled circuits are serialised to results/<algorithm>_<n>q_level<l>.qpy.
    """
    if max_level not in range(6):
        raise ValueError(f"max_level must be 0–5, got {max_level}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    circuit = build_circuit(algorithm, n_qubits, **circuit_kwargs)
    backend = get_backend(backend_name, n_qubits)

    results = {}

    for level in range(max_level + 1):
        pm = get_pass_manager(level, backend, seed=seed)
        compiled = pm.run(circuit)

        qpy_path = os.path.join(RESULTS_DIR, f'{algorithm}_{n_qubits}q_level{level}.qpy')
        with open(qpy_path, 'wb') as f:
            qpy.dump(compiled, f)

        metrics = compute_all_metrics(circuit, compiled, backend, seed=seed)
        results[level] = {
            'circuit': compiled,
            'qpy_path': qpy_path,
            'metrics': metrics,
        }

        tvd_str = f"{metrics['tvd']:.4f}" if metrics['tvd'] is not None else 'N/A (>20q)'
        print(
            f"  Level {level}: "
            f"gates={metrics['gate_count']:4d}  "
            f"cx={metrics['cx_count']:3d}  "
            f"depth={metrics['depth']:4d}  "
            f"TVD={tvd_str}"
        )

    _save_metrics_json(algorithm, n_qubits, results)
    return results


def _save_metrics_json(algorithm: str, n_qubits: int, results: dict):
    path = os.path.join(RESULTS_DIR, f'{algorithm}_{n_qubits}q_metrics.json')
    serialisable = {
        str(level): {k: v for k, v in data['metrics'].items()}
        for level, data in results.items()
    }
    with open(path, 'w') as f:
        json.dump(serialisable, f, indent=2)


def _parse_args():
    parser = argparse.ArgumentParser(description='Transpile and simulate a quantum algorithm at levels 0–3.')
    parser.add_argument('--algorithm', required=True, choices=['grover', 'qft'])
    parser.add_argument('--n_qubits', type=int, required=True)
    parser.add_argument('--backend', default='nairobi', choices=['nairobi', 'sherbrooke'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_level', type=int, default=3, choices=range(6),
                        help='Highest optimization level to run (0–5). '
                             'Level 4 adds TemplateOptimization; '
                             'Level 5 adds HoareOptimizer (needs z3-solver).')
    parser.add_argument('--marked_state', default=None, help='Grover: target state bitstring')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    kwargs = {}
    if args.marked_state:
        kwargs['marked_state'] = args.marked_state

    print(f"\nTranspiling {args.algorithm} ({args.n_qubits}q) on {args.backend} "
          f"— levels 0–{args.max_level}, seed={args.seed}\n")
    run_all(args.algorithm, args.n_qubits, backend_name=args.backend,
            seed=args.seed, max_level=args.max_level, **kwargs)
