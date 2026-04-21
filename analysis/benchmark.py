"""
Benchmark runner that wraps our transpile_all pipeline.

Runs all four algorithms at multiple qubit counts across all 4 optimization
levels and aggregates metrics into a single JSON report.

ARLINE integration is omitted: arline-benchmarks requires Python 3.8–3.10
and is incompatible with the Python 3.14 environment. The benchmark here
replicates ARLINE's structured output (gate_count, depth per circuit instance)
using our own pipeline.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuits import build_circuit
from compiler.preset_pass_managers import get_pass_manager
from hardware.backends import get_backend
from analysis.metrics import gate_count, cx_count, depth


ALGORITHMS = {
    'grover': {'n_qubits_range': [2, 3, 4], 'kwargs': {}},
    'qft':    {'n_qubits_range': [2, 3, 4, 5], 'kwargs': {}},
    'bv':     {'n_qubits_range': [3, 4, 5], 'kwargs': {}},
    'qaoa':   {'n_qubits_range': [3, 4], 'kwargs': {'p': 1}},
}

BACKEND_NAME = 'nairobi'
SEED = 42


def run_benchmark(output_path: str = None) -> dict:
    """
    Run all algorithms × qubit counts × optimization levels.

    Returns nested dict: {algorithm: {n_qubits: {level: {gate_count, cx_count, depth}}}}.
    Structural metrics only (no simulation) — fast enough for full sweeps.
    """
    backend = get_backend(BACKEND_NAME)
    report = {}

    for algo, config in ALGORITHMS.items():
        report[algo] = {}
        for n in config['n_qubits_range']:
            if n > backend.num_qubits:
                continue
            report[algo][n] = {}
            circuit = build_circuit(algo, n, **config['kwargs'])

            for level in range(4):
                pm = get_pass_manager(level, backend, seed=SEED)
                compiled = pm.run(circuit)
                report[algo][n][level] = {
                    'gate_count': gate_count(compiled),
                    'cx_count': cx_count(compiled),
                    'depth': depth(compiled),
                }
                print(f"  {algo:6s} n={n} L{level}  "
                      f"gates={report[algo][n][level]['gate_count']:4d}  "
                      f"cx={report[algo][n][level]['cx_count']:3d}  "
                      f"depth={report[algo][n][level]['depth']:4d}")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

    return report


if __name__ == '__main__':
    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'benchmark_report.json'
    )
    print("Running benchmark…\n")
    run_benchmark(output_path=out)
    print(f"\nReport saved to {out}")
