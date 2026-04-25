"""
Runs Grover and QFT at multiple qubit counts across optimization levels 0–3
and aggregates metrics into a single JSON report.
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
    'grover': {'n_qubits_range': [3, 4, 5, 6], 'kwargs': {}},
    'qft':    {'n_qubits_range': [3, 4, 5, 6, 16, 64], 'kwargs': {}},
}

BACKEND_NAME = 'nairobi'
SEED = 42


def run_benchmark(output_path: str = None) -> dict:
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
