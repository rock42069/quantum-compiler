"""
Graph 3 — Metric Scaling Plots.

3a. Gate count / depth / TVD vs optimization level (per algorithm)
3b. CX count vs n_qubits (for each optimization level)
3c. TVD vs noise scale factor (0.5×, 1×, 2×)

Usage:
    python plots/scaling.py --algorithm grover
    python plots/scaling.py --algorithm grover --plot 3c --n_qubits 4
"""

import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def _load_metrics(algorithm: str, n_qubits: int) -> dict:
    path = os.path.join(RESULTS_DIR, f'{algorithm}_{n_qubits}q_metrics.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No metrics at {path}. Run transpile_all.py first.")
    with open(path) as f:
        return json.load(f)


def plot_3a_metrics_vs_level(algorithm: str, n_qubits: int, save: bool = True):
    """Gate count, depth, and TVD across optimization levels 0–3."""
    metrics = _load_metrics(algorithm, n_qubits)
    levels = [0, 1, 2, 3]

    gate_counts = [metrics[str(l)]['gate_count'] for l in levels]
    depths = [metrics[str(l)]['depth'] for l in levels]
    tvds = [metrics[str(l)]['tvd'] for l in levels]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f'{algorithm.upper()} ({n_qubits}q) — Metrics vs Optimization Level', fontsize=12)

    for ax, values, label, color in zip(
        axes,
        [gate_counts, depths, tvds],
        ['Gate Count', 'Circuit Depth', 'TVD'],
        ['steelblue', 'seagreen', 'tomato'],
    ):
        ax.plot(levels, values, 'o-', color=color, linewidth=2, markersize=7)
        ax.set_xlabel('Optimization Level')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(levels)
        ax.grid(True, alpha=0.3)

    out_path = os.path.join(RESULTS_DIR, f'{algorithm}_{n_qubits}q_scaling_3a.png')
    if save:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved {out_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_3b_cx_vs_qubits(algorithm: str, save: bool = True):
    """CX count vs n_qubits for each optimization level (reads benchmark_report.json)."""
    report_path = os.path.join(RESULTS_DIR, 'benchmark_report.json')
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"No benchmark report. Run analysis/benchmark.py first.")

    with open(report_path) as f:
        report = json.load(f)

    if algorithm not in report:
        raise KeyError(f"Algorithm '{algorithm}' not in benchmark report.")

    algo_data = report[algorithm]
    n_list = sorted(int(n) for n in algo_data)
    levels = [0, 1, 2, 3]
    colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8']

    fig, ax = plt.subplots(figsize=(7, 5))
    for level, color in zip(levels, colors):
        cx_vals = [algo_data[str(n)][str(level)]['cx_count'] for n in n_list
                   if str(n) in algo_data and str(level) in algo_data[str(n)]]
        valid_n = [n for n in n_list if str(n) in algo_data and str(level) in algo_data[str(n)]]
        ax.plot(valid_n, cx_vals, 'o-', color=color, label=f'Level {level}', linewidth=2)

    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('CX Gate Count')
    ax.set_title(f'{algorithm.upper()} — CX Count vs Qubits')
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(RESULTS_DIR, f'{algorithm}_scaling_3b.png')
    if save:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved {out_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_3c_tvd_vs_noise(algorithm: str, n_qubits: int, save: bool = True):
    """TVD vs noise scale factor, computed on-the-fly for Level 3."""
    from circuits import build_circuit
    from compiler.preset_pass_managers import get_pass_manager
    from hardware.backends import get_backend
    from analysis.metrics import compute_tvd_vs_noise

    backend = get_backend('nairobi')
    circuit = build_circuit(algorithm, n_qubits)
    pm = get_pass_manager(3, backend)
    compiled = pm.run(circuit)

    scales = [0.25, 0.5, 1.0, 1.5, 2.0]
    tvd_values = compute_tvd_vs_noise(circuit, compiled, backend, scales=scales)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(scales, [tvd_values[s] for s in scales], 'o-', color='tomato', linewidth=2, markersize=7)
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5, label='Real device noise')
    ax.set_xlabel('Noise Scale Factor')
    ax.set_ylabel('TVD')
    ax.set_title(f'{algorithm.upper()} ({n_qubits}q) — TVD vs Noise Scale (Level 3)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(RESULTS_DIR, f'{algorithm}_{n_qubits}q_scaling_3c.png')
    if save:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved {out_path}")
    else:
        plt.show()
    plt.close(fig)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', required=True)
    parser.add_argument('--n_qubits', type=int, default=4)
    parser.add_argument('--plot', choices=['3a', '3b', '3c', 'all'], default='all')
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    save = not args.show

    if args.plot in ('3a', 'all'):
        plot_3a_metrics_vs_level(args.algorithm, args.n_qubits, save=save)
    if args.plot in ('3b', 'all'):
        plot_3b_cx_vs_qubits(args.algorithm, save=save)
    if args.plot in ('3c', 'all'):
        plot_3c_tvd_vs_noise(args.algorithm, args.n_qubits, save=save)
