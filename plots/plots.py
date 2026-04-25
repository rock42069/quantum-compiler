"""
Three focused plots comparing Grover's Search and QFT across optimization levels.

  1. histogram        — state probability distribution at 4 qubits (L0 vs L5)
  2. metrics_vs_level — gate count + circuit depth vs optimization level (4 qubits)
  3. qubit_scaling    — gate count, CX count, depth vs qubit count (3–6)

Usage:
    python plots/plots.py           # saves PNGs to results/
    python plots/plots.py --show    # display interactively instead
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

ALGORITHMS   = ['grover', 'qft']
ALGO_LABELS  = {'grover': "Grover's Search", 'qft': 'QFT'}
ALGO_COLORS  = {'grover': '#e07b39', 'qft': '#3a7ebf'}
ALGO_MARKERS = {'grover': 'o', 'qft': 's'}

LEVEL_COLORS  = {0: '#c0392b', 1: '#e67e22', 2: '#27ae60',
                 3: '#2980b9', 4: '#8e44ad', 5: '#16a085'}
LEVEL_MARKERS = {0: 'o', 1: 's', 2: '^', 3: 'D', 4: 'P', 5: 'X'}

QUBIT_COUNTS = [2, 3, 4, 5]


# ── helpers ──────────────────────────────────────────────────────────────────

def _load(algorithm: str, n_qubits: int):
    path = os.path.join(RESULTS_DIR, f'{algorithm}_{n_qubits}q_metrics.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save_or_show(fig, fname: str, save: bool):
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, fname)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved {path}')
        plt.close(fig)
    else:
        plt.show()


# ── Plot 1: Probability Histogram (4 qubits, L0 vs L5) ───────────────────────

def plot_histogram(n_qubits: int = 4, save: bool = True):
    """
    2×2 grid showing the output state probability distribution.
    Rows = algorithm (Grover, QFT). Columns = compilation level (L0, L5).
    Blue = ideal statevector, red = noisy simulation.
    """
    levels_to_show = [0, 5]
    n_cols = len(levels_to_show)

    fig, axes = plt.subplots(len(ALGORITHMS), n_cols, figsize=(14, 9), sharey='row')
    fig.suptitle(
        f'Output State Probability Distribution — {n_qubits} Qubits\n'
        'Blue = Ideal  |  Red = Noisy simulation (FakeNairobiV2)',
        fontsize=13, fontweight='bold',
    )

    level_titles = {
        0: 'Level 0 — No optimisation\n(TrivialLayout + BasicSwap)',
        5: 'Level 5 — + HoareOptimizer\n(SabreLayout + SabreSwap)',
    }

    for row, algo in enumerate(ALGORITHMS):
        data = _load(algo, n_qubits)
        if data is None:
            for ax in axes[row]:
                ax.text(0.5, 0.5, f'No data\n({algo} {n_qubits}q)',
                        ha='center', va='center', transform=ax.transAxes, fontsize=11)
            continue

        # Fix x-axis ordering from L0 ideal so levels are visually comparable
        ideal_l0 = data.get('0', {}).get('ideal_probs', {})
        n_show = 2 ** n_qubits
        ordered_states = sorted(ideal_l0, key=ideal_l0.get, reverse=True)[:n_show]
        x = np.arange(len(ordered_states))
        w = 0.38

        for col, level in enumerate(levels_to_show):
            ax = axes[row][col]
            entry = data.get(str(level))
            if entry is None:
                ax.text(0.5, 0.5, f'Level {level} not run',
                        ha='center', va='center', transform=ax.transAxes)
                continue

            ideal_vals = np.array([entry['ideal_probs'].get(s, 0.0) for s in ordered_states])
            noisy_vals = np.array([entry['noisy_probs'].get(s, 0.0) for s in ordered_states])

            ax.bar(x - w / 2, ideal_vals, w, color='#4c8fc0', alpha=0.80, label='Ideal', zorder=3)
            ax.bar(x + w / 2, noisy_vals, w, color='#d94f3d', alpha=0.85, label='Noisy', zorder=3)

            tvd = entry.get('tvd')
            subtitle = (f'Gates={entry["gate_count"]}  CX={entry["cx_count"]}  '
                        f'Depth={entry["depth"]}  TVD={tvd:.4f}' if tvd is not None
                        else f'Gates={entry["gate_count"]}  CX={entry["cx_count"]}  Depth={entry["depth"]}')
            ax.set_title(
                f'{ALGO_LABELS[algo]} — {level_titles[level]}\n{subtitle}',
                fontsize=9, fontweight='bold', pad=5,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(ordered_states, rotation=45, ha='right', fontsize=7.5)
            ax.set_xlabel('Basis state', fontsize=9)
            ax.set_ylabel('Probability', fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.25, zorder=0)
            ax.set_axisbelow(True)

            if row == 0 and col == 0:
                ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    _save_or_show(fig, f'histogram_{n_qubits}q.png', save)


# ── Plot 2: Metrics vs Optimization Level (4 qubits) ─────────────────────────

def plot_metrics_vs_level(n_qubits: int = 4, save: bool = True):
    """
    Gate count and circuit depth vs optimization level for both algorithms.
    Both algorithms are overlaid on the same axes for direct comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f'Compilation Metrics vs Optimization Level — {n_qubits} Qubits\n'
        'FakeNairobiV2 backend, seed=42',
        fontsize=13, fontweight='bold',
    )

    metrics = [('gate_count', 'Gate Count'), ('depth', 'Circuit Depth')]

    for ax, (metric, label) in zip(axes, metrics):
        for algo in ALGORITHMS:
            data = _load(algo, n_qubits)
            if data is None:
                continue
            levels = sorted(int(k) for k in data)
            vals   = [data[str(l)][metric] for l in levels]

            ax.plot(levels, vals,
                    color=ALGO_COLORS[algo],
                    marker=ALGO_MARKERS[algo],
                    linewidth=2.2, markersize=8,
                    label=ALGO_LABELS[algo], zorder=3)

            # Annotate each data point
            for l, v in zip(levels, vals):
                ax.annotate(str(v), xy=(l, v), xytext=(0, 6),
                            textcoords='offset points',
                            ha='center', fontsize=7.5,
                            color=ALGO_COLORS[algo])

        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Optimization Level', fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_xticks(range(6))
        ax.set_xticklabels([f'L{i}' for i in range(6)], fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)

    plt.tight_layout()
    _save_or_show(fig, 'metrics_vs_level.png', save)


# ── Plot 3: Metric Scaling vs Qubit Count ────────────────────────────────────

def plot_qubit_scaling(save: bool = True):
    """
    2×3 grid: rows = algorithm, cols = metric (gate count, CX count, depth).
    Lines for levels 0, 3, and 5 show how compilation benefit changes with scale.
    """
    metrics = [
        ('gate_count', 'Gate Count'),
        ('cx_count',   'CX Gate Count'),
        ('depth',      'Circuit Depth'),
    ]
    levels_to_plot = [0, 3, 5]

    fig, axes = plt.subplots(len(ALGORITHMS), len(metrics), figsize=(16, 9))
    fig.suptitle(
        'Metric Scaling vs Number of Qubits (2–5)\n'
        'FakeNairobiV2 backend — lines = optimization levels 0, 3, 5',
        fontsize=13, fontweight='bold',
    )

    for row, algo in enumerate(ALGORITHMS):
        for col, (metric, label) in enumerate(metrics):
            ax = axes[row][col]

            for level in levels_to_plot:
                xs, ys = [], []
                for n in QUBIT_COUNTS:
                    data = _load(algo, n)
                    if data and str(level) in data:
                        val = data[str(level)].get(metric)
                        if val is not None:
                            xs.append(n)
                            ys.append(val)

                if xs:
                    ax.plot(xs, ys,
                            color=LEVEL_COLORS[level],
                            marker=LEVEL_MARKERS[level],
                            linewidth=2, markersize=8,
                            label=f'Level {level}', zorder=3)
                    for x, y in zip(xs, ys):
                        ax.annotate(str(y), xy=(x, y), xytext=(4, 4),
                                    textcoords='offset points',
                                    fontsize=7, color=LEVEL_COLORS[level])

            if row == 0:
                ax.set_title(label, fontsize=11, fontweight='bold')
            ax.set_xlabel('Number of Qubits', fontsize=9)
            ax.set_ylabel(label, fontsize=9)
            ax.set_xticks(QUBIT_COUNTS)
            ax.grid(True, alpha=0.25)
            ax.set_axisbelow(True)
            if col == 0:
                ax.text(-0.22, 0.5, ALGO_LABELS[algo],
                        transform=ax.transAxes, fontsize=11, fontweight='bold',
                        va='center', ha='center', rotation=90)
            ax.legend(fontsize=8)

    plt.tight_layout()
    _save_or_show(fig, 'qubit_scaling.png', save)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Generate the three benchmark plots for Grover and QFT.')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively instead of saving to results/')
    parser.add_argument('--n_qubits', type=int, default=4,
                        help='Qubit count for histogram and metrics-vs-level plots (default: 4)')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    save = not args.show

    print('Generating plots…\n')

    print('[1/3] Probability histograms (L0 vs L5)')
    plot_histogram(n_qubits=args.n_qubits, save=save)

    print('[2/3] Metrics vs optimization level')
    plot_metrics_vs_level(n_qubits=args.n_qubits, save=save)

    print('[3/3] Qubit scaling (3–6 qubits)')
    plot_qubit_scaling(save=save)

    print('\nDone.')
