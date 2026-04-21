"""
Graph 1 — Probability Histograms: Bad (Level 0) vs Good (Level 3) compilation.

Side-by-side bar charts of output state probabilities under noisy simulation.
Bad = Level 0 (flat/spread distribution), Good = Level 3 (sharp peak at answer).

Usage:
    python plots/histogram.py --algorithm grover --n_qubits 4
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def _load_probs(algorithm: str, n_qubits: int, level: int) -> tuple:
    path = os.path.join(RESULTS_DIR, f'{algorithm}_{n_qubits}q_metrics.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No metrics found at {path}. Run transpile_all.py first.")
    with open(path) as f:
        data = json.load(f)
    entry = data[str(level)]
    return entry['ideal_probs'], entry['noisy_probs']


def plot_histogram(algorithm: str, n_qubits: int, save: bool = True):
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(
        f'{algorithm.upper()} ({n_qubits}q) — Probability Distribution Under Noise',
        fontsize=13, fontweight='bold'
    )

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    for col, (level, label) in enumerate([(0, 'Level 0 (No Optimisation)'), (3, 'Level 3 (Full Optimisation)')]):
        ideal_probs, noisy_probs = _load_probs(algorithm, n_qubits, level)

        # Show at most 16 most probable ideal states to keep plots readable
        top_states = sorted(ideal_probs, key=ideal_probs.get, reverse=True)[:16]
        ideal_vals = [ideal_probs.get(s, 0) for s in top_states]
        noisy_vals = [noisy_probs.get(s, 0) for s in top_states]

        x = np.arange(len(top_states))
        w = 0.38

        ax = fig.add_subplot(gs[col])
        ax.bar(x - w / 2, ideal_vals, w, label='Ideal', color='steelblue', alpha=0.85)
        ax.bar(x + w / 2, noisy_vals, w, label='Noisy', color='tomato', alpha=0.85)

        ax.set_title(label, fontsize=11)
        ax.set_xlabel('Basis state')
        ax.set_ylabel('Probability')
        ax.set_xticks(x)
        ax.set_xticklabels(top_states, rotation=45, ha='right', fontsize=7)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)

    out_path = os.path.join(RESULTS_DIR, f'{algorithm}_{n_qubits}q_histogram.png')
    if save:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved {out_path}")
    else:
        plt.show()
    plt.close(fig)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', required=True)
    parser.add_argument('--n_qubits', type=int, required=True)
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    plot_histogram(args.algorithm, args.n_qubits, save=not args.show)
