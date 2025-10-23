"""
Visualization functions for QPE experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_phase_histogram(counts, true_phase, n_ancilla, save_dir='results'):
    """
    Plot a histogram of measured phases from QPE.

    Converts measurement outcomes to phase values and plots their
    distribution, with the true phase marked.

    Args:
        counts: measurement counts dict from QPE
        true_phase: the true eigenphase for comparison
        n_ancilla: number of ancilla qubits (for phase conversion)
        save_dir: directory to save the plot
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # convert bitstrings to phases
    phases = []
    probs = []
    total_shots = sum(counts.values())

    for bitstring, count in sorted(counts.items()):
        # reverse for qiskit bit ordering
        bits = bitstring[::-1]
        phase = 0.0
        for i, bit in enumerate(bits):
            if bit == '1':
                phase += 1.0 / (2 ** (i + 1))
        phases.append(phase)
        probs.append(count / total_shots)

    # sort by phase
    sorted_pairs = sorted(zip(phases, probs))
    phases, probs = zip(*sorted_pairs)

    # color bars — green for closest to true phase, gray otherwise
    best_phase = min(phases, key=lambda p: min(abs(p - true_phase),
                                                abs(p - true_phase + 1)))
    colors = ['#4CAF50' if p == best_phase else '#90A4AE' for p in phases]

    ax.bar(range(len(phases)), probs, color=colors, alpha=0.85,
           edgecolor='#2C3E50')

    ax.axvline(x=phases.index(best_phase), color='red', linestyle='--',
               linewidth=1.5, alpha=0.7)

    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels([f'{p:.3f}' for p in phases], rotation=45, fontsize=8)
    ax.set_xlabel('Estimated Phase', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'QPE Measurement Distribution (true phase = {true_phase:.4f}, '
                 f'n_ancilla = {n_ancilla})')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path / 'phase_histogram.png', dpi=150)
    plt.close()


def plot_precision_vs_ancilla(results, save_dir='results'):
    """
    Plot phase estimation error vs number of ancilla qubits.

    Shows both the actual error and the theoretical bound (1/2^n).

    Args:
        results: dict with 'n_ancilla', 'errors', 'true_phase'
        save_dir: directory to save the plot
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    n_vals = results['n_ancilla']
    errors = results['errors']

    # actual errors
    ax.semilogy(n_vals, [e + 1e-15 for e in errors], 'o-',
                color='#FF6B6B', linewidth=2, markersize=8,
                label='Actual Error')

    # theoretical bound: 1/2^n
    theoretical = [1.0 / (2 ** n) for n in n_vals]
    ax.semilogy(n_vals, theoretical, 's--', color='#4ECDC4',
                linewidth=1.5, markersize=6, alpha=0.7,
                label='Theoretical Bound ($1/2^n$)')

    ax.set_xlabel('Number of Ancilla Qubits', fontsize=12)
    ax.set_ylabel('Phase Estimation Error (log scale)', fontsize=12)
    ax.set_title(f'QPE Precision vs Ancilla Count '
                 f'(true phase = {results["true_phase"]:.4f})')
    ax.legend(fontsize=11)
    ax.set_xticks(n_vals)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'precision_vs_ancilla.png', dpi=150)
    plt.close()


def plot_iterative_convergence(phase_estimates, true_phase, save_dir='results'):
    """
    Plot how the iterative QPE phase estimate converges bit by bit.

    Args:
        phase_estimates: list of running phase estimates after each iteration
        true_phase: the true eigenphase
        save_dir: directory to save the plot
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    iterations = range(1, len(phase_estimates) + 1)

    # phase estimate convergence
    ax = axes[0]
    ax.plot(iterations, phase_estimates, 'o-', color='#4ECDC4',
            linewidth=2, markersize=8)
    ax.axhline(y=true_phase, color='red', linestyle='--',
               linewidth=1.5, label=f'True phase ({true_phase:.4f})')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Estimated Phase', fontsize=12)
    ax.set_title('Iterative QPE: Phase Convergence')
    ax.legend(fontsize=11)
    ax.set_xticks(list(iterations))
    ax.grid(True, alpha=0.3)

    # error convergence
    ax = axes[1]
    errors = [min(abs(est - true_phase),
                  abs(est - true_phase + 1),
                  abs(est - true_phase - 1))
              for est in phase_estimates]
    ax.semilogy(iterations, [e + 1e-15 for e in errors], 's-',
                color='#FF6B6B', linewidth=2, markersize=8)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Phase Error (log scale)', fontsize=12)
    ax.set_title('Iterative QPE: Error Convergence')
    ax.set_xticks(list(iterations))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / 'iterative_convergence.png', dpi=150)
    plt.close()


def plot_method_comparison(textbook_results, iterative_results, save_dir='results'):
    """
    Side-by-side comparison of textbook vs iterative QPE.

    Args:
        textbook_results: dict with 'n_ancilla' and 'errors'
        iterative_results: dict with 'n_iterations' and 'errors'
        save_dir: directory to save the plot
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # textbook QPE
    ax.semilogy(textbook_results['n_ancilla'],
                [e + 1e-15 for e in textbook_results['errors']],
                'o-', color='#FF6B6B', linewidth=2, markersize=8,
                label='Textbook QPE')

    # iterative QPE
    ax.semilogy(iterative_results['n_iterations'],
                [e + 1e-15 for e in iterative_results['errors']],
                's-', color='#4ECDC4', linewidth=2, markersize=8,
                label='Iterative QPE')

    ax.set_xlabel('Precision Bits (n)', fontsize=12)
    ax.set_ylabel('Phase Estimation Error (log scale)', fontsize=12)
    ax.set_title('Textbook vs Iterative QPE: Error Comparison')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'method_comparison.png', dpi=150)
    plt.close()
