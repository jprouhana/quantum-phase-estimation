"""
Iterative Quantum Phase Estimation using a single ancilla qubit
with classical feedback.

References:
    Kitaev (1995) - "Quantum measurements and the Abelian stabilizer problem"
    Dobsicek et al. (2007) - "Arbitrary accuracy iterative quantum phase estimation"
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from .unitaries import eigenstate_circuit_one, get_true_phase, matrix_power


def _build_iteration_circuit(unitary, k, phase_correction, eigenstate_circuit=None):
    """
    Build the circuit for one iteration of iterative QPE.

    Each iteration estimates one bit of the phase:
    1. Prepare ancilla in |+>
    2. Apply controlled-U^(2^k) to the eigenstate
    3. Apply a phase correction based on previously estimated bits
    4. Apply Hadamard and measure the ancilla

    Args:
        unitary: 2x2 unitary matrix
        k: iteration index (estimates bit k of the phase)
        phase_correction: accumulated phase from previous iterations
        eigenstate_circuit: circuit to prepare eigenstate

    Returns:
        QuantumCircuit for this iteration
    """
    if eigenstate_circuit is None:
        eigenstate_circuit = eigenstate_circuit_one()

    n_eigenstate = eigenstate_circuit.num_qubits

    # qubit 0 is the ancilla, rest is eigenstate
    qc = QuantumCircuit(1 + n_eigenstate, 1)

    # prepare eigenstate
    qc.compose(eigenstate_circuit, qubits=range(1, 1 + n_eigenstate),
               inplace=True)

    # put ancilla in |+>
    qc.h(0)

    # controlled-U^(2^k)
    power = 2 ** k
    U_powered = matrix_power(unitary, power)
    gate = QuantumCircuit(1, name=f'U^{power}')
    gate.unitary(U_powered, 0)
    controlled_gate = gate.to_gate().control(1)
    qc.append(controlled_gate, [0, 1])

    # phase correction from previously estimated bits
    if phase_correction != 0:
        qc.p(-phase_correction, 0)

    # hadamard + measure
    qc.h(0)
    qc.measure(0, 0)

    return qc


def iterative_phase_estimation(unitary, n_iterations, shots=4096, seed=42,
                                eigenstate_circuit=None):
    """
    Run iterative QPE using a single ancilla qubit.

    Estimates the phase one bit at a time, from the least significant
    bit to the most significant. Each iteration uses the results of
    previous iterations for phase correction.

    The key advantage: only 1 ancilla qubit is needed, regardless
    of the desired precision.

    Args:
        unitary: 2x2 unitary matrix
        n_iterations: number of iterations (= number of precision bits)
        shots: measurement shots per iteration
        seed: random seed for simulator
        eigenstate_circuit: optional eigenstate prep circuit

    Returns:
        dict with 'estimated_phase', 'true_phase', 'phase_error',
        'bit_estimates', 'phase_estimates', 'confidence'
    """
    backend = AerSimulator()
    true_phase = get_true_phase(unitary, eigenstate='|1>')

    bit_estimates = []
    phase_estimates = []  # running phase estimate after each iteration

    for k in range(n_iterations - 1, -1, -1):
        # compute phase correction from previously estimated bits
        phase_correction = 0.0
        for j, bit in enumerate(bit_estimates):
            # the j-th estimated bit corresponds to a specific phase contribution
            if bit == 1:
                phase_correction += 2 * np.pi / (2 ** (j + 1))

        # run one iteration
        qc = _build_iteration_circuit(unitary, k, phase_correction,
                                       eigenstate_circuit)
        job = backend.run(qc, shots=shots, seed_simulator=seed)
        result = job.result()
        counts = result.get_counts()

        # estimate this bit from majority vote
        count_0 = counts.get('0', 0)
        count_1 = counts.get('1', 0)
        bit = 0 if count_0 > count_1 else 1
        bit_estimates.append(bit)

        # compute running phase estimate
        current_phase = 0.0
        for j, b in enumerate(bit_estimates):
            current_phase += b / (2 ** (j + 1))
        phase_estimates.append(current_phase)

    # final phase estimate from all bits
    estimated_phase = 0.0
    for j, bit in enumerate(bit_estimates):
        estimated_phase += bit / (2 ** (j + 1))

    # confidence from the last iteration's measurement statistics
    last_counts = counts
    total = sum(last_counts.values())
    majority_count = max(last_counts.values())
    confidence = majority_count / total

    # phase error
    error = min(abs(estimated_phase - true_phase),
                abs(estimated_phase - true_phase + 1),
                abs(estimated_phase - true_phase - 1))

    return {
        'estimated_phase': estimated_phase,
        'true_phase': true_phase,
        'phase_error': error,
        'bit_estimates': bit_estimates,
        'phase_estimates': phase_estimates,
        'confidence': confidence,
        'n_iterations': n_iterations,
    }


def iterative_precision_sweep(unitary, iteration_range, shots=4096, seed=42):
    """
    Run iterative QPE with different numbers of iterations.

    Args:
        unitary: 2x2 unitary matrix
        iteration_range: iterable of iteration counts
        shots: shots per iteration
        seed: random seed

    Returns:
        dict with 'n_iterations', 'estimated_phases', 'errors',
        'confidences', 'true_phase'
    """
    true_phase = get_true_phase(unitary, eigenstate='|1>')

    estimated_phases = []
    errors = []
    confidences = []

    for n in iteration_range:
        result = iterative_phase_estimation(unitary, n, shots=shots, seed=seed)
        estimated_phases.append(result['estimated_phase'])
        errors.append(result['phase_error'])
        confidences.append(result['confidence'])

        print(f"n_iter={n}: estimated={result['estimated_phase']:.6f}, "
              f"error={result['phase_error']:.6f}, "
              f"confidence={result['confidence']:.3f}")

    return {
        'n_iterations': list(iteration_range),
        'estimated_phases': estimated_phases,
        'errors': errors,
        'confidences': confidences,
        'true_phase': true_phase,
    }
# Based on Kitaev's iterative approach
