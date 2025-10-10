"""
Textbook Quantum Phase Estimation using the standard circuit with
inverse QFT on the ancilla register.

References:
    Nielsen & Chuang (2010) - Section 5.2
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator

from .unitaries import eigenstate_circuit_one, get_true_phase, matrix_power


def _controlled_unitary(qc, control, target, unitary, power):
    """
    Apply controlled-U^power to the circuit.

    For a 2x2 unitary, this creates a controlled version of U^power
    using Qiskit's unitary gate.

    Args:
        qc: QuantumCircuit to add the gate to
        control: control qubit index
        target: target qubit index
        unitary: 2x2 unitary matrix
        power: integer power (U is applied this many times)
    """
    U_powered = matrix_power(unitary, power)
    gate = QuantumCircuit(1, name=f'U^{power}')
    gate.unitary(U_powered, 0)
    controlled_gate = gate.to_gate().control(1)
    qc.append(controlled_gate, [control, target])


def build_qpe_circuit(unitary, n_ancilla, eigenstate_circuit=None):
    """
    Build the standard QPE circuit.

    The circuit consists of:
    1. Hadamard on all ancilla qubits
    2. Controlled-U^(2^k) from ancilla k to the eigenstate register
    3. Inverse QFT on the ancilla register
    4. Measurement of ancilla qubits

    Args:
        unitary: the unitary operator (2x2 numpy array)
        n_ancilla: number of ancilla qubits (determines precision)
        eigenstate_circuit: circuit to prepare the eigenstate (defaults to |1>)

    Returns:
        QuantumCircuit for QPE
    """
    if eigenstate_circuit is None:
        eigenstate_circuit = eigenstate_circuit_one()

    n_eigenstate = eigenstate_circuit.num_qubits

    # total qubits: ancilla + eigenstate
    qc = QuantumCircuit(n_ancilla + n_eigenstate, n_ancilla)

    # prepare eigenstate
    qc.compose(eigenstate_circuit, qubits=range(n_ancilla, n_ancilla + n_eigenstate),
               inplace=True)

    # hadamard on ancilla qubits
    qc.h(range(n_ancilla))

    # controlled unitaries
    for k in range(n_ancilla):
        power = 2 ** k
        _controlled_unitary(qc, k, n_ancilla, unitary, power)

    # inverse QFT on ancilla register
    qft_inv = QFT(n_ancilla, inverse=True)
    qc.compose(qft_inv, qubits=range(n_ancilla), inplace=True)

    # measure ancilla qubits
    qc.measure(range(n_ancilla), range(n_ancilla))

    return qc


def counts_to_phase(counts, n_ancilla):
    """
    Convert measurement counts to a phase estimate.

    Takes the most frequent measurement outcome and interprets it
    as a binary fraction: if we measure b_{n-1}...b_1 b_0, the
    phase estimate is 0.b_0 b_1 ... b_{n-1} in binary.

    Args:
        counts: measurement counts dict
        n_ancilla: number of ancilla qubits

    Returns:
        estimated phase (float in [0, 1))
    """
    # most frequent outcome
    best = max(counts, key=counts.get)

    # qiskit returns bits in reverse order — reverse the string
    bits = best[::-1]

    # convert binary fraction to decimal
    phase = 0.0
    for i, bit in enumerate(bits):
        if bit == '1':
            phase += 1.0 / (2 ** (i + 1))

    return phase


def run_qpe(unitary, n_ancilla, shots=4096, seed=42, eigenstate_circuit=None):
    """
    Run QPE and return the estimated phase.

    Args:
        unitary: 2x2 unitary matrix
        n_ancilla: number of ancilla qubits
        shots: number of measurement shots
        seed: random seed for simulator
        eigenstate_circuit: optional eigenstate prep circuit

    Returns:
        dict with 'estimated_phase', 'true_phase', 'phase_error',
        'counts', 'n_ancilla', 'circuit'
    """
    qc = build_qpe_circuit(unitary, n_ancilla, eigenstate_circuit)

    backend = AerSimulator()
    job = backend.run(qc, shots=shots, seed_simulator=seed)
    result = job.result()
    counts = result.get_counts()

    estimated_phase = counts_to_phase(counts, n_ancilla)
    true_phase = get_true_phase(unitary, eigenstate='|1>')

    # handle wraparound for phase error
    error = min(abs(estimated_phase - true_phase),
                abs(estimated_phase - true_phase + 1),
                abs(estimated_phase - true_phase - 1))

    return {
        'estimated_phase': estimated_phase,
        'true_phase': true_phase,
        'phase_error': error,
        'counts': counts,
        'n_ancilla': n_ancilla,
        'circuit': qc,
    }


def precision_sweep(unitary, ancilla_range, shots=4096, seed=42):
    """
    Run QPE with different numbers of ancilla qubits to study precision.

    Args:
        unitary: 2x2 unitary matrix
        ancilla_range: iterable of ancilla counts to try
        shots: shots per run
        seed: random seed

    Returns:
        dict with 'n_ancilla', 'estimated_phases', 'errors', 'true_phase'
    """
    true_phase = get_true_phase(unitary, eigenstate='|1>')

    estimated_phases = []
    errors = []

    for n in ancilla_range:
        result = run_qpe(unitary, n, shots=shots, seed=seed)
        estimated_phases.append(result['estimated_phase'])
        errors.append(result['phase_error'])

        print(f"n_ancilla={n}: estimated={result['estimated_phase']:.6f}, "
              f"error={result['phase_error']:.6f}")

    return {
        'n_ancilla': list(ancilla_range),
        'estimated_phases': estimated_phases,
        'errors': errors,
        'true_phase': true_phase,
    }
