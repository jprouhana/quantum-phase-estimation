"""
Unitary operators and eigenstate preparation circuits for QPE experiments.

Provides standard unitaries with known eigenphases for testing and
benchmarking phase estimation algorithms.
"""

import numpy as np
from qiskit import QuantumCircuit


def t_gate_unitary():
    """
    Return the T gate unitary matrix.

    T = diag(1, e^{i*pi/4})

    Eigenvalues: 1 and e^{i*pi/4}
    Eigenphases: 0 and 1/8

    The T gate is the standard test case for QPE because its phase
    (1/8) is exactly representable in 3 binary digits.

    Returns:
        2x2 unitary matrix (numpy array)
    """
    return np.array([[1, 0],
                     [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def phase_gate(theta):
    """
    Return the phase gate P(theta) unitary matrix.

    P(theta) = diag(1, e^{i*theta})

    Eigenvalues: 1 and e^{i*theta}
    Eigenphases: 0 and theta / (2*pi)

    Args:
        theta: phase angle in radians

    Returns:
        2x2 unitary matrix (numpy array)
    """
    return np.array([[1, 0],
                     [0, np.exp(1j * theta)]], dtype=complex)


def custom_unitary(theta):
    """
    Build a general single-qubit unitary with a specified eigenphase.

    Creates a diagonal unitary with eigenphase theta/(2*pi).
    This is equivalent to phase_gate but parameterized differently
    for convenience in QPE experiments.

    Args:
        theta: desired eigenphase (the phase will be 2*pi*theta)

    Returns:
        2x2 unitary matrix (numpy array)
    """
    phase = 2 * np.pi * theta
    return np.array([[1, 0],
                     [0, np.exp(1j * phase)]], dtype=complex)


def eigenstate_circuit_zero():
    """
    Prepare the |0> eigenstate.

    |0> is an eigenstate of any diagonal unitary with eigenvalue 1
    (eigenphase 0). This is the trivial eigenstate.

    Returns:
        QuantumCircuit that prepares |0>
    """
    qc = QuantumCircuit(1, name='|0>')
    # |0> is the default — nothing to do
    return qc


def eigenstate_circuit_one():
    """
    Prepare the |1> eigenstate.

    |1> is an eigenstate of any diagonal unitary with eigenvalue e^{i*theta}.
    For the T gate, this gives eigenphase 1/8.

    Returns:
        QuantumCircuit that prepares |1>
    """
    qc = QuantumCircuit(1, name='|1>')
    qc.x(0)
    return qc


def get_true_phase(unitary, eigenstate='|1>'):
    """
    Compute the true eigenphase for a given unitary and eigenstate.

    For diagonal unitaries:
    - |0> eigenstate: phase = arg(U[0,0]) / (2*pi)
    - |1> eigenstate: phase = arg(U[1,1]) / (2*pi)

    Args:
        unitary: 2x2 unitary matrix
        eigenstate: '|0>' or '|1>'

    Returns:
        eigenphase in [0, 1)
    """
    if eigenstate == '|0>':
        eigenvalue = unitary[0, 0]
    elif eigenstate == '|1>':
        eigenvalue = unitary[1, 1]
    else:
        raise ValueError(f"Unknown eigenstate: {eigenstate}")

    phase = np.angle(eigenvalue) / (2 * np.pi)
    # normalize to [0, 1)
    phase = phase % 1.0
    return float(phase)


def matrix_power(U, power):
    """
    Compute U^power for a unitary matrix.

    For diagonal unitaries this is exact. For general unitaries,
    uses eigendecomposition.

    Args:
        U: unitary matrix
        power: integer power

    Returns:
        U^power as a numpy array
    """
    eigenvalues, eigenvectors = np.linalg.eig(U)
    powered_eigenvalues = eigenvalues ** power
    return eigenvectors @ np.diag(powered_eigenvalues) @ np.linalg.inv(eigenvectors)
