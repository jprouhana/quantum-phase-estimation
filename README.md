# Quantum Phase Estimation

Implementation of textbook and iterative Quantum Phase Estimation (QPE) algorithms. Estimates eigenphases of unitary operators with controllable precision, and compares the two approaches in terms of ancilla requirements and accuracy. Built as part of independent study work on fundamental quantum algorithms.

## Background

### The Phase Estimation Problem

Given a unitary operator $U$ and one of its eigenstates $|\psi\rangle$ such that $U|\psi\rangle = e^{2\pi i \phi}|\psi\rangle$, the goal is to estimate the eigenphase $\phi \in [0, 1)$.

Phase estimation is one of the most important subroutines in quantum computing — it's the core of Shor's factoring algorithm, quantum chemistry simulations, and many other quantum algorithms.

### Textbook QPE

The standard QPE algorithm uses:
1. $n$ ancilla qubits initialized in $|+\rangle$
2. Controlled-$U^{2^k}$ operations for $k = 0, 1, \ldots, n-1$
3. Inverse Quantum Fourier Transform on the ancilla register
4. Measurement to read out a binary approximation to $\phi$

The precision scales as $1/2^n$ — each additional ancilla qubit doubles the precision.

### Iterative QPE

The iterative version (Kitaev's approach) uses only a **single ancilla qubit**, estimating $\phi$ one bit at a time from least significant to most significant. Each iteration:
1. Prepare the ancilla in $|+\rangle$
2. Apply a controlled-$U^{2^k}$ with a phase correction based on previously estimated bits
3. Measure the ancilla to extract one bit of $\phi$

This trades ancilla qubits for classical feedback — very useful on near-term hardware.

## Project Structure

```
quantum-phase-estimation/
├── src/
│   ├── textbook_qpe.py        # Standard QPE with QFT
│   ├── iterative_qpe.py       # Single-ancilla iterative QPE
│   ├── unitaries.py           # Unitary operators and eigenstates
│   └── plotting.py            # Visualization functions
├── notebooks/
│   └── qpe_experiments.ipynb  # Full analysis walkthrough
├── results/
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

```bash
git clone https://github.com/jrouhana/quantum-phase-estimation.git
cd quantum-phase-estimation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from src.textbook_qpe import run_qpe
from src.unitaries import t_gate_unitary

# estimate the T gate phase (should be 1/8 = 0.125)
U = t_gate_unitary()
result = run_qpe(U, n_ancilla=4, shots=4096, seed=42)

print(f"Estimated phase: {result['estimated_phase']:.4f}")
print(f"True phase: 0.125")
print(f"Error: {result['phase_error']:.6f}")
```

### Jupyter Notebook

```bash
jupyter notebook notebooks/qpe_experiments.ipynb
```

## Results

### Precision vs Ancilla Count

Estimating the T gate phase ($\phi = 1/8$) with increasing ancilla qubits:

| Ancilla Qubits | Estimated Phase | Absolute Error | Binary Resolution |
|---------------|----------------|----------------|-------------------|
| 2             | 0.000          | 0.1250         | 1/4               |
| 3             | 0.125          | 0.0000         | 1/8               |
| 4             | 0.125          | 0.0000         | 1/16              |
| 5             | 0.125          | 0.0000         | 1/32              |
| 6             | 0.125          | 0.0000         | 1/64              |

*T gate phase is exactly representable in 3 bits, so QPE nails it at n=3.*

### Key Findings

- QPE achieves exact results when the phase is exactly representable in $n$ bits
- For non-dyadic phases, the error decreases as $O(1/2^n)$ with additional ancillas
- Iterative QPE matches textbook QPE precision with only 1 ancilla qubit
- The textbook version is more robust to shot noise since all bits are estimated simultaneously
- Iterative QPE is better suited for near-term hardware with limited qubit count

## References

1. Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*. Cambridge University Press. Section 5.2.
2. Kitaev, A. Y. (1995). "Quantum measurements and the Abelian stabilizer problem." [arXiv:quant-ph/9511026](https://arxiv.org/abs/quant-ph/9511026)
3. Dobsicek, M., et al. (2007). "Arbitrary accuracy iterative quantum phase estimation algorithm." *Physical Review A*, 76(3), 030306.

## License

MIT License — see [LICENSE](LICENSE) for details.
