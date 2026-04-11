import time, tracemalloc, numpy as np
import matplotlib.pyplot as plt
from math import gcd, ceil, log2
from fractions import Fraction
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFTGate

N             = 15          # number to factor
MIN_COUNT_Q   = 3           # smallest counting-qubit count to try
MAX_COUNT_Q   = 23           # largest  counting-qubit count to try
SHOTS         = 2048        # simulator shots per run

N_WORK = ceil(log2(N + 1))

def c_amod(a, power, N, n_work):
    """Build a controlled-U gate for |x> -> |a^power * x mod N>
       by constructing the full controlled unitary matrix directly."""
    dim = 2 ** n_work
    # permutation matrix for a^power mod N
    perm = np.eye(dim)
    for x in range(N):
        perm[x, x] = 0
        perm[(pow(a, power, N) * x) % N, x] = 1

    # controlled version: |0><0| ⊗ I  +  |1><1| ⊗ U   (control is qubit 0)
    cdim = 2 * dim
    cu = np.eye(cdim, dtype=complex)
    cu[dim:, dim:] = perm

    gate = QuantumCircuit(n_work + 1)
    gate.unitary(cu, range(n_work + 1))
    return gate.to_gate()

def run_shor(N, n_count, n_work, shots=2048):
    for a in [x for x in range(2, N) if gcd(x, N) == 1]:
        tracemalloc.start()
        t0 = time.perf_counter()

        total_q = n_count + n_work
        qc = QuantumCircuit(total_q, n_count)
        qc.x(n_count)                              # work register = |1>

        for q in range(n_count):
            qc.h(q)

        for q in range(n_count):
            qc.append(c_amod(a, 2**q, N, n_work),
                       list(range(n_count, total_q)) + [q])

        qc.append(QFTGate(n_count).inverse(), range(n_count))
        qc.measure(range(n_count), range(n_count))

        sim    = AerSimulator()
        result = sim.run(transpile(qc, sim), shots=shots).result()
        counts = result.get_counts()
        rt     = time.perf_counter() - t0
        _, mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        for measured in sorted(counts, key=counts.get, reverse=True):
            phase = int(measured, 2) / (2**n_count)
            if phase == 0:
                continue
            r = Fraction(phase).limit_denominator(N).denominator
            if r % 2 == 0:
                g1, g2 = gcd(a**(r//2) - 1, N), gcd(a**(r//2) + 1, N)
                if g1 not in [1, N]: return (g1, N//g1), rt, mem
                if g2 not in [1, N]: return (g2, N//g2), rt, mem

    return None, None, None

# ── Benchmarking ──────────────────────────────────────────────
qubit_counts = []
times        = []
memories     = []

print(f"Factoring N = {N}  |  work qubits = {N_WORK}")
print(f"{'Count Q':<10}{'Total Q':<10}{'Time (s)':<12}{'Mem (MB)':<12}{'Factors'}")
print("-" * 56)

for nc in range(MIN_COUNT_Q, MAX_COUNT_Q + 1):
    factors, rt, mem = run_shor(N, nc, N_WORK, SHOTS)
    total_q = nc + N_WORK
    if factors:
        mem_mb = mem / 1048576
        qubit_counts.append(total_q)
        times.append(rt)
        memories.append(mem_mb)
        print(f"{nc:<10}{total_q:<10}{rt:<12.4f}{mem_mb:<12.4f}{factors}")
    else:
        print(f"{nc:<10}{total_q:<10}{'FAIL':<12}{'N/A':<12}N/A")

# ── Plotting ──────────────────────────────────────────────────
if qubit_counts:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(qubit_counts, times, 'o-', color='tab:blue')
    ax1.set_xlabel('Total Qubits')
    ax1.set_ylabel('Run Time (s)')
    ax1.set_title(f"Shor's Algorithm (N={N}) – Run Time vs Qubits")
    ax1.grid(True)

    ax2.plot(qubit_counts, memories, 's-', color='tab:red')
    ax2.set_xlabel('Total Qubits')
    ax2.set_ylabel('Peak Memory (MB)')
    ax2.set_title(f"Shor's Algorithm (N={N}) – Memory vs Qubits")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('shor_benchmarks.png', dpi=150)
    plt.show()
    print("\nGraph saved to shor_benchmarks.png")
else:
    print("\nNo successful runs to plot.")