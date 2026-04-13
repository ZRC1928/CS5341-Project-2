import time, tracemalloc, numpy as np
import matplotlib.pyplot as plt
from math import gcd
from fractions import Fraction
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFTGate

def c_amod15(a, power):
    U = QuantumCircuit(8)
    for _ in range(power):
        if a in [2, 13]: U.swap(2,3); U.swap(1,2); U.swap(0,1)
        if a in [7, 8]:  U.swap(0,1); U.swap(1,2); U.swap(2,3)
        if a in [4, 11]: U.swap(1,3); U.swap(0,2)
        if a in [7, 11, 13]:
            for q in range(4): U.x(q)
    return U.to_gate().control()

def run_shor(N, n_count, shots=2048):
    for a in [x for x in range(2, N) if gcd(x, N) == 1]:
        tracemalloc.start()
        t0 = time.time()
        qc = QuantumCircuit(n_count + 4, n_count)
        qc.x(n_count)
        
        
        
        for q in range(n_count): qc.h(q)
        
        for q in range(n_count):
            qc.append(c_amod15(a, 2**q), [q] + list(range(n_count, n_count+4)))
        
        qc.append(QFTGate(n_count).inverse(), range(n_count))
        qc.measure(range(n_count), range(n_count))
        result = AerSimulator().run(qc, shots=shots).result()
        counts = result.get_counts()
        rt = time.time() - t0
        mem = tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        
        
        
        for measured in sorted(counts, key=counts.get, reverse=True):
            phase = int(measured, 2) / (2**n_count)
            if phase == 0: continue
            r = Fraction(phase).limit_denominator(N).denominator
            if r % 2 == 0:
                g1, g2 = gcd(a**(r//2)-1, N), gcd(a**(r//2)+1, N)
                if g1 not in [1,N]: return (g1, N//g1), rt, mem
                if g2 not in [1,N]: return (g2, N//g2), rt, mem
    
    return None, None, None

print(f"{'Count Q':<10}{'Total Q':<10}{'Time (s)':<12}{'Mem (MB)':<12}{'Factors'}")
print("-"*56)

counts, times, mems = [], [], []
for nc in range(3, 10):
    factors, rt, mem = run_shor(15, nc)
    if factors:
        print(f"{nc:<10}{nc+4:<10}{rt:<12.4f}{mem/1048576:<12.4f}{factors}")
        counts.append(nc)
        times.append(rt)
        mems.append(mem / 1048576)
    else:
        print(f"{nc:<10}{nc+4:<10}{'FAIL':<12}{'N/A':<12}N/A")

# --- Graph ---
fig, ax1 = plt.subplots(figsize=(8, 5))

color_time = "#1f77b4"
color_mem = "#ff7f0e"

ax1.set_xlabel("Counting Qubits")
ax1.set_ylabel("Time (s)", color=color_time)
ax1.plot(counts, times, "o-", color=color_time, label="Time")
ax1.tick_params(axis="y", labelcolor=color_time)

ax2 = ax1.twinx()
ax2.set_ylabel("Tracemalloc Memory (MB)", color=color_mem)
ax2.plot(counts, mems, "s--", color=color_mem, label="Memory")
ax2.tick_params(axis="y", labelcolor=color_mem)

fig.suptitle("Shor's Algorithm   Scaling by Counting Qubits")
fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
fig.tight_layout()
plt.show()