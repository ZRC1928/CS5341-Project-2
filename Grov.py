import time, csv, tracemalloc
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

sim = AerSimulator()
shots = 1024
results = []

for n in range(3, 21):
    target = np.random.randint(0, 2**n)
    target_bits = format(target, f"0{n}b")[::-1]
    iters = max(1, int(np.floor(np.pi / 4 * np.sqrt(2**n))))

    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    for _ in range(iters):
        for i, b in enumerate(target_bits):
            if b == "0": qc.x(i)
        qc.h(n-1)
        qc.mcx(list(range(n-1)), n-1)
        qc.h(n-1)
        for i, b in enumerate(target_bits):
            if b == "0": qc.x(i)

        qc.h(range(n))
        qc.x(range(n))
        qc.h(n-1)
        qc.mcx(list(range(n-1)), n-1)
        qc.h(n-1)
        qc.x(range(n))
        qc.h(range(n))

    qc.measure(range(n), range(n))

    tracemalloc.start()
    t0 = time.time()
    counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
    elapsed = time.time() - t0
    mem = tracemalloc.get_tracemalloc_memory() / 1048576
    tracemalloc.stop()

    top = max(counts, key=counts.get)
    rate = counts[top] / shots * 100
    results.append((n, target, iters, elapsed, mem, rate))
    print(f"n={n:>2}  target={target:>8}  iters={iters:>4}  time={elapsed:>8.3f}s  mem={mem:>8.3f}MB  success={rate:.1f}%")

    if elapsed > 300:
        print("Stopped - exceeded 5 min")
        break

with open("grover_benchmarks.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["qubits","target","iterations","runtime_s","tracemalloc_memory_mb","success_rate"])
    w.writerows(results)

qubits = [r[0] for r in results]
times = [r[3] for r in results]
mems  = [r[4] for r in results]

fig, ax1 = plt.subplots(figsize=(8, 5))

color_time = "#1f77b4"
color_mem  = "#ff7f0e"

ax1.set_xlabel("Qubits")
ax1.set_ylabel("Time (s)", color=color_time)
ax1.plot(qubits, times, "o-", color=color_time, label="Time")
ax1.tick_params(axis="y", labelcolor=color_time)

ax2 = ax1.twinx()
ax2.set_ylabel("Tracemalloc Memory (MB)", color=color_mem)
ax2.plot(qubits, mems, "s--", color=color_mem, label="Memory")
ax2.tick_params(axis="y", labelcolor=color_mem)

fig.suptitle("Grover's Algorithm - Scaling by Qubits")
fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
fig.tight_layout()
plt.show()