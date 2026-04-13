import simon
import matplotlib.pyplot as plt

def main():
    
    n = 3
    samplingCounts = []
    totalTimes = []
    totalSpaces = []
    nVals = []
    while n <= 150:
        output, samplingCount, totalTime, totalSpace = simon.simonAlg(n)
        nVals.append(n)
        samplingCounts.append(samplingCount)
        totalTimes.append(totalTime)
        totalSpaces.append((totalSpace/1000))

        if totalTime > 300:
            break

        n = n + 1
    
    plt.plot(nVals, samplingCounts)
    plt.title("Sampling Counts over Qubits")
    plt.xlabel("Input Qubits")
    plt.ylabel("Sampling Count")
    plt.tight_layout()
    plt.show()

    plt.plot(nVals, totalTimes)
    plt.title("Runtimes over Qubits")
    plt.xlabel("Input Qubits")
    plt.ylabel("Time (Seconds)")
    plt.tight_layout()
    plt.show()

    plt.plot(nVals, totalSpaces)
    plt.title("Memory Usage over Qubits")
    plt.xlabel("Input Qubits")
    plt.ylabel("Memory Usage (KB)")
    plt.tight_layout()
    plt.show()

main()