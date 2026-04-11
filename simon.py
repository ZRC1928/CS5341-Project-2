import matplotlib.pyplot as plt
from qiskit import *
from qiskit.circuit.library import *
from qiskit_aer import AerSimulator
import numpy as np
import time
import tracemalloc

def modN(n):
    input = QuantumRegister(n, name="in")
    output = QuantumRegister(n, name="out")
    helper = QuantumRegister(1, name="help")

    circuit = QuantumCircuit(input, output, helper)
    circuit.cx(0, (2*n))
    circuit.cx((2*n), 0)
    circuit.barrier()
    for i in range(0, n):
        circuit.cx(i, (n+i))

    return circuit

def simon(n, function_circ):
    input = QuantumRegister(n, name="in")
    output = QuantumRegister(n, name="out")
    helper = QuantumRegister(1, name="help")
    functionMeasure = ClassicalRegister(n, "f(x)")
    sampleMeasure = ClassicalRegister(n, "y")

    simon_circ = QuantumCircuit(input, output, helper, functionMeasure, sampleMeasure)
    for i in range(0, n):
        simon_circ.h(i)
    simon_circ.barrier()

    qubitList = []
    for i in range(0, (2*n)+1):
        qubitList.append(i)

    simon_circ.compose(other=function_circ, qubits=qubitList, clbits=[], inplace=True)
    simon_circ.barrier()

    simon_circ.reset((2*n))
    for i in range(n, (2*n)):
        simon_circ.measure(i, (i-n))
    simon_circ.barrier()

    for i in range(0, n):
        simon_circ.h(i)
    simon_circ.barrier()

    for i in range(0, n):
        simon_circ.measure(i, (i+n))
    
    return simon_circ


def main():

    startTime = time.time()
    tracemalloc.start()

    print("Number of qubits: ")
    n = input()

    function_circuit = modN(int(n))
    circuit = simon(int(n), function_circuit)

    sim = AerSimulator()
    sim_result = sim.run(circuit, shots=1, memory=True).result().get_memory()[0]
    firstSample = sim_result.split(" ")[0]

    # print(sim_result)

    tempMatrix = []
    for bit in firstSample:
        tempMatrix.append(int(bit))
    output = [tempMatrix]
    del tempMatrix

    # print(output)

    samplingCount = 1
    independentSamples = 1
    while independentSamples < (int(n)-1):
        sim_result = sim.run(circuit, shots=1, memory=True).result().get_memory()[0]
        samplingCount = samplingCount + 1
        sample = sim_result.split(" ")[0]
        vectorArr = []
        for bit in sample:
            vectorArr.append(int(bit))

        testMatrix = output.copy()
        testMatrix.append(vectorArr)

        print("Sample: " + sample)

        numpyTestMatrix = np.array(testMatrix)
        if np.linalg.matrix_rank(testMatrix) == len(testMatrix):
            print("Sample added!")
            output.append(vectorArr)
            independentSamples = independentSamples + 1
        del vectorArr
        del testMatrix
        del numpyTestMatrix

    endTime = time.time()
    totalTime = endTime - startTime
    totalSpace = tracemalloc.get_tracemalloc_memory()

    tracemalloc.stop()

    print(output)
    print(samplingCount)
    print(totalTime)
    print(totalSpace)

    return output, samplingCount, totalTime, totalSpace

    # circuit.draw(output="mpl")
    # plt.tight_layout()
    # plt.show()

main()