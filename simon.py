import matplotlib.pyplot as plt
from qiskit import *
from qiskit.circuit.library import *
from qiskit_aer import AerSimulator
import numpy as np
import time
import tracemalloc
import sys

def modN(n):
    #create n qubit inputs, n qubit outputs, and a single helper qubit
    input = QuantumRegister(n, name="in")
    output = QuantumRegister(n, name="out")
    helper = QuantumRegister(1, name="help")

    #create the quantum circuit that mods the input by the most significant bit
    circuit = QuantumCircuit(input, output, helper)
    #use a cNot to flip the helper if the most significant bit is 1, then flip that 1 into a 0 using the helper
    circuit.cx(0, (2*n))
    circuit.cx((2*n), 0)
    #create a barrier
    circuit.barrier()
    #use cNots to flip all output values whose corresponding input value is a 1
    for i in range(0, n):
        circuit.cx(i, (n+i))

    return circuit

def simon(n, function_circ):
    #instantiate the registers to use in the circuit
    input = QuantumRegister(n, name="in")
    output = QuantumRegister(n, name="out")
    helper = QuantumRegister(1, name="help")
    functionMeasure = ClassicalRegister(n, "f(x)")
    sampleMeasure = ClassicalRegister(n, "y")

    #create the circuit and apply a Hadamard to each of the input qubits
    simon_circ = QuantumCircuit(input, output, helper, functionMeasure, sampleMeasure)
    for i in range(0, n):
        simon_circ.h(i)
    simon_circ.barrier()

    #get a list of every qubit in the circuit
    qubitList = []
    for i in range(0, (2*n)+1):
        qubitList.append(i)

    #combine the circuit with the circuit for modN
    simon_circ.compose(other=function_circ, qubits=qubitList, clbits=[], inplace=True)
    simon_circ.barrier()

    #reset the helper qubit to get it back to |0>, then measure all the output qubits to get |f(x)>
    simon_circ.reset((2*n))
    for i in range(n, (2*n)):
        simon_circ.measure(i, (i-n))
    simon_circ.barrier()

    #apply a Hadamard to each input qubit again
    for i in range(0, n):
        simon_circ.h(i)
    simon_circ.barrier()

    #measure all the input qubits
    for i in range(0, n):
        simon_circ.measure(i, (i+n))
    
    #return the circuit
    return simon_circ


def simonAlg(n):

    #start tracking time and memory usage
    startTime = time.time()
    tracemalloc.start()

    #create the modN circuit using the n parameter, then create the simon circuit
    function_circuit = modN(int(n))
    circuit = simon(int(n), function_circuit)

    #create the AerSimulator and run it once to get a single sample, then extract only the input qubit values
    sim = AerSimulator()
    sim_result = sim.run(circuit, shots=1, memory=True).result().get_memory()[0]
    firstSample = sim_result.split(" ")[0]

    #add the measured sample to a matrix and add that matrix to the output
    tempMatrix = []
    for bit in firstSample:
        tempMatrix.append(int(bit))
    output = [tempMatrix]
    del tempMatrix

    #iterate through each sample, until the number of independent samples equals n-1
    samplingCount = 1
    independentSamples = 1
    while independentSamples < (int(n)-1):
        #for each sample, extract the input qubit values
        sim_result = sim.run(circuit, shots=1, memory=True).result().get_memory()[0]
        samplingCount = samplingCount + 1
        sample = sim_result.split(" ")[0]
        #transform the qubit values into a vector
        vectorArr = []
        for bit in sample:
            vectorArr.append(int(bit))

        #create a copy of the output and add this vector to it
        testMatrix = output.copy()
        testMatrix.append(vectorArr)

        #calculate the rank of the testMatrix. If it equals the nubmer of vectors, then the vectors are independent
        numpyTestMatrix = np.array(testMatrix)
        if np.linalg.matrix_rank(testMatrix) == len(testMatrix):
            output.append(vectorArr)
            independentSamples = independentSamples + 1
        #clean up unused arrays
        del vectorArr
        del testMatrix
        del numpyTestMatrix

    #extract the total time taken and total space taken
    endTime = time.time()
    totalTime = endTime - startTime
    totalSpace = tracemalloc.get_tracemalloc_memory()

    #stop the memory tracking
    tracemalloc.stop()

    # print(output)
    # print(samplingCount)
    # print(totalTime)
    # print(totalSpace)
    print("n = " + str(n) + ", runtime = " + str(totalTime) + " seconds, memory usage = " + str(totalSpace) + " bytes")

    # circuit.draw(output="mpl")
    # plt.tight_layout()
    # plt.show()

    #return the samples, the number of samples taken to get the n-1 independent samples, the total time, and the total space
    return output, samplingCount, totalTime, totalSpace

def main():
    #extract n from the command line arguments, call simonAlg with it, and return the values from simonAlg
    n = sys.argv[1]
    output, samplingCount, totalTime, totalSpace = simonAlg(n)
    #print the n-1 samples
    print(output)
    return output, samplingCount, totalTime, totalSpace

main()