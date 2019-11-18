#importing required modules
import numpy as np
import week2modules as wk2
import week3module as wk3
import matplotlib.pyplot as plt

#creating  combined_H
def combine_H(n):
    #where n is no. Hardamard gates
    combined_h = wk3.combine_gate((1/np.sqrt(2))*np.array([[1, 1], [1, -1]]), (1/np.sqrt(2))*np.array([[1, 1], [1, -1]]))
    for i in range(0, n-2):
        combined_h = wk3.combine_gate(combined_h, (1/np.sqrt(2))*np.array([[1, 1], [1, -1]]))
    return combined_h


#creating  combined_X
def combine_X(n):
    #where n is no. pauli's X gates
    combined_x = wk3.combine_gate(np.array([[0, 1], [1, 0]]), np.array([[0, 1],[1, 0]]))
    for i in range(0, n-2):
        combined_x = wk3.combine_gate(combined_x, np.array([[0, 1],[1, 0]]))
    return combined_x


#creating a superpostion state
def superposition(n):
    #n is no. of state 0
    combined_q = wk3.combine_qubits(np.array([[1], [0]]), np.array([[1], [0]]))
    for i in range(0, n-2):
        combined_q = wk3.combine_qubits(combined_q, np.array([[1], [0]]))
    combined_g = wk3.combine_gate((1/np.sqrt(2))*np.array([[1, 1], [1, -1]]), (1/np.sqrt(2))*np.array([[1, 1], [1, -1]]))
    for i in range(0, n-2):
        combined_g = wk3.combine_gate(combined_g, (1/np.sqrt(2))*np.array([[1, 1], [1, -1]]))
    return np.dot(combined_g, combined_q)


#defining oracle function
def oracle(w, s):
    #where w is the winner state, where s is the no. of supersition states
    orcl_1 = np.identity(s, dtype=int)
    orcl_1[w][w] = -orcl_1[w][w]
    return orcl_1

#creating reflection over reduced mean
def mean_reflector(n):
    #where s is the superposition
    combined_q = wk3.combine_qubits(np.array([[1], [0]]), np.array([[1], [0]]))
    for i in range(0, n - 2):
        combined_q = wk3.combine_qubits(combined_q, np.array([[1], [0]]))
    return wk2.construct_density_matrix(combined_q)*2 - np.identity(2**n, dtype=int)

#let's N(no. of data is 8, then no. of (n) qubit used 3
N = 8
n = int(np.log2(N))
#let's say the winner item is 1
w = 1
#finding the superposition given the no. qubits
S_position = superposition(n)
qubits = tuple(wk2.construct_standard_basis(n))
Y_pos = np.arange(len(qubits))
probalility = (S_position*S_position).flatten()

plt.bar(Y_pos,  probalility, align='center', alpha=0.5)
plt.xticks(Y_pos, qubits)
plt.ylabel('probalility')
plt.xlabel('qubits')
plt.title('Before amplitude amplification')
plt.show()


# iterating over the qubits sqrt(N) no. of times perfroming amplitude amplification
for i in range(int((3.14/4)*np.sqrt(N))):
    interm_1 = np.dot(oracle(w, N), S_position)
    interm_2 = np.dot(combine_H(n), interm_1)
    interm_3 = np.dot(mean_reflector(n), interm_2)
    interm_4 = np.dot(combine_H(n), interm_3)
    S_position = interm_4
    print()
    print('amplified superposition in iteration'+str(i+1)+'\n')
    print(interm_4)
    qubits = tuple(wk2.construct_standard_basis(n))
    Y_pos = np.arange(len(qubits))
    amplitude = (S_position*S_position).flatten()

    plt.bar(Y_pos,  amplitude, align='center', alpha=1.0)
    plt.xticks(Y_pos, qubits)
    plt.ylabel('probalility')
    plt.xlabel('qubits')
    plt.title('After two reflection in'+str(i+1)+'th iteration')
    plt.show()
