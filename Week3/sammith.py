import numpy as np
import math


#cheaking validity
def gate_validator(Q):
    Q1=Q.conjugate().transpose()
    Q2=np.array([[1,0],[0,1]])
    if np.dot(Q,Q1).all()==Q2.all():
        return 'valid'
    else:
        return 'invalid'
        
        
        
#defining pauli X gate
def pauli_X(Q):
    gate_validator(Q)
    X=np.array([[0,1],[1,0]])
    return np.matmul(X,Q)


# defining pauli Y gate
def pauli_Y(Q):
    gate_validator(Q)
    Y=np.array([[0,-1j],[1j,0]])
    return np.matmul(Y,Q)



# defining pauli Z gate
def pauli_Z(Q):
    gate_validator(Q)
    Z=np.array([[1,0],[0,-1]])
    return np.matmul(Z,Q)


# defining Hadamard gate
def pauli_H(Q):
    gate_validator(Q)
    H=np.array([[1/math.sqrt(2),1/math.sqrt(2)],[1/math.sqrt(2),-1/math.sqrt(2)]])
    return np.matmul(H,Q)



# defining CNOT gate
# enter Q as crossproduct of the two qubits i.e np.kron(qubit,qubit2)
def CNOT1(Q):
    gate_validator(Q)
    CNOT=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    return np.matmul(CNOT,Q)

def CNOT2(Q1,Q2):
    gate_validator(Q1)
    gate_validator(Q2)
    CNOT=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    Q=np.kron(Q1,Q2)
    return np.matmul(CNOT,Q)
    
    
    
# defining CCNOT gate(type1)
# take crossproduct first two qubits as np.kron(qubit1,qubit2) and then take crossproduct of the result with third qubit in the same way as done before
def CCNOT1(Q):
    gate_validator(Q)
    CCNOT=np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]])
    return np.matmul(CCNOT,Q)



# defining CCNOT gate(type2)
# enter three qubits in the function
def CCNOT2(Q1,Q2,Q3):
    gate_validator(Q1)
    gate_validator(Q2)
    gate_validator(Q3)
    CCNOT=np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]])
    Q=np.kron(Q1,Q2)
    Q=np.kron(Q,Q3)
    return np.matmul(CCNOT,Q)


# defining swap gate(type1)
def SWAP1(Q1,Q2):
    gate_validator(Q1)
    gate_validator(Q2)
    SWAP1=np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
    Q=np.kron(Q1,Q2)
    return np.matmul(SWAP1,Q)


# defining swap gate(type2)
# Q=np.kron(Q1,Q2)
def SWAP2(Q):
    gate_validator(Q)
    SWAP2=np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
    return np.matmul(SWAP2,Q)



# Combining gates

def XZ(Q):
    gate_validator(Q)
    XZ=pauli_X(pauli_Z(Q))
    return XZ

def YZ(Q):
    gate_validator(Q)
    YZ=pauli_Y(pauli_Z(Q))
    return YZ

def XY(Q):
    gate_validator(Q)
    XY=pauli_X(pauli_Y(Q))
    return XY
    
def HX(Q):
    gate_validator(Q)
    HX=pauli_H(pauli_X(Q))
    return HX

def HY(Q):
    gate_validator(Q)
    HY=pauli_H(pauli_Y(Q))
    return HY

def HZ(Q):
    gate_validator(Q)
    HZ=pauli_H(pauli_Z(Q))
    return HZ


