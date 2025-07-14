import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit import QuantumCircuit
import itertools
from qiskit.quantum_info import SparsePauliOp
from itertools import product
import networkx as nx

class Portfolio:
    def __init__(self, sigma, mu, q, B):
        """
        sigma: covariance matrix (numpy 2D array)
        mu: expected returns (numpy array)
        q: risk aversion parameter (float)
        """

        self.sigma = sigma
        self.mu = mu
        self.q = q
        self.B = B

        # This is the variable for the number of qubits and nodes in the problem
        self.n = len(mu)

    def construct_qp(self):

        sigma = self.sigma
        mu = self.mu
        q = self.q
        n = self.n

        # We use quadratic program from qiskit_optimization to build a qp. Using qiskit's QP class will easily let us convert this to an ising problem
        # When building the qp, we do not include constraints because the goal is to include those in the mixer hamiltonian
        qp = QuadraticProgram() 

        #For the quadratic program, we need a binary variable for each node in the graph. Each binary variable represents whether each node is included in the solution
        for i in range(n):
            qp.binary_var(str(i))

        # This adds the quadratic term to the quadrratic program. Check out Markowitz portfolio optimization for more info
        qp.objective.quadratic = q * sigma

        # This adds the linear term to the quadratic
        qp.objective.linear = -mu

        qp.linear_constraint(
        linear={str(i): 1 for i in range(self.n)},  # Coefficients: 1*x_0 + 1*x_1 + ... + 1*x_{n-1}
        sense='==',                                 # Equality constraint
        rhs=self.B,                                 # Right-hand side = B
        name='hamming_weight_constraint'
)

        return qp
    
    def build_initial_state(self):
        """Return a circuit that initializes a Dicke state of Hamming weight k over n qubits"""
        # Create a statevector with 1s at positions with Hamming weight k
        k = self.B
        n = len(self.mu)
        state = np.zeros(2**n)
        for bits in itertools.combinations(range(n), k):
            index = sum([1 << (n - 1 - b) for b in bits])  # convert bit positions to integer
            state[index] = 1

        state = state / np.linalg.norm(state)  # normalize

        # Create circuit and initialize state
        qc = QuantumCircuit(n)
        qc.initialize(state, range(n))
        return qc
    
    # Apologies for switiching between driver and mixer. The words are interchangable. Ill try to stick with mixer because Qiskit uses mixer
    def build_XY_mixer(self):
        paulis_for_mixer = [] # We need a list of paulis and coefficients for the operator 
        list_of_paulis = ['X', 'Y'] # We need to repeat the following loop for each pauli in our hamiltonian, so we create this list to do it more easily.

        #The driver hamiltonian is summation (X_i * X_j + Y_i*Y_j) for all i < j

        for pauli in list_of_paulis:
            for i in range(self.n):
                temp = None # I probably dont need this, but I dont want to accidentally point to something else when I go use the variable in the next loop.

                for j in range (i+1, self.n):
                    temp = ['I'] * self.n # Puts an identity gate on each qubit
                    temp[i] = pauli # Per the hamiltonian, we put either an X or Y pauli gate on the ith qubit

                    temp[j] = pauli # Per the hamiltonian, we put either an X or Y pauli gate on the jth qubit
                    temp = "".join(temp) # This turns the temp char array into a string
                    paulis_for_mixer.append(temp) # Combines the strings into an array of strings. Each string represents one level of paulis on the entire circuit.

        coefficients = [1.0 for i in range(len(paulis_for_mixer))] # Coefficients for each operator. They are all one. 

        mixer_list = list(zip(paulis_for_mixer, coefficients)) # Creates a list of tuples combinining the pauli strings and the coefficients
        mixer_hamiltonian = SparsePauliOp.from_list(mixer_list) #Createas the pauli operator that can later be passed to QAOAAnsatz's mixer_operator constructor
                
        return mixer_hamiltonian
    
#Sigma is the covariance matrix
# Mu is the return value for each investment
#This builds the networkx graph to be saved via pickle for training
def build_portfolio_graph(mu, sigma):
    n = len(mu)
    G = nx.Graph()

    # Add nodes with return attribute
    for i in range(n):
        G.add_node(i, mu=mu[i])

    # Add edges with weight equal to covariance
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=sigma[i][j])

    return G
# Generates random values for sigma and mu
#Sigma is the covariance matrix
# Mu is the return value for each investment
def generate_portfolio(n, isRandom = False):
    """
    n: Int to represent the size of the problem (How many nodes are in the graph?)
    isRandom: Bool If false, it will generate the same graph every time for the appropriate n value. It is set to false by default to easily compare between trials
    """
    if not isRandom:
        np.random.seed(42) 
    sigma = np.zeros([n,n])
    for ii in range(n):
        for jj in range(ii+1,n):
            r = np.random.random()
            sigma[ii][jj] = r
            sigma[jj][ii] = r

    mu = [np.random.random() * 10 for i in range(n)]
    # Return order: Mu, Sigma
    return np.array(mu), np.array(sigma)

    
# This function calculates the cost of a specific solution 
def markowitz_cost(x, sigma, mu, q):

    """
    x: Binary decision variable (numpy array)
    sigma: covariance matrix (numpy 2D array)
    mu: expected returns (numpy array)
    q: risk aversion parameter (float)
    
    """

    x = np.array(x)
    sigma = np.array(sigma)
    mu = np.array(mu)

    quadratic_term = q * x.T @ sigma @ x
    linear_term = mu.T @ x
    return quadratic_term - linear_term

# This function finds the best solution via brute force
def brute_force_markowitz(sigma, mu, q, B):
    n = len(mu)
    best_cost = float('inf')
    best_x = None

    for x in product([0, 1], repeat=n):
        if sum(x) != B:
            continue
        cost = markowitz_cost(x, sigma, mu, q)
        if cost < best_cost:
            best_cost = cost
            best_x = x
    return best_x, best_cost



