from qiskit import transpile
from qiskit_optimization.converters import QuadraticProgramToQubo

# This is the cost function clas used for all my QAOAs
from qiskit import transpile

class CostFunction:
    def __init__(self, ansatz, hamiltonian, backend, method='aer', shots=1000):
        """
        Initializes a callable cost function for optimizers like scipy.minimize.

        Stores state for reuse and evaluation.
        """
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.backend = backend
        self.method = method
        self.shots = shots
        self.costs = []  # Stores expectation value at each iteration

    def __call__(self, params):
        """
        Computes and returns the expectation value of the Hamiltonian for given parameters.
        """
        if self.method == 'aer':
            bound_ansatz = self.ansatz.assign_parameters(params)
            circuit = transpile(bound_ansatz, backend=self.backend)
            circuit.save_statevector()
            result = self.backend.run(circuit, shots=self.shots).result()
            statevector = result.get_statevector(circuit)
            cost = statevector.expectation_value(self.hamiltonian).real
            print(cost)
            self.costs.append(cost)
            return cost

        elif self.method == 'qiskit_ibm_runtime':
            hamiltonian_isa = self.hamiltonian.apply_layout(layout=self.ansatz.layout)
            pub = (self.ansatz, hamiltonian_isa, params)
            estimator_results = self.backend.run(pubs=[pub], shots=self.shots).result()[0]
            cost = estimator_results.data.evs
            print(cost)
            self.costs.append(cost)
            return cost

        else:
            raise ValueError(f"method must be either 'aer' or 'qiskit_ibm_runtime'. Got '{self.method}'")

    def run_final_circuit(self, optimal_params):
        """
        Runs the optimized ansatz circuit on the backend using the given parameters.

        Adds measurements if using AerSimulator. Returns the result object.
        """
        optimized_circuit = self.ansatz.assign_parameters(optimal_params)
        optimized_circuit.measure_all()
        self.final_circuit = optimized_circuit

        if self.method == 'aer':
            optimized_circuit = transpile(optimized_circuit, backend=self.backend)
            job = self.backend.run(optimized_circuit, shots=self.shots)
            return job

        elif self.method == 'qiskit_ibm_runtime':
            pub = (optimized_circuit)
            job = self.backend.run(circuits=[pub], shots=self.shots)
            return job

        else:
            raise ValueError(f"method must be either 'aer' or 'qiskit_ibm_runtime'. Got '{self.method}'")
        


################################################################################################################################################
# Converts my portfolio class from Define_Portfolio to a qiskit quadratic program
def portfolio_to_ising(portfolio):
    qp = portfolio.construct_qp() # Turn the portfolio to a quadratic program as explained in a previous cell
    qp2qubo = QuadraticProgramToQubo()
    qubo = qp2qubo.convert(qp)
    cost_hamiltonian = qubo.to_ising()[0]
    return cost_hamiltonian # Converts the qp to an ising hamiltonian. This hamiltonian is directly loaded into QAOAAnsatz
