import sys
import os

# Add parent directory (one level up) to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from qiskit_aer import AerSimulator
from qiskit import transpile
from graphs_and_parsing.Define_PostProcessing import process_sampler_results, approximation_ratio
from graphs_and_parsing.Define_Portfolio import markowitz_cost, brute_force_markowitz
import networkx as nx
import numpy as np

def circuit_to_job(qc, shots):
    sim = AerSimulator(method="statevector")
    qc = qc.decompose(reps=6)
    qc.measure_all()
    isa_qc = transpile(qc)
    job = sim.run(isa_qc, shots=shots)
    return job

def job_to_binary_vector(job):
    filtered_counts = process_sampler_results(job, 4, isAer=True)
    binary_decision_variables = []
    for item in filtered_counts:
        binary_decision_variables.append([int(bit) for bit in item[1]])
    return binary_decision_variables

def get_costs(binary_decision_variables, po_problem):
    sigma = po_problem['cov']
    mu = po_problem['means']
    q = po_problem['q']
    states_cost_dict = {}
    for array in binary_decision_variables:
        c = markowitz_cost(array, sigma, mu, q)
        states_cost_dict[tuple(array)] = c
    return states_cost_dict

def min_cost_min_key(states_cost_dict):
    if not states_cost_dict:
        print("[WARNING] states_cost_dict is empty. Approximation Ratio will return None")
        return None, None
    min_key = min(states_cost_dict, key=states_cost_dict.get)
    min_cost = states_cost_dict[min_key]
    return min_cost, min_key

def print_po_data(po_problem, min_cost, min_key):
    sigma = po_problem['cov']
    mu = po_problem['means']
    q = po_problem['q']
    B = po_problem['K']
    classical_eigenvector, classical_cost = brute_force_markowitz(sigma, mu, q, B)
    alpha = approximation_ratio(quantum_cost=min_cost, classical_cost=classical_cost)
    return alpha

def run_gpt_circuit(qc, po_problem, shots = 1000):
    job = circuit_to_job(qc, shots = shots)
    binary_decision_variables = job_to_binary_vector(job)
    states_cost_dict = get_costs(binary_decision_variables, po_problem)
    min_cost, min_key = min_cost_min_key(states_cost_dict)
    if min_cost is None:
        return None
    alpha = print_po_data(po_problem, min_cost, min_key)
    return alpha
