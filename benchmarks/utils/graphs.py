import sys
import os

qokit_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../QOKit"))
if qokit_path not in sys.path:
    sys.path.insert(0, qokit_path)


from qokit.portfolio_optimization import get_problem
import networkx as nx
import pickle
import numpy as np
import csv

def generate_random_po_graph(n, K, q):
    # n is number of assets (number of nodes)
    # K is the budget constraint (how many assests can be in the portfolio)
    # q is the risk tolerance. A number between 0 and 1
    po_problem = get_problem(N=n,K=K,q=q,seed=1,pre=1, isRandom = True) # Creates PO instance
    
    G = nx.Graph() # Initializes empty graph
    
    returns = po_problem['means'] # Grabs "mu," the return values, from the problem instance
    sigma = po_problem['cov'] # Grabs the covariance matrix from the problem instance
    # Add nodes with returns as attributes
    for i, r in enumerate(returns):
        G.add_node(i, return_=r)
    
    # Add edges with covariances as weights
    n = len(returns)
    for i in range(n):
        for j in range(i + 1, n):
            weight = sigma[i, j]
            if weight != 0:
                G.add_edge(i, j, weight=weight)
    return G


def graph_to_po_problem(graph, K=4, q=0.5, seed=1, pre=1, scale=1):
    """
    Reconstructs the PO problem dictionary from a portfolio graph.

    Parameters:
    - graph: networkx.Graph with node attribute 'return_' and edge attribute 'weight'
    - K, q, seed, pre, scale: same as original generation parameters

    Returns:
    - Dictionary with keys: N, K, q, seed, means, cov, pre, scale
    """
    N = graph.number_of_nodes()

    # Extract returns
    means = np.array([graph.nodes[i]['return_'] for i in range(N)])

    # Reconstruct covariance matrix
    cov = np.zeros((N, N))
    for i, j, attr in graph.edges(data=True):
        cov[i, j] = attr['weight']
        cov[j, i] = attr['weight']  # Ensure symmetry

    return {
        'N': N,
        'K': K,
        'q': q,
        'seed': seed,
        'means': means * scale,
        'cov': cov * scale,
        'pre': pre,
        'scale': scale
    }
    
def save_graphs(graphs, filename = 'graphs.pkl'):
    try:
        with open(filename, 'wb') as file:
            pickle.dump(graphs, file)
        print(f'{len(graphs)} have been saved to {filename}.')
    except Exception as e:
        print(f'Error: {e}')
def load_graphs(filename):
    try:
        with open(filename, 'rb') as file:
            graphs = pickle.load(file)
        print(f'{len(graphs)} loaded from {filename}.')
        return graphs
    except Exception as e:
        print(f'Error: {e}')

def append_csv(rows, filename):
    file_exists = os.path.exists(filename)

    try:
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(['Node Count', 'QOKit time (s)', 'GPT time (s)', 'QOKit AR', 'GPT AR', 'QOKit Peak Memory (MB)', 'GPT Peak Memory (MB)'])

            writer.writerows(rows)
        print(f'{len(rows)} rows added to {filename}')

    except Exception as e:
        print(f"Error writing to CSV: {e}")
###############################################################################################
################################### DEPRECATED ###############################################
###############################################################################################

def print_results(qokit_solver_time, gpt_alphas, qokit_alphas, num_graphs):

    qokit_time = sum(qokit_solver_time)
    average_ar_gpt = sum(gpt_alphas) / len(gpt_alphas)
    average_ar_qokit = sum(qokit_alphas) / len(qokit_alphas)
    print(f"Number of Graphs: {num_graphs}")
    print(f"GPT Time: {gpt_duration}")
    print(f"QOKit Time: {qokit_time}")
    print()
    print(f"GPT Mean Approximation Ratio: {average_ar_gpt}")
    print(f"QOKit Mean Approximation Ratio: {average_ar_qokit}")