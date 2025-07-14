import matplotlib.pyplot as plt
import os
import pandas as pd

# This is a wrapper function that combines the processing functions below into one callable function that processes the sampler job and eliminates any constraint violating results.
# B is the budget constraint
def process_sampler_results(job, B, isAer = False):
    counts = get_counts(job, isAer)
    filtered_counts = filter_counts(counts, B)
    return filtered_counts

# Takes the counts directly from the sampler job and converts it to a sorted list.
# The bitstrings are reversed to match logical qubit order (qubit 0 on the right).
def get_counts(job, isAer = False):
    counts_bin = None
    if isAer == True:
        counts_bin = job.result().get_counts()
    else:
        counts_bin = job.result()[0].data.meas.get_counts()

    # Reverse each bitstring to fix Qiskit’s default little-endian format
    counts = [(count, bitstring[::-1]) for bitstring, count in counts_bin.items()]
    counts.sort(reverse=True)
    return counts

# Filters results to include only those that satisfy the constraint of exactly B ones.
def filter_counts(counts, B):
    return [result for result in counts if result[1].count('1') == B]
    
# Calculates the approximation ratio. This gives a measure of the quality of the quantum solution. 
def approximation_ratio(classical_cost, quantum_cost):
    ratio = abs(classical_cost / quantum_cost)
    if ratio > 1:
        # 1 is the max ratio, so if the ratio is over 1, we flip it.
        if ratio == 0:
            print("Error: Ratio is 0.")
            return
        ratio = 1 / ratio
    return ratio
    
def draw_results(counts_list, B, shots, classical_solution = None, filename = None):
    # counts_list is [(count, bitstring), ...]
    counts_list = sorted(counts_list, key=lambda x: x[1])  # Optional: sort by bitstring for nicer plots

    # Separate into two lists for plotting
    counts = [count / shots for count, bitstring in counts_list]
    bitstrings = [bitstring for count, bitstring in counts_list]
    colors = ['green' if bitstring == classical_solution else 'blue' for bitstring in bitstrings]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.bar(bitstrings, counts, color=colors)
    plt.xlabel('Potential Solution Bitstrings')
    plt.ylabel('Probability')
    plt.title(f'Measurements of States with Exactly {B} Ones')
    plt.grid(axis='y')
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
        plt.close()
        return
    plt.show()
        
def log_column_csv(csv_path, optimizer_name, expectations):
    """
    Log a full list of expectation values for an optimizer as a column.

    If the file exists, it appends the new column to it by iteration index.
    If it doesn't exist, it creates it with iteration as index.

    Parameters:
        csv_path (str): Path to the CSV file.
        optimizer_name (str): Name of the optimizer (column header).
        expectations (list or array-like): Expectation values per iteration.
    """
    new_data = pd.DataFrame({optimizer_name: expectations})

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df[optimizer_name] = pd.Series(expectations)
    else:
        df = new_data

    df.to_csv(csv_path, index=False)

def log_row_csv(csv_path, row_dict):
    """
    Logs a row of data (dict format) into a CSV.

    Adds headers if the file doesn't exist.

    Parameters:
        csv_path (str): CSV file path.
        row_dict (dict): Dictionary of key-value pairs to log.
    """
    file_exists = os.path.exists(csv_path)
    df = pd.DataFrame([row_dict])
    df.to_csv(csv_path, mode='a', index=False, header=not file_exists)


def read_expectation_csv(csv_path):
    """
    Reads a CSV file of optimizer expectation values per iteration.

    Parameters:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame where each column is an optimizer and each row is an iteration.
    """
    return pd.read_csv(csv_path)

def plot_expectation_data(df, filename = None):
    """
    Plots expectation value vs iteration for each optimizer.

    Parameters:
        df (pd.DataFrame): DataFrame where each column is an optimizer and rows are iterations.
    """
    plt.figure(figsize=(10, 6))

    for optimizer in df.columns:
        plt.plot(df[optimizer], marker='o', label=optimizer)  # Only dots

    plt.xlabel("Iteration")
    plt.ylabel("Expectation Value")
    plt.title("QAOA Expectation Value by Optimizer")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
        plt.close()
        return
    plt.show()