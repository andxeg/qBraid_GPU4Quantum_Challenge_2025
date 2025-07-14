import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# def convert_string_list_to_ints_and_floats(data):
#     for i in range(1, len(data)):
#         data[i][0] = int(data[i][0])
#         for j in range(1, len(data[1])):
#             data[i][j] = float(data[i][j])
#     return data

def open_csv_results(filename='results.csv'):
    """
    Reads a CSV into a pandas DataFrame.
    Automatically handles headers and type conversion.
    """
    df = pd.read_csv(filename)
    return df
        
def compute_average(lst):
    total = sum(lst)
    length = len(lst)
    return total / length


def df_to_slices(data):
    # Convert columns to lists
    nodes = data['Node Count'].tolist()
    qokit_times = data["QOKit time (s)"].tolist()
    gpt_times = data['GPT time (s)'].tolist()
    qokit_ars = data["QOKit AR"].tolist()
    gpt_ars = data["GPT AR"].tolist()
    qokit_peak_memory = data['QOKit Peak Memory (MB)'].tolist()
    gpt_peak_memory = data["GPT Peak Memory (MB)"].tolist()
    
    # Slice lists into chunks of 5
    step = 5
    sliced_nodes = [nodes[i:i+step] for i in range(0, len(nodes), step)]
    sliced_qokit_times = [qokit_times[i:i+step] for i in range(0, len(nodes), step)]
    sliced_gpt_times = [gpt_times[i:i+step] for i in range(0, len(nodes), step)]
    sliced_qokit_ars = [qokit_ars[i:i+step] for i in range(0, len(nodes), step)]
    sliced_gpt_ars = [gpt_ars[i:i+step] for i in range(0, len(nodes), step)]
    sliced_qokit_peak_memory = [qokit_peak_memory[i:i+step] for i in range(0, len(nodes), step)]
    sliced_gpt_peak_memory = [gpt_peak_memory[i:i+step] for i in range(0, len(nodes), step)]
    
    # Zip the slices
    zipped = list(zip(
        sliced_nodes,
        sliced_qokit_times,
        sliced_gpt_times,
        sliced_qokit_ars,
        sliced_gpt_ars,
        sliced_qokit_peak_memory,
        sliced_gpt_peak_memory
    ))

    # Calculate standard deviations (sample std dev)
    zipped_stds = [
        (
            np.std(node, ddof=1),
            np.std(qokit_time, ddof=1),
            np.std(gpt_time, ddof=1),
            np.std(qokit_ar, ddof=1),
            np.std(gpt_ar, ddof=1),
            np.std(qokit_mem, ddof=1),
            np.std(gpt_mem, ddof=1)
        )
        for node, qokit_time, gpt_time, qokit_ar, gpt_ar, qokit_mem, gpt_mem in zipped
    ]
    return zipped, zipped_stds


def df_to_full_lists(data):
    # Convert columns to full lists
    nodes = data['Node Count'].tolist()
    qokit_times = data["QOKit time (s)"].tolist()
    gpt_times = data['GPT time (s)'].tolist()
    qokit_ars = data["QOKit AR"].tolist()
    gpt_ars = data["GPT AR"].tolist()
    qokit_peak_memory = data['QOKit Peak Memory (MB)'].tolist()
    gpt_peak_memory = data["GPT Peak Memory (MB)"].tolist()
    
    # Full data tuple (like zipped but not sliced)
    full_data = (
        nodes,
        qokit_times,
        gpt_times,
        qokit_ars,
        gpt_ars,
        qokit_peak_memory,
        gpt_peak_memory
    )

    # Compute standard deviation for each full list
    full_stds = (
        np.std(nodes, ddof=1),
        np.std(qokit_times, ddof=1),
        np.std(gpt_times, ddof=1),
        np.std(qokit_ars, ddof=1),
        np.std(gpt_ars, ddof=1),
        np.std(qokit_peak_memory, ddof=1),
        np.std(gpt_peak_memory, ddof=1)
    )

    return full_data, full_stds



def draw_plot(
    x,
    y_series,
    labels,
    xlabel="X",
    ylabel="Y",
    title="Plot",
    marker="o-",
    show_grid=True,
    filename='',
    y_errors=None,
    yscale="linear"  # New argument
):
    for i, (y, label) in enumerate(zip(y_series, labels)):
        if y_errors is not None and y_errors[i] is not None:
            plt.errorbar(x, y, yerr=y_errors[i], fmt=marker, label=label, capsize=5)
        else:
            plt.plot(x, y, marker, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.yscale(yscale)  # Set y-axis scale here
    if show_grid:
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()