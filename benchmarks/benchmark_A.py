import sys
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from memory_profiler import memory_usage
import torch

# --- SETUP PATHS TO IMPORT MODULES ---
# Get absolute path to the notebook directory
notebook_dir = os.path.dirname(os.path.abspath(__file__))
if notebook_dir not in sys.path:
    sys.path.insert(0, notebook_dir)

# Get absolute path to the QOKit directory (assumes it's one level up from notebook_dir)
qokit_path = os.path.abspath(os.path.join(notebook_dir, "../QOKit"))
if qokit_path not in sys.path:
    sys.path.insert(0, qokit_path)
sys.path.append("../gpt-qaoa")
print("Notebook directory:", notebook_dir)
print("QOKit path:", qokit_path)

# --- IMPORT FUNCTIONS FROM LOCAL FILES ---
from utils.graphs import (
    append_csv,
    generate_random_po_graph,
    save_graphs,
    load_graphs,
    graph_to_po_problem,
)
from inference import (
    inference,
    load_meta_info,
    load_model,
    generate_batch,
    MODELS_INFO,
    graph_to_tokens_old_format,
    graph_to_tokens_v1,
)
from functools import partial
from utils.run_circuit_qiskit import run_gpt_circuit
from utils.run_circuit_qokit import solve_with_qokit
from utils.parsing import tokens_to_circuit, get_max_token_count
GRAPH_TOKENIZERS: dict[str, Callable] = {
    "graph_to_tokens_old_format": graph_to_tokens_old_format,
    "graph_to_tokens_v1": graph_to_tokens_v1,
    "graph_to_tokens_v1_nasdaq": partial(graph_to_tokens_v1, version_token="<format_v3_nasdaq>"),
}

def get_graph_tokenizer(model_id: str) -> Callable:
    if model_id not in MODELS_INFO:
        raise ValueError(f"Unknown model_id: {model_id}")
    tokenizer_name = MODELS_INFO[model_id]["graph_tokenizer"]
    if tokenizer_name not in GRAPH_TOKENIZERS:
        raise ValueError(f"Tokenizer '{tokenizer_name}' not found in GRAPH_TOKENIZERS")
    return GRAPH_TOKENIZERS[tokenizer_name]

n = int(input("Enter number of assets (n, e.g. 10-20): "))
K = int(input(f"Enter number of assets to select (K, default {n // 5}): ") or (n // 5))
q = float(input("Enter risk aversion parameter (q, e.g. 0.5): "))
number_of_trials = int(input("Enter number of trials (e.g. 3-5): "))

max_token_count = get_max_token_count(n)
print(f"Graph Size: {n}, Max Number of Tokens: {max_token_count}")
# --- GENERATE RANDOM GRAPHS FOR PORTFOLIO OPTIMIZATION ---
graphs = []
for number in range(number_of_trials):
    G = generate_random_po_graph(n=n, K=K, q=q)
    graphs.append(G)

# --- LOAD GPT MODEL AND TOKENIZER ---
model_id = "50m_new_ft_nasdaq"  # Can change to another model like "20m_new"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
cached = True          # Use a cached version of the model if possible

# Step 1: Get model config
model_params = MODELS_INFO[model_id]

# Step 2: Load meta info and tokenizer
meta, stoi, itos = load_meta_info(model_params["meta_path"])
graph_tokenizer = get_graph_tokenizer(model_id)

# Step 3: Load model itself
model, config = load_model(
    model_params["ckpt_path"], meta, device=device, compile=True, cached=True
)

# --- GENERATE QUANTUM CIRCUITS FROM GRAPHS USING GPT ---
generated_circuits, generated_times, generated_memory = generate_batch(
    graphs,
    model,
    config,
    stoi,
    itos,
    graph_tokenizer,
    cached=cached,
    device=device,
    max_total_tokens = max_token_count
)
print("DONE GPT")
# Convert generated tokens to actual Qiskit circuits
circuits = []
for tokens in generated_circuits:
    qc = tokens_to_circuit(tokens, num_qubits = n)
    circuits.append(qc)

# --- RUN QOKIT AND GPT CIRCUITS ON EACH GRAPH ---
qokit_solver_times = []
gpt_alphas = []
qokit_alphas = []
qokit_peak_memories = []

# Loop over each graph and its corresponding circuit
for g, (graph, qc) in enumerate(zip(graphs, circuits)):
    # Convert graph to PO dictionary (portfolio optimization problem)
    po_problem = graph_to_po_problem(graph)

    if g != 0:
        # Time and profile QOKit execution
        start = time.time()
        mem_info = memory_usage(
            lambda: solve_with_qokit(po_problem),
            retval=True,
            max_usage=True
        )
        end = time.time()
        qokit_peak_memory = mem_info[0]
        _, __, qokit_alpha = mem_info[1]  # result from solve_with_qokit
    
        print("QOKit Approximation Ratio:", qokit_alpha)
    
        # Record QOKit results
        qokit_alphas.append(qokit_alpha)
        qokit_solver_times.append(end - start)
        qokit_peak_memories.append(qokit_peak_memory)

    # Run GPT-generated circuit and compute approximation ratio
    gpt_alpha = run_gpt_circuit(po_problem=po_problem, qc=qc, shots= 100)
    gpt_alphas.append(gpt_alpha)

# --- SAVE RESULTS TO CSV FILE ---
# Skip the first entry (typically used for debugging or warm-up)
rows = [[n,
    qokit_solver_times[i],
    generated_times[i + 1],
    qokit_alphas[i],
    gpt_alphas[i + 1],
    qokit_peak_memories[i],
    generated_memory[i+1]
] for i in range(len(qokit_alphas))]

append_csv(rows=rows, filename=f'results.csv')

