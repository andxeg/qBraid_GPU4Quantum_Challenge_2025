
import time
from memory_profiler import memory_usage

# --------------------------------------------
# SETUP PATHS TO IMPORT LOCAL MODULES
# --------------------------------------------

# --------------------------------------------
# IMPORT FUNCTIONS FROM LOCAL MODULES
# --------------------------------------------

from utils.graphs import (
    append_csv,
    generate_random_po_graph,
    save_graphs,
    load_graphs,
    graph_to_po_problem,
)
# THis file needs to be adjusted to import from ../gpt-qaoa/inference.py
from inference import (
    inference,
    load_meta_info,
    load_model,
    generate_batch,
    MODELS_INFO,
    graph_to_tokens_old_format,
)
from utils.run_circuit_qiskit import run_gpt_circuit
from utils.run_circuit_qokit import solve_with_qokit
from utils.parsing import tokens_to_circuit, get_max_token_count

# --------------------------------------------
# USER INPUT PROMPTS
# --------------------------------------------

# Ask user to input number of assets (nodes), number to select, risk aversion, and trials
n = int(input("Enter number of assets (n): "))
K = int(input(f"Enter number of assets to select (K, default {n // 5}): ") or (n // 5))
q = float(input("Enter risk aversion parameter (q, e.g. 0.5): "))
number_of_trials = int(input("Enter number of trials: "))

# Estimate maximum number of tokens needed based on n
max_token_count = get_max_token_count(n)
print(f"Graph Size: {n}, Max Number of Tokens: {max_token_count}")

# --------------------------------------------
# GENERATE RANDOM PORTFOLIO OPTIMIZATION GRAPHS
# --------------------------------------------

graphs = []
for trial in range(number_of_trials):
    G = generate_random_po_graph(n=n, K=K, q=q)
    graphs.append(G)

# --------------------------------------------
# LOAD GPT MODEL AND TOKENIZER
# --------------------------------------------

model_id = "50m_new_ft_nasdaq"  # Can be changed to other models
device = "cuda"                 # or "cpu" if no GPU available
cached = True                   # Use cached model if possible

# Load model config
model_params = MODELS_INFO[model_id]

# Load metadata and tokenizer
meta, stoi, itos = load_meta_info(model_params["meta_path"])
graph_tokenizer = eval(model_params["graph_tokenizer"])

# Load GPT model
model, config = load_model(
    model_params["ckpt_path"],
    meta,
    device=device,
    compile=True,
    cached=True
)

# --------------------------------------------
# GENERATE QUANTUM CIRCUITS FROM GRAPHS USING GPT
# --------------------------------------------

generated_circuits, generated_times, generated_memory = generate_batch(
    graphs,
    model,
    config,
    stoi,
    itos,
    graph_tokenizer,
    cached=cached,
    device=device,
    max_total_tokens=max_token_count
)

print("DONE GPT")

# Convert tokens to Qiskit circuits
circuits = []
for tokens in generated_circuits:
    qc = tokens_to_circuit(tokens, num_qubits=n)
    circuits.append(qc)

# --------------------------------------------
# RUN QOKIT AND GPT CIRCUITS ON EACH GRAPH
# --------------------------------------------

qokit_solver_times = []
gpt_alphas = []
qokit_alphas = []
qokit_peak_memories = []

for g, (graph, qc) in enumerate(zip(graphs, circuits)):
    po_problem = graph_to_po_problem(graph)

    if g != 0:
        # Measure time and memory for QOKit circuit
        start = time.time()
        mem_info = memory_usage(
            lambda: solve_with_qokit(po_problem),
            retval=True,
            max_usage=True
        )
        end = time.time()

        # Extract results
        qokit_peak_memory = mem_info[0]
        _, __, qokit_alpha = mem_info[1]

        print("QOKit Approximation Ratio:", qokit_alpha)

        # Store QOKit metrics
        qokit_alphas.append(qokit_alpha)
        qokit_solver_times.append(end - start)
        qokit_peak_memories.append(qokit_peak_memory)

    # Run GPT circuit on same problem and compute approximation ratio
    gpt_alpha = run_gpt_circuit(po_problem=po_problem, qc=qc, shots=100)
    gpt_alphas.append(gpt_alpha)

# --------------------------------------------
# SAVE RESULTS TO CSV FILE
# --------------------------------------------

# Skip first entry (used as warm-up or burn-in)
rows = [[
    n,
    qokit_solver_times[i],
    generated_times[i + 1],
    qokit_alphas[i],
    gpt_alphas[i + 1],
    qokit_peak_memories[i],
    generated_memory[i + 1]
] for i in range(len(qokit_alphas))]

# Save to CSV
append_csv(rows=rows, filename='100_shots.csv')
