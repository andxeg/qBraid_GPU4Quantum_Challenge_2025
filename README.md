[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/andxeg/qBraid_GPU4Quantum_Challenge_2025.git&redirectUrl=start_here.ipynb)

# 2025 qBraid GPU4Quantum Challenge Submission

Qvengers Team
Members: Andrew Maciejunes | Marlon Jost | Andrei Chupakhin | Pranik Chainani | Vlad Gaylun

**Single-shot circuit generation + graph-partitioning pipeline that delivers up to 137× speed-ups on 20-asset problems and scales cleanly to 150-asset instances.**


## Key ideas
- **GPT-based circuit generator** - swaps the slow, iterative QAOA optimiser for a single transformer inference, eliminating hundreds of gradient-descent steps.  
- **Warm-start option** - feed the GPT suggestion into any classical optimiser when you want a final polish.  
- **Hardware-aware graph partitioner** - decomposes a large portfolio into *p*-node sub-graphs that fit comfortably on your GPU/QPU, then stitches the sub-solutions back together.  
- **Full pipeline notebook** - demonstrates end-to-end optimisation of a `150`-asset Nasdaq portfolio, including RMT denoising, spectral clustering, and recombination of sub-solutions. 


## How to Use This Repository

1. First, start an instance on [qBraid](https://account.qbraid.com).
2. Go back to this repository and click on the `Launch on qBraid` button. You will be redirected to qBraid (repository cloning might take up to 5 minutes).
3. When the repository is cloned, open the Jupyter Notebook [start_here.ipynb](./start_here.ipynb) and follow the instructions inside.
4. You're all set!


## The Bottleneck: Slow QAOA Optimization
The Quantum Approximate Optimization Algorithm (**QAOA**) is a promising approach for solving complex optimization problems like portfolio management on near-term quantum hardware. However, a critical bottleneck has slowed its practical use: **the iterative parameter-tuning loop**.

The standard QAOA workflow involves:
1. **Defining** a quantum circuit with a set of variable parameters (β,γ).
2. **Running** the circuit on a quantum processor (QPU) or simulator.
3. **Measuring** the cost function (expectation value).
4. Using a **classical optimizer** to adjust the parameters based on the measurement, which requires **hundreds of sequential optimization steps**.

This constant back-and-forth between quantum simulation/execution and classical optimization makes the process extremely slow and computationally expensive, especially as problem sizes grow.


## Our Challenge: Scaling and Speeding Up Quantum Optimization
Our goal was to eliminate this slow, sequential tuning loop and address the scalability issue, demonstrating a path toward practical quantum advantage.

We faced two primary challenges:
1. **Eliminating Iteration**: Develop a method to find near-optimal QAOA circuits in a **single inference step**, bypassing the time-consuming classical optimization loop.
2. **Overcoming Hardware Limits**: Create a strategy to scale quantum optimization beyond the qubit limits of current Noisy Intermediate-Scale Quantum (**NISQ**) devices.


## Our Solution: GPT-QAOA and Finance-Aware Decomposition
To solve these problems, we introduced two key innovations:
1. **GPT-based Circuit Generation (GPT-QAOA)**

We trained a **GPT-driven generative model** to learn the relationship between an optimization problem's structure (specifically, the graph/asset correlation) and its optimal QAOA parameters. This allows the model to produce a near-optimal QAOA circuit in a **single, rapid inference**, effectively replacing the hundreds of sequential steps previously required.

2. **Large-Scale Problem Decomposition**

To tackle problems larger than what current quantum hardware or simulators can handle (like a 180-asset portfolio), we implemented a **finance-aware partitioning pipeline**. This approach  decomposes large problems into a set of smaller, manageable subproblems that can be solved independently and potentially in parallel across multiple QPUs while maintaining high solution quality.

By integrating our **GPT-QAOA model** with this **decomposition pipeline**, we created a scalable and accelerated quantum optimization framework that extends QOkit's applicability to complex, real-world problems.


## Repository layout
- **benchmarks** - Scripts and Jupyter notebooks for comparing QOkit and GPT in terms of approximation ratio (AR), execution time, and memory usage. Also includes benchmarks for graph partitioning.
    - `benchmark_A.py` - This file is used to generated `results.csv`. It has been modified to take in user input when run from the terminal. It requires GPU support to run.
    - `results.csv` - Holds the results from running `benchmark_A.py` on `75` graphs. `5` graphs of each node size from `5` to `20`. 
    - `utils/graphs.py` - Contains supporting functions related to saving, loading, and generating graphs used by `benchmark_A.py`
    - `utils/parsing.py` - Contains logic for parsing the GPT tokens generated by the model. This converts tokens into quantum circuits.
    - `utils/run_circuit_qiskit.py` - A file that holds all the qiskit related code. Our model is evaluated on qiskit code in `benchmark_A.py`.
    - `utils/run_circuit_qokit.py` - A wrapper for `QOKit/qokit/examples/portfolio_optimization.ipynb` from the original QOKit repo. We compare our model to this function's output.
    - `utils/utils.py` - Miscileneous helper functions.
- **gpt-qaoa** - Training a custom model to generate quantum circuits from graphs (nodes represent assets with result attributes; edges represent correlations between assets).
- **partitioning** - Graph decomposition: solve the optimization problem for each subgraph using QOkit or the custom GPT model, then concatenate the results to obtain the final solution.
    - `Partitioning_classical_approach_210 nodes.ipynb` - demonstration of decomposition pipeline
    - `partitioning_with_gpt.py` - GPT-based circuit generator in paritioning workflow (working progress)
    - `partitioning_with_QOKit.py` - QOKit-based circuit simulator in partitioning workflow (working progress)


## ⚠️ Important Note about Checkpoints and Git LFS

> **Note:** Large binary assets (e.g. model checkpoints) are stored via **Git LFS**.  
> Due to GitHub bandwidth and file size limits, you may **encounter errors** when trying to download `gpt-qaoa/checkpoints` directly through `git clone` or `git lfs pull`.


### 🔧 Recommended workaround

1. Manually go to the [GitHub repo](https://github.com/Marlon-Jost/JPMorgan-Challenge-Submission/tree/main/gpt-qaoa/checkpoints)
2. Download the all files to your **local machine**
3. Upload them to your **qBraid instance** using the web interface

This ensures you get all the necessary files without hitting GitHub's LFS restrictions.
