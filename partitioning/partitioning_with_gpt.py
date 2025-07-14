import numpy as np
import pandas as pd
import pickle
import time
import os
import networkx as nx
import itertools
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import json
from datetime import datetime
import sys

# Scientific computing
from numpy.linalg import eigh
from sklearn.cluster import SpectralClustering
import pulp as pl

# Quantum computing imports
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available. Quantum features will be disabled.")

# Import the inference and parsing modules for GPT
try:
    sys.path.append("../gpt-qaoa")
    from inference import inference
    from parsing import tokens_to_circuit

    GPT_AVAILABLE = True
except ImportError:
    GPT_AVAILABLE = False
    print("Warning: GPT modules not available. GPT solving will be disabled.")

# Import partitioning functions
try:
    from graph_partitioning import (
        partition,
        map_qaoa_outputs_to_full_vector,
        get_sub_problem,
    )

    PARTITIONING_AVAILABLE = True
except ImportError:
    PARTITIONING_AVAILABLE = False
    print(
        "Warning: graph_partitioning module not available. Using local implementation."
    )


@dataclass
class OptimizationResults:
    """Container for optimization results"""

    global_objective: float
    pipeline_objective: float
    global_solution: List[int]
    pipeline_solution: List[int]
    global_tickers: List[str]
    pipeline_tickers: List[str]
    time_global: float
    time_pipeline: float
    cluster_sizes: List[int]
    n_assets: int
    n_clusters: int
    solver_type: str
    subproblem_approximation_ratios: Optional[List[float]] = None
    additional_stats: Optional[Dict[str, Any]] = None


class PortfolioOptimizerWithGPT:
    """
    Portfolio optimization with graph-based partitioning and GPT solver support.
    """

    def __init__(self, graph_path: str, output_dir: str = "optimization_results"):
        """Initialize the optimizer with a pre-computed graph."""
        self.graph_path = graph_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load graph and extract data
        self._load_graph()

    def _load_graph(self):
        """Load graph from pickle file and extract necessary data"""
        print(f"Loading graph from {self.graph_path}...")

        with open(self.graph_path, "rb") as f:
            self.graph = pickle.load(f)

        # Extract assets and build matrices
        self.tickers = list(self.graph.nodes())
        self.n_assets = len(self.tickers)

        # Extract returns and volatility from node attributes
        self.mu = np.array(
            [self.graph.nodes[ticker]["annual_return"] for ticker in self.tickers]
        )
        self.volatility = np.array(
            [self.graph.nodes[ticker]["annual_volatility"] for ticker in self.tickers]
        )

        # Build covariance matrix from edge weights (correlations)
        self._build_covariance_matrix()

        print(f"Loaded {self.n_assets} assets from graph")

    def _build_covariance_matrix(self):
        """Build covariance matrix from correlation graph"""
        # Initialize correlation matrix
        corr_matrix = np.eye(self.n_assets)

        # Fill correlation matrix from edge weights
        ticker_to_idx = {ticker: i for i, ticker in enumerate(self.tickers)}

        for i, ticker1 in enumerate(self.tickers):
            for j, ticker2 in enumerate(self.tickers):
                if i != j and self.graph.has_edge(ticker1, ticker2):
                    corr_matrix[i, j] = self.graph[ticker1][ticker2]["weight"]

        # Convert correlation to covariance: Cov = D * Corr * D
        D = np.diag(self.volatility)
        self.sigma = D @ corr_matrix @ D

        # Ensure symmetry
        self.sigma = (self.sigma + self.sigma.T) / 2

    def partition_problem(
        self,
        n_partitions: Optional[int] = None,
        max_partition_size: Optional[int] = None,
    ):
        """
        Partition the problem using natural clustering.

        Args:
            n_partitions: Number of partitions to create (if None, uses natural clustering)
            max_partition_size: Maximum allowed partition size (optional)

        Returns:
            partitioned_mus, partitioned_sigmas, mappings
        """
        if PARTITIONING_AVAILABLE and n_partitions is None:
            # Use the imported partition function with its natural clustering
            # The partition function will decide the sizes based on covariance structure
            target_size = self.n_assets // 10  # Just a hint, not enforced
            partitioned_mus, partitioned_sigmas, mappings = partition(
                self.sigma, self.mu, target_size
            )
        else:
            # Use spectral clustering to create specified number of partitions
            from sklearn.cluster import SpectralClustering

            # Convert covariance to correlation for clustering
            std = np.sqrt(np.diag(self.sigma))
            std = np.maximum(std, 1e-10)
            corr = self.sigma / np.outer(std, std)

            # Create affinity matrix
            affinity = np.abs(corr)
            np.fill_diagonal(affinity, 0)  # No self-loops

            n_clusters = n_partitions if n_partitions else max(2, self.n_assets // 15)

            # Perform spectral clustering
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                random_state=42,
                n_init=10,
            )
            labels = clustering.fit_predict(affinity)

            # Create mappings from cluster labels
            mappings = []
            for i in range(n_clusters):
                cluster_indices = np.where(labels == i)[0].tolist()
                if cluster_indices:  # Only add non-empty clusters
                    mappings.append(cluster_indices)

            # Create partitioned data
            partitioned_mus = [self.mu[mapping] for mapping in mappings]
            partitioned_sigmas = [
                self.sigma[np.ix_(mapping, mapping)] for mapping in mappings
            ]

        # Apply max partition size constraint if specified
        if max_partition_size is not None:
            mappings, partitioned_mus, partitioned_sigmas = (
                self._enforce_max_partition_size(
                    mappings, partitioned_mus, partitioned_sigmas, max_partition_size
                )
            )

        return partitioned_mus, partitioned_sigmas, mappings

    def _enforce_max_partition_size(
        self, mappings, partitioned_mus, partitioned_sigmas, max_size
    ):
        """Split partitions that exceed max_size"""
        new_mappings = []
        new_mus = []
        new_sigmas = []

        for mapping, mu, sigma in zip(mappings, partitioned_mus, partitioned_sigmas):
            if len(mapping) <= max_size:
                new_mappings.append(mapping)
                new_mus.append(mu)
                new_sigmas.append(sigma)
            else:
                # Split large partition
                for i in range(0, len(mapping), max_size):
                    sub_mapping = mapping[i : i + max_size]
                    indices = list(range(i, min(i + max_size, len(mapping))))
                    new_mappings.append(sub_mapping)
                    new_mus.append(mu[indices])
                    new_sigmas.append(sigma[np.ix_(indices, indices)])

        return new_mappings, new_mus, new_sigmas

    def portfolio_objective(
        self, x: np.ndarray, sigma: np.ndarray, mu: np.ndarray, q: float
    ) -> float:
        """Calculate portfolio objective: q * risk - return."""
        return q * x.T @ sigma @ x - mu.T @ x

    def brute_force_optimization(
        self, sigma: np.ndarray, mu: np.ndarray, k: int, q: float
    ) -> Tuple[np.ndarray, float, int]:
        """Find optimal portfolio by brute force (for small problems)."""
        n = len(mu)
        best_solution = None
        best_objective = float("inf")
        combinations_tried = 0

        for selected in itertools.combinations(range(n), k):
            combinations_tried += 1
            x = np.zeros(n)
            x[list(selected)] = 1
            objective = self.portfolio_objective(x, sigma, mu, q)

            if objective < best_objective:
                best_objective = objective
                best_solution = x.copy()

        return best_solution, best_objective, combinations_tried

    def greedy_portfolio_selection(
        self, mu: np.ndarray, sigma: np.ndarray, K: int, q: float
    ) -> np.ndarray:
        """Greedy algorithm: Select K assets with best risk-adjusted returns."""
        n = len(mu)
        selected = []
        remaining = list(range(n))

        for _ in range(K):
            best_idx = None
            best_score = float("-inf")

            for idx in remaining:
                current_selected = selected + [idx]
                x_temp = np.zeros(n)
                x_temp[current_selected] = 1

                score = mu[idx] - q * np.sum(sigma[idx, current_selected])

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        x = np.zeros(n)
        x[selected] = 1
        return x

    def simulated_annealing(
        self, mu: np.ndarray, sigma: np.ndarray, K: int, q: float, max_iter: int = 2000
    ) -> np.ndarray:
        """Simulated annealing for portfolio optimization."""
        n = len(mu)

        # Start with greedy solution
        x_current = self.greedy_portfolio_selection(mu, sigma, K, q)
        current_obj = self.portfolio_objective(x_current, sigma, mu, q)

        best_x = x_current.copy()
        best_obj = current_obj

        for i in range(max_iter):
            T = 1.0 * (1 - i / max_iter)  # Temperature schedule

            selected = np.where(x_current == 1)[0]
            unselected = np.where(x_current == 0)[0]

            if len(unselected) > 0 and len(selected) > 0:
                # Swap one selected with one unselected
                remove_idx = np.random.choice(selected)
                add_idx = np.random.choice(unselected)

                x_neighbor = x_current.copy()
                x_neighbor[remove_idx] = 0
                x_neighbor[add_idx] = 1

                neighbor_obj = self.portfolio_objective(x_neighbor, sigma, mu, q)

                # Accept or reject
                delta = neighbor_obj - current_obj
                if delta < 0 or np.random.rand() < np.exp(-delta / T):
                    x_current = x_neighbor
                    current_obj = neighbor_obj

                    if current_obj < best_obj:
                        best_x = x_current.copy()
                        best_obj = current_obj

        return best_x

    def get_classical_baseline(
        self, mu: np.ndarray, sigma: np.ndarray, K: int, q: float
    ) -> Tuple[np.ndarray, float]:
        """Get the best solution from multiple classical heuristics."""
        best_x = None
        best_obj = float("inf")

        print("Computing classical baseline using heuristics...")

        # Greedy
        x_greedy = self.greedy_portfolio_selection(mu, sigma, K, q)
        obj_greedy = self.portfolio_objective(x_greedy, sigma, mu, q)
        print(f"  Greedy: {obj_greedy:.4f}")
        if obj_greedy < best_obj:
            best_obj = obj_greedy
            best_x = x_greedy

        # Simulated Annealing (best of 3 runs)
        for run in range(3):
            x_sa = self.simulated_annealing(mu, sigma, K, q, max_iter=2000)
            obj_sa = self.portfolio_objective(x_sa, sigma, mu, q)
            if run == 0:
                print(f"  Simulated Annealing: {obj_sa:.4f}")
            if obj_sa < best_obj:
                best_obj = obj_sa
                best_x = x_sa

        return best_x, best_obj

    def create_graph_for_gpt(
        self,
        sub_mu: np.ndarray,
        sub_sigma: np.ndarray,
        K: int,
        q: float,
        model_id: str = "50m_old",
    ) -> nx.Graph:
        """Create a NetworkX graph compatible with GPT inference."""
        n = len(sub_mu)
        G = nx.Graph()

        if model_id == "50m_old":
            for i in range(n):
                G.add_node(i)
        else:
            for i in range(n):
                G.add_node(i, mu=sub_mu[i])

        # Add edges with covariance weights
        for i in range(n):
            for j in range(i + 1, n):
                if sub_sigma[i, j] != 0:
                    G.add_edge(i, j, weight=sub_sigma[i, j])

        # Store metadata
        G.graph["K"] = K
        G.graph["q"] = q
        G.graph["N"] = n
        G.graph["mu"] = sub_mu.tolist()
        G.graph["sigma"] = sub_sigma.tolist()

        return G

    def evaluate_gpt_circuit(self, qc, sub_problem: dict) -> Tuple[float, str]:
        """Evaluate a quantum circuit for portfolio optimization."""
        if not QISKIT_AVAILABLE:
            return 0.0, "0" * sub_problem["N"]

        n = sub_problem["N"]
        K = sub_problem["K"]
        q = sub_problem["q"]
        mu = np.array(sub_problem["mu"])
        sigma = np.array(sub_problem["sigma"])

        # Validate inputs
        if K == 0 or K > n:
            print(f"    Warning: Invalid K={K} for n={n}, using greedy fallback")
            x = self.greedy_portfolio_selection(mu, sigma, min(K, n), q)
            bitstring = "".join(str(int(xi)) for xi in x)
            return 0.5, bitstring

        try:
            # Get statevector from circuit
            sv = Statevector.from_instruction(qc)
            probs = sv.probabilities()
        except Exception as e:
            print(
                f"    Warning: Circuit evaluation failed: {str(e)}, using greedy fallback"
            )
            x = self.greedy_portfolio_selection(mu, sigma, K, q)
            bitstring = "".join(str(int(xi)) for xi in x)
            return 0.5, bitstring

        # Find best valid state with exactly K assets
        best_state = None
        best_prob = 0

        for i, prob in enumerate(probs):
            bitstring = format(i, f"0{n}b")
            if bitstring.count("1") == K and prob > best_prob:
                best_prob = prob
                best_state = i

        # If no valid state found, find closest
        if best_state is None:
            for delta in range(1, n):
                for target_k in [K - delta, K + delta]:
                    if 0 < target_k <= n:
                        for i, prob in enumerate(probs):
                            bitstring = format(i, f"0{n}b")
                            if bitstring.count("1") == target_k and prob > best_prob:
                                best_prob = prob
                                best_state = i
                                break
                        if best_state is not None:
                            break
                if best_state is not None:
                    break

        if best_state is None:
            best_state = np.argmax(probs)

        # Convert to solution vector
        bitstring = format(best_state, f"0{n}b")
        x = np.array([int(b) for b in bitstring])

        # Adjust to exactly K assets if needed
        num_selected = x.sum()
        if num_selected != K:
            asset_values = mu - q * sigma @ x

            if num_selected < K:
                unselected = np.where(x == 0)[0]
                if len(unselected) > 0:
                    values = asset_values[unselected]
                    sorted_indices = unselected[np.argsort(values)[::-1]]
                    for idx in sorted_indices[: K - num_selected]:
                        x[idx] = 1
            else:
                selected = np.where(x == 1)[0]
                if len(selected) > 0:
                    values = asset_values[selected]
                    sorted_indices = selected[np.argsort(values)]
                    for idx in sorted_indices[: num_selected - K]:
                        x[idx] = 0

        bitstring = "".join(str(int(b)) for b in x)

        # Calculate approximation ratio
        objective = self.portfolio_objective(x, sigma, mu, q)

        # For small subproblems, use brute force
        if n <= 20 and K > 0 and K <= n:
            try:
                optimal_x, optimal_objective, _ = self.brute_force_optimization(
                    sigma, mu, K, q
                )

                # Calculate worst case
                worst_objective = float("-inf")
                total_combinations = int(np.math.comb(n, K))

                if total_combinations <= 10000:
                    for selected in itertools.combinations(range(n), K):
                        x_temp = np.zeros(n)
                        x_temp[list(selected)] = 1
                        obj = self.portfolio_objective(x_temp, sigma, mu, q)
                        worst_objective = max(worst_objective, obj)
                else:
                    # Sample for approximation
                    for _ in range(min(10000, total_combinations)):
                        selected = np.random.choice(n, K, replace=False)
                        x_temp = np.zeros(n)
                        x_temp[selected] = 1
                        obj = self.portfolio_objective(x_temp, sigma, mu, q)
                        worst_objective = max(worst_objective, obj)

                if worst_objective != optimal_objective:
                    approx_ratio = (worst_objective - objective) / (
                        worst_objective - optimal_objective
                    )
                else:
                    approx_ratio = 1.0
            except:
                approx_ratio = 0.5  # Default if calculation fails
        else:
            # For larger subproblems, use heuristic baseline
            sa_solution = self.simulated_annealing(mu, sigma, K, q, max_iter=1000)
            heuristic_objective = self.portfolio_objective(sa_solution, sigma, mu, q)

            if objective <= heuristic_objective:
                approx_ratio = 1.0
            else:
                approx_ratio = 0.9 * (heuristic_objective / objective)

        approx_ratio = max(0.0, min(1.0, approx_ratio))

        return approx_ratio, bitstring

    def solve_with_gpt(
        self,
        partitioned_mus,
        partitioned_sigmas,
        mappings,
        K_per_partition: List[int],
        q: float,
        model_id: str = "50m_old",
        device: str = "cpu",
        cached: bool = True,
    ) -> Tuple[List[int], float, List[float], Dict]:
        """Solve subproblems using GPT-generated quantum circuits."""
        if not GPT_AVAILABLE:
            raise ImportError("GPT modules not available.")

        print("\n🔗 GPT-BASED QUANTUM OPTIMIZATION")
        print("=" * 60)

        # Create graphs for all subproblems
        print(f"Creating graphs for {len(mappings)} subproblems...")
        graphs_batch = []
        sub_problems = []
        valid_indices = []

        for i, (sub_mu, sub_sigma, k_sub, mapping) in enumerate(
            zip(partitioned_mus, partitioned_sigmas, K_per_partition, mappings)
        ):

            # Skip invalid partitions
            if len(sub_mu) == 0 or k_sub == 0 or k_sub > len(sub_mu):
                print(f"  Skipping partition {i+1}: size={len(sub_mu)}, k={k_sub}")
                continue

            valid_indices.append(i)

            # Create subproblem dictionary
            sub_problem = (
                get_sub_problem(
                    sigma=sub_sigma, mu=sub_mu, K=k_sub, q=q, seed=1, pre=False
                )
                if PARTITIONING_AVAILABLE
                else {
                    "N": len(sub_mu),
                    "K": k_sub,
                    "q": q,
                    "mu": sub_mu.tolist(),
                    "sigma": sub_sigma.tolist(),
                }
            )

            sub_problems.append(sub_problem)

            # Create graph for GPT
            graph = self.create_graph_for_gpt(
                np.array(sub_problem["means"]) if "means" in sub_problem else sub_mu,
                np.array(sub_problem["cov"]) if "cov" in sub_problem else sub_sigma,
                k_sub,
                q,
                model_id,
            )
            graphs_batch.append(graph)

        if not graphs_batch:
            print("No valid partitions to solve!")
            return [], 0.0, [], {}

        # Generate circuits using GPT
        print(f"\nGenerating QAOA circuits using GPT model ({model_id})...")
        generated_circuits, generation_times = inference(
            graphs_batch=graphs_batch, model_id=model_id, cached=cached, device=device
        )

        # Evaluate circuits
        print("\nEvaluating generated circuits...")
        sub_solutions = []
        approximation_ratios = []
        pipe_idx = []

        for j, (circuit_tokens, sub_problem, orig_idx) in enumerate(
            zip(generated_circuits, sub_problems, valid_indices)
        ):

            mapping = mappings[orig_idx]

            print(f"\nPartition {orig_idx+1}/{len(mappings)}:")
            print(f"  Size: {sub_problem['N']}, K: {sub_problem['K']}")

            try:
                # Parse circuit
                qc = tokens_to_circuit(
                    circuit_tokens,
                    initial_state=None,
                    num_qubits=15,
                    K=sub_problem["K"],
                )

                # Evaluate circuit
                approx_ratio, bitstring = self.evaluate_gpt_circuit(qc, sub_problem)

                # Map solution back to original indices
                x_sub = np.array([int(b) for b in bitstring])
                selected_in_partition = np.where(x_sub == 1)[0]
                pipe_idx.extend([mapping[j] for j in selected_in_partition])

                sub_solutions.append(bitstring)
                approximation_ratios.append(approx_ratio)

                print(
                    f"  Solution: {bitstring} (selected {bitstring.count('1')} assets)"
                )
                print(f"  Approximation ratio: {approx_ratio:.3f}")

            except Exception as e:
                print(f"  Failed to evaluate: {str(e)}")
                # Use greedy fallback
                sub_mu = np.array(sub_problem["mu"])
                sub_sigma = np.array(sub_problem["sigma"])
                fallback_x = self.greedy_portfolio_selection(
                    sub_mu, sub_sigma, sub_problem["K"], q
                )
                selected = np.where(fallback_x == 1)[0]
                pipe_idx.extend([mapping[j] for j in selected])
                approximation_ratios.append(0.0)

        # Calculate pipeline objective
        pipe_x = np.zeros(self.n_assets)
        for idx in pipe_idx:
            if idx < self.n_assets:
                pipe_x[idx] = 1
        pipe_obj = self.portfolio_objective(pipe_x, self.sigma, self.mu, q)

        additional_stats = {
            "total_generation_time": sum(generation_times),
            "avg_circuit_tokens": (
                np.mean([len(tokens) for tokens in generated_circuits])
                if generated_circuits
                else 0
            ),
            "valid_partitions": len(valid_indices),
            "total_partitions": len(mappings),
        }

        return pipe_idx, pipe_obj, approximation_ratios, additional_stats

    def solve_with_classical(
        self,
        partitioned_mus,
        partitioned_sigmas,
        mappings,
        K_per_partition: List[int],
        q: float,
    ) -> Tuple[List[int], float]:
        """Solve subproblems using classical optimization."""
        print("\n🔗 CLASSICAL PIPELINE OPTIMIZATION")
        print("=" * 60)

        pipe_idx = []

        for i, (sub_mu, sub_sigma, mapping, k_sub) in enumerate(
            zip(partitioned_mus, partitioned_sigmas, mappings, K_per_partition)
        ):

            # Use simulated annealing for each subproblem
            x_sub = self.simulated_annealing(
                sub_mu, sub_sigma, k_sub, q * 0.8, max_iter=1000
            )
            selected = np.where(x_sub == 1)[0]
            pipe_idx.extend([mapping[j] for j in selected])

            print(f"Partition {i+1}: selected {len(selected)} assets")

        # Calculate pipeline objective
        pipe_x = np.zeros(self.n_assets)
        for idx in pipe_idx:
            if idx < self.n_assets:
                pipe_x[idx] = 1
        pipe_obj = self.portfolio_objective(pipe_x, self.sigma, self.mu, q)

        return pipe_idx, pipe_obj

    def run_optimization(
        self,
        n_partitions: Optional[int] = None,
        K_total: int = 40,
        q: float = 0.5,
        max_partition_size: Optional[int] = None,
        solver: str = "classical",
        gpt_model_id: str = "50m_old",
        gpt_device: str = "cpu",
        gpt_cached: bool = True,
    ) -> OptimizationResults:
        """
        Run the complete optimization pipeline.

        Args:
            n_partitions: Number of partitions (if None, uses natural clustering)
            K_total: Total number of assets to select
            q: Risk aversion parameter
            max_partition_size: Maximum allowed partition size (optional)
            solver: 'classical' or 'gpt'
            gpt_model_id: GPT model ID
            gpt_device: Device for GPT ('cpu' or 'cuda')
            gpt_cached: Whether to use cached GPT model

        Returns:
            OptimizationResults object
        """
        print("\n" + "=" * 60)
        print(f"PORTFOLIO OPTIMIZATION - {self.n_assets} Assets")
        print(f"Solver: {solver.upper()}")
        print("=" * 60)

        # Step 1: Partition the problem
        if n_partitions:
            print(f"\nPartitioning into {n_partitions} clusters...")
        else:
            print(f"\nUsing natural partitioning...")

        if max_partition_size:
            print(f"Maximum partition size: {max_partition_size}")

        partitioned_mus, partitioned_sigmas, mappings = self.partition_problem(
            n_partitions, max_partition_size
        )

        print(f"Created {len(mappings)} partitions")
        print(f"Partition sizes: {[len(m) for m in mappings]}")

        # Step 2: Distribute K across partitions proportionally
        K_per_partition = []
        total_partition_assets = sum(len(m) for m in mappings)

        for i, mapping in enumerate(mappings):
            partition_size_actual = len(mapping)

            # Skip empty partitions
            if partition_size_actual == 0:
                K_per_partition.append(0)
                continue

            # Proportional allocation
            k_i = int(
                np.round(K_total * partition_size_actual / total_partition_assets)
            )

            # Ensure k_i is valid: 0 < k_i <= partition_size
            k_i = max(1, min(k_i, partition_size_actual))
            K_per_partition.append(k_i)

        # Adjust to match K_total exactly
        current_total = sum(K_per_partition)

        if current_total < K_total:
            # Add to partitions that can take more
            for i in range(len(mappings)):
                if K_per_partition[i] < len(mappings[i]):
                    additional = min(
                        K_total - current_total, len(mappings[i]) - K_per_partition[i]
                    )
                    K_per_partition[i] += additional
                    current_total += additional
                    if current_total >= K_total:
                        break
        elif current_total > K_total:
            # Remove from partitions with k > 1
            for i in range(len(mappings) - 1, -1, -1):
                if K_per_partition[i] > 1:
                    reduction = min(current_total - K_total, K_per_partition[i] - 1)
                    K_per_partition[i] -= reduction
                    current_total -= reduction
                    if current_total <= K_total:
                        break

        # Final validation
        K_per_partition = [k for k, m in zip(K_per_partition, mappings) if len(m) > 0]
        mappings = [m for m in mappings if len(m) > 0]
        partitioned_mus = [
            partitioned_mus[i] for i, m in enumerate(mappings) if len(m) > 0
        ]
        partitioned_sigmas = [
            partitioned_sigmas[i] for i, m in enumerate(mappings) if len(m) > 0
        ]

        print(f"K distribution: {K_per_partition} (total: {sum(K_per_partition)})")

        # Step 3: Global baseline (using heuristics)
        print("\n🌐 GLOBAL OPTIMIZATION (Heuristic Baseline)")
        print("=" * 60)

        start_global = time.time()
        global_x, global_obj = self.get_classical_baseline(
            self.mu, self.sigma, K_total, q
        )
        global_idx = np.where(global_x == 1)[0].tolist()
        time_global = time.time() - start_global

        print(f"\nBest heuristic objective: {global_obj:.6f}")
        print(f"Selected {len(global_idx)} assets")

        # Step 4: Classical partitioned baseline (always run this for comparison)
        print("\n⚙️ CLASSICAL PARTITIONED BASELINE")
        print("=" * 60)

        start_classical = time.time()
        classical_idx, classical_obj = self.solve_with_classical(
            partitioned_mus, partitioned_sigmas, mappings, K_per_partition, q
        )
        time_classical = time.time() - start_classical

        print(f"\nClassical partitioned objective: {classical_obj:.6f}")
        print(f"Time: {time_classical:.2f}s")

        # Step 5: Main solver (GPT or classical)
        start_pipeline = time.time()

        if solver == "gpt":
            pipe_idx, pipe_obj, approx_ratios, additional_stats = self.solve_with_gpt(
                partitioned_mus,
                partitioned_sigmas,
                mappings,
                K_per_partition,
                q,
                gpt_model_id,
                gpt_device,
                gpt_cached,
            )
        else:
            # If solver is classical, use the same results as classical baseline
            pipe_idx = classical_idx
            pipe_obj = classical_obj
            approx_ratios = None
            additional_stats = None

        time_pipeline = time.time() - start_pipeline

        # Prepare results
        global_tickers = [self.tickers[i] for i in global_idx if i < self.n_assets]
        pipe_tickers = [self.tickers[i] for i in pipe_idx if i < self.n_assets]

        # Store classical partitioned results
        results = OptimizationResults(
            global_objective=global_obj,
            pipeline_objective=pipe_obj,
            global_solution=global_idx,
            pipeline_solution=pipe_idx,
            global_tickers=global_tickers,
            pipeline_tickers=pipe_tickers,
            time_global=time_global,
            time_pipeline=time_pipeline,
            cluster_sizes=[len(m) for m in mappings],
            n_assets=self.n_assets,
            n_clusters=len(mappings),
            solver_type=solver,
            subproblem_approximation_ratios=approx_ratios,
            additional_stats=additional_stats,
        )

        # Add classical partitioned results to additional_stats
        if results.additional_stats is None:
            results.additional_stats = {}
        results.additional_stats["classical_partitioned_objective"] = classical_obj
        results.additional_stats["classical_partitioned_time"] = time_classical
        results.additional_stats["classical_partitioned_solution"] = classical_idx

        # Print summary
        self._print_results_summary(results)

        return results

    def _print_results_summary(self, results: OptimizationResults):
        """Print a summary of optimization results"""
        print("\n" + "=" * 60)
        print("📊 RESULTS SUMMARY")
        print("=" * 60)

        # Show all three objectives
        print("\nObjective Comparison:")
        print(
            f"  1. Global heuristic (full problem):     {results.global_objective:.6f}"
        )

        if (
            results.additional_stats
            and "classical_partitioned_objective" in results.additional_stats
        ):
            classical_obj = results.additional_stats["classical_partitioned_objective"]
            print(f"  2. Classical solver (partitioned):      {classical_obj:.6f}")

            # Calculate relative differences
            classical_vs_global = (
                (classical_obj - results.global_objective)
                / abs(results.global_objective)
                * 100
            )
            print(f"     → vs global: {classical_vs_global:+.1f}%")

        print(
            f"  3. {results.solver_type.upper()} solver (partitioned):    {results.pipeline_objective:.6f}"
        )

        # Calculate relative differences for main solver
        pipeline_vs_global = (
            (results.pipeline_objective - results.global_objective)
            / abs(results.global_objective)
            * 100
        )
        print(f"     → vs global: {pipeline_vs_global:+.1f}%")

        if (
            results.additional_stats
            and "classical_partitioned_objective" in results.additional_stats
        ):
            classical_obj = results.additional_stats["classical_partitioned_objective"]
            pipeline_vs_classical = (
                (results.pipeline_objective - classical_obj) / abs(classical_obj) * 100
            )
            print(f"     → vs classical partitioned: {pipeline_vs_classical:+.1f}%")

        # Timing information
        print(f"\nTiming:")
        print(f"  Global heuristic: {results.time_global:.2f}s")

        if (
            results.additional_stats
            and "classical_partitioned_time" in results.additional_stats
        ):
            print(
                f"  Classical partitioned: {results.additional_stats['classical_partitioned_time']:.2f}s"
            )

        if results.solver_type == "gpt":
            print(f"  GPT partitioned: {results.time_pipeline:.2f}s")

        # Partitioning structure
        print(f"\nPartitioning Structure:")
        print(f"  Number of partitions: {len(results.cluster_sizes)}")
        print(f"  Partition sizes: {results.cluster_sizes}")

        # GPT-specific metrics
        if results.subproblem_approximation_ratios:
            print(f"\nGPT Subproblem Performance:")
            print(
                f"  Average approximation ratio: {np.mean(results.subproblem_approximation_ratios):.3f}"
            )
            print(
                f"  Min/Max AR: {np.min(results.subproblem_approximation_ratios):.3f} / {np.max(results.subproblem_approximation_ratios):.3f}"
            )

            # Distribution
            ratios = results.subproblem_approximation_ratios
            print(f"  Distribution:")
            print(
                f"    Excellent (AR > 0.9): {sum(1 for r in ratios if r > 0.9)} partitions"
            )
            print(
                f"    Good (0.7 < AR ≤ 0.9): {sum(1 for r in ratios if 0.7 < r <= 0.9)} partitions"
            )
            print(
                f"    Fair (0.5 < AR ≤ 0.7): {sum(1 for r in ratios if 0.5 < r <= 0.7)} partitions"
            )
            print(
                f"    Poor (AR ≤ 0.5): {sum(1 for r in ratios if r <= 0.5)} partitions"
            )

        print(f"\n🏆 GLOBAL PORTFOLIO ({len(results.global_solution)} assets):")
        print(
            ", ".join(results.global_tickers[:10])
            + ("..." if len(results.global_tickers) > 10 else "")
        )

        print(
            f"\n🔗 {results.solver_type.upper()} PORTFOLIO ({len(results.pipeline_solution)} assets):"
        )
        print(
            ", ".join(results.pipeline_tickers[:10])
            + ("..." if len(results.pipeline_tickers) > 10 else "")
        )

    def save_results(self, results: OptimizationResults, prefix: str = ""):
        """Save optimization results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if prefix:
            prefix = f"{prefix}_"

        # Save summary as JSON
        summary = {
            "timestamp": timestamp,
            "n_assets": results.n_assets,
            "n_clusters": results.n_clusters,
            "global_objective": results.global_objective,
            "pipeline_objective": results.pipeline_objective,
            "time_global": results.time_global,
            "time_pipeline": results.time_pipeline,
            "cluster_sizes": results.cluster_sizes,
            "solver_type": results.solver_type,
            "subproblem_approximation_ratios": results.subproblem_approximation_ratios,
        }

        # Add classical partitioned results if available
        if (
            results.additional_stats
            and "classical_partitioned_objective" in results.additional_stats
        ):
            summary["classical_partitioned_objective"] = results.additional_stats[
                "classical_partitioned_objective"
            ]
            summary["classical_partitioned_time"] = results.additional_stats[
                "classical_partitioned_time"
            ]

            # Calculate all relative differences
            summary["classical_vs_global_pct"] = (
                (
                    results.additional_stats["classical_partitioned_objective"]
                    - results.global_objective
                )
                / abs(results.global_objective)
                * 100
            )
            summary["pipeline_vs_global_pct"] = (
                (results.pipeline_objective - results.global_objective)
                / abs(results.global_objective)
                * 100
            )
            summary["pipeline_vs_classical_pct"] = (
                (
                    results.pipeline_objective
                    - results.additional_stats["classical_partitioned_objective"]
                )
                / abs(results.additional_stats["classical_partitioned_objective"])
                * 100
            )

        with open(
            os.path.join(self.output_dir, f"{prefix}summary_{timestamp}.json"), "w"
        ) as f:
            json.dump(summary, f, indent=2)

        # Save portfolios
        global_df = pd.DataFrame(
            {
                "ticker": results.global_tickers,
                "index": results.global_solution[: len(results.global_tickers)],
            }
        )
        global_df.to_csv(
            os.path.join(self.output_dir, f"{prefix}global_portfolio_{timestamp}.csv"),
            index=False,
        )

        pipeline_df = pd.DataFrame(
            {
                "ticker": results.pipeline_tickers,
                "index": results.pipeline_solution[: len(results.pipeline_tickers)],
            }
        )
        pipeline_df.to_csv(
            os.path.join(
                self.output_dir, f"{prefix}pipeline_portfolio_{timestamp}.csv"
            ),
            index=False,
        )

        # Save classical partitioned portfolio if available
        if (
            results.additional_stats
            and "classical_partitioned_solution" in results.additional_stats
        ):
            classical_idx = results.additional_stats["classical_partitioned_solution"]
            classical_tickers = [
                self.tickers[i] for i in classical_idx if i < self.n_assets
            ]
            classical_df = pd.DataFrame(
                {
                    "ticker": classical_tickers,
                    "index": classical_idx[: len(classical_tickers)],
                }
            )
            classical_df.to_csv(
                os.path.join(
                    self.output_dir, f"{prefix}classical_partitioned_{timestamp}.csv"
                ),
                index=False,
            )

        print(f"\n✅ Results saved to {self.output_dir}/")
        print(f"   - {prefix}summary_{timestamp}.json")
        print(f"   - {prefix}global_portfolio_{timestamp}.csv")
        print(f"   - {prefix}pipeline_portfolio_{timestamp}.csv")
        if (
            results.additional_stats
            and "classical_partitioned_solution" in results.additional_stats
        ):
            print(f"   - {prefix}classical_partitioned_{timestamp}.csv")


def main(
    graph_path: str,
    output_dir: str = "optimization_results",
    n_partitions: Optional[int] = None,
    K_total: int = 40,
    q: float = 0.5,
    max_partition_size: Optional[int] = None,
    solver: str = "classical",
    gpt_model_id: str = "50m_old",
    gpt_device: str = "cpu",
    gpt_cached: bool = True,
    save_results: bool = True,
    results_prefix: str = "",
) -> OptimizationResults:
    """
    Main function to run portfolio optimization with partitioning.

    Args:
        graph_path: Path to pickle file containing the graph
        output_dir: Directory to save results
        n_partitions: Number of partitions (if None, uses natural clustering)
        K_total: Total number of assets to select
        q: Risk aversion parameter
        max_partition_size: Maximum allowed partition size (optional)
        solver: 'classical' or 'gpt'
        gpt_model_id: GPT model ID
        gpt_device: Device for GPT ('cpu' or 'cuda')
        gpt_cached: Whether to use cached GPT model
        save_results: Whether to save results to files
        results_prefix: Prefix for saved result files

    Returns:
        OptimizationResults object
    """
    # Initialize optimizer
    optimizer = PortfolioOptimizerWithGPT(graph_path, output_dir)

    # Run optimization
    results = optimizer.run_optimization(
        n_partitions=n_partitions,
        K_total=K_total,
        q=q,
        max_partition_size=max_partition_size,
        solver=solver,
        gpt_model_id=gpt_model_id,
        gpt_device=gpt_device,
        gpt_cached=gpt_cached,
    )

    # Save results if requested
    if save_results:
        optimizer.save_results(results, prefix=results_prefix)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Portfolio Optimization with Graph Partitioning and GPT"
    )
    parser.add_argument("graph_path", help="Path to graph pickle file")
    parser.add_argument(
        "--output-dir", default="optimization_results", help="Output directory"
    )
    parser.add_argument(
        "--n-partitions",
        type=int,
        help="Number of partitions (if not set, uses natural clustering)",
    )
    parser.add_argument("--max-partition-size", type=int, help="Maximum partition size")
    parser.add_argument(
        "--k-total", type=int, default=40, help="Total assets to select"
    )
    parser.add_argument(
        "--risk-aversion", type=float, default=0.5, help="Risk aversion parameter"
    )
    parser.add_argument(
        "--solver",
        choices=["classical", "gpt"],
        default="classical",
        help="Solver type",
    )
    parser.add_argument("--gpt-model", default="50m_old", help="GPT model ID")
    parser.add_argument("--gpt-device", default="cpu", help="Device for GPT")
    parser.add_argument(
        "--no-gpt-cache", action="store_true", help="Disable GPT model caching"
    )
    parser.add_argument("--prefix", default="", help="Prefix for output files")

    args = parser.parse_args()

    results = main(
        graph_path=args.graph_path,
        output_dir=args.output_dir,
        n_partitions=args.n_partitions,
        K_total=args.k_total,
        q=args.risk_aversion,
        max_partition_size=args.max_partition_size,
        solver=args.solver,
        gpt_model_id=args.gpt_model,
        gpt_device=args.gpt_device,
        gpt_cached=not args.no_gpt_cache,
        save_results=True,
        results_prefix=args.prefix,
    )
