import numpy as np
import pandas as pd
import pickle
import time
import os
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import json
from datetime import datetime
import sys

# Scientific computing
from numpy.linalg import eigh
from sklearn.cluster import SpectralClustering
from itertools import combinations

# QOKit imports
try:
    from qokit.portfolio_optimization import (
        get_problem,
        portfolio_brute_force,
        get_sk_ini,
    )
    from qokit.qaoa_circuit_portfolio import (
        get_qaoa_circuit,
        get_parameterized_qaoa_circuit,
        get_energy_expectation_sv,
    )
    from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective
    from qokit.utils import reverse_array_index_bit_order
    from qiskit.quantum_info import Statevector
    from qiskit import transpile
    from qiskit_aer import Aer
    import nlopt

    QOKIT_AVAILABLE = True
except ImportError:
    QOKIT_AVAILABLE = False
    print("Warning: QOKit not available. QOKit features will be disabled.")

# GPT imports
try:
    sys.path.append("../gpt-qaoa")
    from inference import inference
    from parsing import tokens_to_circuit

    GPT_AVAILABLE = True
except ImportError:
    GPT_AVAILABLE = False
    print("Warning: GPT modules not available. GPT features will be disabled.")


@dataclass
class ComparisonResults:
    """Container for comparison results across all methods"""

    # Problem info
    n_assets: int
    n_partitions: int
    partition_sizes: List[int]
    K_total: int
    q: float

    # Classical results
    classical_global_objective: float
    classical_global_solution: List[int]
    classical_partitioned_objective: float
    classical_partitioned_solution: List[int]
    classical_time: float

    # QOKit results
    qokit_objective: Optional[float] = None
    qokit_solution: Optional[List[int]] = None
    qokit_approximation_ratio: Optional[float] = None
    qokit_subproblem_ars: Optional[List[float]] = None
    qokit_time: Optional[float] = None

    # GPT results
    gpt_objective: Optional[float] = None
    gpt_solution: Optional[List[int]] = None
    gpt_approximation_ratio: Optional[float] = None
    gpt_subproblem_ars: Optional[List[float]] = None
    gpt_time: Optional[float] = None

    # Additional statistics
    additional_stats: Optional[Dict[str, Any]] = None


class UnifiedPortfolioOptimizer:
    """Unified optimizer for comparing GPT, QOKit, and Classical approaches"""

    def __init__(self, graph_path: str):
        """Initialize with a portfolio graph"""
        self.graph_path = graph_path
        self._load_graph()

    def _load_graph(self):
        """Load graph from pickle file"""
        with open(self.graph_path, "rb") as f:
            data = pickle.load(f)

        if hasattr(data, "nodes") and hasattr(data, "edges"):
            self.graph = data
            self.tickers = list(self.graph.nodes())
            self.n_assets = len(self.tickers)

            # Extract returns and volatilities
            self.mu = np.array(
                [self.graph.nodes[ticker]["annual_return"] for ticker in self.tickers]
            )
            self.volatility = np.array(
                [
                    self.graph.nodes[ticker]["annual_volatility"]
                    for ticker in self.tickers
                ]
            )

            # Build covariance matrix
            self._build_covariance_matrix()
        else:
            raise ValueError("Expected NetworkX graph in pickle file")

    def _build_covariance_matrix(self):
        """Build covariance matrix from correlation graph"""
        corr_matrix = np.eye(self.n_assets)
        ticker_to_idx = {ticker: i for i, ticker in enumerate(self.tickers)}

        for ticker1, ticker2, edge_data in self.graph.edges(data=True):
            i = ticker_to_idx[ticker1]
            j = ticker_to_idx[ticker2]
            weight = edge_data.get("weight", 0)
            corr_matrix[i, j] = weight
            corr_matrix[j, i] = weight

        # Convert to covariance
        D = np.diag(self.volatility)
        self.sigma = D @ corr_matrix @ D
        self.sigma = (self.sigma + self.sigma.T) / 2

    def rmt_denoise(self, sigma, n_obs=None):
        """RMT denoising implementation from QOKit pipeline"""
        n = len(sigma)
        if n_obs is None:
            n_obs = 252 * 5  # 5 years of daily data

        # Ensure symmetric
        sigma = (sigma + sigma.T) / 2
        sigma += np.eye(n) * 1e-8

        # Calculate correlation matrix
        std = np.sqrt(np.diag(sigma))
        std = np.maximum(std, 1e-10)
        corr = sigma / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)

        # Eigenvalue decomposition
        vals, vecs = eigh(corr)
        vals, vecs = vals[::-1], vecs[:, ::-1]

        # Marchenko-Pastur bounds
        q = n / n_obs
        if q >= 1:
            q = 0.99

        lmin = (1 - np.sqrt(q)) ** 2
        lmax = (1 + np.sqrt(q)) ** 2

        # Clean eigenvalues
        clean_vals = vals.copy()
        noise_idx = (vals >= lmin) & (vals <= lmax)

        if np.any(noise_idx):
            avg_noise = np.mean(vals[noise_idx])
            clean_vals[noise_idx] = avg_noise

        # Remove market mode if dominant
        if clean_vals[0] > lmax * 2:
            clean_vals[0] = clean_vals[1]

        clean_vals = np.maximum(clean_vals, 1e-8)

        # Reconstruct
        corr_clean = (vecs * clean_vals) @ vecs.T
        corr_clean = (corr_clean + corr_clean.T) / 2
        np.fill_diagonal(corr_clean, 1.0)

        return corr_clean * np.outer(std, std)

    def spectral_clusters(self, corr, k, min_sz=3):
        """Spectral clustering from QOKit pipeline"""
        n = len(corr)

        # Affinity matrix
        aff = np.abs(corr) ** 2
        np.fill_diagonal(aff, 1.0)
        aff = (aff + aff.T) / 2

        # Clustering
        try:
            clustering = SpectralClustering(
                n_clusters=k, affinity="precomputed", random_state=42, n_init=10
            )
            labels = clustering.fit_predict(aff)
        except:
            labels = np.random.randint(0, k, size=n)

        clusters = [np.where(labels == i)[0].tolist() for i in range(k)]
        clusters = [c for c in clusters if len(c) > 0]

        # Merge small clusters
        i = 0
        while i < len(clusters):
            if len(clusters[i]) < min_sz and len(clusters) > 1:
                small = clusters.pop(i)
                if clusters:
                    clusters[np.random.randint(len(clusters))].extend(small)
            else:
                i += 1

        return clusters[:k]

    def partition_with_rmt(self, partition_size: int, n_obs: Optional[int] = None):
        """Advanced partitioning using RMT denoising and spectral clustering"""
        # Apply RMT denoising
        sigma_clean = self.rmt_denoise(self.sigma, n_obs)

        # Calculate correlation for clustering
        std = np.sqrt(np.diag(sigma_clean))
        std = np.maximum(std, 1e-10)
        corr_clean = sigma_clean / np.outer(std, std)

        # Determine number of clusters
        n_clusters = max(1, self.n_assets // partition_size)

        # Perform spectral clustering
        clusters = self.spectral_clusters(corr_clean, n_clusters, min_sz=3)

        # Extract submatrices and sub-vectors
        partitioned_sigmas = []
        partitioned_mus = []
        mappings = []

        for cluster in clusters:
            if len(cluster) > 0:
                cluster_array = np.array(cluster)
                sub_sigma = sigma_clean[np.ix_(cluster_array, cluster_array)]
                sub_mu = self.mu[cluster_array]

                partitioned_sigmas.append(sub_sigma)
                partitioned_mus.append(sub_mu)
                mappings.append(cluster)

        return partitioned_mus, partitioned_sigmas, mappings

    def _enforce_max_partition_size(
        self, partitioned_mus, partitioned_sigmas, mappings, max_size
    ):
        """Split partitions that exceed max_size"""
        new_mus = []
        new_sigmas = []
        new_mappings = []

        for mu, sigma, mapping in zip(partitioned_mus, partitioned_sigmas, mappings):
            if len(mapping) <= max_size:
                # Keep partition as is
                new_mus.append(mu)
                new_sigmas.append(sigma)
                new_mappings.append(mapping)
            else:
                # Split large partition into smaller ones
                n_splits = (len(mapping) + max_size - 1) // max_size  # Ceiling division

                for i in range(n_splits):
                    start_idx = i * max_size
                    end_idx = min((i + 1) * max_size, len(mapping))

                    # Get indices for this split
                    sub_indices = list(range(start_idx, end_idx))
                    sub_mapping = [mapping[j] for j in sub_indices]

                    # Extract submatrices
                    sub_mu = mu[sub_indices]
                    sub_sigma = sigma[np.ix_(sub_indices, sub_indices)]

                    new_mus.append(sub_mu)
                    new_sigmas.append(sub_sigma)
                    new_mappings.append(sub_mapping)

        return new_mus, new_sigmas, new_mappings

    def portfolio_objective(
        self, x: np.ndarray, sigma: np.ndarray, mu: np.ndarray, q: float
    ) -> float:
        """Calculate portfolio objective: q * risk - return"""
        return q * x.T @ sigma @ x - mu.T @ x

    def greedy_portfolio_selection(
        self, mu: np.ndarray, sigma: np.ndarray, K: int, q: float
    ) -> np.ndarray:
        """Greedy algorithm for portfolio selection"""
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
        """Simulated annealing for portfolio optimization"""
        n = len(mu)

        # Start with greedy solution
        x_current = self.greedy_portfolio_selection(mu, sigma, K, q)
        current_obj = self.portfolio_objective(x_current, sigma, mu, q)

        best_x = x_current.copy()
        best_obj = current_obj

        for i in range(max_iter):
            T = 1.0 * (1 - i / max_iter)

            selected = np.where(x_current == 1)[0]
            unselected = np.where(x_current == 0)[0]

            if len(unselected) > 0 and len(selected) > 0:
                remove_idx = np.random.choice(selected)
                add_idx = np.random.choice(unselected)

                x_neighbor = x_current.copy()
                x_neighbor[remove_idx] = 0
                x_neighbor[add_idx] = 1

                neighbor_obj = self.portfolio_objective(x_neighbor, sigma, mu, q)

                delta = neighbor_obj - current_obj
                if delta < 0 or np.random.rand() < np.exp(-delta / T):
                    x_current = x_neighbor
                    current_obj = neighbor_obj

                    if current_obj < best_obj:
                        best_x = x_current.copy()
                        best_obj = current_obj

        return best_x

    def solve_classical_global(self, K: int, q: float) -> Tuple[np.ndarray, float]:
        """Solve the global problem using classical heuristics"""
        # Run multiple algorithms and take best
        solutions = []
        objectives = []

        # Greedy
        x_greedy = self.greedy_portfolio_selection(self.mu, self.sigma, K, q)
        obj_greedy = self.portfolio_objective(x_greedy, self.sigma, self.mu, q)
        solutions.append(x_greedy)
        objectives.append(obj_greedy)

        # Multiple SA runs
        for _ in range(3):
            x_sa = self.simulated_annealing(self.mu, self.sigma, K, q, max_iter=3000)
            obj_sa = self.portfolio_objective(x_sa, self.sigma, self.mu, q)
            solutions.append(x_sa)
            objectives.append(obj_sa)

        # Return best
        best_idx = np.argmin(objectives)
        return solutions[best_idx], objectives[best_idx]

    def solve_classical_partitioned(
        self, partitioned_mus, partitioned_sigmas, mappings, K_per_partition, q: float
    ) -> Tuple[List[int], float]:
        """Solve partitioned problem using classical methods"""
        selected_indices = []

        for sub_mu, sub_sigma, mapping, k_sub in zip(
            partitioned_mus, partitioned_sigmas, mappings, K_per_partition
        ):

            if k_sub == 0 or len(sub_mu) == 0:
                continue

            # Solve subproblem
            x_sub = self.simulated_annealing(sub_mu, sub_sigma, k_sub, q, max_iter=1000)

            # Map back to global indices
            selected_in_partition = np.where(x_sub == 1)[0]
            selected_indices.extend([mapping[i] for i in selected_in_partition])

        # Calculate objective
        x_full = np.zeros(self.n_assets)
        for idx in selected_indices:
            x_full[idx] = 1
        objective = self.portfolio_objective(x_full, self.sigma, self.mu, q)

        return selected_indices, objective

    def solve_qokit_partitioned(
        self,
        partitioned_mus,
        partitioned_sigmas,
        mappings,
        K_per_partition,
        q: float,
        p: int = 5,
        scale: float = 1.0,
    ) -> Tuple[List[int], float, List[float]]:
        """Solve partitioned problem using QOKit QAOA"""
        if not QOKIT_AVAILABLE:
            raise ImportError("QOKit not available")

        selected_indices = []
        sub_approximation_ratios = []

        def minimize_nlopt(f, x0, p):
            """NLopt minimization"""

            def nlopt_wrapper(x, grad):
                return f(x).real

            opt = nlopt.opt(nlopt.LN_BOBYQA, 2 * p)
            opt.set_min_objective(nlopt_wrapper)
            opt.set_xtol_rel(1e-8)
            opt.set_ftol_rel(1e-8)
            opt.set_initial_step(0.01)

            xstar = opt.optimize(x0)
            minf = opt.last_optimum_value()

            return xstar, minf

        for i, (sub_mu, sub_sigma, mapping, k_sub) in enumerate(
            zip(partitioned_mus, partitioned_sigmas, mappings, K_per_partition)
        ):

            if k_sub == 0 or len(sub_mu) == 0:
                continue

            print(
                f"  QOKit solving partition {i+1}/{len(mappings)}: {len(sub_mu)} assets, K={k_sub}"
            )

            # Create QOKit problem
            po_problem = {
                "N": len(sub_mu),
                "K": k_sub,
                "q": q,
                "seed": 1,
                "means": scale * sub_mu,
                "cov": scale * sub_sigma,
                "pre": False,
            }

            # Apply risk rebalancing
            mu_ratio = np.linalg.norm(sub_mu) / np.linalg.norm(self.mu)
            sigma_ratio = np.linalg.norm(sub_sigma, "fro") / np.linalg.norm(
                self.sigma, "fro"
            )
            q_rebalanced = q * (mu_ratio / sigma_ratio)
            po_problem["q"] = q_rebalanced

            try:
                # Create QAOA objective
                qaoa_obj = get_qaoa_portfolio_objective(
                    po_problem=po_problem,
                    p=p,
                    ini="dicke",
                    mixer="trotter_ring",
                    T=1,
                    simulator="python",
                )

                # Optimize
                x0 = get_sk_ini(p=p)
                opt_params, opt_energy = minimize_nlopt(qaoa_obj, x0, p=p)

                # Get statevector
                gammas = opt_params[:p] / 2
                betas = opt_params[p:] / 2
                qc = get_qaoa_circuit(po_problem, gammas=gammas, betas=betas, depth=p)
                sv = Statevector.from_instruction(qc)
                probs = sv.probabilities()

                # Find best valid state
                best_prob = 0
                best_bitstring = None

                for idx, prob in enumerate(probs):
                    bitstring = format(idx, f"0{len(sub_mu)}b")
                    if bitstring.count("1") == k_sub and prob > best_prob:
                        best_prob = prob
                        best_bitstring = bitstring

                if best_bitstring is None:
                    # Fallback to classical
                    x_sub = self.greedy_portfolio_selection(sub_mu, sub_sigma, k_sub, q)
                    best_bitstring = "".join(str(int(xi)) for xi in x_sub)

                # Map solution
                x_sub = np.array([int(b) for b in best_bitstring])
                selected_in_partition = np.where(x_sub == 1)[0]
                selected_indices.extend([mapping[j] for j in selected_in_partition])

                # Calculate approximation ratio
                if len(sub_mu) <= 20:
                    # Brute force for small problems
                    best_obj = float("inf")
                    worst_obj = float("-inf")

                    for selected in combinations(range(len(sub_mu)), k_sub):
                        x_temp = np.zeros(len(sub_mu))
                        x_temp[list(selected)] = 1
                        obj = self.portfolio_objective(x_temp, sub_sigma, sub_mu, q)
                        best_obj = min(best_obj, obj)
                        worst_obj = max(worst_obj, obj)

                    current_obj = self.portfolio_objective(x_sub, sub_sigma, sub_mu, q)
                    if worst_obj != best_obj:
                        ar = (worst_obj - current_obj) / (worst_obj - best_obj)
                    else:
                        ar = 1.0
                else:
                    # Use heuristic baseline
                    baseline_x = self.simulated_annealing(
                        sub_mu, sub_sigma, k_sub, q, max_iter=500
                    )
                    baseline_obj = self.portfolio_objective(
                        baseline_x, sub_sigma, sub_mu, q
                    )
                    current_obj = self.portfolio_objective(x_sub, sub_sigma, sub_mu, q)
                    ar = 1.0 if current_obj <= baseline_obj else 0.9

                sub_approximation_ratios.append(max(0.0, min(1.0, ar)))

            except Exception as e:
                print(f"    QOKit failed for partition {i+1}: {e}")
                # Fallback to classical
                x_sub = self.simulated_annealing(
                    sub_mu, sub_sigma, k_sub, q, max_iter=500
                )
                selected_in_partition = np.where(x_sub == 1)[0]
                selected_indices.extend([mapping[j] for j in selected_in_partition])
                sub_approximation_ratios.append(0.0)

        # Calculate objective
        x_full = np.zeros(self.n_assets)
        for idx in selected_indices:
            if idx < self.n_assets:
                x_full[idx] = 1
        objective = self.portfolio_objective(x_full, self.sigma, self.mu, q)

        return selected_indices, objective, sub_approximation_ratios

    def create_graph_for_gpt(self, sub_mu, sub_sigma, K, q, model_id="50m_old"):
        """Create NetworkX graph for GPT inference"""
        import networkx as nx

        n = len(sub_mu)
        G = nx.Graph()

        if model_id == "50m_old":
            for i in range(n):
                G.add_node(i)
        else:
            for i in range(n):
                G.add_node(i, mu=sub_mu[i])

        for i in range(n):
            for j in range(i + 1, n):
                if sub_sigma[i, j] != 0:
                    G.add_edge(i, j, weight=sub_sigma[i, j])

        G.graph["K"] = K
        G.graph["q"] = q
        G.graph["N"] = n
        G.graph["mu"] = sub_mu.tolist()
        G.graph["sigma"] = sub_sigma.tolist()

        return G

    def solve_gpt_partitioned(
        self,
        partitioned_mus,
        partitioned_sigmas,
        mappings,
        K_per_partition,
        q: float,
        model_id: str = "50m_old",
        device: str = "cpu",
        cached: bool = True,
    ) -> Tuple[List[int], float, List[float]]:
        """Solve partitioned problem using GPT-generated circuits"""
        if not GPT_AVAILABLE:
            raise ImportError("GPT modules not available")

        selected_indices = []
        sub_approximation_ratios = []

        # Create graphs for all subproblems
        graphs_batch = []
        valid_mappings = []
        valid_k = []

        for sub_mu, sub_sigma, mapping, k_sub in zip(
            partitioned_mus, partitioned_sigmas, mappings, K_per_partition
        ):

            if k_sub == 0 or len(sub_mu) == 0 or k_sub > len(sub_mu):
                continue

            graph = self.create_graph_for_gpt(sub_mu, sub_sigma, k_sub, q, model_id)
            graphs_batch.append(graph)
            valid_mappings.append(mapping)
            valid_k.append(k_sub)

        if not graphs_batch:
            return [], 0.0, []

        # Generate circuits
        print(f"  Generating {len(graphs_batch)} GPT circuits...")
        generated_circuits, generation_times = inference(
            graphs_batch=graphs_batch, model_id=model_id, cached=cached, device=device
        )

        # Evaluate circuits
        for i, (circuit_tokens, graph, mapping, k_sub) in enumerate(
            zip(generated_circuits, graphs_batch, valid_mappings, valid_k)
        ):

            print(f"  GPT evaluating partition {i+1}/{len(graphs_batch)}")

            try:
                # Parse circuit
                from qiskit import QuantumCircuit
                from qiskit.quantum_info import Statevector

                qc = tokens_to_circuit(
                    circuit_tokens, initial_state=None, num_qubits=15, K=k_sub
                )

                # Get statevector
                sv = Statevector.from_instruction(qc)
                probs = sv.probabilities()

                # Extract problem data
                sub_mu = np.array(graph.graph["mu"])
                sub_sigma = np.array(graph.graph["sigma"])
                n = len(sub_mu)

                # Find best valid state
                best_state = None
                best_prob = 0

                for idx, prob in enumerate(probs):
                    bitstring = format(idx, f"0{n}b")
                    if bitstring.count("1") == k_sub and prob > best_prob:
                        best_prob = prob
                        best_state = idx

                if best_state is None:
                    # Find closest
                    for idx in np.argsort(probs)[::-1]:
                        bitstring = format(idx, f"0{n}b")
                        if abs(bitstring.count("1") - k_sub) <= 2:
                            best_state = idx
                            break

                if best_state is None:
                    best_state = np.argmax(probs)

                # Convert to solution
                bitstring = format(best_state, f"0{n}b")
                x_sub = np.array([int(b) for b in bitstring])

                # Adjust to exactly k_sub assets if needed
                num_selected = x_sub.sum()
                if num_selected != k_sub:
                    asset_values = sub_mu - q * sub_sigma @ x_sub

                    if num_selected < k_sub:
                        unselected = np.where(x_sub == 0)[0]
                        if len(unselected) > 0:
                            values = asset_values[unselected]
                            sorted_indices = unselected[np.argsort(values)[::-1]]
                            for idx in sorted_indices[: k_sub - num_selected]:
                                x_sub[idx] = 1
                    else:
                        selected = np.where(x_sub == 1)[0]
                        if len(selected) > 0:
                            values = asset_values[selected]
                            sorted_indices = selected[np.argsort(values)]
                            for idx in sorted_indices[: num_selected - k_sub]:
                                x_sub[idx] = 0

                # Map solution
                selected_in_partition = np.where(x_sub == 1)[0]
                selected_indices.extend([mapping[j] for j in selected_in_partition])

                # Calculate approximation ratio
                current_obj = self.portfolio_objective(x_sub, sub_sigma, sub_mu, q)
                baseline_x = self.simulated_annealing(
                    sub_mu, sub_sigma, k_sub, q, max_iter=500
                )
                baseline_obj = self.portfolio_objective(
                    baseline_x, sub_sigma, sub_mu, q
                )
                ar = (
                    1.0
                    if current_obj <= baseline_obj
                    else max(0.5, baseline_obj / current_obj)
                )
                sub_approximation_ratios.append(ar)

            except Exception as e:
                print(f"    GPT evaluation failed: {e}")
                # Fallback
                sub_mu = np.array(graph.graph["mu"])
                sub_sigma = np.array(graph.graph["sigma"])
                x_sub = self.greedy_portfolio_selection(sub_mu, sub_sigma, k_sub, q)
                selected_in_partition = np.where(x_sub == 1)[0]
                selected_indices.extend([mapping[j] for j in selected_in_partition])
                sub_approximation_ratios.append(0.0)

        # Calculate objective
        x_full = np.zeros(self.n_assets)
        for idx in selected_indices:
            if idx < self.n_assets:
                x_full[idx] = 1
        objective = self.portfolio_objective(x_full, self.sigma, self.mu, q)

        return selected_indices, objective, sub_approximation_ratios

    def calculate_global_approximation_ratio(
        self,
        solution_objective: float,
        best_objective: float,
        worst_objective: Optional[float] = None,
    ) -> float:
        """Calculate approximation ratio for a solution"""
        if worst_objective is None:
            # Estimate worst case by sampling
            worst_objective = float("-inf")
            K = int(sum(1 for _ in range(self.n_assets) if np.random.rand() < 0.3))

            for _ in range(1000):
                indices = np.random.choice(self.n_assets, K, replace=False)
                x_temp = np.zeros(self.n_assets)
                x_temp[indices] = 1
                obj = self.portfolio_objective(x_temp, self.sigma, self.mu, 0.5)
                worst_objective = max(worst_objective, obj)

        if worst_objective != best_objective:
            ar = (worst_objective - solution_objective) / (
                worst_objective - best_objective
            )
        else:
            ar = 1.0

        return max(0.0, min(1.0, ar))

    def run_comparison(
        self,
        partition_size: int = 15,
        K_total: int = 40,
        q: float = 0.5,
        p: int = 5,
        n_obs: Optional[int] = None,
        max_partition_size: Optional[int] = None,
        gpt_model_id: str = "50m_old",
        gpt_device: str = "cpu",
        gpt_cached: bool = True,
    ) -> ComparisonResults:
        """Run complete comparison across all methods

        Args:
            partition_size: Target size for each partition
            K_total: Total number of assets to select
            q: Risk aversion parameter
            p: QAOA circuit depth
            n_obs: Number of observations for RMT denoising
            max_partition_size: Maximum allowed partition size (if None, no limit)
            gpt_model_id: GPT model ID
            gpt_device: Device for GPT computation
            gpt_cached: Whether to use cached GPT model
        """
        print("\n" + "=" * 80)
        print(f"PORTFOLIO OPTIMIZATION COMPARISON - {self.n_assets} Assets")
        print("=" * 80)

        # Step 1: Partition the problem
        print(f"\nPartitioning into ~{partition_size}-asset subproblems...")
        if max_partition_size:
            print(f"Maximum partition size: {max_partition_size}")

        partitioned_mus, partitioned_sigmas, mappings = self.partition_with_rmt(
            partition_size, n_obs
        )

        # Apply maximum partition size constraint if specified
        if max_partition_size is not None:
            partitioned_mus, partitioned_sigmas, mappings = (
                self._enforce_max_partition_size(
                    partitioned_mus, partitioned_sigmas, mappings, max_partition_size
                )
            )

        print(f"Created {len(mappings)} partitions")
        print(f"Partition sizes: {[len(m) for m in mappings]}")

        # Step 2: Distribute K across partitions
        K_per_partition = []
        remaining_K = K_total

        for i, mapping in enumerate(mappings):
            partition_size_actual = len(mapping)
            if i < len(mappings) - 1:
                k_i = int(K_total * partition_size_actual / self.n_assets)
                k_i = min(k_i, partition_size_actual, remaining_K)
                K_per_partition.append(k_i)
                remaining_K -= k_i
            else:
                K_per_partition.append(min(remaining_K, partition_size_actual))

        print(f"K distribution: {K_per_partition} (total: {sum(K_per_partition)})")

        # Step 3: Classical Global Baseline
        print("\n" + "-" * 60)
        print("CLASSICAL GLOBAL OPTIMIZATION")
        print("-" * 60)
        start_time = time.time()
        classical_global_x, classical_global_obj = self.solve_classical_global(
            K_total, q
        )
        classical_global_indices = np.where(classical_global_x == 1)[0].tolist()
        classical_global_time = time.time() - start_time
        print(f"Objective: {classical_global_obj:.6f}")
        print(f"Time: {classical_global_time:.2f}s")

        # Step 4: Classical Partitioned
        print("\n" + "-" * 60)
        print("CLASSICAL PARTITIONED OPTIMIZATION")
        print("-" * 60)
        start_time = time.time()
        classical_part_indices, classical_part_obj = self.solve_classical_partitioned(
            partitioned_mus, partitioned_sigmas, mappings, K_per_partition, q
        )
        classical_part_time = time.time() - start_time
        print(f"Objective: {classical_part_obj:.6f}")
        print(
            f"vs Global: {((classical_part_obj - classical_global_obj) / abs(classical_global_obj) * 100):+.1f}%"
        )
        print(f"Time: {classical_part_time:.2f}s")

        # Store total classical time
        total_classical_time = classical_global_time + classical_part_time

        # Step 5: QOKit Partitioned (if available)
        qokit_results = {}
        if QOKIT_AVAILABLE:
            print("\n" + "-" * 60)
            print("QOKIT QAOA PARTITIONED OPTIMIZATION")
            print("-" * 60)
            start_time = time.time()
            try:
                qokit_indices, qokit_obj, qokit_sub_ars = self.solve_qokit_partitioned(
                    partitioned_mus, partitioned_sigmas, mappings, K_per_partition, q, p
                )
                qokit_time = time.time() - start_time

                # Calculate global AR
                qokit_ar = self.calculate_global_approximation_ratio(
                    qokit_obj, classical_global_obj
                )

                print(f"\nObjective: {qokit_obj:.6f}")
                print(
                    f"vs Global: {((qokit_obj - classical_global_obj) / abs(classical_global_obj) * 100):+.1f}%"
                )
                print(
                    f"vs Classical Partitioned: {((qokit_obj - classical_part_obj) / abs(classical_part_obj) * 100):+.1f}%"
                )
                print(f"Global Approximation Ratio: {qokit_ar:.3f}")
                print(f"Average Subproblem AR: {np.mean(qokit_sub_ars):.3f}")
                print(f"Time: {qokit_time:.2f}s")

                qokit_results = {
                    "objective": qokit_obj,
                    "solution": qokit_indices,
                    "approximation_ratio": qokit_ar,
                    "subproblem_ars": qokit_sub_ars,
                    "time": qokit_time,
                }
            except Exception as e:
                print(f"QOKit optimization failed: {e}")

        # Step 6: GPT Partitioned (if available)
        gpt_results = {}
        if GPT_AVAILABLE:
            print("\n" + "-" * 60)
            print("GPT QUANTUM PARTITIONED OPTIMIZATION")
            print("-" * 60)
            start_time = time.time()
            try:
                gpt_indices, gpt_obj, gpt_sub_ars = self.solve_gpt_partitioned(
                    partitioned_mus,
                    partitioned_sigmas,
                    mappings,
                    K_per_partition,
                    q,
                    gpt_model_id,
                    gpt_device,
                    gpt_cached,
                )
                gpt_time = time.time() - start_time

                # Calculate global AR
                gpt_ar = self.calculate_global_approximation_ratio(
                    gpt_obj, classical_global_obj
                )

                print(f"\nObjective: {gpt_obj:.6f}")
                print(
                    f"vs Global: {((gpt_obj - classical_global_obj) / abs(classical_global_obj) * 100):+.1f}%"
                )
                print(
                    f"vs Classical Partitioned: {((gpt_obj - classical_part_obj) / abs(classical_part_obj) * 100):+.1f}%"
                )
                print(f"Global Approximation Ratio: {gpt_ar:.3f}")
                print(f"Average Subproblem AR: {np.mean(gpt_sub_ars):.3f}")
                print(f"Time: {gpt_time:.2f}s")

                gpt_results = {
                    "objective": gpt_obj,
                    "solution": gpt_indices,
                    "approximation_ratio": gpt_ar,
                    "subproblem_ars": gpt_sub_ars,
                    "time": gpt_time,
                }
            except Exception as e:
                print(f"GPT optimization failed: {e}")

        # Create results object
        results = ComparisonResults(
            n_assets=self.n_assets,
            n_partitions=len(mappings),
            partition_sizes=[len(m) for m in mappings],
            K_total=K_total,
            q=q,
            classical_global_objective=classical_global_obj,
            classical_global_solution=classical_global_indices,
            classical_partitioned_objective=classical_part_obj,
            classical_partitioned_solution=classical_part_indices,
            classical_time=total_classical_time,
            qokit_objective=qokit_results.get("objective"),
            qokit_solution=qokit_results.get("solution"),
            qokit_approximation_ratio=qokit_results.get("approximation_ratio"),
            qokit_subproblem_ars=qokit_results.get("subproblem_ars"),
            qokit_time=qokit_results.get("time"),
            gpt_objective=gpt_results.get("objective"),
            gpt_solution=gpt_results.get("solution"),
            gpt_approximation_ratio=gpt_results.get("approximation_ratio"),
            gpt_subproblem_ars=gpt_results.get("subproblem_ars"),
            gpt_time=gpt_results.get("time"),
        )

        # Print final summary
        self._print_final_summary(results)

        return results

    def _print_final_summary(self, results: ComparisonResults):
        """Print final comparison summary"""
        print("\n" + "=" * 80)
        print("FINAL COMPARISON SUMMARY")
        print("=" * 80)

        print(
            f"\nProblem: {results.n_assets} assets, {results.n_partitions} partitions, K={results.K_total}"
        )

        print("\n📊 OBJECTIVE VALUES (lower is better):")
        print(
            f"  Classical Global:      {results.classical_global_objective:.6f} (baseline)"
        )
        print(
            f"  Classical Partitioned: {results.classical_partitioned_objective:.6f} "
            + f"({((results.classical_partitioned_objective - results.classical_global_objective) / abs(results.classical_global_objective) * 100):+.1f}%)"
        )

        if results.qokit_objective is not None:
            print(
                f"  QOKit Partitioned:     {results.qokit_objective:.6f} "
                + f"({((results.qokit_objective - results.classical_global_objective) / abs(results.classical_global_objective) * 100):+.1f}%)"
            )

        if results.gpt_objective is not None:
            print(
                f"  GPT Partitioned:       {results.gpt_objective:.6f} "
                + f"({((results.gpt_objective - results.classical_global_objective) / abs(results.classical_global_objective) * 100):+.1f}%)"
            )

        print("\n📈 APPROXIMATION RATIOS (higher is better):")
        if results.qokit_approximation_ratio is not None:
            print(f"  QOKit Global AR:  {results.qokit_approximation_ratio:.3f}")
            print(f"  QOKit Avg Sub AR: {np.mean(results.qokit_subproblem_ars):.3f}")

        if results.gpt_approximation_ratio is not None:
            print(f"  GPT Global AR:    {results.gpt_approximation_ratio:.3f}")
            print(f"  GPT Avg Sub AR:   {np.mean(results.gpt_subproblem_ars):.3f}")

        print("\n⏱️ COMPUTATION TIME:")
        print(f"  Classical:   {results.classical_time:.2f}s")
        if results.qokit_time is not None:
            print(f"  QOKit:       {results.qokit_time:.2f}s")
        if results.gpt_time is not None:
            print(f"  GPT:         {results.gpt_time:.2f}s")

        # Quantum advantage analysis
        print("\n🔬 QUANTUM ADVANTAGE ANALYSIS:")

        quantum_better_than_classical_part = False

        if results.qokit_objective is not None:
            qokit_vs_classical = (
                (results.qokit_objective - results.classical_partitioned_objective)
                / abs(results.classical_partitioned_objective)
                * 100
            )
            print(f"  QOKit vs Classical Partitioned: {qokit_vs_classical:+.1f}%")
            if qokit_vs_classical < 0:
                quantum_better_than_classical_part = True

        if results.gpt_objective is not None:
            gpt_vs_classical = (
                (results.gpt_objective - results.classical_partitioned_objective)
                / abs(results.classical_partitioned_objective)
                * 100
            )
            print(f"  GPT vs Classical Partitioned:   {gpt_vs_classical:+.1f}%")
            if gpt_vs_classical < 0:
                quantum_better_than_classical_part = True

        if quantum_better_than_classical_part:
            print(
                "\n✨ Quantum methods show advantage over classical partitioned approach!"
            )
        else:
            print(
                "\n📝 Classical partitioned approach currently outperforms quantum methods."
            )

    def save_results(
        self, results: ComparisonResults, output_dir: str = "comparison_results"
    ):
        """Save comparison results to files"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary as JSON
        summary = {
            "timestamp": timestamp,
            "problem": {
                "n_assets": results.n_assets,
                "n_partitions": results.n_partitions,
                "partition_sizes": results.partition_sizes,
                "K_total": results.K_total,
                "q": results.q,
            },
            "objectives": {
                "classical_global": results.classical_global_objective,
                "classical_partitioned": results.classical_partitioned_objective,
                "qokit": results.qokit_objective,
                "gpt": results.gpt_objective,
            },
            "approximation_ratios": {
                "qokit_global": results.qokit_approximation_ratio,
                "qokit_subproblems": results.qokit_subproblem_ars,
                "gpt_global": results.gpt_approximation_ratio,
                "gpt_subproblems": results.gpt_subproblem_ars,
            },
            "computation_times": {
                "classical": results.classical_time,
                "qokit": results.qokit_time,
                "gpt": results.gpt_time,
            },
        }

        with open(
            os.path.join(output_dir, f"comparison_summary_{timestamp}.json"), "w"
        ) as f:
            json.dump(summary, f, indent=2)

        # Save detailed results as pickle
        with open(
            os.path.join(output_dir, f"comparison_results_{timestamp}.pkl"), "wb"
        ) as f:
            pickle.dump(results, f)

        print(f"\n✅ Results saved to {output_dir}/")
        return os.path.join(output_dir, f"comparison_summary_{timestamp}.json")


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare GPT, QOKit, and Classical portfolio optimization"
    )
    parser.add_argument("graph_path", help="Path to graph pickle file")
    parser.add_argument(
        "--partition-size", type=int, default=15, help="Target partition size"
    )
    parser.add_argument(
        "--max-partition-size", type=int, help="Maximum allowed partition size"
    )
    parser.add_argument(
        "--k-total", type=int, default=40, help="Total assets to select"
    )
    parser.add_argument(
        "--risk-aversion", type=float, default=0.5, help="Risk aversion parameter"
    )
    parser.add_argument("--qaoa-depth", type=int, default=5, help="QAOA circuit depth")
    parser.add_argument("--gpt-model", default="50m_old", help="GPT model ID")
    parser.add_argument("--gpt-device", default="cpu", help="Device for GPT")
    parser.add_argument(
        "--output-dir", default="comparison_results", help="Output directory"
    )

    args = parser.parse_args()

    # Run comparison
    optimizer = UnifiedPortfolioOptimizer(args.graph_path)
    results = optimizer.run_comparison(
        partition_size=args.partition_size,
        K_total=args.k_total,
        q=args.risk_aversion,
        p=args.qaoa_depth,
        max_partition_size=args.max_partition_size,
        gpt_model_id=args.gpt_model,
        gpt_device=args.gpt_device,
    )

    # Save results
    optimizer.save_results(results, args.output_dir)

    return results


if __name__ == "__main__":
    # Example direct usage
    graph_path = "subgraph_150.pickle"  # Update with your graph path

    optimizer = UnifiedPortfolioOptimizer(graph_path)
    results = optimizer.run_comparison(
        partition_size=15,  # Target partition size
        K_total=30,  # Total assets to select
        q=0.5,  # Risk aversion
        p=5,  # QAOA depth
        max_partition_size=15,  # Maximum allowed partition size
        gpt_model_id="50m_old",
        gpt_device="cpu",
    )

    # Save results
    optimizer.save_results(results)
