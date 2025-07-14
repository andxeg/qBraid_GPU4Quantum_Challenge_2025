from qokit.portfolio_optimization import portfolio_brute_force, get_sk_ini
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective
import nlopt  # A numerical optimization library (no gradients needed)

# Comments are AI generated

# This file wraps functionality from the Jupyter notebook example in QOKit:
# QOKit/examples/QAOA_portfolio_optimization.ipynb

# Generates the QAOA objective function used for optimizing a quantum portfolio
def get_objective(po_problem, reps=5, simulator='python'):
    # reps = number of QAOA layers (a.k.a. depth)
    # simulator = which backend to use; default is Python
    qaoa_obj = get_qaoa_portfolio_objective(
        po_problem=po_problem,
        p=reps,
        ini='dicke',  # Uses Dicke state as the initial state
        mixer='trotter_ring',  # Mixer used for QAOA
        T=1,  # Total time of evolution
        simulator=simulator
    )
    return qaoa_obj

# Gets the brute-force (classical) best portfolio value and/or bitstring
def classical_solution(po_problem, return_bitstring=False):
    best_portfolio = portfolio_brute_force(
        po_problem,
        return_bitstring=return_bitstring
    )
    return best_portfolio 

# Generates a random initial parameter vector for QAOA based on number of layers
def set_initial_parameters(reps):
    return get_sk_ini(p=reps)

# Minimizes the QAOA objective function using gradient-free optimizer (BOBYQA)
def minimize_nlopt(f, x0, rhobeg=None, p=None):
    # nlopt requires us to wrap the function in a format it expects
    def nlopt_wrapper(x, grad):
        if grad.size > 0:
            import sys
            sys.exit("Shouldn't be calling a gradient!")  # Defensive exit
        return f(x).real  # nlopt expects real numbers only

    # BOBYQA is a derivative-free optimizer from nlopt
    opt = nlopt.opt(nlopt.LN_BOBYQA, 2 * p)
    opt.set_min_objective(nlopt_wrapper)
    opt.set_xtol_rel(1e-8)  # Relative tolerance on x
    opt.set_ftol_rel(1e-8)  # Relative tolerance on function value
    opt.set_initial_step(rhobeg)  # Optional initial step size

    xstar = opt.optimize(x0)  # Runs the optimization
    minf = opt.last_optimum_value()  # Minimum function value found

    return xstar, minf

# Runs the optimization for QAOA and compares to the classical best solution
def minimize(po_problem, qaoa_obj, initial_parameters, reps, rhobeg=0.01):
    optimal_parameters, opt_energy = minimize_nlopt(
        qaoa_obj,
        initial_parameters,  # ← Was mistakenly named 'initial_objective'
        p=reps,
        rhobeg=rhobeg
    )

    # Get the classical best portfolio value for approximation ratio calculation
    best_portfolio = classical_solution(po_problem)
    best_val, worst_val = best_portfolio[0], best_portfolio[1]

    # Compute approximation ratio (normalized between worst and best case)
    opt_ar = (opt_energy - worst_val) / (best_val - worst_val)

    return optimal_parameters, opt_energy, opt_ar

# High-level function to solve a portfolio optimization problem with QAOA
def solve_with_qokit(po_problem, reps=5, simulator='python'):
    # 1. Get the quantum objective function
    objective_function = get_objective(po_problem, reps, simulator)

    # 2. Get the classical (brute-force) best solution for comparison
    best_portfolio = classical_solution(po_problem)

    # 3. Set up the initial random QAOA parameters
    initial_parameters = set_initial_parameters(reps)

    # 4. Minimize the QAOA objective
    optimal_parameters, opt_energy, opt_ar = minimize(
        po_problem,
        objective_function,
        initial_parameters,
        reps,
        rhobeg=0.01
    )

    return optimal_parameters, opt_energy, opt_ar
