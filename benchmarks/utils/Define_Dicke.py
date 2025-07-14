import numpy as np

# main classes and functions
from qiskit import QuantumRegister, QuantumCircuit

# gates
from qiskit.circuit.library import RYGate


def build_gate_i_circuit(n):
    theta = 2 * np.arccos(np.sqrt(1 / n))
    
    qc_i = QuantumCircuit(2)
    qc_i.cx(0, 1)
    
    cry = RYGate(theta).control()
    qc_i.append(cry, [1, 0])  # cry(target=1, control=0)
    
    qc_i.cx(0, 1)
    return qc_i


# ===============================================

def build_gate_ii_l_circuit(l, n, draw=False):
    """
    Returns a flattened circuit for the (ii)_l gate: a ccRY between 2 CXs.
    """
    qc = QuantumCircuit(3)

    theta = 2 * np.arccos(np.sqrt(l / n))

    # First CX
    qc.cx(0, 2)

    # Flattened CC-RY (control qubits 0,1 and target 2)
    ccry = RYGate(theta).control(num_ctrl_qubits=2, ctrl_state="11")
    qc.append(ccry, [2, 1, 0])  # Note: ordering is target, control1, control0 (Qiskit reverses)

    # Second CX
    qc.cx(0, 2)

    if draw:
        show_figure(qc.draw("mpl"))

    return qc

# ===============================================

def gate_scs_nk(n, k):
    qc_scs = QuantumCircuit(k+1)
    qc_scs.compose(build_gate_i_circuit(n), [k-1, k], inplace=True)
    
    for l in range(2, k+1):
        qc_scs.compose(build_gate_ii_l_circuit(l, n), [k-l, k-l+1, k], inplace=True)

    return qc_scs

# ===============================================

def first_block(n, k, l):
    qr = QuantumRegister(n)
    qc_first_block = QuantumCircuit(qr)

    n_first = l - k - 1
    n_last = n - l

    idxs_scs = list(range(n))

    if n_first != 0:
        idxs_scs = idxs_scs[n_first:]
        qc_first_block.id(qr[:n_first])

    if n_last != 0:
        idxs_scs = idxs_scs[:-n_last]
        qc_first_block.compose(gate_scs_nk(l, k), idxs_scs, inplace=True)
        qc_first_block.id(qr[-n_last:])
    else:
        qc_first_block.compose(gate_scs_nk(l, k), idxs_scs, inplace=True)

    return qc_first_block


# ===============================================

def second_block(n, k, l):
    qr = QuantumRegister(n)
    qc_second_block = QuantumCircuit(qr)

    n_last = n - l
    idxs_scs = list(range(n))

    if n_last != 0:
        idxs_scs = idxs_scs[:-n_last]
        qc_second_block.compose(gate_scs_nk(l, l - 1), idxs_scs, inplace=True)
        qc_second_block.id(qr[-n_last:])
    else:
        qc_second_block.compose(gate_scs_nk(l, l - 1), idxs_scs, inplace=True)

    return qc_second_block

# ===============================================

def dicke_state(n, k, barrier=False):
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)

    # Step 1: Prepare |111...000⟩ state
    qc.x(qr[-k:])

    if barrier:
        qc.barrier()

    # Step 2: Apply U_{n,k}
    for l in range(k + 1, n + 1)[::-1]:
        block = first_block(n, k, l)
        qc.compose(block, qubits=range(n), inplace=True)
        if barrier:
            qc.barrier()

    for l in range(2, k + 1)[::-1]:
        block = second_block(n, k, l)
        qc.compose(block, qubits=range(n), inplace=True)
        if barrier:
            qc.barrier()

    return qc