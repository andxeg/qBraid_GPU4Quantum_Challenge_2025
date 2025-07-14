import ast
from qiskit.circuit.library import RZGate, RZZGate, RYYGate, RXXGate
from qiskit.circuit import QuantumCircuit
from graphs_and_parsing.Define_Dicke import dicke_state
from qiskit.circuit.library import RYGate, XGate, CXGate

VALID_GATES = {"rz": 1, "rzz": 2, "ryy": 2, "rxx": 2}

def is_qubit_list(s):
    try:
        q = ast.literal_eval(s)
        return isinstance(q, list) and all(isinstance(i, int) for i in q)
    except:
        return False

def is_float(s):
    try:
        float(s)
        return True
    except:
        return False

def fix_and_parse_tokens(tokens, num_qubits):
    errors_counter = 0
    parsed = []
    i = 0

    while i < len(tokens):
        if tokens[i] == "<new_layer_p>":
            i += 1
            if i >= len(tokens): break

            gate = tokens[i]
            if gate not in VALID_GATES:
                i += 1
                errors_counter += 1
                continue

            i += 1
            if i >= len(tokens):
                errors_counter += 1
                continue

            # Parse qubit list
            qubits = [0]
            if is_qubit_list(tokens[i]):
                qubits = ast.literal_eval(tokens[i])
                i += 1
            else:
                errors_counter += 1
                i += 1
                continue

            # Check that qubit indices are in range
            if any(q < 0 or q >= num_qubits for q in qubits):
                errors_counter += 1
                continue

            # Parse parameter
            param = 1.0
            if i < len(tokens) and is_float(tokens[i]):
                param = float(tokens[i])
                i += 1

            if len(qubits) != VALID_GATES[gate]:
                errors_counter += 1
                continue

            parsed.append(("<new_layer_p>", gate, qubits, param))
        else:
            i += 1

    print("Number of Errors:", errors_counter)
    return parsed



# Gate class mapping
GATE_MAP = {
    "rz": RZGate,
    "rzz": RZZGate,
    "ryy": RYYGate,
    "rxx": RXXGate
}

def build_circuit(parsed_instructions, initial_state=None, num_qubits=None, K = None):
    """
    Builds the circuit with instructions fully flattened using `compose`.
    parsed_instructions: output from fix_and_parse_tokens
    initial_state: initial state for the circuit to build off of
    
    """
    if initial_state is None:
        qc = dicke_state(n = num_qubits, k = K)
    else:
        qc = initial_state

    for _, gate_name, qubits, param in parsed_instructions:
        gate_class = GATE_MAP.get(gate_name)
        if gate_class is None:
            raise ValueError(f"Unknown gate: {gate_name}")

        gate = gate_class(param)

        # Create a single-gate circuit and flatten it into qc
        temp = QuantumCircuit(qc.num_qubits)
        temp.append(gate, qubits)
        qc.compose(temp, inplace=True)

    return qc



# === NASDAQ Gate Metadata ===
NASDAQ_VALID_GATES = {
    "rz": 1,
    "ry": 1,
    "x": 1,
    "cx": 2,
    "rzz": 2,
    "xx_plus_yy": 2
}

# Custom NASDAQ gate for "xx_plus_yy" = RXX + RYY
def nasdaq_XXPlusYYGate(theta):
    qc = QuantumCircuit(2)
    qc.rxx(theta, 0, 1)
    qc.ryy(theta, 0, 1)
    return qc.to_gate(label="xx+yy")

NASDAQ_GATE_MAP = {
    "rz": RZGate,
    "ry": RYGate,
    "x": lambda _: XGate(),
    "cx": lambda _: CXGate(),
    "rzz": RZZGate,
    "xx_plus_yy": nasdaq_XXPlusYYGate
}

# === NASDAQ Helper Functions ===
def nasdaq_is_qubit_list(s):
    try:
        q = ast.literal_eval(s)
        return isinstance(q, list) and all(isinstance(i, int) for i in q)
    except:
        return False

def nasdaq_is_float(s):
    try:
        float(s)
        return True
    except:
        return False

def nasdaq_fix_and_parse_tokens(tokens, num_qubits):
    errors_counter = 0
    parsed = []
    i = 0

    while i < len(tokens):
        if tokens[i] == "<new_layer_p>":
            i += 1
            if i >= len(tokens): break

            gate = tokens[i]
            if gate not in NASDAQ_VALID_GATES:
                i += 1
                errors_counter += 1
                continue

            i += 1
            if i >= len(tokens):
                errors_counter += 1
                continue

            qubits = [0]
            if nasdaq_is_qubit_list(tokens[i]):
                qubits = ast.literal_eval(tokens[i])
                i += 1
            else:
                errors_counter += 1
                i += 1
                continue

            # Check qubit index bounds
            if any(q < 0 or q >= num_qubits for q in qubits):
                errors_counter += 1
                continue

            param = 1.0
            if i < len(tokens) and nasdaq_is_float(tokens[i]):
                param = float(tokens[i])
                i += 1

            if len(qubits) != NASDAQ_VALID_GATES[gate]:
                errors_counter += 1
                continue

            parsed.append(("<new_layer_p>", gate, qubits, param))
        else:
            i += 1

    print("NASDAQ Parse Errors:", errors_counter)
    return parsed

# === NASDAQ Circuit Builder ===
def nasdaq_build_circuit(parsed_instructions, initial_state=None, num_qubits=None, K=None):
    if initial_state is None:
        qc = dicke_state(n=num_qubits, k=K)
    else:
        qc = initial_state

    for _, gate_name, qubits, param in parsed_instructions:
        gate_factory = NASDAQ_GATE_MAP.get(gate_name)
        if gate_factory is None:
            raise ValueError(f"NASDAQ: Unknown gate: {gate_name}")

        try:
            gate = gate_factory(param)
        except TypeError:
            gate = gate_factory()

        temp = QuantumCircuit(qc.num_qubits)
        temp.append(gate, qubits)
        qc.compose(temp, inplace=True)

    return qc

# === NASDAQ Circuit Builder ===
def nasdaq_build_circuit(parsed_instructions, initial_state=None, num_qubits=None, K=None):
    if initial_state is None:
        qc = dicke_state(n=num_qubits, k=K)
    else:
        qc = initial_state

    for _, gate_name, qubits, param in parsed_instructions:
        gate_factory = NASDAQ_GATE_MAP.get(gate_name)
        if gate_factory is None:
            raise ValueError(f"NASDAQ: Unknown gate: {gate_name}")

        try:
            gate = gate_factory(param)
        except TypeError:
            gate = gate_factory()

        temp = QuantumCircuit(qc.num_qubits)
        temp.append(gate, qubits)
        qc.compose(temp, inplace=True)

    return qc

def tokens_to_circuit(tokens, initial_state=None, num_qubits=15, K=4, isNasdaq=False):
    """
    Converts a list of tokens into a Qiskit QuantumCircuit.
    
    If isNasdaq=True, it uses the NASDAQ-style parsing logic and gate map.
    """
    if isNasdaq:
        parsed_tokens = nasdaq_fix_and_parse_tokens(tokens, num_qubits)
        qc = nasdaq_build_circuit(parsed_tokens, initial_state, num_qubits=num_qubits, K=K)
    else:
        parsed_tokens = fix_and_parse_tokens(tokens, num_qubits)
        qc = build_circuit(parsed_tokens, initial_state, num_qubits=num_qubits, K=K)
    return qc

    
def get_max_token_count(n):
    token_map = {
        5: 820,
        10: 2050,
        15: 4100,
        20: 7175,
        25: 10660,
        30: 14350
    }
    # token_map = {
    #     5: 1386,
    #     10: 3465,
    #     15: 6930,
    #     20: 12127,
    #     25: 18018,
    #     30: 24255
    # }

    # Step 1: Sort the keys
    sorted_keys = sorted(token_map.keys())

    # Step 2: Find smallest key >= n
    for k in sorted_keys:
        if n <= k:
            return token_map[k]
    
    # If n is greater than all keys, use the largest
    return token_map[sorted_keys[-1]]


    