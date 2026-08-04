import math
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import qiskit.qasm2
import qiskit.qasm3
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit.library import U3Gate, UnitaryGate
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator


BASIS_MODE_GLOBAL = "global"
BASIS_MODE_SELECTIVE = "selective"
DEFAULT_SELECTIVE_FRACTION = 0.3
PRESERVED_INSTRUCTIONS = {"barrier", "measure", "reset", "delay"}
DEFAULT_SHOTS = 1024


def add_measurements(circuit):
    """Add measurements only when the input circuit has none."""
    has_measurements = any(
        instruction.operation.name == "measure"
        for instruction in circuit.data
    )
    if has_measurements:
        return circuit

    if not circuit.clbits:
        print("Warning: No classical registers found. Adding one.")
        circuit.add_register(ClassicalRegister(len(circuit.qubits), "c"))

    num_clbits = len(circuit.clbits)
    qubits_to_measure = circuit.qubits[: min(num_clbits, len(circuit.qubits))]

    for index, qubit in enumerate(qubits_to_measure):
        circuit.measure(qubit, circuit.clbits[index])

    print(
        f"Added measurements for {len(qubits_to_measure)} qubits "
        "to classical register."
    )
    return circuit


def generate_random_u3_params():
    theta = random.uniform(0, math.pi)
    phi = random.uniform(0, 2 * math.pi)
    lam = random.uniform(0, 2 * math.pi)
    return theta, phi, lam


def tensor_power(matrix, num_qubits):
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")

    result = matrix
    for _ in range(num_qubits - 1):
        result = np.kron(result, matrix)
    return result


def basis_gates(u3_params):
    theta, phi, lam = u3_params
    basis_gate = U3Gate(theta, phi, lam)
    return basis_gate, basis_gate.inverse()


def is_classically_controlled(operation):
    return getattr(operation, "condition", None) is not None


def should_preserve_instruction(operation):
    return (
        operation.name in PRESERVED_INSTRUCTIONS
        or operation.num_qubits == 0
        or operation.num_clbits > 0
    )


def is_transformable_instruction(instruction):
    operation = instruction.operation

    if is_classically_controlled(operation) or should_preserve_instruction(operation):
        return False

    try:
        Operator(operation)
    except Exception:
        return False

    return True


def transformed_gate_label(gate, block_id):
    gate_name_map = {
        "h": "H",
        "cx": "CNOT",
        "x": "X",
        "y": "Y",
        "z": "Z",
        "t": "T",
        "s": "S",
        "rx": "RX",
        "ry": "RY",
        "rz": "RZ",
        "cu": "CU",
        "cp": "CP",
        "cry": "CRY",
        "swap": "SWAP",
    }
    gate_name = gate_name_map.get(gate.name.lower(), gate.name.upper())
    return f"Obf_{gate_name}_{block_id}"


def create_transformed_gate(gate, u3_params, block_id, mode):
    basis_gate, inverse_basis_gate = basis_gates(u3_params)

    basis_matrix = tensor_power(
        Operator(basis_gate).data,
        gate.num_qubits,
    )
    inverse_basis_matrix = tensor_power(
        Operator(inverse_basis_gate).data,
        gate.num_qubits,
    )
    gate_matrix = Operator(gate).data

    if mode == BASIS_MODE_GLOBAL:
        transformed_matrix = basis_matrix @ gate_matrix @ inverse_basis_matrix #type: ignore
    elif mode == BASIS_MODE_SELECTIVE:
        transformed_matrix = inverse_basis_matrix @ gate_matrix @ basis_matrix #type: ignore
    else:
        raise ValueError(f"Unsupported basis transformation mode: {mode}")

    transformed_name = transformed_gate_label(gate, block_id)
    transformed_gate = UnitaryGate(
        transformed_matrix,
        label=transformed_name,
    )
    transformed_gate.name = transformed_name
    return transformed_gate


def append_single_qubit_gate(circuit, gate, qubits):
    for qubit in qubits:
        circuit.append(gate, [qubit])


def append_instruction(circuit, instruction):
    circuit.append(
        instruction.operation,
        instruction.qubits,
        instruction.clbits,
    )


def apply_selective_basis_obfuscation(circuit, basis_k=None):
    """Obfuscate a selected number of eligible gates independently."""
    obfuscated_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

    eligible_indices = [
        index
        for index, instruction in enumerate(circuit.data)
        if is_transformable_instruction(instruction)
    ]

    if basis_k is None:
        basis_k = math.ceil(
            len(eligible_indices) * DEFAULT_SELECTIVE_FRACTION
        )

    basis_k = max(0, min(int(basis_k), len(eligible_indices)))
    selected_indices = set(random.sample(eligible_indices, basis_k))
    block_id = 0

    for index, instruction in enumerate(circuit.data):
        if index not in selected_indices:
            append_instruction(obfuscated_circuit, instruction)
            continue

        u3_params = generate_random_u3_params()
        basis_gate, inverse_basis_gate = basis_gates(u3_params)
        transformed_gate = create_transformed_gate(
            instruction.operation,
            u3_params,
            block_id,
            BASIS_MODE_SELECTIVE,
        )

        append_single_qubit_gate(
            obfuscated_circuit,
            inverse_basis_gate,
            instruction.qubits,
        )
        obfuscated_circuit.append(transformed_gate, instruction.qubits)
        append_single_qubit_gate(
            obfuscated_circuit,
            basis_gate,
            instruction.qubits,
        )
        block_id += 1

    return obfuscated_circuit


def apply_global_basis_obfuscation(circuit):
    """Apply one basis per executable segment, closing it before measurements/resets."""
    obfuscated_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    measured_qubits = set()
    segment_open = False
    segment_params = None
    block_id = 0

    def active_qubits():
        return [
            qubit
            for qubit in circuit.qubits
            if qubit not in measured_qubits
        ]

    def open_segment_if_needed():
        nonlocal segment_open, segment_params

        if segment_open:
            return

        segment_qubits = active_qubits()
        if not segment_qubits:
            return

        segment_params = generate_random_u3_params()
        basis_gate, _ = basis_gates(segment_params)
        append_single_qubit_gate(
            obfuscated_circuit,
            basis_gate,
            segment_qubits,
        )
        segment_open = True

    def close_segment_if_needed():
        nonlocal segment_open, segment_params

        if not segment_open:
            return

        _, inverse_basis_gate = basis_gates(segment_params)
        append_single_qubit_gate(
            obfuscated_circuit,
            inverse_basis_gate,
            active_qubits(),
        )
        segment_open = False
        segment_params = None

    for instruction in circuit.data:
        operation_name = instruction.operation.name

        if operation_name in {"measure", "reset"}:
            close_segment_if_needed()
            append_instruction(obfuscated_circuit, instruction)

            if operation_name == "measure":
                measured_qubits.update(instruction.qubits)
            else:
                measured_qubits.difference_update(instruction.qubits)
            continue

        if not is_transformable_instruction(instruction):
            append_instruction(obfuscated_circuit, instruction)
            continue

        if any(qubit in measured_qubits for qubit in instruction.qubits):
            append_instruction(obfuscated_circuit, instruction)
            continue

        open_segment_if_needed()

        if segment_params is None:
            append_instruction(obfuscated_circuit, instruction)
            continue

        transformed_gate = create_transformed_gate(
            instruction.operation,
            segment_params,
            block_id,
            BASIS_MODE_GLOBAL,
        )
        obfuscated_circuit.append(transformed_gate, instruction.qubits)
        block_id += 1

    close_segment_if_needed()
    return obfuscated_circuit


def apply_basis_obfuscation(
    circuit,
    basis_mode=BASIS_MODE_SELECTIVE,
    basis_k=None,
):
    mode = (basis_mode or BASIS_MODE_SELECTIVE).lower()

    if mode == BASIS_MODE_GLOBAL:
        return apply_global_basis_obfuscation(circuit)

    if mode == BASIS_MODE_SELECTIVE:
        return apply_selective_basis_obfuscation(circuit, basis_k)

    raise ValueError("basis_mode must be either 'selective' or 'global'")


def parse_qasm(input_qasm):
    input_qasm = input_qasm.strip()

    if input_qasm.startswith("OPENQASM 2.0"):
        return QuantumCircuit.from_qasm_str(input_qasm), 2

    if input_qasm.startswith("OPENQASM 3"):
        return qiskit.qasm3.loads(input_qasm), 3

    raise ValueError(
        "Invalid QASM version: Must start with "
        "'OPENQASM 2.0;' or 'OPENQASM 3;'"
    )


def execute_circuit(circuit, shots=DEFAULT_SHOTS):
    if shots <= 0:
        raise ValueError("Number of shots must be positive")

    simulator = AerSimulator()
    transpiled_circuit = transpile(circuit, simulator)

    start_time = time.time()
    result = simulator.run(
        transpiled_circuit,
        shots=shots,
    ).result()
    execution_time = time.time() - start_time

    return result.get_counts(), execution_time


def compare_results(original, obfuscated):
    keys = set(original).union(obfuscated)
    total = sum(original.values())
    overlap = sum(
        min(original.get(key, 0), obfuscated.get(key, 0))
        for key in keys
    )
    return 100 * overlap / total if total > 0 else 0


def compute_tvd_dfc(original_results, obfuscated_results, shots=DEFAULT_SHOTS):
    if shots <= 0:
        raise ValueError("Number of shots must be positive")

    all_keys = set(original_results).union(obfuscated_results)
    tvd_sum = sum(
        abs(
            original_results.get(key, 0)
            - obfuscated_results.get(key, 0)
        )
        for key in all_keys
    )
    tvd = tvd_sum / (2 * shots)

    correct_bitstrings = set(original_results)
    correct_count_sum = sum(
        obfuscated_results.get(bitstring, 0)
        for bitstring in correct_bitstrings
    )
    incorrect_counts = [
        count
        for bitstring, count in obfuscated_results.items()
        if bitstring not in correct_bitstrings
    ]
    max_incorrect_count = max(incorrect_counts, default=0)
    dfc = (correct_count_sum - max_incorrect_count) / shots

    return tvd, dfc


def circuit_style():
    return {
        "fontsize": 12,
        "displaycolor": {
            "u3": "#FB0202",
            "Obf_H": "#FF5733",
            "Obf_CNOT": "#FF5733",
            "Obf_X": "#FF5733",
            "Obf_Y": "#FF5733",
            "Obf_Z": "#FF5733",
            "Obf_CU": "#FF5733",
            "Obf_CP": "#FF5733",
            "Obf_CRY": "#FF5733",
            "Obf_SWAP": "#FF5733",
        },
    }


def plot_result_comparison(original_results, obfuscated_results):
    all_keys = sorted(set(original_results).union(obfuscated_results))
    original_counts = [original_results.get(key, 0) for key in all_keys]
    obfuscated_counts = [obfuscated_results.get(key, 0) for key in all_keys]

    x_positions = np.arange(len(all_keys))
    width = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar(
        x_positions - width / 2,
        original_counts,
        width,
        label="Original",
    )
    plt.bar(
        x_positions + width / 2,
        obfuscated_counts,
        width,
        label="Obfuscated",
    )
    plt.xlabel("Measurement Outcome")
    plt.ylabel("Counts")
    plt.title("Original vs Obfuscated Results")
    plt.xticks(x_positions, all_keys, rotation=45)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def basis_obfuscate_and_execute(
    input_qasm,
    basis_mode=BASIS_MODE_SELECTIVE,
    basis_k=None,
    shots=DEFAULT_SHOTS,
    draw_circuit=True,
):
    try:
        original_circuit, qasm_version = parse_qasm(input_qasm)
    except Exception as error:
        raise ValueError(f"Error parsing QASM: {error}") from error

    original_circuit = add_measurements(original_circuit)
    obfuscated_circuit = apply_basis_obfuscation(
        original_circuit,
        basis_mode=basis_mode,
        basis_k=basis_k,
    )

    original_results, original_time = execute_circuit(
        original_circuit,
        shots=shots,
    )
    obfuscated_results, obfuscated_time = execute_circuit(
        obfuscated_circuit,
        shots=shots,
    )

    if draw_circuit:
        obfuscated_circuit.draw("mpl", style=circuit_style())
        plt.show()

    semantic_accuracy = compare_results(
        original_results,
        obfuscated_results,
    )
    tvd, dfc = compute_tvd_dfc(
        original_results,
        obfuscated_results,
        shots=shots,
    )

    if qasm_version == 2:
        obfuscated_qasm = qiskit.qasm2.dumps(obfuscated_circuit)
    else:
        obfuscated_qasm = qiskit.qasm3.dumps(obfuscated_circuit)

    return {
        "original_circuit": original_circuit,
        "obfuscated_circuit": obfuscated_circuit,
        "obfuscated_qasm": obfuscated_qasm,
        "original_results": original_results,
        "obfuscated_results": obfuscated_results,
        "semantic_accuracy": semantic_accuracy,
        "tvd": tvd,
        "dfc": dfc,
        "original_time": original_time,
        "obfuscated_time": obfuscated_time,
        "basis_mode": (basis_mode or BASIS_MODE_SELECTIVE).lower(),
        "basis_k": basis_k,
        "shots": shots,
    }


if __name__ == "__main__":
    file_path = "QASM Circuits/BV(1011).qasm"
    basis_mode = BASIS_MODE_SELECTIVE
    basis_k = None

    try:
        with open(file_path, "r", encoding="utf-8") as qasm_file:
            test_qasm = qasm_file.read()

        print("\nTesting QASM Circuit:")
        print(f"Basis mode: {basis_mode}")
        if basis_mode == BASIS_MODE_SELECTIVE:
            print(
                "Selected gate count: "
                f"{basis_k if basis_k is not None else '30% of eligible gates'}"
            )

        results = basis_obfuscate_and_execute(
            test_qasm,
            basis_mode=basis_mode,
            basis_k=basis_k,
        )
    except (OSError, ValueError) as error:
        print(error)
        sys.exit(1)

    print("\n--- Original Results ---")
    for key, value in results["original_results"].items():
        print(f"Result: {key}, Count: {value}")

    print("\n--- Obfuscated Results ---")
    for key, value in results["obfuscated_results"].items():
        print(f"Result: {key}, Count: {value}")

    print(f"\nSemantic Accuracy: {results['semantic_accuracy']:.2f}%")
    print(f"Original Time: {results['original_time']} s")
    print(f"Obfuscated Time: {results['obfuscated_time']} s")
    print(f"Total Variation Distance (TVD): {results['tvd']}")
    print(f"Degree of Functional Corruption (DFC): {results['dfc']:.4f}")

    plot_result_comparison(
        results["original_results"],
        results["obfuscated_results"],
    )
