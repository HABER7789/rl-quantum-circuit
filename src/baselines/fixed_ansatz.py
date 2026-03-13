# fixed_ansatz.py
# Fixed-structure baselines: Hardware-Efficient (HE) ansatz and QAOA.
# Human designs the circuit template, COBYLA finds the best angles.
# No RL, no learning — just classical optimization on a fixed structure.

import numpy as np
from src.environment.vqe_utils import optimize_parameters, count_params


def build_he_ansatz(n_qubits, n_layers):
    """
    HE ansatz: alternating Ry rotations and linear CNOT chains.
    Final Ry layer at the end (no trailing CNOTs).
    """
    gates = []
    for _ in range(n_layers):
        for q in range(n_qubits):
            gates.append({"type": "Ry", "target": q, "control": None})
        for q in range(n_qubits - 1):
            gates.append({"type": "CNOT", "target": q + 1, "control": q})
    for q in range(n_qubits):
        gates.append({"type": "Ry", "target": q, "control": None})
    return gates


def run_he_ansatz(
    hamiltonian,
    n_qubits,
    ground_state_energy,
    n_layers=2,
    energy_threshold=0.0016,
    cobyla_max_iter=1000,
    n_restarts=5,
    seed=42,
):
    """Run HE ansatz with multiple random restarts, keep the best."""
    rng = np.random.default_rng(seed)
    gate_seq = build_he_ansatz(n_qubits, n_layers)
    n_params = count_params(gate_seq)
    depth = len(gate_seq)

    best_energy = float("inf")
    best_params = None

    for _ in range(n_restarts):
        init = rng.uniform(-np.pi, np.pi, size=n_params)
        params, energy = optimize_parameters(
            n_qubits=n_qubits,
            gate_sequence=gate_seq,
            hamiltonian=hamiltonian,
            initial_params=init,
            max_iterations=cobyla_max_iter,
        )
        if energy < best_energy:
            best_energy = energy
            best_params = params

    energy_error = abs(best_energy - ground_state_energy)
    return {
        "best_energy":  best_energy,
        "best_params":  best_params,
        "energy_error": energy_error,
        "success":      energy_error <= energy_threshold,
        "depth":        depth,
        "n_params":     n_params,
        "n_layers":     n_layers,
    }


def build_qaoa_ansatz(n_qubits, graph_edges, p):
    """
    QAOA circuit for MaxCut with p layers.

    Each layer: cost unitary (CNOT-Rz-CNOT per edge) + mixer (Rx per qubit).
    Initial state: Ry(pi/2) on all qubits ≈ uniform superposition.
    Parameters: [gamma_1, beta_1, gamma_2, beta_2, ...] (2*p total)
    """
    gates = []

    # approx Hadamard to get |+>^n
    for q in range(n_qubits):
        gates.append({"type": "Ry", "target": q, "control": None})

    for _ in range(p):
        # exp(-i*gamma*Z_i*Z_j/2) = CNOT Rz CNOT
        for (i, j) in graph_edges:
            gates.append({"type": "CNOT", "target": j, "control": i})
            gates.append({"type": "Rz",   "target": j, "control": None})
            gates.append({"type": "CNOT", "target": j, "control": i})

        for q in range(n_qubits):
            gates.append({"type": "Rx", "target": q, "control": None})

    return gates


def run_qaoa(
    hamiltonian,
    n_qubits,
    graph_edges,
    optimal_cut,
    p=1,
    cobyla_max_iter=1000,
    n_restarts=10,
    seed=42,
):
    """Optimize QAOA angles with multiple random restarts."""
    rng = np.random.default_rng(seed)
    gate_seq = build_qaoa_ansatz(n_qubits, graph_edges, p)
    n_params = count_params(gate_seq)
    depth = len(gate_seq)

    best_energy = float("inf")
    best_params = None

    for _ in range(n_restarts):
        init = rng.uniform(0, 2 * np.pi, size=n_params)
        params, energy = optimize_parameters(
            n_qubits=n_qubits,
            gate_sequence=gate_seq,
            hamiltonian=hamiltonian,
            initial_params=init,
            max_iterations=cobyla_max_iter,
        )
        if energy < best_energy:
            best_energy = energy
            best_params = params

    # H = -C(Z), so E_0 = -MaxCut → MaxCut found = -best_energy
    maxcut_found = -best_energy
    approx_ratio = maxcut_found / optimal_cut if optimal_cut > 0 else 0.0

    return {
        "best_energy":  best_energy,
        "best_params":  best_params,
        "maxcut_found": maxcut_found,
        "approx_ratio": approx_ratio,
        "depth":        depth,
        "n_params":     n_params,
        "p":            p,
    }
