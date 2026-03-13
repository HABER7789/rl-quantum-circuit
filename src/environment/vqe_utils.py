# vqe_utils.py
# VQE machinery: build circuits, evaluate energies, run COBYLA.
#
# Flow: RL picks a gate → append to sequence → COBYLA finds best angles →
# compute E(theta) = <psi(theta)|H|psi(theta)> → reward = delta_E
#
# Using statevector sim (no shot noise) since we're on a laptop and real
# hardware is both slow and noisy. COBYLA because it's gradient-free and
# worked better than gradient methods when we tested it.

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

GATE_Rx   = "Rx"
GATE_Ry   = "Ry"
GATE_Rz   = "Rz"
GATE_CNOT = "CNOT"

ROTATION_GATES = {GATE_Rx, GATE_Ry, GATE_Rz}


def build_circuit(n_qubits, gate_sequence, params):
    """
    Build a Qiskit circuit from a list of gate dicts + angles.

    Gate dicts look like: {"type": "Rx", "target": 0, "control": None}
    Params are assigned in order, one per rotation gate.
    """
    n_rot = sum(1 for g in gate_sequence if g["type"] in ROTATION_GATES)
    if len(params) != n_rot:
        raise ValueError(
            f"Got {len(params)} params but sequence has {n_rot} rotation gates."
        )

    qc = QuantumCircuit(n_qubits)
    param_idx = 0

    for gate in gate_sequence:
        gtype = gate["type"]
        tgt = gate["target"]

        if gtype == GATE_Rx:
            qc.rx(params[param_idx], tgt); param_idx += 1
        elif gtype == GATE_Ry:
            qc.ry(params[param_idx], tgt); param_idx += 1
        elif gtype == GATE_Rz:
            qc.rz(params[param_idx], tgt); param_idx += 1
        elif gtype == GATE_CNOT:
            ctrl = gate["control"]
            if ctrl is None:
                raise ValueError("CNOT needs a control qubit.")
            qc.cx(ctrl, tgt)
        else:
            raise ValueError(f"Unknown gate type: '{gtype}'")

    return qc


def evaluate_energy(n_qubits, gate_sequence, params, hamiltonian):
    """Compute <psi(theta)|H|psi(theta)> via exact statevector simulation."""
    if len(gate_sequence) == 0:
        qc = QuantumCircuit(n_qubits)
    else:
        qc = build_circuit(n_qubits, gate_sequence, params)

    sv = Statevector.from_instruction(qc)
    energy = sv.expectation_value(hamiltonian)
    return float(energy.real)


def count_params(gate_sequence):
    return sum(1 for g in gate_sequence if g["type"] in ROTATION_GATES)


def optimize_parameters(
    n_qubits,
    gate_sequence,
    hamiltonian,
    initial_params=None,
    max_iterations=500,
    rhobeg=0.5,
):
    """Run COBYLA to find the best rotation angles for this gate sequence."""
    n_params = count_params(gate_sequence)

    if n_params == 0:
        energy = evaluate_energy(n_qubits, gate_sequence, np.array([]), hamiltonian)
        return np.array([]), energy

    if initial_params is None:
        rng = np.random.default_rng()
        initial_params = rng.uniform(-np.pi, np.pi, size=n_params)

    result = minimize(
        lambda theta: evaluate_energy(n_qubits, gate_sequence, theta, hamiltonian),
        x0=initial_params,
        method="COBYLA",
        options={"maxiter": max_iterations, "rhobeg": rhobeg},
    )

    return result.x, float(result.fun)


def optimize_parameters_warm_start(
    n_qubits,
    gate_sequence,
    hamiltonian,
    prev_params,
    prev_gate_count,
    max_iterations=300,
):
    """
    Reuse params from the previous step as a warm start.

    When we add one new gate, old optimized angles are still pretty good.
    We just randomly init the new angle(s) and re-run COBYLA from there.
    Makes training noticeably faster than starting cold every time.
    """
    n_params = count_params(gate_sequence)
    n_new = n_params - prev_gate_count

    if prev_params is None or len(prev_params) == 0:
        return optimize_parameters(n_qubits, gate_sequence, hamiltonian,
                                   max_iterations=max_iterations)

    rng = np.random.default_rng()
    new_vals = rng.uniform(-np.pi, np.pi, size=n_new)
    starting_point = np.concatenate([prev_params, new_vals])

    return optimize_parameters(
        n_qubits, gate_sequence, hamiltonian,
        initial_params=starting_point,
        max_iterations=max_iterations,
    )


def get_statevector(n_qubits, gate_sequence, params):
    """Return the full state vector as a complex array. Handy for debugging."""
    if len(gate_sequence) == 0:
        qc = QuantumCircuit(n_qubits)
    else:
        qc = build_circuit(n_qubits, gate_sequence, params)
    return Statevector.from_instruction(qc).data
