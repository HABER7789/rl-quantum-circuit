# circuit_env.py
# The RL environment. Agent builds a quantum circuit one gate at a time.
#
# Each episode: start with empty circuit, agent picks gates, we run COBYLA
# after each gate to find the best angles, reward = energy improvement.
# Episode ends when we hit chemical accuracy (success) or run out of depth.
#
# State: flat one-hot vector over gate slots
#   Each slot: [5 gate types | n target qubits | n+1 control/none]
#
# Actions: discrete
#   [0, n)     → Rx on qubit i
#   [n, 2n)    → Ry on qubit i
#   [2n, 3n)   → Rz on qubit i
#   [3n, ...)  → CNOT(ctrl, tgt)

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from qiskit.quantum_info import SparsePauliOp

from .vqe_utils import (
    optimize_parameters_warm_start,
    evaluate_energy,
    count_params,
    ROTATION_GATES,
)

REWARD_SUCCESS = +5.0
REWARD_FAILURE = -5.0

GATE_TYPE_INDEX = {
    "EMPTY": 0,
    "Rx":    1,
    "Ry":    2,
    "Rz":    3,
    "CNOT":  4,
}
N_GATE_TYPES = len(GATE_TYPE_INDEX)


class QuantumCircuitEnv(gym.Env):
    """
    Gymnasium env where the agent builds a VQE/MaxCut circuit gate by gate.

    Usage:
        ham, e0 = build_h2_hamiltonian_2q(0.735)
        env = QuantumCircuitEnv(ham, n_qubits=2, ground_state_energy=e0, max_depth=10)
        obs, info = env.reset(seed=42)
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        hamiltonian,
        n_qubits,
        ground_state_energy,
        max_depth=15,
        energy_threshold=0.0016,
        cobyla_max_iter=300,
        reward_success=REWARD_SUCCESS,
        reward_failure=REWARD_FAILURE,
        render_mode=None,
    ):
        super().__init__()

        if hamiltonian.num_qubits != n_qubits:
            raise ValueError(
                f"Hamiltonian has {hamiltonian.num_qubits} qubits but n_qubits={n_qubits}."
            )

        self.hamiltonian = hamiltonian
        self.n_qubits = n_qubits
        self.ground_state_energy = ground_state_energy
        self.max_depth = max_depth
        self.energy_threshold = energy_threshold
        self.cobyla_max_iter = cobyla_max_iter
        self.reward_success = reward_success
        self.reward_failure = reward_failure
        self.render_mode = render_mode

        n = n_qubits
        self._cnot_pairs = [
            (ctrl, tgt)
            for ctrl in range(n)
            for tgt in range(n)
            if ctrl != tgt
        ]

        self._n_actions = 3 * n + len(self._cnot_pairs)
        self.action_space = spaces.Discrete(self._n_actions)

        self._gate_slot_dim = N_GATE_TYPES + n + (n + 1)
        obs_dim = max_depth * self._gate_slot_dim
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._gate_sequence = []
        self._current_params = None
        self._current_energy = 0.0
        self._prev_param_count = 0
        self._step_count = 0
        self._rng = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        self._gate_sequence = []
        self._current_params = None
        self._current_energy = self._compute_empty_circuit_energy()
        self._prev_param_count = 0
        self._step_count = 0

        return self._encode_state(), self._build_info()

    def step(self, action):
        assert self.action_space.contains(action), f"Action {action} out of range."

        gate = self._decode_action(action)
        self._gate_sequence.append(gate)
        self._step_count += 1

        new_params, new_energy = optimize_parameters_warm_start(
            n_qubits=self.n_qubits,
            gate_sequence=self._gate_sequence,
            hamiltonian=self.hamiltonian,
            prev_params=self._current_params,
            prev_gate_count=self._prev_param_count,
            max_iterations=self.cobyla_max_iter,
        )

        # positive reward if energy went down
        reward = float(self._current_energy - new_energy)

        self._current_params = new_params
        self._current_energy = new_energy
        self._prev_param_count = count_params(self._gate_sequence)

        energy_error = abs(new_energy - self.ground_state_energy)
        terminated = False
        success = False

        if energy_error <= self.energy_threshold:
            reward += self.reward_success
            terminated = True
            success = True
        elif self._step_count >= self.max_depth:
            reward += self.reward_failure
            terminated = True

        return self._encode_state(), reward, terminated, False, self._build_info(success=success)

    def render(self):
        from .vqe_utils import build_circuit

        if len(self._gate_sequence) == 0:
            circuit_str = f"Empty circuit ({self.n_qubits} qubits)"
        else:
            params = (
                self._current_params
                if self._current_params is not None
                else np.zeros(count_params(self._gate_sequence))
            )
            qc = build_circuit(self.n_qubits, self._gate_sequence, params)
            circuit_str = str(qc.draw(output="text"))

        err = abs(self._current_energy - self.ground_state_energy)
        status = (
            f"Step {self._step_count}/{self.max_depth}  |  "
            f"Energy: {self._current_energy:.6f}  |  "
            f"Error: {err:.6f}  |  "
            f"Threshold: {self.energy_threshold:.4f}"
        )
        output = f"{circuit_str}\n{status}"

        if self.render_mode == "human":
            print(output)
        elif self.render_mode == "ansi":
            return output

    def close(self):
        pass

    def _decode_action(self, action):
        n = self.n_qubits
        if action < n:
            return {"type": "Rx", "target": action, "control": None}
        elif action < 2 * n:
            return {"type": "Ry", "target": action - n, "control": None}
        elif action < 3 * n:
            return {"type": "Rz", "target": action - 2 * n, "control": None}
        else:
            ctrl, tgt = self._cnot_pairs[action - 3 * n]
            return {"type": "CNOT", "target": tgt, "control": ctrl}

    def _encode_state(self):
        n = self.n_qubits
        obs = np.zeros(self.max_depth * self._gate_slot_dim, dtype=np.float32)

        for k, gate in enumerate(self._gate_sequence):
            if k >= self.max_depth:
                break
            offset = k * self._gate_slot_dim
            obs[offset + GATE_TYPE_INDEX[gate["type"]]] = 1.0
            obs[offset + N_GATE_TYPES + gate["target"]] = 1.0
            # n = "no control"
            ctrl = gate["control"] if gate["control"] is not None else n
            obs[offset + N_GATE_TYPES + n + ctrl] = 1.0

        return obs

    def _compute_empty_circuit_energy(self):
        return evaluate_energy(
            n_qubits=self.n_qubits,
            gate_sequence=[],
            params=np.array([]),
            hamiltonian=self.hamiltonian,
        )

    def _build_info(self, success=False):
        return {
            "energy": self._current_energy,
            "depth": self._step_count,
            "energy_error": abs(self._current_energy - self.ground_state_energy),
            "ground_state_energy": self.ground_state_energy,
            "success": success,
            "gate_sequence": list(self._gate_sequence),
        }

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def obs_dim(self):
        return self.max_depth * self._gate_slot_dim

    @property
    def current_circuit_depth(self):
        return self._step_count

    @property
    def current_energy(self):
        return self._current_energy
