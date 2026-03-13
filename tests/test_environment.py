# test_environment.py
# Sanity checks for Phase 1. Run with: pytest tests/ -v

import numpy as np
import pytest

from src.environment.hamiltonian import (
    build_h2_hamiltonian_2q,
    build_maxcut_hamiltonian,
    exact_ground_state_energy,
    get_benchmark_graph,
    _make_zz_string,
    _brute_force_maxcut,
    H2_GROUND_STATE_2Q,
    CHEMICAL_ACCURACY_HA,
)
from src.environment.vqe_utils import (
    build_circuit,
    evaluate_energy,
    optimize_parameters,
    count_params,
    GATE_Rx, GATE_Ry, GATE_Rz, GATE_CNOT,
)
from src.environment.circuit_env import QuantumCircuitEnv


class TestH2Hamiltonian:

    def test_build_2q_equilibrium(self):
        ham, e0 = build_h2_hamiltonian_2q(0.735)
        assert ham.num_qubits == 2
        assert abs(e0 - (-1.8572)) < 0.001, f"Expected ~-1.8572 Ha, got {e0}"

    def test_exact_diagonalization_matches_hardcoded(self):
        ham, e0_hardcoded = build_h2_hamiltonian_2q(0.735)
        e0_exact = exact_ground_state_energy(ham)
        assert abs(e0_exact - e0_hardcoded) < 1e-2, (
            f"Exact diag {e0_exact:.4f} vs hardcoded {e0_hardcoded:.4f} — too far apart"
        )

    def test_hamiltonian_is_hermitian(self):
        # physical requirement
        ham, _ = build_h2_hamiltonian_2q(0.735)
        mat = ham.to_matrix()
        assert np.allclose(mat, mat.conj().T, atol=1e-10), "H2 Hamiltonian is not Hermitian!"

    def test_multiple_bond_distances(self):
        for dist in [0.5, 0.735, 1.0, 1.5, 2.0]:
            ham, e0 = build_h2_hamiltonian_2q(dist)
            assert ham.num_qubits == 2
            assert e0 < 0, f"Ground state energy at R={dist} Å should be negative"

    def test_invalid_bond_distance_raises(self):
        with pytest.raises(ValueError):
            build_h2_hamiltonian_2q(10.0)


class TestMaxCutHamiltonian:

    def test_c4_maxcut_value(self):
        # C4 should have MaxCut = 4
        g = get_benchmark_graph("C4")
        ham, optimal_cut = build_maxcut_hamiltonian(g)
        assert optimal_cut == 4, f"C4 MaxCut should be 4, got {optimal_cut}"
        assert ham.num_qubits == 4

    def test_k4_maxcut_value(self):
        g = get_benchmark_graph("K4")
        _, optimal_cut = build_maxcut_hamiltonian(g)
        assert optimal_cut == 4, f"K4 MaxCut should be 4, got {optimal_cut}"

    def test_ground_state_energy_equals_negative_maxcut(self):
        # key identity: E_0 = -MaxCut for our Ising formulation
        g = get_benchmark_graph("C4")
        ham, optimal_cut = build_maxcut_hamiltonian(g)
        e0 = exact_ground_state_energy(ham)
        assert abs(e0 - (-optimal_cut)) < 1e-6, (
            f"E_0={e0:.4f} but -MaxCut={-optimal_cut:.4f} — these should match exactly"
        )

    def test_hamiltonian_hermitian(self):
        g = get_benchmark_graph("K4")
        ham, _ = build_maxcut_hamiltonian(g)
        mat = ham.to_matrix()
        assert np.allclose(mat, mat.conj().T, atol=1e-10)

    def test_zz_pauli_string_little_endian(self):
        # took a while to get this right — qubit 0 is rightmost char
        s = _make_zz_string(4, 0, 2)
        assert s == "ZIZI", f"Expected 'ZIZI', got '{s}' — check little-endian ordering!"

    def test_brute_force_path_graph(self):
        import networkx as nx
        g = nx.path_graph(3)
        assert _brute_force_maxcut(g) == 2


class TestVQEUtils:

    def test_build_circuit_empty(self):
        from qiskit import QuantumCircuit
        qc = build_circuit(2, [], np.array([]))
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

    def test_build_circuit_wrong_param_count_raises(self):
        gates = [{"type": "Rx", "target": 0, "control": None}]
        with pytest.raises(ValueError):
            build_circuit(2, gates, np.array([1.0, 2.0]))

    def test_count_params(self):
        gates = [
            {"type": "Ry",   "target": 0, "control": None},
            {"type": "CNOT", "target": 1, "control": 0},  # no params
            {"type": "Rz",   "target": 1, "control": None},
        ]
        assert count_params(gates) == 2

    def test_evaluate_energy_empty_circuit(self):
        ham, _ = build_h2_hamiltonian_2q(0.735)
        energy = evaluate_energy(2, [], np.array([]), ham)
        assert np.isfinite(energy)

    def test_rotation_changes_energy(self):
        ham, _ = build_h2_hamiltonian_2q(0.735)
        e_baseline = evaluate_energy(2, [], np.array([]), ham)
        gates = [{"type": "Ry", "target": 0, "control": None}]
        e_rotated = evaluate_energy(2, gates, np.array([np.pi / 2]), ham)
        assert e_baseline != pytest.approx(e_rotated, abs=1e-6), \
            "Energy didn't change with rotation — something is wrong"

    def test_cobyla_converges_h2_2q(self):
        """
        VQE should reach near-chemical-accuracy on H2 2Q with this ansatz.
        Ry(q0) - Ry(q1) - CNOT(0,1) - Ry(q0) - Ry(q1) — basically a truncated UCCSD.
        Testing provided good convergence (error < 5x chemical accuracy).
        """
        ham, e0_exact = build_h2_hamiltonian_2q(0.735)
        ansatz = [
            {"type": "Ry",   "target": 0, "control": None},
            {"type": "Ry",   "target": 1, "control": None},
            {"type": "CNOT", "target": 1, "control": 0},
            {"type": "Ry",   "target": 0, "control": None},
            {"type": "Ry",   "target": 1, "control": None},
        ]
        _, energy = optimize_parameters(2, ansatz, ham, max_iterations=1000)
        error = abs(energy - e0_exact)
        assert error < CHEMICAL_ACCURACY_HA * 5, (
            f"COBYLA didn't converge! error={error:.6f} Ha "
            f"(got {energy:.4f}, target {e0_exact:.4f})"
        )


class TestQuantumCircuitEnv:

    @pytest.fixture
    def h2_env(self):
        ham, e0 = build_h2_hamiltonian_2q(0.735)
        return QuantumCircuitEnv(
            hamiltonian=ham,
            n_qubits=2,
            ground_state_energy=e0,
            max_depth=10,
            energy_threshold=0.0016,
        )

    def test_reset_returns_correct_shapes(self, h2_env):
        obs, info = h2_env.reset(seed=0)
        assert obs.shape == (h2_env.obs_dim,)
        assert obs.dtype == np.float32
        assert "energy" in info

    def test_reset_obs_is_all_zeros(self, h2_env):
        obs, _ = h2_env.reset(seed=0)
        assert np.all(obs == 0.0), "Empty circuit state should be all zeros"

    def test_step_return_types(self, h2_env):
        h2_env.reset(seed=0)
        obs, reward, terminated, truncated, info = h2_env.step(0)
        assert obs.shape == (h2_env.obs_dim,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_one_hot_encoding(self, h2_env):
        # each gate slot should have exactly 3 ones (type + target + control)
        h2_env.reset(seed=0)
        obs, _, _, _, _ = h2_env.step(0)  # Rx on qubit 0

        slot_dim = h2_env._gate_slot_dim
        first_slot = obs[:slot_dim]
        remaining = obs[slot_dim:]

        assert first_slot.sum() == pytest.approx(3.0, abs=1e-6), \
            f"First gate slot should have exactly 3 ones, got sum={first_slot.sum()}"
        assert np.all(remaining == 0.0), "Unfilled gate slots should be all zeros"

    def test_action_space_size_n2(self, h2_env):
        # n=2: 3*2 rotation actions + 2*1 CNOT actions = 8 total
        assert h2_env.n_actions == 8

    def test_decode_rotation_action(self, h2_env):
        gate = h2_env._decode_action(0)
        assert gate["type"] == "Rx"
        assert gate["target"] == 0
        assert gate["control"] is None

    def test_decode_cnot_action(self, h2_env):
        gate = h2_env._decode_action(6)  # CNOT actions start at 3*n=6 for n=2
        assert gate["type"] == "CNOT"

    def test_at_least_one_action_improves_energy(self, h2_env):
        rewards = []
        for action in range(h2_env.n_actions):
            obs, _ = h2_env.reset(seed=action)
            _, reward, _, _, _ = h2_env.step(action)
            rewards.append(reward)

        assert any(r > 0 for r in rewards), \
            "No action improved the energy. Reward signal might be wrong."

    def test_max_depth_terminates_episode(self, h2_env):
        obs, _ = h2_env.reset(seed=0)
        terminated = False
        steps = 0
        while not terminated:
            _, _, terminated, _, info = h2_env.step(0)
            steps += 1
        assert steps == h2_env.max_depth or info["success"], \
            f"Episode ended after {steps} steps but max_depth={h2_env.max_depth}"

    def test_maxcut_env_construction(self):
        g = get_benchmark_graph("C4")
        ham, optimal_cut = build_maxcut_hamiltonian(g)
        e0 = -optimal_cut
        env = QuantumCircuitEnv(
            hamiltonian=ham,
            n_qubits=4,
            ground_state_energy=e0,
            max_depth=15,
            energy_threshold=0.1,  # I relaxed this for test speed
        )
        obs, info = env.reset(seed=0)
        assert obs.shape == (env.obs_dim,)
        assert env.n_actions == 24  # 3*4 + 4*3


class TestFullEpisode:

    def test_episode_runs_without_errors(self):
        """Full random episode — just checking nothing crashes."""
        ham, e0 = build_h2_hamiltonian_2q(0.735)
        env = QuantumCircuitEnv(
            hamiltonian=ham,
            n_qubits=2,
            ground_state_energy=e0,
            max_depth=5,
            energy_threshold=0.0016,
            cobyla_max_iter=100,  # fewer iters for test speed
        )
        rng = np.random.default_rng(0)
        obs, _ = env.reset(seed=0)
        done = False
        total_reward = 0.0

        while not done:
            action = int(rng.integers(0, env.n_actions))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        assert np.isfinite(total_reward)
        assert np.isfinite(info["energy"])
        assert info["depth"] <= env.max_depth
