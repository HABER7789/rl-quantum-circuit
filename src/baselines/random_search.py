# random_search.py
# Dumbest possible baseline: pick random gates at every step.
# If our RL agent can't beat this, something is seriously wrong.

import numpy as np
from src.environment.circuit_env import QuantumCircuitEnv


def run_random_search(
    hamiltonian,
    n_qubits,
    ground_state_energy,
    max_depth=15,
    energy_threshold=0.0016,
    n_episodes=100,
    seed=42,
):
    env = QuantumCircuitEnv(
        hamiltonian=hamiltonian,
        n_qubits=n_qubits,
        ground_state_energy=ground_state_energy,
        max_depth=max_depth,
        energy_threshold=energy_threshold,
    )
    rng = np.random.default_rng(seed)

    best_energy = float("inf")
    best_depth = 0
    energy_errors = []
    depths = []
    n_success = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31)))
        done = False

        while not done:
            action = int(rng.integers(0, env.n_actions))
            obs, reward, done, truncated, info = env.step(action)

        energy_errors.append(info["energy_error"])
        depths.append(info["depth"])

        if info["success"]:
            n_success += 1

        if info["energy"] < best_energy:
            best_energy = info["energy"]
            best_depth = info["depth"]

    return {
        "best_energy":   best_energy,
        "best_depth":    best_depth,
        "energy_errors": energy_errors,
        "depths":        depths,
        "success_rate":  n_success / n_episodes,
    }
