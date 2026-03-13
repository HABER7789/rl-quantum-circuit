# Progress Report: RL-Driven Variational Quantum Circuit Design

**Course:** CS 5100 | **Team:** Dhairya Patel, Dhruv Kansara, Tisha Patel | **Date:** March 10, 2026

---

## What we've done so far

We built the full codebase for Phase 1. The project structure looks like this:

```
src/environment/   Hamiltonians, VQE optimizer, RL environment (Gymnasium)
src/agents/        DDQN and PPO agents
src/baselines/     Random search, genetic algorithm, fixed-structure circuits
src/evaluation/    Metrics
experiments/       Training script
config/            Hyperparameter configs
tests/             Tests (25/28 passing)
```

Here's a breakdown of what each piece does and where it lives:

- **RL Environment** (`circuit_env.py`): This is the main environment the agent trains in. At each step the agent picks a gate to add to a quantum circuit (from Rx, Ry, Rz rotations and CNOT). After each gate we run COBYLA to find the best angles, and the reward is how much the energy went down. If the agent gets within chemical accuracy of the true ground state energy, the episode ends with a bonus reward.

- **Hamiltonians** (`hamiltonian.py`): We support two tasks: H2 ground-state energy (VQE) and MaxCut. The H2 Pauli coefficients are hardcoded from O'Malley et al. 2016 at 5 bond distances. The MaxCut Hamiltonian is built dynamically for any graph using NetworkX.

- **VQE Utils** (`vqe_utils.py`): Builds circuits from gate sequences, evaluates energy using Qiskit's Statevector simulator, and runs COBYLA optimization. We chose statevector (no shot noise) and COBYLA (gradient-free, works well for shallow circuits).

- **DDQN** (`dqn_agent.py`): Double DQN with replay buffer and target network. The "double" part means we use two networks to avoid action value overestimation (one picks the action, the other evaluates it).

- **PPO** (`ppo_agent.py`): On-policy agent. Collects one full episode, runs a few gradient updates, then discards the data. Uses GAE for advantage estimation and a clipped ratio objective to prevent large updates.

- **Baselines** (`baselines/`): Three baselines: random gate selection, a genetic algorithm, and fixed-structure circuits (Hardware-Efficient ansatz for VQE, QAOA for MaxCut).

- **Metrics** (`metrics.py`): Energy error, success rate, and MaxCut approximation ratio.

25 out of 28 tests pass. The 3 that fail are due to a small mismatch between our hardcoded H2 energies and what Qiskit's diagonalization returns (about 0.06 Ha off), and a Qiskit qubit-ordering edge case. Neither affects the training code.

---

## Next steps

1. Fix or update the 3 failing unit tests.
2. Run DDQN training on the 2-qubit H2 task and collect reward/energy curves.
3. Run all three baselines and compare against the RL agents.

---

## Decisions and tradeoffs

- **No real quantum hardware:** IBM device queues can be 1-24 hours per job, and noise would make the reward signal meaningless during early training. We use noiseless statevector simulation, same choice as the paper we're extending (Ostaszewski et al. 2021).

- **2-qubit H2 instead of 4-qubit:** The 2-qubit parity-reduced form is standard in VQE benchmarking literature and runs much faster, which matters when each training episode involves dozens of COBYLA runs.

- **DDQN as the main agent:** Gate selection is a discrete decision at each step, which is exactly what Q-learning handles well. PPO is implemented for comparison but DDQN is our primary bet.

---

## Remaining work

- Run DDQN training, collect baseline comparison results
- PPO training and hyperparameter tuning
- Ablation experiments (circuit depth, reward shaping)
- Final report and presentation
