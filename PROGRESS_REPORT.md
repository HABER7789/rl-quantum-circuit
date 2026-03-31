# Progress Report: RL-Driven Variational Quantum Circuit Design

**Course:** CS 5100 | **Team:** Dhairya Patel, Dhruv Kansara, Tisha Patel | **Date:** March 31, 2026

---

## What we have done so far

We have completed the full Phase 1 codebase and run the first round of training experiments.

**Environment and Hamiltonians.** The core RL environment (`src/environment/circuit_env.py`) is a custom Gymnasium environment where the agent adds one gate per step from the set {Rx, Ry, Rz, CNOT}. After each gate, COBYLA finds the best rotation angles, and the reward is the resulting energy decrease. We implement two tasks: H2 ground-state energy using the 2-qubit parity-reduced Hamiltonian from O'Malley et al. [1], and MaxCut using the standard Ising formulation. Both are unit tested — we now have **28/28 tests passing**.

We had a bug worth documenting: we initially seeded `H2_GROUND_STATE_2Q` with the full-CI energy (−1.8572 Ha at R=0.735 Å) reported in [1], but our Pauli coefficients go through the parity reduction and a two-qubit tapering, which shifts the zero of energy. The correct reference for our encoding is the smallest eigenvalue of the matrix we actually encode, which is −1.9154 Ha. The fix was to update the lookup table to values obtained by exact diagonalization (`np.linalg.eigvalsh`) of each SparsePauliOp. This is not a science error — our quantum chemistry is unchanged — but it was causing VQE to compare its output against the wrong target, making every episode look like a failure.

**Agents.** We implemented Double DQN (DDQN) [2] and PPO with GAE [3] in `src/agents/`. DDQN uses a replay buffer and a delayed target network that syncs every 200 steps. PPO collects one full episode, computes advantages using GAE (λ=0.95), and runs 4 epochs of clipped gradient updates (ε=0.2). Both agents are designed following Ostaszewski et al. [4], who showed DDQN works well for discrete gate selection because each gate choice is a one-shot action with no continuous parameter.

**Training run.** We completed a 2000-episode DDQN run on H2 at R=0.735 Å. The checkpoint and full episode log are included in `checkpoints/` and `logs/`. The agent reliably reaches energies near −1.9154 Ha but does not yet consistently cross the chemical accuracy threshold (0.0016 Ha = 1 kcal/mol) — we believe this is a hyperparameter issue (depth budget of 10 gates may be too tight) that we plan to address next.

**Baselines.** All three baseline methods are implemented and verified to run:

| Method | Energy (Ha) | Error (Ha) | Depth | Succeeded |
|---|---|---|---|---|
| Random search (n=200 episodes) | −1.915371 | 0.000029 | 10 | Yes |
| Genetic algorithm (gen=30, pop=20) | −1.915371 | 0.000029 | 7 | Yes |
| HE ansatz (L=2, 5 COBYLA restarts) | −1.915371 | 0.000029 | 8 | Yes |

All three reach chemical accuracy on H2 2Q without learning. This is partly expected for such a small system — 2 qubits with a well-known UCCSD-like structure almost any flexible optimizer can find. The interesting comparison will come when we extend to more bond distances and larger graphs, where fixed-structure methods struggle more.

**Tests.** The full test suite covers Hamiltonian correctness, Hermiticity, MaxCut Ising energy identity, VQE convergence, state encoding, action decoding, episode termination logic, and a full random episode smoke test.

---

## Next steps

1. Fix the DDQN hyperparameters: extend max depth from 10 to 15, lower epsilon decay to allow longer exploration.
2. Run training across all 5 H2 bond distances and compare RL against the baselines at each distance.
3. Start MaxCut experiments on C4, K4, and 4-node random graphs.
4. Add a short analysis notebook with reward curves, energy convergence plots, and the baseline comparison table.
5. Run PPO training and compare convergence speed against DDQN.

---

## Challenges and decisions

The main surprise was the Hamiltonian reference energy issue described above. It wasted some debugging time but the fix is correct and documented in the code.

We are sticking with statevector simulation throughout. IBM device queues are typically 1–24 hours per job, and shot noise during early random exploration would make the reward signal meaningless. The Ostaszewski et al. paper and essentially all VQE RL results in the literature use noiseless simulation [4]. For ≤5 qubits, exact expectation values are fast and informative.

The 2-qubit H2 reduction was a deliberate choice. The parity-reduced form [1] is well-validated in the VQE literature and reduces qubit count from 4 (Jordan-Wigner) to 2, which matters because each COBYLA call inside the environment re-evaluates the full circuit, and training involves thousands of episodes.

---

## Plan for the next month

We will spend the next two weeks finishing training experiments across all bond distances and tasks, then analyze results and write the final report. The core infrastructure (environment, agents, baselines, metrics) is complete and working, so the remaining effort is running experiments, interpreting results, and writing. We expect the final comparison to show DDQN outperforming random search and matching the HE ansatz on H2 2Q, with a clearer advantage on larger MaxCut graphs where the search space grows.

---

## References

[1] O'Malley, P. J. J. et al. "Scalable Quantum Simulation of Molecular Energies." *Physical Review X*, 6, 031007 (2016).

[2] van Hasselt, H., Guez, A., and Silver, D. "Deep Reinforcement Learning with Double Q-Learning." *AAAI*, 2016.

[3] Schulman, J. et al. "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*, 2017.

[4] Ostaszewski, M. et al. "Reinforcement Learning for Optimization of Variational Quantum Circuit Architectures." *NeurIPS 2021*.
