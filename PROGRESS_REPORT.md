# Progress Report: RL-Driven Variational Quantum Circuit Design

**Course:** CS 5100 | **Team:** Dhairya Patel, Dhruv Kansara, Tisha Patel | **Date:** March 31, 2026

---

## What we have done so far

We finished the full Phase 1 codebase and ran the first round of training experiments.

**Environment and Hamiltonians.** The core RL environment (`src/environment/circuit_env.py`) is a custom Gymnasium environment where the agent adds one gate at a time from {Rx, Ry, Rz, CNOT}. After each gate is placed, COBYLA finds the best rotation angles and the reward is the resulting energy decrease. We have two tasks: H2 ground-state energy using the 2-qubit parity-reduced Hamiltonian from O'Malley et al. [1], and MaxCut using the standard Ising formulation. Both are unit tested; all **28/28 tests now pass**.

There was a bug worth documenting. We initially set `H2_GROUND_STATE_2Q` to the full-CI energy of −1.8572 Ha (at R=0.735 Å) from [1], but our Pauli coefficients go through the parity reduction and two-qubit tapering, which shifts the zero of energy. The correct reference for our encoding is the smallest eigenvalue of the matrix we actually diagonalize, which turns out to be −1.9154 Ha. We fixed this by computing ground state energies via exact diagonalization (`np.linalg.eigvalsh`) at each bond distance and updating the lookup table. The quantum chemistry is unchanged (the H2 Hamiltonian is still the same), but we were comparing VQE output against the wrong target, which made every episode look like a failure.

**Agents.** We implemented Double DQN (DDQN) [2] and PPO with GAE [3] in `src/agents/`. DDQN uses a replay buffer and a target network that syncs every 200 steps. PPO collects one full episode, computes advantages using GAE (λ=0.95), and runs 4 epochs of clipped gradient updates (ε=0.2). We followed Ostaszewski et al. [4] in using DDQN for gate selection; they showed it works better than policy gradient methods for discrete one-shot gate choices.

**Training run.** We ran 2000 episodes of DDQN on H2 at R=0.735 Å. The checkpoint and full episode log are in `checkpoints/` and `logs/`. The agent gets close to −1.9154 Ha but does not consistently hit chemical accuracy (0.0016 Ha = 1 kcal/mol). We think the depth budget of 10 gates is too tight and plan to relax it.

**Baselines.** All three baselines are implemented and verified:

| Method | Energy (Ha) | Error (Ha) | Depth | Succeeded |
|---|---|---|---|---|
| Random search (n=200 episodes) | −1.915371 | 0.000029 | 10 | Yes |
| Genetic algorithm (gen=30, pop=20) | −1.915371 | 0.000029 | 7 | Yes |
| HE ansatz (L=2, 5 COBYLA restarts) | −1.915371 | 0.000029 | 8 | Yes |

All three reach chemical accuracy on H2 2Q without any learning. This is not too surprising for a 2-qubit system; the search space is small enough that almost any optimizer with a flexible enough circuit will find it. The real comparison will be at multiple bond distances and on larger MaxCut graphs, where fixed-structure methods are harder to tune.

**Tests.** The test suite covers Hamiltonian correctness, Hermiticity, MaxCut Ising energy identity, VQE convergence, state encoding, action decoding, episode termination, and a full random episode smoke test.

---

## Next steps

1. Extend max gate depth from 10 to 15 and slow down epsilon decay to give the agent more room to explore.
2. Run training across all 5 H2 bond distances and compare RL against baselines at each one.
3. Start MaxCut experiments on C4, K4, and 4-node random graphs.
4. Build a short analysis notebook with reward curves, energy convergence plots, and the baseline comparison table.
5. Run PPO training and compare convergence with DDQN.

---

## Challenges and decisions

The Hamiltonian reference energy issue took the most debugging time. We were getting failing tests and strange reward signals before tracing the problem to the mismatch between the literature energy and the eigenvalue of our reduced operator. The fix is correct and is now documented in the code.

We decided to stick with statevector simulation throughout. IBM device queues are typically 1–24 hours per job, and shot noise during random exploration early in training would make the reward signal noisy enough to break learning. Every VQE RL paper we found uses noiseless simulation [4], and for circuits with 5 or fewer qubits, exact expectation values are fast.

The 2-qubit H2 reduction was a deliberate choice. The parity-reduced form [1] is well-validated in the VQE literature and cuts the qubit count from 4 (Jordan-Wigner) to 2. This matters because COBYLA runs inside the environment at every step, so lower qubit count means faster per-episode wall time.

---

## Plan for the next month

The next two weeks will go toward training experiments across all bond distances and MaxCut graphs, followed by analysis and writing. The environment, agents, baselines, and metrics are all working, so the remaining work is running the experiments, interpreting the results, and writing up what we find. We expect DDQN to match or slightly outperform random search on H2 2Q, and to show a larger advantage on MaxCut graphs where the action space is bigger.

---

## References

[1] O'Malley, P. J. J. et al. "Scalable Quantum Simulation of Molecular Energies." *Physical Review X*, 6, 031007 (2016).

[2] van Hasselt, H., Guez, A., and Silver, D. "Deep Reinforcement Learning with Double Q-Learning." *AAAI*, 2016.

[3] Schulman, J. et al. "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*, 2017.

[4] Ostaszewski, M. et al. "Reinforcement Learning for Optimization of Variational Quantum Circuit Architectures." *NeurIPS 2021*.
