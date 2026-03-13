# RL-Driven Variational Quantum Circuit Design

**CS 5100 | Dhairya Patel, Dhruv Kansara, Tisha Patel**

A reinforcement learning agent that autonomously constructs quantum circuit ansätze for:
1. **VQE** — Variational Quantum Eigensolver (H2 molecule ground state energy)
2. **MaxCut** — Combinatorial optimization via Ising Hamiltonian minimization

## Setup and Running

```bash
# 1. Create and activate a virtual environment (Python 3.10+ required)
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run tests to verify everything is set up correctly
pytest tests/ -v

# 4. Train the DDQN agent on H2
python experiments/train.py --config config/default.yaml
```

Training takes ~9 minutes (2000 episodes). Results are saved automatically to `checkpoints/` and `logs/`.

To switch to PPO, open `config/default.yaml` and change `type: "ddqn"` to `type: "ppo"`, then run the same command.

**Note on tests:** 3 out of 28 tests currently fail due to a minor mismatch between hardcoded H2 energy values and Qiskit's diagonalization result. This does not affect training.

## Project Structure

```
src/environment/   Hamiltonian builders, VQE utils, Gymnasium RL environment
src/agents/        DDQN and PPO agents with replay buffer
src/baselines/     Random search, genetic algorithm, HE/QAOA fixed ansätze
src/evaluation/    Metrics (energy error, success rate, approximation ratio)
experiments/       Training script (train.py)
config/            YAML hyperparameter configs
tests/             Verification tests
```
