# RL-Driven Variational Quantum Circuit Design

**CS 5100   Dhairya Patel, Dhruv Kansara, Tisha Patel**

We built an RL agent that learns to construct quantum circuit ansätze from scratch, without assuming any fixed circuit structure. The agent is tested on two problems:

1. **VQE**: finding the ground state energy of H2 using the 2-qubit parity-reduced Hamiltonian
2. **MaxCut**: combinatorial optimization via Ising Hamiltonian minimization

The agent picks one gate per step from {Rx, Ry, Rz, CNOT}. After each gate is placed, COBYLA optimizes the rotation angles and returns the energy as feedback. Reward is the energy decrease at each step.

## Setup

```bash
# Python 3.10+ required
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
# 1. Verify everything is working
PYTHONPATH=. pytest tests/ -v

# 2. Fastest standard run (runs tests, DDQN, and PPO sequentially)
./run_all.sh

# 3. Manual training runs
python experiments/train.py --config config/default.yaml
```

Training runs for 2000 episodes per architectural pass and takes roughly 8-10 minutes. Checkpoints and logs go to `checkpoints/` and `logs/`. To manually swap between Double DQN and PPO, edit `config/default.yaml` and change `type: "ddqn"` to `type: "ppo"`.

## Layout

```
src/environment/   Hamiltonian definitions, VQE circuit builder, Gymnasium env
src/agents/        DDQN (with target network + replay buffer) and PPO (with GAE)
src/baselines/     Random search, genetic algorithm, HE ansatz, QAOA
src/evaluation/    Energy error, success rate, approximation ratio metrics
experiments/       train.py (main entry point)
config/            YAML configs for hyperparameters
tests/             28 unit/integration tests, all passing
```

## A note on tests

Run tests with `PYTHONPATH=.` from the project root; pytest doesn't add the project root to the path by default.

All 28 tests pass as of the current commit.
