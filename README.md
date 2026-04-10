# RL-Driven Variational Quantum Circuit Design

**CS 5100   Dhairya Patel, Dhruv Kansara, Tisha Patel**

An RL agent that builds quantum circuit ansätze gate by gate. The agent picks one gate per step from {Rx, Ry, Rz, CNOT}. After each gate, COBYLA finds the best rotation angles and the reward is the resulting energy decrease. Tested on H2 ground-state energy (VQE) using the 2-qubit parity-reduced Hamiltonian.

## Setup

```bash
# Python 3.10+ required
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
# 1. Run tests first
PYTHONPATH=. pytest tests/ -v

# 2. Run everything (tests → DDQN training → PPO training)
./run_all.sh

# 3. Train a single agent manually
python experiments/train.py --config config/default.yaml

# 4. Run baselines
python experiments/run_baselines.py

# 5. Regenerate plots from existing logs
python generate_plots.py
```

Training is 2000 episodes per agent, roughly 8-10 minutes on CPU. Checkpoints save to `checkpoints/`, logs and plots to `logs/`. To switch between DDQN and PPO, change `type:` in `config/default.yaml`.

Pre-trained results are already in `logs/` if you just want to look at the data without re-running.

## Layout

```
src/environment/   Hamiltonian definitions, VQE circuit builder, Gymnasium env
src/agents/        DDQN (target network + replay buffer) and PPO (GAE)
src/baselines/     Random search, genetic algorithm, HE ansatz, QAOA
src/evaluation/    Metrics: energy error, success rate, approximation ratio
experiments/       train.py and run_baselines.py
config/            Hyperparameters (default.yaml)
logs/              Training logs (JSON) and plots (PNG) from completed runs
checkpoints/       Saved model weights
tests/             28 unit/integration tests
notebooks/         analysis.ipynb — results walkthrough
final_paper.tex    Write-up (compile in Overleaf)
```

## Tests

Run from the project root with `PYTHONPATH=.` — pytest doesn't add the root to the path by default.

All 28 tests pass.
