#!/bin/bash
# run_all.sh - Automated pipeline execution for CS 5100 Final Project

echo "=========================================================="
echo "    RL-Driven Quantum Circuit Design Pipeline"
echo "=========================================================="

echo -e "\n[1/4] Running Unit Tests..."
PYTHONPATH=. pytest tests/ -v -W ignore

echo -e "\n[2/4] Running DDQN Training..."
# Ensure default starts as ddqn
sed -i '' 's/type: "ppo"/type: "ddqn"/g' config/default.yaml
python experiments/train.py --config config/default.yaml

echo -e "\n[3/4] Switching architecture to PPO and training..."
# Swap config to PPO
sed -i '' 's/type: "ddqn"/type: "ppo"/g' config/default.yaml
python experiments/train.py --config config/default.yaml

echo -e "\n[4/4] Restoring configuration and exiting..."
# Restore back to default
sed -i '' 's/type: "ppo"/type: "ddqn"/g' config/default.yaml

echo -e "\n=========================================================="
echo " Pipeline Complete. Check the 'logs/' folder for results."
echo "=========================================================="
