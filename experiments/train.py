# train.py
# Main training script.
# Usage: python experiments/train.py --config config/default.yaml [--seed N]

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.hamiltonian import (
    build_h2_hamiltonian_2q,
    build_maxcut_hamiltonian,
    get_benchmark_graph,
)
from src.environment.circuit_env import QuantumCircuitEnv
from src.agents.dqn_agent import DDQNAgent
from src.agents.ppo_agent import PPOAgent


def load_config(path, seed_override=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if seed_override is not None:
        cfg["training"]["seed"] = seed_override
    return cfg


def build_environment(cfg):
    task = cfg["task"]
    n_qubits  = task["n_qubits"]
    max_depth = task["max_depth"]
    threshold = task["energy_threshold"]
    cobyla_iter = cfg["training"]["cobyla_max_iter"]

    if task["name"] == "h2_vqe":
        bond_dist = cfg["h2"]["bond_distance"]
        hamiltonian, e0 = build_h2_hamiltonian_2q(bond_dist)
        print(f"Task: H2 VQE at R={bond_dist} Å  |  Ground state: {e0:.6f} Ha  |  Qubits: {n_qubits}")

    elif task["name"] == "maxcut":
        graph_name = cfg["maxcut"]["graph"]
        graph = get_benchmark_graph(graph_name)
        hamiltonian, optimal_cut = build_maxcut_hamiltonian(graph)
        e0 = -optimal_cut  # minimize H to maximize cut
        print(f"Task: MaxCut on {graph_name}  |  Optimal cut: {optimal_cut}  |  Qubits: {n_qubits}")

    else:
        raise ValueError(f"Unknown task '{task['name']}'. Choose 'h2_vqe' or 'maxcut'.")

    env = QuantumCircuitEnv(
        hamiltonian=hamiltonian,
        n_qubits=n_qubits,
        ground_state_energy=e0,
        max_depth=max_depth,
        energy_threshold=threshold,
        cobyla_max_iter=cobyla_iter,
    )
    return env, e0


def build_agent(cfg, obs_dim, n_actions):
    agent_type = cfg["agent"]["type"]
    seed = cfg["training"]["seed"]
    device = cfg["device"]

    if agent_type == "ddqn":
        dc = cfg["ddqn"]
        return DDQNAgent(
            obs_dim=obs_dim,
            n_actions=n_actions,
            gamma=dc["gamma"],
            lr=dc["lr"],
            epsilon_start=dc["epsilon_start"],
            epsilon_end=dc["epsilon_end"],
            epsilon_decay=dc["epsilon_decay"],
            batch_size=dc["batch_size"],
            buffer_size=dc["buffer_size"],
            target_update=dc["target_update"],
            min_buffer=dc["min_buffer"],
            hidden_dim=cfg["agent"]["hidden_dim"],
            device=device,
            seed=seed,
        )

    elif agent_type == "ppo":
        pc = cfg["ppo"]
        return PPOAgent(
            obs_dim=obs_dim,
            n_actions=n_actions,
            lr=pc["lr"],
            gamma=pc["gamma"],
            gae_lambda=pc["gae_lambda"],
            clip_eps=pc["clip_eps"],
            n_epochs=pc["n_epochs"],
            batch_size=pc["batch_size"],
            vf_coef=pc["vf_coef"],
            ent_coef=pc["ent_coef"],
            max_grad_norm=pc["max_grad_norm"],
            hidden_dim=cfg["agent"]["hidden_dim"],
            device=device,
            seed=seed,
        )

    raise ValueError(f"Unknown agent type '{agent_type}'. Choose 'ddqn' or 'ppo'.")


def evaluate_greedy(env, agent, n_eval, seed):
    """No exploration — just run the learned policy and see how it does."""
    errors, depths, successes = [], [], []

    for i in range(n_eval):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            if isinstance(agent, DDQNAgent):
                action = agent.select_action(obs, greedy=True)
            else:
                action, _, _ = agent.select_action(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        errors.append(info["energy_error"])
        depths.append(info["depth"])
        successes.append(info["success"])

    return {
        "mean_error":   float(np.mean(errors)),
        "success_rate": float(np.mean(successes)),
        "mean_depth":   float(np.mean(depths)),
    }


def train_ddqn(env, agent, cfg):
    tc = cfg["training"]
    n_episodes    = tc["n_episodes"]
    eval_interval = tc["eval_interval"]
    eval_eps      = tc["eval_episodes"]
    seed          = tc["seed"]
    rng = np.random.default_rng(seed)

    logs = []
    t0 = time.time()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, done)
            agent.update()  # no-op until buffer fills up

            ep_reward += reward
            obs = next_obs

        logs.append({
            "episode":      ep,
            "reward":       ep_reward,
            "energy":       info["energy"],
            "energy_error": info["energy_error"],
            "depth":        info["depth"],
            "success":      info["success"],
            "epsilon":      agent.epsilon,
        })

        if (ep + 1) % eval_interval == 0:
            eval_stats = evaluate_greedy(env, agent, eval_eps, seed=seed + ep)
            elapsed = time.time() - t0
            print(
                f"[Ep {ep+1:5d}/{n_episodes}]  "
                f"eps={agent.epsilon:.3f}  |  "
                f"eval_error={eval_stats['mean_error']:.4f}  |  "
                f"success={eval_stats['success_rate']:.2f}  |  "
                f"depth={eval_stats['mean_depth']:.1f}  |  "
                f"time={elapsed:.0f}s"
            )

    return logs


def train_ppo(env, agent, cfg):
    tc = cfg["training"]
    n_episodes    = tc["n_episodes"]
    eval_interval = tc["eval_interval"]
    eval_eps      = tc["eval_episodes"]
    seed          = tc["seed"]
    rng = np.random.default_rng(seed)

    logs = []
    t0 = time.time()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        ep_reward = 0.0
        done = False
        last_value = 0.0

        while not done:
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, log_prob, reward, value, done)
            ep_reward += reward
            obs = next_obs
            last_value = value

        # PPO update at end of episode — bootstrap with 0 if it succeeded
        agent.update(last_value=0.0 if info["success"] else last_value)

        logs.append({
            "episode":      ep,
            "reward":       ep_reward,
            "energy":       info["energy"],
            "energy_error": info["energy_error"],
            "depth":        info["depth"],
            "success":      info["success"],
        })

        if (ep + 1) % eval_interval == 0:
            eval_stats = evaluate_greedy(env, agent, eval_eps, seed=seed + ep)
            elapsed = time.time() - t0
            print(
                f"[Ep {ep+1:5d}/{n_episodes}]  "
                f"eval_error={eval_stats['mean_error']:.4f}  |  "
                f"success={eval_stats['success_rate']:.2f}  |  "
                f"depth={eval_stats['mean_depth']:.1f}  |  "
                f"time={elapsed:.0f}s"
            )

    return logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, seed_override=args.seed)
    seed = cfg["training"]["seed"]

    np.random.seed(seed)

    env, e0 = build_environment(cfg)
    agent = build_agent(cfg, env.obs_dim, env.n_actions)

    agent_type = cfg["agent"]["type"].upper()
    print(f"Agent: {agent_type}  |  obs_dim={env.obs_dim}  |  n_actions={env.n_actions}  |  seed={seed}")
    print("-" * 60)

    if cfg["agent"]["type"] == "ddqn":
        logs = train_ddqn(env, agent, cfg)
    else:
        logs = train_ppo(env, agent, cfg)

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{cfg['task']['name']}_{cfg['agent']['type']}_seed{seed}"
    agent.save(str(ckpt_dir / f"{run_name}.pt"))

    log_dir = Path(cfg["training"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"{run_name}.json", "w") as f:
        json.dump(logs, f, indent=2)

    print(f"\nDone. Checkpoint: {ckpt_dir / f'{run_name}.pt'}")
    print(f"Training log:   {log_dir / f'{run_name}.json'}")


if __name__ == "__main__":
    main()
