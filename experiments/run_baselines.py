# run_baselines.py
# Runs all three baselines on H2 at equilibrium and prints a comparison table.
# Usage: python experiments/run_baselines.py [--seed N] [--out path/to/results.json]
#
# This is separate from the RL training loop. Baselines don't learn — they
# just try to find a good circuit via random search, genetic evolution, or
# a hand-designed template. We use them to benchmark how much the RL agents
# actually help over dumb alternatives.

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.hamiltonian import (
    build_h2_hamiltonian_2q,
    CHEMICAL_ACCURACY_HA,
)
from src.baselines.random_search import run_random_search
from src.baselines.genetic_algorithm import run_genetic_algorithm
from src.baselines.fixed_ansatz import run_he_ansatz


def _fmt_row(label, result, e0):
    err = result["energy_error"]
    ok  = "YES" if result["success"] else "no"
    depth = result.get("depth", result.get("best_depth", "?"))
    return f"  {label:<22}  E = {result['best_energy']:+.6f} Ha  err = {err:.6f}  depth = {depth:<4}  success = {ok}"


def run_all(bond_distance=0.735, seed=42, cobyla_iter=500, verbose=True):
    ham, e0 = build_h2_hamiltonian_2q(bond_distance)

    if verbose:
        print(f"\nH2 VQE benchmark  |  R = {bond_distance} Å  |  E_0 = {e0:.6f} Ha  |  seed = {seed}")
        print(f"Chemical accuracy threshold: {CHEMICAL_ACCURACY_HA} Ha (1 kcal/mol)")
        print("-" * 72)

    results = {}

    # --- random search ---
    if verbose:
        print("Running random search ...")
    t0 = time.time()
    rs = run_random_search(
        hamiltonian=ham,
        n_qubits=2,
        ground_state_energy=e0,
        max_depth=10,
        n_episodes=200,
        energy_threshold=CHEMICAL_ACCURACY_HA,
        seed=seed,
    )
    # random_search returns best_depth + success_rate, normalize to common shape
    rs["energy_error"] = abs(rs["best_energy"] - e0)
    rs["success"]      = rs["success_rate"] > 0
    rs["depth"]        = rs["best_depth"]
    rs["wall_time_s"]  = round(time.time() - t0, 2)
    results["random_search"] = rs
    if verbose:
        print(_fmt_row("random search (n=200)", rs, e0))

    # --- genetic algorithm ---
    if verbose:
        print("Running genetic algorithm ...")
    t0 = time.time()
    ga = run_genetic_algorithm(
        hamiltonian=ham,
        n_qubits=2,
        ground_state_energy=e0,
        max_depth=10,
        energy_threshold=CHEMICAL_ACCURACY_HA,
        population_size=20,
        n_generations=30,
        mutation_rate=0.3,
        seed=seed,
    )
    # GA doesn't return a fixed depth — use best circuit length
    ga["best_depth"] = len(ga.get("best_circuit", [])) if ga.get("best_circuit") else 0
    ga["wall_time_s"] = round(time.time() - t0, 2)
    results["genetic_algorithm"] = ga
    if verbose:
        print(_fmt_row("genetic algorithm (g=30)", ga, e0))

    # --- hardware-efficient ansatz ---
    if verbose:
        print("Running HE ansatz (2 layers, 5 restarts) ...")
    t0 = time.time()
    he = run_he_ansatz(
        hamiltonian=ham,
        n_qubits=2,
        ground_state_energy=e0,
        n_layers=2,
        energy_threshold=CHEMICAL_ACCURACY_HA,
        cobyla_max_iter=cobyla_iter,
        n_restarts=5,
        seed=seed,
    )
    he["wall_time_s"] = round(time.time() - t0, 2)
    results["he_ansatz_2l"] = he
    if verbose:
        print(_fmt_row("HE ansatz (L=2, 5 restarts)", he, e0))

    if verbose:
        print("-" * 72)
        any_success = any(r["success"] for r in results.values())
        print(f"Chemical accuracy ({CHEMICAL_ACCURACY_HA} Ha): " +
              ("at least one method succeeded." if any_success else "none reached it."))
        print()

    return results


def main():
    parser = argparse.ArgumentParser(description="Run all baselines on H2 VQE")
    parser.add_argument("--seed",  type=int,   default=42)
    parser.add_argument("--dist",  type=float, default=0.735,
                        help="H2 bond distance in Angstrom (one of 0.5, 0.735, 1.0, 1.5, 2.0)")
    parser.add_argument("--out",   type=str,   default=None,
                        help="Path to write JSON output (default: logs/baseline_comparison.json)")
    args = parser.parse_args()

    results = run_all(bond_distance=args.dist, seed=args.seed)

    out_path = args.out or "logs/baseline_comparison.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        # strip non-serializable numpy arrays before dumping
        clean = {}
        for method, r in results.items():
            clean[method] = {
                k: v for k, v in r.items()
                if not hasattr(v, "tolist") and k not in {"best_params", "best_circuit"}
            }
        json.dump(clean, f, indent=2)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
