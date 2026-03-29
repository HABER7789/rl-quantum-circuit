# genetic_algorithm.py
# GA baseline for circuit search. Evolves a population of circuits over
# N generations using tournament selection, crossover, and mutation.
# Slower than random search but actually finds decent solutions.

import copy
import numpy as np
from src.environment.vqe_utils import optimize_parameters, evaluate_energy


def _random_gate(n_qubits, rng):
    gate_type = rng.choice(["Rx", "Ry", "Rz", "CNOT"])
    target = int(rng.integers(0, n_qubits))

    if gate_type == "CNOT":
        others = [q for q in range(n_qubits) if q != target]
        control = int(rng.choice(others))
        return {"type": "CNOT", "target": target, "control": control}

    return {"type": gate_type, "target": target, "control": None}


def _evaluate_circuit(gate_sequence, n_qubits, hamiltonian):
    # fewer COBYLA iters than full training to keep this fast
    if len(gate_sequence) == 0:
        return evaluate_energy(n_qubits, [], np.array([]), hamiltonian)
    _, energy = optimize_parameters(n_qubits, gate_sequence, hamiltonian, max_iterations=200)
    return energy


def run_genetic_algorithm(
    hamiltonian,
    n_qubits,
    ground_state_energy,
    max_depth=15,
    energy_threshold=0.0016,
    population_size=20,
    n_generations=50,
    mutation_rate=0.3,
    tournament_k=3,
    seed=42,
):
    rng = np.random.default_rng(seed)

    # random initial population
    population = []
    for _ in range(population_size):
        depth = int(rng.integers(1, max_depth + 1))
        population.append([_random_gate(n_qubits, rng) for _ in range(depth)])

    fitnesses = [_evaluate_circuit(c, n_qubits, hamiltonian) for c in population]

    best_energy = min(fitnesses)
    best_circuit = copy.deepcopy(population[fitnesses.index(best_energy)])
    energy_history = [best_energy]

    for gen in range(n_generations):
        next_gen = []

        while len(next_gen) < population_size:
            # tournament selection — pick k random circuits, keep the best-energy one
            cands1 = rng.choice(population_size, size=tournament_k, replace=False)
            p1 = population[int(cands1[np.argmin([fitnesses[i] for i in cands1])])]

            cands2 = rng.choice(population_size, size=tournament_k, replace=False)
            p2 = population[int(cands2[np.argmin([fitnesses[i] for i in cands2])])]

            # single-point crossover
            if len(p1) > 1 and len(p2) > 1:
                cut1 = int(rng.integers(1, len(p1)))
                cut2 = int(rng.integers(1, len(p2)))
                child = copy.deepcopy(p1[:cut1]) + copy.deepcopy(p2[cut2:])
            else:
                child = copy.deepcopy(p1)

            # randomly mutate
            if rng.random() < mutation_rate and len(child) > 0:
                mut = rng.choice(["replace", "insert", "delete"])
                idx = int(rng.integers(0, len(child)))
                if mut == "replace":
                    child[idx] = _random_gate(n_qubits, rng)
                elif mut == "insert" and len(child) < max_depth:
                    child.insert(idx, _random_gate(n_qubits, rng))
                elif mut == "delete" and len(child) > 1:
                    child.pop(idx)

            next_gen.append(child[:max_depth])

        population = next_gen
        fitnesses = [_evaluate_circuit(c, n_qubits, hamiltonian) for c in population]

        gen_best = min(fitnesses)
        if gen_best < best_energy:
            best_energy = gen_best
            best_circuit = copy.deepcopy(population[fitnesses.index(gen_best)])

        energy_history.append(best_energy)

    energy_error = abs(best_energy - ground_state_energy)
    return {
        "best_energy":    best_energy,
        "best_circuit":   best_circuit,
        "energy_history": energy_history,
        "energy_error":   energy_error,
        "success":        energy_error <= energy_threshold,
    }
