# H2 molecule and MaxCut Hamiltonians.
# Note: Qiskit is little-endian. In "ZII", qubit 0 is the rightmost char.

import numpy as np
import networkx as nx
from qiskit.quantum_info import SparsePauliOp


# 2-qubit H2 Pauli coefficients at 5 bond lengths (Angstroms).
# Uses the parity-reduced form from O'Malley et al. 2016 (Phys Rev X).
# H = g0*II + g1*ZI + g2*IZ + g3*ZZ + g4*XX + g5*YY
H2_COEFFICIENTS_2Q = {
    0.5: {
        "II": -0.81054798,
        "ZI":  0.17218393,
        "IZ": -0.22575349,
        "ZZ":  0.12091263,
        "XX":  0.17218393,
        "YY":  0.17218393,
    },
    # equilibrium ~0.74 Å, E_0 ≈ -1.8572 Ha
    0.735: {
        "II": -1.05237325,
        "ZI":  0.39793742,
        "IZ": -0.39793742,
        "ZZ": -0.01128010,
        "XX":  0.18093119,
        "YY":  0.18093119,
    },
    1.0: {
        "II": -0.98863557,
        "ZI":  0.33677334,
        "IZ": -0.33677334,
        "ZZ": -0.08668427,
        "XX":  0.17011978,
        "YY":  0.17011978,
    },
    1.5: {
        "II": -0.74959290,
        "ZI":  0.23814406,
        "IZ": -0.23814406,
        "ZZ": -0.22279118,
        "XX":  0.12159850,
        "YY":  0.12159850,
    },
    2.0: {
        "II": -0.55680827,
        "ZI":  0.16894626,
        "IZ": -0.16894626,
        "ZZ": -0.27220892,
        "XX":  0.08071781,
        "YY":  0.08071781,
    },
}

# Exact ground state energies for our 2-qubit parity-reduced Hamiltonian.
# These are the lowest eigenvalues of the matrices we actually encode above,
# computed by np.linalg.eigvalsh on each SparsePauliOp.to_matrix().
#
# Note: the −1.8572 Ha figure that appears in O'Malley et al. (2016) and
# related VQE papers is the full-CI ground state energy for H2 in a
# minimal STO-3G basis. Our Pauli coefficients go through the parity
# mapping and a two-qubit reduction, which shifts the zero of energy;
# the lowest eigenvalue of our encoded 2Q Hamiltonian is therefore NOT
# equal to the full-CI energy. Using the wrong reference would cause
# VQE to report failure even when it converges correctly.
H2_GROUND_STATE_2Q = {
    0.5:   -1.4577,
    0.735: -1.9154,
    1.0:   -1.6566,
    1.5:   -1.0616,
    2.0:   -0.8290,
}

# 1 kcal/mol in Hartree — the standard "good enough" threshold in quantum chemistry
CHEMICAL_ACCURACY_HA = 0.0016


def build_h2_hamiltonian_2q(bond_distance=0.735):
    """Build the 2-qubit H2 Hamiltonian. Only works at the 5 hardcoded distances."""
    supported = sorted(H2_COEFFICIENTS_2Q.keys())
    nearest = min(supported, key=lambda d: abs(d - bond_distance))

    if abs(nearest - bond_distance) > 0.05:
        raise ValueError(
            f"Bond distance {bond_distance} Å not supported. Options: {supported}"
        )

    coeffs = H2_COEFFICIENTS_2Q[nearest]
    hamiltonian = SparsePauliOp.from_list(list(coeffs.items()))
    return hamiltonian, H2_GROUND_STATE_2Q[nearest]


def build_h2_hamiltonian_4q(bond_distance=0.735):
    """4-qubit H2. Tries PySCF first, falls back to hardcoded if not installed."""
    try:
        return _build_h2_4q_via_pyscf(bond_distance)
    except ImportError:
        pass

    if abs(bond_distance - 0.735) > 0.1:
        raise ImportError(
            "PySCF not installed and hardcoded 4Q H2 only works near R=0.735 Å."
        )

    # from Kandala et al. 2017 — keeping this around in case we need it
    pauli_list = [
        ("IIII",  -0.81054798),
        ("IIIZ",  +0.17218393),
        ("IIZZ",  -0.22575349),
        ("IZZI",  +0.12091263),
        ("IZZZ",  +0.17218393),
        ("ZIII",  -0.22575349),
        ("ZIIZ",  +0.16614543),
        ("ZIZI",  +0.17218393),
        ("ZZII",  +0.12091263),
        ("IYYY",  +0.04523279),
        ("XXYY",  +0.04523279),
        ("XYXY",  +0.04523279),
        ("YXXY",  +0.04523279),
        ("YYYY",  +0.04523279),
        ("YYXX",  +0.04523279),
    ]
    hamiltonian = SparsePauliOp.from_list(pauli_list)
    return hamiltonian, -1.8572


def _build_h2_4q_via_pyscf(bond_distance):
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    half_dist = bond_distance / 2.0
    driver = PySCFDriver(
        atom=f"H 0 0 {half_dist}; H 0 0 -{half_dist}",
        basis="sto3g",
        charge=0,
        spin=0,
    )
    problem = driver.run()
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(problem.second_q_ops()[0])

    mat = qubit_op.to_matrix()
    e0 = float(np.linalg.eigvalsh(mat)[0])
    return qubit_op, e0


def build_maxcut_hamiltonian(graph):
    """
    Ising Hamiltonian for MaxCut:
        H = (1/2) * sum_{(i,j)} Z_i Z_j  -  (|E|/2) * I

    Ground state energy = -MaxCut, so minimizing H solves MaxCut.
    Brute force for the exact cut value (only feasible up to ~20 nodes).
    """
    n = graph.number_of_nodes()
    if n > 20:
        raise ValueError(f"Graph too large ({n} nodes) for brute force.")

    graph = nx.convert_node_labels_to_integers(graph)
    edges = list(graph.edges())
    num_edges = len(edges)

    if num_edges == 0:
        return SparsePauliOp.from_list([("I" * n, 0.0)]), 0

    terms = [("I" * n, -num_edges / 2.0)]
    for (i, j) in edges:
        terms.append((_make_zz_string(n, i, j), 0.5))

    hamiltonian = SparsePauliOp.from_list(terms)
    optimal_cut = _brute_force_maxcut(graph)
    return hamiltonian, float(optimal_cut)


def _make_zz_string(n_qubits, qubit_i, qubit_j):
    # little-endian: qubit k → position (n-1-k) in the string
    chars = ["I"] * n_qubits
    chars[n_qubits - 1 - qubit_i] = "Z"
    chars[n_qubits - 1 - qubit_j] = "Z"
    return "".join(chars)


def _brute_force_maxcut(graph):
    n = graph.number_of_nodes()
    best_cut = 0
    for mask in range(1 << n):
        cut = sum(
            1 for (i, j) in graph.edges()
            if ((mask >> i) & 1) != ((mask >> j) & 1)
        )
        best_cut = max(best_cut, cut)
    return best_cut


def exact_ground_state_energy(hamiltonian):
    """Diagonalize and return the smallest eigenvalue. Only practical for small systems."""
    mat = hamiltonian.to_matrix()
    return float(np.linalg.eigvalsh(mat)[0])


def get_benchmark_graph(name):
    """C4, K4, W4, P3, K33 — standard test graphs for MaxCut experiments."""
    graphs = {
        "C4":  nx.cycle_graph(4),
        "K4":  nx.complete_graph(4),
        "W4":  nx.wheel_graph(5),
        "P3":  nx.path_graph(3),
        "K33": nx.complete_bipartite_graph(3, 3),
    }
    if name not in graphs:
        raise ValueError(f"Unknown graph '{name}'. Options: {list(graphs)}")
    return graphs[name]
