# metrics.py
# Aggregation helpers used to compare the different methods.

import numpy as np


def energy_error(found, ground):
    return abs(found - ground)


def success_rate(errors, threshold=0.0016):
    # fraction of runs that hit chemical accuracy (1 kcal/mol = 0.0016 Ha)
    return sum(e <= threshold for e in errors) / len(errors)


def approximation_ratio(cut_found, cut_optimal):
    # for MaxCut: 1.0 = perfect, anything above ~0.87 is good
    if cut_optimal <= 0:
        return 0.0
    return cut_found / cut_optimal


def convergence_step(energy_history, ground, threshold=0.0016):
    """First step where we crossed the success threshold. -1 if never."""
    for t, e in enumerate(energy_history):
        if abs(e - ground) <= threshold:
            return t
    return -1


def summarize_runs(energy_errors, depths, threshold=0.0016):
    errs = np.array(energy_errors)
    deps = np.array(depths)
    return {
        "mean_energy_error":   float(errs.mean()),
        "std_energy_error":    float(errs.std()),
        "median_energy_error": float(np.median(errs)),
        "mean_depth":          float(deps.mean()),
        "std_depth":           float(deps.std()),
        "success_rate":        float(success_rate(energy_errors, threshold)),
    }
