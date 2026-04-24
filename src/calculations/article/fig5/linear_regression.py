import numpy as np
from scipy.stats import linregress
from src.y_operator.params import kB, HBAR, OM_small


def fit_temperature(density_matrix_diagonal):
    energies = HBAR * OM_small * (np.arange(len(density_matrix_diagonal)) + 0.5)

    # Filter out near-zero probabilities to avoid log(0)
    mask = density_matrix_diagonal > 1e-10
    log_probs = np.log(density_matrix_diagonal[mask])
    E_filtered = energies[mask]

    # Linear regression
    slope, intercept, _, _, _ = linregress(E_filtered, log_probs)
    T_fit = -1 / (kB * slope)
    print(T_fit)
    return T_fit