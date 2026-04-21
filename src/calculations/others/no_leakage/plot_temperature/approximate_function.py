import numpy as np
from scipy.stats import linregress
from src.y_operator.params import kB, HBAR, OM_small


# Function to extract temperature from a density matrix
def get_temperature(rho):
    probs = np.diag(rho).real
    probs = probs / np.sum(probs)  # Normalize
    energies = HBAR * OM_small * (np.arange(len(probs)) + 0.5)

    mask = probs > 1e-10
    # if np.sum(mask) < 2:  # Need at least 2 points for linear regression
    #     return np.nan

    log_probs = np.log(probs[mask])
    E_filtered = energies[mask]

    slope, intercept, _, _, _ = linregress(E_filtered, log_probs)
    return -1 / (kB * slope)

