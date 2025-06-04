import numpy as np
from scipy.stats import linregress
from src.initial_state.new_code.plot_fideliry_vs_iter.main_code import rho_iter15_to_save
from src.y_operator.params import kB, HBAR, OM_small

probs = np.diag(rho_iter15_to_save).real
probs = probs / np.sum(probs)  # Normalize
energies = HBAR * OM_small * (np.arange(len(probs)) + 0.5)

# Filter out near-zero probabilities to avoid log(0)
mask = probs > 1e-10
log_probs = np.log(probs[mask])
E_filtered = energies[mask]

# Linear regression
slope, intercept, _, _, _ = linregress(E_filtered, log_probs)
T_fit = -1 / (kB * slope)
print(f"Fitted temperature: T = {T_fit:.3e} K")