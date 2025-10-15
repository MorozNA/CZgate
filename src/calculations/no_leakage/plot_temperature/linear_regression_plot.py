from src.initial_state.new_code.plot_temperature.linear_regression import energies, probs, T_fit, kB, HBAR, OM_small
import numpy as np


def gibbs_probabilities(T, energies):
    """Returns Gibbs probabilities for given T and energies."""
    E_normalized = energies / (kB * T)
    probs = np.exp(-E_normalized)
    return probs / np.sum(probs)


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.style.use('seaborn')

# x = energies
x = np.arange(len(probs))

# Original distribution (rho_iter1)
plt.bar(
    x, probs,
    width=0.1 * HBAR * OM_small,
    color='skyblue', edgecolor='navy',
    alpha=0.6, label='Original (rho_iter1)'
)

# Best-fit Gibbs ensemble
gibbs_probs = gibbs_probabilities(T_fit, energies)
plt.plot(
    x, gibbs_probs,
    'o-', color='red', linewidth=2,
    markersize=6, label=f'Gibbs Fit (T = {T_fit:.2e} K)'
)

plt.xlim((x[0], x[14]))
plt.xlabel('Energy Levels (J)', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Comparison with Gibbs Ensemble', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()