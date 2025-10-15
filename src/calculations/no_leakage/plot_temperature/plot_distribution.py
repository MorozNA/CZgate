import numpy as np
import matplotlib.pyplot as plt
from src.initial_state.initial_state import get_rho_T0
from src.y_operator.params import n, HBAR, OM_small
from src.initial_state.new_code.plot_temperature.approximate_function import best_T



T = 5e-6

# Get the density matrix
rho_T0 = get_rho_T0(T)
rho_fitted = get_rho_T0(best_T)

# Extract diagonal elements (probabilities)
probabilities = np.diag(rho_T0).real  # Ensure real part (should already be real)

# Energy levels for the x-axis
energy_levels = HBAR * OM_small * (np.arange(n) + 0.5)

# Plotting
plt.figure(figsize=(10, 6))
plt.style.use('seaborn')  # Beautiful style

# Bar plot (for discrete energy levels)
plt.bar(energy_levels, probabilities, width=0.1 * HBAR * OM_small,
        color='skyblue', edgecolor='navy', alpha=0.7, label=f'T = {T} K')

# Alternatively, smooth curve (for visualization)
# plt.plot(energy_levels, probabilities, 'o-', color='royalblue', linewidth=2, markersize=5, label=f'T = {T} K')

plt.xlabel('Energy Levels (J)', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Gibbs Ensemble Distribution', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()