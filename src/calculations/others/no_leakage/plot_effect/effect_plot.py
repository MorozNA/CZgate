from src.calculations.no_leakage.plot_effect.effect_calc import tau_array, fidelity0, fidelity1, fidelity2, fidelity3
from src.calculations.no_leakage.plot_effect.effect_calc import f_expected, expected_fidelity_flag
import numpy as np
from matplotlib import pyplot as plt


plt.figure(figsize=(10, 6))

plt.plot(tau_array * 1e9, 1 - np.array(fidelity0), color='purple', linestyle='dashed', linewidth=2, label=r'$T \approx 0 K$')
plt.plot(tau_array * 1e9, 1 - np.array(fidelity1), color='red', linewidth=2, label=r'$T = 1 \mu K$')
plt.plot(tau_array * 1e9, 1 - np.array(fidelity2), color='green', linewidth=2, label=r'$T = 5 \mu K$')
plt.plot(tau_array * 1e9, 1 - np.array(fidelity3), color='blue', linewidth=2, label=r'$T = 10 \mu K$')
if expected_fidelity_flag:
    plt.plot(tau_array * 1e9, 1 - np.array(f_expected), 'k--', linewidth=2, label='eq. (2.20)')

plt.title('Infidelity vs. Gate Time for Inhomogeneous Spatial Profiles', fontsize=16)
plt.ylabel('Infidelity', fontsize=14)
plt.xlabel(r'$\tau$ (ns)', fontsize=14)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()