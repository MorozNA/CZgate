from src.calculations.delta_R.test_effects.cz_comparison.effect_calc import fidelity0, fidelity1, fidelity5, tau_array

import numpy as np
from matplotlib import pyplot as plt


plt.figure(figsize=(10, 6))

plt.plot(np.array(tau_array) * 1e9, 1 - np.array(fidelity0), color='purple', linestyle='dashed', linewidth=2, label=r'$T \approx 0 K$')
plt.plot(np.array(tau_array) * 1e9, 1 - np.array(fidelity1), color='red', linewidth=2, label=r'$T = 1 \mu K$')
plt.plot(np.array(tau_array) * 1e9, 1 - np.array(fidelity5), color='green', linewidth=2, label=r'$T = 5 \mu K$')

plt.title('Infidelity vs. Tau for W effect', fontsize=16)
plt.ylabel('Infidelity', fontsize=14)
plt.xlabel(r'$\tau$ (ns)', fontsize=14)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()