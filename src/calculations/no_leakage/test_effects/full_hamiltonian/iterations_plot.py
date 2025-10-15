from src.calculations.no_leakage.test_effects.full_hamiltonian.iterations_calc import fidelity0, fidelity0_decomp, iterations


import numpy as np
from matplotlib import pyplot as plt


plt.figure(figsize=(10, 6))

plt.plot(range(iterations), 1 - np.array(fidelity0), color='purple', linestyle='dashed', linewidth=2, label=r'full \rho')
plt.plot(range(iterations), 1 - np.array(fidelity0_decomp), color='red', linewidth=2, label=r'$decomp. \rho$')

plt.title(r'Infidelity vs. Iteration number for W effect ($T \approx 0$)', fontsize=16)
plt.ylabel('Infidelity', fontsize=14)
plt.xlabel(r'Num. of iter.', fontsize=14)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()