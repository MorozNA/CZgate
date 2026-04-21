import numpy as np
from tqdm import tqdm
from src.y_operator.construct_U0 import construct_U0
from src.y_operator.params import get_params
from src.algorithm.other_tools import get_U0_ideal, exact_evolution, generalized_fidelity


# INITIAL PARAMETERS
iterations = 30
om = 2 * np.pi * 5e6
tau, delta, xi = get_params(om)

# EVOLUTION OPERATORS
U0 = construct_U0(2 * tau, om)
U0_ideal = get_U0_ideal(tau, delta, xi)

# INITIAL STATE
rho_S0 = np.zeros((9, 9), dtype=complex)
rho_S0[4, 4] = 1.0
rho_calc = np.copy(rho_S0)
rho_ideal = np.copy(rho_S0)
# rho_S0[0, 0] = 1 / 2
# rho_S0[4, 4] = 1 / 2
# rho_S0[0, 4] = 1 / 2
# rho_S0[4, 0] = 1 / 2

fidelity = []

for i in tqdm(range(iterations)):
    rho_calc = exact_evolution(rho_calc, U0)
    rho_ideal = exact_evolution(rho_ideal, U0_ideal)
    fidelity.append(generalized_fidelity(rho_calc, rho_ideal))


import numpy as np
from matplotlib import pyplot as plt


plt.figure(figsize=(10, 6))

plt.plot(range(iterations), 1 - np.array(fidelity), color='purple', linestyle='dashed', linewidth=2)

plt.title('Infidelity vs. Iteration number for spin matrices', fontsize=16)
plt.ylabel('Infidelity', fontsize=14)
plt.xlabel(r'Num. of iter.', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()