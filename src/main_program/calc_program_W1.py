import numpy as np
from calc_fun import get_rho_expected, get_rho_t, get_fidelity_expected
from tqdm import tqdm

q = 0.0
temperature0 = 0.3e-9
temperature1 = 1e-6
temperature2 = 5e-6
temperature3 = 10e-6

tau_array = np.linspace(50, 250, 25) * 1e-9

rho_S0 = np.zeros((9, 9), dtype=complex)
rho_S0[0, 0] = 1 / 2
rho_S0[4, 4] = 1 / 2
rho_S0[0, 4] = 1 / 2
rho_S0[4, 0] = 1 / 2

fidelity0 = []


for tau in tqdm(tau_array):
    omega = 4.292682 / tau
    rho_expected = get_rho_expected(rho_S0, omega)
    rho_ST0 = get_rho_t(rho_S0, omega, q, temperature0)
    fidelity0.append(abs(np.trace(rho_ST0 @ rho_expected)))

from src.y_operator.params import HBAR, M, OM_small, z_r1, z_r2, Z_ast, w01
parameter_sq = HBAR / 2 / M / OM_small * (1 / 2 / (z_r1 ** 2) + 1 / 2 / (z_r2 ** 2))
parameter_z_ast = HBAR / 2 / M / OM_small / (Z_ast ** 2)
print(parameter_sq)
print(parameter_z_ast)
print(1 - np.amax(fidelity0))
print((1 - np.amax(fidelity0)) / parameter_sq)

from matplotlib import pyplot as plt

plt.figure(figsize=(10, 6))


plt.plot(tau_array * 1e9, 1 - np.array(fidelity0), color='purple', linestyle='dashed', linewidth=2,
         label=r'$T \approx 0 K$')
plt.title('Infidelity vs. Gate Time for Inhomogeneous Spatial Profiles', fontsize=16)
plt.ylabel('Infidelity', fontsize=14)
plt.xlabel(r'$2 \cdot \tau$ (ns)', fontsize=14)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()
