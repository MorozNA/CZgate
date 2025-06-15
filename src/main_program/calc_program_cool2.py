import numpy as np
from calc_fun import get_rho_expected, get_rho_t, get_fidelity_expected
from tqdm import tqdm


q = 5e6
temperature0 = 0.3e-9
temperature1 = 1e-6
temperature2 = 5e-6
temperature3 = 10e-6

tau_array = np.linspace(50, 250, 10) * 1e-9

rho_S0 = np.zeros((9, 9), dtype=complex)
rho_S0[0, 0] = 1 / 2
rho_S0[4, 4] = 1 / 2
rho_S0[0, 4] = 1 / 2
rho_S0[4, 0] = 1 / 2

fidelity0 = []
fidelity1 = []
fidelity2 = []
fidelity3 = []
f_expected = []
y_unitarity = []
for tau in tqdm(tau_array):
    omega = 4.292682 / tau
    rho_expected = get_rho_expected(rho_S0, omega)
    rho_temp0 = get_rho_t(rho_S0, omega, q, temperature0)
    rho_S1 = get_rho_t(rho_S0, omega, q, temperature1)

    rho_S2 = get_rho_t(rho_S0, omega, q, temperature2)
    rho_S3 = get_rho_t(rho_S0, omega, q, temperature3)

    fidelity0.append(abs(np.trace(rho_temp0 @ rho_expected)))
    fidelity1.append(abs(np.trace(rho_S1 @ rho_expected)))
    fidelity2.append(abs(np.trace(rho_S2 @ rho_expected)))
    fidelity3.append(abs(np.trace(rho_S3 @ rho_expected)))
    f_expected.append(get_fidelity_expected(tau, q))


from matplotlib import pyplot as plt

plt.figure(figsize=(10, 6))

plt.plot(tau_array * 1e9, 1 - np.array(fidelity0), color='purple', linestyle='dashed', linewidth=2, label=r'$T \approx 0 K$')
plt.plot(tau_array * 1e9, 1 - np.array(fidelity1), color='red', linewidth=2, label='$T = 1 \mu K$')
plt.plot(tau_array * 1e9, 1 - np.array(fidelity2), color='green', linewidth=2, label='$T = 5 \mu K$')
plt.plot(tau_array * 1e9, 1 - np.array(fidelity3), color='blue', linewidth=2, label='$T = 10 \mu K$')
plt.plot(tau_array * 1e9, 1 - np.array(f_expected), 'k--', linewidth=2, label='eq. (2.20)')

plt.title('Infidelity vs. Gate Time for Homogeneous Spatial Profiles', fontsize=16)
plt.ylabel('Infidelity', fontsize=14)
plt.xlabel(r'$2 \cdot \tau$ (ns)', fontsize=14)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()