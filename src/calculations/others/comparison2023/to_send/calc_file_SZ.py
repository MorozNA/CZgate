import numpy as np
from hamiltonian_SZ import get_evolution


initial_state = np.zeros((9, 1), dtype=complex)

initial_state[0] = 1.0 / 2
initial_state[1] = 1.0 / 2
initial_state[3] = 1.0 / 2
initial_state[5] = 1.0 / 2


delta_rydberg = 2 * np.pi * 30 * 1e6
om_array = np.loadtxt('data/om_array.csv', delimiter=',')
delta_to_om_arr = np.loadtxt('data/delta_to_om_arr.csv', delimiter=',')
tau_times_omega_arr = np.loadtxt('data/tau_times_omega_arr.csv', delimiter=',')
xi_arr = np.loadtxt('data/xi_arr.csv', delimiter=',')
xi_arr = - xi_arr

fidelities = []

for i in range(len(om_array)):
    om = om_array[i]
    delta = delta_to_om_arr[i] * om
    tau = tau_times_omega_arr[i] / om
    xi = xi_arr[i]

    U0_with_leakage = get_evolution(om, delta, xi, delta_rydberg, 2 * tau)

    delta_renorm = delta + (om ** 2) / (2 * delta_rydberg)
    phi2 = delta_renorm * tau
    phi1 = (phi2 + np.pi) / 2
    CZ_expected = np.diag([1, np.exp(1j * phi1), 1, np.exp(1j * phi1), 1, np.exp(1j * phi2), 1, 1, 1])

    state_with_leakage = U0_with_leakage @ initial_state
    state_exact = CZ_expected @ initial_state

    fidelity = np.abs(np.vdot(state_exact, state_with_leakage)) ** 2
    fidelities.append(fidelity)
