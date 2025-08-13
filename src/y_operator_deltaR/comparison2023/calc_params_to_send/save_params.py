import numpy as np
from src.y_operator_deltaR.calc_params import get_delta_renorm
from src.y_operator_deltaR.params import get_params
from src.y_operator_deltaR.comparison2023.calc_params_to_send.hamiltonian import get_evolution


initial_state = np.zeros((9, 1), dtype=complex)

initial_state[0] = 0.5
initial_state[1] = 0.5
initial_state[3] = 0.5
initial_state[5] = 0.5


delta_R = 2 * np.pi * 30 * 1e6
om_array = 2 * np.pi * np.linspace(3, 10, 100) * 1e6

delta_to_om_arr = []
tau_times_omega_arr = []
xi_arr = []

fidelities = []
cz_time = []

for om in om_array:
    tau, delta, xi = get_params(om, delta_R)
    U0_with_leakage = get_evolution(om, delta, xi, delta_R, tau)

    delta_renorm = get_delta_renorm(delta, om, delta_R)
    phi2 = delta_renorm * tau
    phi1 = (phi2 + np.pi) / 2
    CZ_expected = np.diag([1, np.exp(1j * phi1), 1, np.exp(1j * phi1), 1, np.exp(1j * phi2), 1, 1, 1])

    state_with_leakage = U0_with_leakage @ initial_state
    state_exact = CZ_expected @ initial_state

    fidelity = np.abs(np.vdot(state_exact, state_with_leakage)) ** 2
    fidelities.append(fidelity)
    cz_time.append(2 * tau)

    print(r'DATA FOR $\delta_R$ = ', delta_R)
    print('delta / omega = ', delta / om)
    print('tau * omega = ', tau * om)
    print('xi = ', xi)
    print('fidelity = ', fidelity)

    delta_to_om_arr.append(delta / om)
    tau_times_omega_arr.append(tau * om)
    xi_arr.append(xi)

    phase_exact_0 = np.angle(state_exact[0])
    phase_approx_0 = np.angle(state_with_leakage[0])
    phase_error_0 = np.abs(phase_exact_0 - phase_approx_0)
    phase_exact_1 = np.angle(state_exact[1])
    phase_approx_1 = np.angle(state_with_leakage[1])
    phase_error_1 = np.abs(phase_exact_1 - phase_approx_1)
    phase_exact_3 = np.angle(state_exact[3])
    phase_approx_3 = np.angle(state_with_leakage[3])
    phase_error_3 = np.abs(phase_exact_3 - phase_approx_3)
    phase_exact_5 = np.angle(state_exact[5])
    phase_approx_5 = np.angle(state_with_leakage[5])
    phase_error_5 = np.abs(phase_exact_5 - phase_approx_5)
    print('phase error 0: ', phase_error_0)
    print('phase error 1: ', phase_error_1)
    print('phase error 3: ', phase_error_3)
    print('phase error 4: ', phase_error_5)
    # phi1 = (phi2 - np.pi) / 2
    print('phi1_approx = ', phase_approx_1)
    print('phi1_approx = ', phase_approx_3)
    print('(phi2_approx - pi) / 2 = ', (phase_approx_5 - np.pi) / 2)
    print('\n')
    print('phi1_exact = ', phase_exact_1)
    print('phi1_exact = ', phase_exact_3)
    print('(phi2_exact - pi) / 2 = ', (phase_exact_5 - np.pi) / 2)
    print('______________________________________________________', '\n')



# Save each array as a CSV file
np.savetxt('data/om_array.csv', om_array, delimiter=',')
np.savetxt('data/delta_to_om_arr.csv', delta_to_om_arr, delimiter=',')
np.savetxt('data/tau_times_omega_arr.csv', tau_times_omega_arr, delimiter=',')
np.savetxt('data/xi_arr.csv', xi_arr, delimiter=',')