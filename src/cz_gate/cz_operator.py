import numpy as np
from calc_tools import get_U, calc_tau, calc_xi, calc_delta


delta_R = 2 * np.pi * 30e6
om = 2 * np.pi * 10e6
delta = calc_delta(om, delta_R)
tau = calc_tau(delta, om, delta_R)
xi2 = calc_xi(delta, om, tau)
print('xi2 + pi =', xi2 + np.pi)

# xi2 = 2.380762 # DOI: https://doi.org/10.1103/PhysRevA.106.042410
# xi2 = 3.90242 # DOI: https://doi.org/10.1103/PhysRevLett.123.170503

U1 = np.zeros((9, 9), dtype=complex)
U2 = np.zeros((9, 9), dtype=complex)

delta_renorm = delta - om ** 2 / 2 / delta_R

U1_1block = 1
U1_2block = get_U(delta, om, 0.0, tau)
U1_3block = U1_2block
U1_4block = get_U(delta_renorm, np.sqrt(2) * om, 0.0, tau)
U1_5block = np.eye(2, dtype=complex)

U2_1block = 1
U2_2block = get_U(delta, om, xi2, tau)
U2_3block = U2_2block
U2_4block = get_U(delta_renorm, np.sqrt(2) * om, xi2, tau)
U2_5block = np.eye(2, dtype=complex)

U1[0, 0] = U1_1block
U1[1:3, 1:3] = U1_2block
U1[3:5, 3:5] = U1_3block
U1[5:7, 5:7] = U1_4block
U1[7:9, 7:9] = U1_5block

U2[0, 0] = U2_1block
U2[1:3, 1:3] = U2_2block
U2[3:5, 3:5] = U2_3block
U2[5:7, 5:7] = U2_4block
U2[7:9, 7:9] = U2_5block

U_final = U2 @ U1
CZ_final = np.diag([U_final[0, 0], U_final[1, 1], U_final[3, 3], U_final[5, 5]])

phi2 = -delta_renorm * tau
phi1 = (phi2 - np.pi) / 2
CZ_expected = np.diag([1, np.exp(1j * phi1), np.exp(1j * phi1), np.exp(1j * phi2)])

print('check norm [3,3]: ', abs(CZ_final[3, 3]))
print('check norm [2,2]: ', abs(CZ_final[2, 2]))
phi1_new = np.angle(CZ_final[1, 1])
phi2_new = np.angle(CZ_final[3, 3])
phi2_expected = 2 * phi1_new + np.pi
print('difference = ', np.round(phi2_new - phi2_expected, 3))
print(np.round(CZ_final, 3))
print('\n')
print(np.round(CZ_expected, 3))

initial_state = np.zeros((9, 1), dtype=complex)
initial_state[0] = 1/np.sqrt(2)
initial_state[5] = 1/np.sqrt(2)
# rho_S0[4, 4] = 1.0
final_state = U_final @ initial_state
final_state_expected = initial_state
final_state_expected[5] = initial_state[5] * np.exp(1j * phi2)
print('\n')
print('Final state: ', '\n', np.round(final_state, 3))
print(np.diag(CZ_expected)[1])
print('Final state norm: ', np.sum(abs(final_state) ** 2), '\n')
print('Fidelity = ', (abs(final_state.conj().T @ final_state_expected) ** 2).item())
