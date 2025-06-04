import numpy as np
from initial_state import rho_T0
from evolution_fun import get_rho_T_partial, get_rho_S_partial
from evolution_fun import get_rho_s
from src.y_operator.construct_U0 import construct_U0
from src.y_operator.construct_Y import construct_Y_A, construct_Y_B
from src.y_operator.params import get_params

OM = 3.5e6
tau, delta = get_params(OM)

Q = 1.0 * 1e7
t_initial = 0.0
t_final = 2.0 * tau * 10
Y_A = construct_Y_A(t_initial, t_final, OM, Q)
Y_B = construct_Y_B(t_initial, t_final, OM, Q)

U0 = construct_U0(t_final, OM)
rho_S0 = np.zeros((9, 9), dtype=complex)
rho_S0[0, 0] = 1/2
rho_S0[4, 4] = 1/2
rho_S0[0, 4] = 1/2
rho_S0[4, 0] = 1/2
print('shapes are: \n', 'Y_A = ', Y_A.shape, '\n rho_S0 = ', rho_S0.shape, '\n rho_T0 = ', rho_T0.shape)

rho_expected = U0 @ rho_S0 @ U0.conj().T
print('expected matrix: \n', np.round(rho_expected, 3))

### MAIN CODE ###
rho_T_part_A = get_rho_T_partial(rho_S0, rho_T0, Y_A)
rho_T_part_B = get_rho_T_partial(rho_S0, rho_T0, Y_B)
print('________')

rho_S_part_B = get_rho_S_partial(rho_S0, rho_T0, Y_B)
rho_S = get_rho_s(rho_S_part_B, rho_T0, Y_A, U0) # TODO: check whether T0 or part
print(np.trace(rho_S_part_B))
print(np.trace(rho_S))
rho_S = rho_S / np.trace(rho_S)
# rho_S = U0 @ rho_S @ U0.conj().T
# print('YA', np.amax(Y_A @ Y_A.conj().T - np.eye(len(Y_A))))
print('final matrix: \n', np.round(rho_S, 3))

print('\n')

print(np.trace(rho_S @ rho_expected))
print(abs(np.trace(rho_S @ rho_expected)))

from src.y_operator.params import HBAR, OM_small, M
p = np.sqrt(HBAR * OM_small * M / 2) * 1
alpha = (HBAR * Q + p) / M * tau * 2 * 1e6 * 10
print(alpha)
F_expected = 1 / 2 * (1 + np.exp(-2 * abs(alpha) ** 2 / 2))
print(F_expected)